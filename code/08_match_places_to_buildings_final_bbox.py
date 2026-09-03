import os
import json
import math
import random
import time

import duckdb
import pandas as pd
import folium

COUNTRY = "CA"
SAMPLE_SIZE = 1000
MIN_CONFIDENCE = 0.33
TILE_SIZE = 0.05
SEARCH_MARGIN = 0.002
MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 3

OUTPUT_DIR = "/Users/stephs/Desktop"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "overture_canada_1000_final_bbox.csv")
OUTPUT_HTML = os.path.join(OUTPUT_DIR, "overture_canada_final_bbox_test.html")
TEMP_DIR = os.path.join(OUTPUT_DIR, "duckdb_tmp")
os.makedirs(TEMP_DIR, exist_ok=True)


def bbox_text(xmin, ymin, xmax, ymax):
    values = [xmin, ymin, xmax, ymax]
    if any(pd.isna(v) for v in values):
        return None
    return f"[{float(xmin)}, {float(ymin)}, {float(xmax)}, {float(ymax)}]"


def parse_bbox_string(value):
    return [float(v.strip()) for v in str(value).replace("[", "").replace("]", "").split(",")]


# ============================================================
# PART 1 — CONNECT TO DUCKDB / OVERTURE
# ============================================================
db = duckdb.connect()
db.execute(f"SET temp_directory='{TEMP_DIR}';")

for extension in ("spatial", "httpfs"):
    try:
        db.execute(f"INSTALL {extension};")
    except Exception:
        pass
    db.execute(f"LOAD {extension};")

db.execute("SET s3_region='us-west-2';")

latest = db.execute("""
    SELECT latest
    FROM read_json('https://stac.overturemaps.org/catalog.json')
""").fetchone()[0]

print("Latest Overture release:", latest)

PLACES_PATH = (
    f"s3://overturemaps-us-west-2/release/{latest}/"
    "theme=places/type=place/*"
)
BUILDINGS_PATH = (
    f"s3://overturemaps-us-west-2/release/{latest}/"
    "theme=buildings/type=building/*"
)


# ============================================================
# PART 2 — EXTRACT 1,000 CANADIAN PLACE ROWS
# ============================================================
print("\nSTEP 1: Loading 1,000 Canadian Overture Places...")

places_df = db.execute(
    f"""
    SELECT
        id AS place_id,
        names.primary AS location_name,
        addresses[1].freeform AS address,
        ST_Y(geometry) AS latitude,
        ST_X(geometry) AS longitude,
        bbox.xmin AS place_xmin,
        bbox.xmax AS place_xmax,
        bbox.ymin AS place_ymin,
        bbox.ymax AS place_ymax
    FROM read_parquet(?, hive_partitioning = 1)
    WHERE addresses[1].country = ?
      AND confidence > ?
      AND names.primary IS NOT NULL
      AND geometry IS NOT NULL
    LIMIT {SAMPLE_SIZE}
    """,
    [PLACES_PATH, COUNTRY, MIN_CONFIDENCE]
).df()

print("Places loaded:", len(places_df))
if places_df.empty:
    db.close()
    raise RuntimeError("No Canadian Place rows were loaded.")

places_df["original_place_bbox"] = places_df.apply(
    lambda row: bbox_text(
        row["place_xmin"], row["place_ymin"],
        row["place_xmax"], row["place_ymax"]
    ),
    axis=1
)


# ============================================================
# PART 3 — GROUP PLACES INTO SMALL GEOGRAPHIC TILES
# ============================================================
places_df["tile_x"] = (places_df["longitude"] / TILE_SIZE).apply(math.floor)
places_df["tile_y"] = (places_df["latitude"] / TILE_SIZE).apply(math.floor)
tile_groups = list(places_df.groupby(["tile_x", "tile_y"], sort=False))

print("\nSTEP 2: Geographic tiles to process:", len(tile_groups))

matches_by_place_id = {}
successful_tiles = 0
failed_tiles = 0


# ============================================================
# PART 4 — QUERY BUILDINGS TILE BY TILE AND MATCH LOCALLY
# 1) remote query only nearby Buildings for a small tile
# 2) local bbox filter
# 3) find candidates: Place point inside of both building bbox and polygon
# 4) if multiple matches, choose smallest polygon
# ============================================================
print("\nSTEP 3: Matching Places to Overture Buildings...")

for tile_number, ((tile_x, tile_y), tile_places) in enumerate(tile_groups, start=1):
    west = float(tile_places["longitude"].min()) - SEARCH_MARGIN
    east = float(tile_places["longitude"].max()) + SEARCH_MARGIN
    south = float(tile_places["latitude"].min()) - SEARCH_MARGIN
    north = float(tile_places["latitude"].max()) + SEARCH_MARGIN

    print(f"\nTile {tile_number}/{len(tile_groups)} - Places: {len(tile_places)}")

    buildings_df = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            buildings_df = db.execute(
                """
                SELECT
                    id AS building_id,
                    bbox.xmin AS building_xmin,
                    bbox.xmax AS building_xmax,
                    bbox.ymin AS building_ymin,
                    bbox.ymax AS building_ymax,
                    ST_AsWKB(geometry) AS building_wkb
                FROM read_parquet(?, hive_partitioning = 1)
                WHERE bbox.xmax >= ?
                  AND bbox.xmin <= ?
                  AND bbox.ymax >= ?
                  AND bbox.ymin <= ?
                  AND geometry IS NOT NULL
                """,
                [BUILDINGS_PATH, west, east, south, north]
            ).df()
            break
        except Exception as error:
            print(f"  Buildings query attempt {attempt}/{MAX_RETRIES} failed:")
            print(" ", error)
            if attempt < MAX_RETRIES:
                print(f"  Waiting {RETRY_WAIT_SECONDS} seconds and retrying...")
                time.sleep(RETRY_WAIT_SECONDS)

    if buildings_df is None:
        failed_tiles += 1
        print("  Tile skipped after repeated remote query failures.")
        continue

    print("  Candidate Buildings loaded:", len(buildings_df))
    if buildings_df.empty:
        successful_tiles += 1
        continue

    tile_places_local = tile_places[["place_id", "location_name", "latitude", "longitude"]].copy()
    db.register("tile_places_df", tile_places_local)
    db.register("tile_buildings_df", buildings_df)

    local_matches = db.execute(
        """
        WITH candidates AS (
            SELECT
                p.place_id,
                b.building_id,
                b.building_xmin,
                b.building_xmax,
                b.building_ymin,
                b.building_ymax,
                ST_GeomFromWKB(b.building_wkb) AS building_geometry,
                ST_Area_Spheroid(ST_GeomFromWKB(b.building_wkb)) AS building_area_m2
            FROM tile_places_df p
            JOIN tile_buildings_df b
              ON p.longitude >= b.building_xmin
             AND p.longitude <= b.building_xmax
             AND p.latitude >= b.building_ymin
             AND p.latitude <= b.building_ymax
             AND ST_Covers(
                    ST_GeomFromWKB(b.building_wkb),
                    ST_Point(p.longitude, p.latitude)
                 )
        ), ranked AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY place_id
                       ORDER BY building_area_m2 ASC
                   ) AS building_rank
            FROM candidates
        )
        SELECT
            place_id,
            building_id,
            building_xmin,
            building_ymin,
            building_xmax,
            building_ymax,
            building_area_m2,
            ST_AsGeoJSON(building_geometry) AS building_geojson
        FROM ranked
        WHERE building_rank = 1
        """
    ).df()

    print("  Places matched to a Building:", len(local_matches))

    for row in local_matches.itertuples(index=False):
        matches_by_place_id[row.place_id] = {
            "building_id": row.building_id,
            "building_xmin": float(row.building_xmin),
            "building_ymin": float(row.building_ymin),
            "building_xmax": float(row.building_xmax),
            "building_ymax": float(row.building_ymax),
            "building_area_m2": float(row.building_area_m2),
            "building_geojson": row.building_geojson,
        }

    successful_tiles += 1

    try:
        db.unregister("tile_places_df")
    except Exception:
        pass
    try:
        db.unregister("tile_buildings_df")
    except Exception:
        pass


# ============================================================
# PART 5 — CREATE FINAL CSV
# matched row -> final_bbox = Building bbox
# unmatched row -> final_bbox = original Place bbox
# ============================================================
print("\nSTEP 4: Creating the final CSV...")

output_rows = []

for row in places_df.itertuples(index=False):
    building_match = matches_by_place_id.get(row.place_id)

    if building_match is not None:
        building_bbox = bbox_text(
            building_match["building_xmin"],
            building_match["building_ymin"],
            building_match["building_xmax"],
            building_match["building_ymax"],
        )
        final_bbox = building_bbox
        bbox_method = "building"
        building_id = building_match["building_id"]
        building_area_m2 = building_match["building_area_m2"]
    else:
        building_bbox = None
        final_bbox = row.original_place_bbox
        bbox_method = "original_place"
        building_id = None
        building_area_m2 = None

    output_rows.append({
        "place_id": row.place_id,
        "location_name": row.location_name,
        "address": row.address,
        "latitude": row.latitude,
        "longitude": row.longitude,
        "original_place_bbox": row.original_place_bbox,
        "building_id": building_id,
        "building_bbox": building_bbox,
        "final_bbox": final_bbox,
        "bbox_method": bbox_method,
        "building_area_m2": building_area_m2,
    })

result_df = pd.DataFrame(output_rows)
result_df.to_csv(OUTPUT_CSV, index=False)

print("\nCSV successfully saved:")
print(OUTPUT_CSV)

matched_count = result_df["bbox_method"].eq("building").sum()
unmatched_count = result_df["bbox_method"].eq("original_place").sum()

print("\n========================================")
print("RESULT SUMMARY")
print("========================================")
print("Total Place rows:", len(result_df))
print("Rows matched to Buildings:", matched_count)
print("Rows keeping original Place bbox:", unmatched_count)
print("Successful geographic tiles:", successful_tiles)
print("Failed geographic tiles:", failed_tiles)
if len(result_df) > 0:
    print("Building-match percentage:", round(100 * matched_count / len(result_df), 2), "%")


# ============================================================
# PART 6 — RANDOMLY SELECT ONE MATCHED ROW AND MAKE HTML
# compare satellite imagery + old bbox + Building polygon +
# final Building-derived bbox.
# ============================================================
matched_result_df = result_df[result_df["bbox_method"] == "building"].copy()

if matched_result_df.empty:
    print("\nNo Building-matched row exists, so the HTML map cannot be created.")
    db.close()
    raise SystemExit()

random_row = matched_result_df.sample(
    n=1,
    random_state=random.randint(1, 1_000_000)
).iloc[0]

test_place_id = random_row["place_id"]
test_name = random_row["location_name"]
test_latitude = float(random_row["latitude"])
test_longitude = float(random_row["longitude"])

building_match = matches_by_place_id[test_place_id]
building_geojson = json.loads(building_match["building_geojson"])

place_xmin, place_ymin, place_xmax, place_ymax = parse_bbox_string(
    random_row["original_place_bbox"]
)
building_xmin, building_ymin, building_xmax, building_ymax = parse_bbox_string(
    random_row["building_bbox"]
)

print("\nSTEP 5: Creating HTML test map...")

store_map = folium.Map(
    location=[test_latitude, test_longitude],
    zoom_start=20,
    tiles=None,
    control_scale=True,
)

folium.TileLayer(
    tiles=(
        "https://server.arcgisonline.com/"
        "ArcGIS/rest/services/"
        "World_Imagery/MapServer/"
        "tile/{z}/{y}/{x}"
    ),
    attr="Esri World Imagery",
    name="Satellite imagery",
    overlay=False,
    control=True,
    max_zoom=23,
    max_native_zoom=18,
).add_to(store_map)

# Red = original Place bbox
folium.Rectangle(
    bounds=[[place_ymin, place_xmin], [place_ymax, place_xmax]],
    tooltip="Original Overture Places bbox",
    popup=f"<b>Original Places bbox</b><br>{random_row['original_place_bbox']}",
    color="red",
    weight=5,
    fill=False,
).add_to(store_map)

# Blue = actual Building polygon
folium.GeoJson(
    building_geojson,
    name="Overture Building Polygon",
    tooltip="Overture Building footprint",
    style_function=lambda feature: {
        "color": "blue",
        "weight": 4,
        "fill": True,
        "fillOpacity": 0.12,
    },
).add_to(store_map)

# Green = final Building-derived bbox
folium.Rectangle(
    bounds=[[building_ymin, building_xmin], [building_ymax, building_xmax]],
    tooltip="Final Building-derived bbox",
    popup=(
        f"<b>Building-derived bbox</b><br>{random_row['building_bbox']}<br>"
        f"Building area: {random_row['building_area_m2']:.2f} m²"
    ),
    color="green",
    weight=5,
    fill=False,
).add_to(store_map)

# Blue point = original Place coordinate
folium.CircleMarker(
    location=[test_latitude, test_longitude],
    radius=7,
    tooltip=f"Overture Place: {test_name}",
    popup=(
        f"<b>{test_name}</b><br>"
        f"Latitude: {test_latitude}<br>"
        f"Longitude: {test_longitude}<br>"
        f"bbox method: {random_row['bbox_method']}"
    ),
    color="white",
    weight=2,
    fill=True,
    fill_color="#0066ff",
    fill_opacity=1,
).add_to(store_map)

legend_html = f"""
<div style="
    position: fixed;
    top: 10px;
    left: 50px;
    z-index: 9999;
    background-color: white;
    border: 2px solid gray;
    border-radius: 6px;
    padding: 10px;
    font-size: 14px;
    max-width: 430px;
">
<b>Building bbox validation: {test_name}</b><br><br>
<span style="color:#0066ff;">●</span> Blue point = Overture Place coordinate<br>
<span style="color:red;">■</span> Red = original Places bbox<br>
<span style="color:blue;">■</span> Blue outline = Overture Building polygon<br>
<span style="color:green;">■</span> Green = final Building-derived bbox<br><br>
Building area: {random_row['building_area_m2']:.2f} m²
</div>
"""
store_map.get_root().html.add_child(folium.Element(legend_html))
folium.LayerControl().add_to(store_map)
store_map.fit_bounds(
    [[building_ymin, building_xmin], [building_ymax, building_xmax]],
    padding=[60, 60],
)

store_map.save(OUTPUT_HTML)

print("\nHTML successfully saved:")
print(OUTPUT_HTML)

print("\n========================================")
print("RANDOM HTML TEST ROW")
print("========================================")
print("Location:", test_name)
print("Latitude:", test_latitude)
print("Longitude:", test_longitude)
print("Original Place bbox:", random_row["original_place_bbox"])
print("Building bbox:", random_row["building_bbox"])
print("Final bbox:", random_row["final_bbox"])
print("\nOpen the HTML file in Chrome or Safari.")
print("Check whether the green bbox follows the physical building in the satellite image.")
print("IMPORTANT: if this Place is inside a mall/shared building, the green bbox may represent the whole building rather than the individual store.")

db.close()
print("\nDONE.")
