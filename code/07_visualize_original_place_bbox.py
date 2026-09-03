import ast
import math

import folium
import pandas as pd


# ----------------------------------------------------
# 1. File settings
# ----------------------------------------------------
location_file = (
    "/Users/stephs/Desktop/"
    "overture_location_level_3000.csv"
)

map_output_path = (
    "/Users/stephs/Desktop/"
    "overture_store_bbox_check.html"
)

# Change this to inspect another location.
STORE_QUERY = "Regal Keauhou"


# ----------------------------------------------------
# 2. Read the location-level CSV
# ----------------------------------------------------
location_df = pd.read_csv(location_file)

required_columns = {
    "brand_name",
    "location_name",
    "address",
    "latitude",
    "longitude",
    "bounding_box"
}

missing_columns = required_columns - set(location_df.columns)

if missing_columns:
    raise ValueError(
        "The CSV is missing these required columns: "
        + ", ".join(sorted(missing_columns))
    )


# ----------------------------------------------------
# 3. Find the selected store
# ----------------------------------------------------
store_matches = location_df[
    location_df["location_name"]
    .fillna("")
    .str.contains(
        STORE_QUERY,
        case=False,
        regex=False
    )
].copy()

if store_matches.empty:
    raise ValueError(
        f"No location containing '{STORE_QUERY}' "
        "was found in the CSV."
    )


print("\nMatching locations:")

print(
    store_matches[
        [
            "brand_name",
            "location_name",
            "address",
            "latitude",
            "longitude",
            "bounding_box"
        ]
    ].to_string(index=False)
)


# Use the first matching row.
store = store_matches.iloc[0]


# ----------------------------------------------------
# 4. Read the point and bounding-box coordinates
# ----------------------------------------------------
latitude = float(store["latitude"])
longitude = float(store["longitude"])

if pd.isna(store["bounding_box"]):
    raise ValueError(
        "The selected location does not have a bounding box."
    )

bbox = ast.literal_eval(
    str(store["bounding_box"])
)

if (
    not isinstance(bbox, (list, tuple))
    or len(bbox) != 4
):
    raise ValueError(
        "The bounding box must contain four values: "
        "[xmin, ymin, xmax, ymax]."
    )


xmin, ymin, xmax, ymax = map(float, bbox)


# ----------------------------------------------------
# 5. Calculate approximate bbox dimensions in meters
# ----------------------------------------------------
def haversine_distance(
    latitude_1,
    longitude_1,
    latitude_2,
    longitude_2
):
    """
    Calculate the approximate distance between two
    latitude/longitude coordinates in meters.
    """

    earth_radius_meters = 6_371_000

    lat_1_radians = math.radians(latitude_1)
    lat_2_radians = math.radians(latitude_2)

    latitude_difference = math.radians(
        latitude_2 - latitude_1
    )

    longitude_difference = math.radians(
        longitude_2 - longitude_1
    )

    a = (
        math.sin(latitude_difference / 2) ** 2
        + math.cos(lat_1_radians)
        * math.cos(lat_2_radians)
        * math.sin(longitude_difference / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    return earth_radius_meters * c


bbox_width_meters = haversine_distance(
    latitude,
    xmin,
    latitude,
    xmax
)

bbox_height_meters = haversine_distance(
    ymin,
    longitude,
    ymax,
    longitude
)


print("\nSelected location:")
print(store["location_name"])

print("\nPoint coordinates:")
print("Latitude:", latitude)
print("Longitude:", longitude)

print("\nActual Overture bounding box:")
print("xmin:", xmin)
print("ymin:", ymin)
print("xmax:", xmax)
print("ymax:", ymax)

print("\nApproximate bbox dimensions:")
print(f"Width: {bbox_width_meters:.2f} meters")
print(f"Height: {bbox_height_meters:.2f} meters")


# ----------------------------------------------------
# 6. Prepare clean popup values
# ----------------------------------------------------
brand_value = (
    ""
    if pd.isna(store["brand_name"])
    else str(store["brand_name"])
)

address_value = (
    ""
    if pd.isna(store["address"])
    else str(store["address"])
)

location_value = str(
    store["location_name"]
)


# ----------------------------------------------------
# 7. Create the satellite map
# ----------------------------------------------------
store_map = folium.Map(
    location=[
        latitude,
        longitude
    ],
    zoom_start=20,
    tiles=None,
    control_scale=True
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
    max_native_zoom=18
).add_to(store_map)


# ----------------------------------------------------
# 8. Add the yellow enlarged viewing guide
# ----------------------------------------------------
# This yellow dashed rectangle is NOT the real bbox.
# It only makes the relevant area easier to locate.
display_padding = 0.00008

guide_bounds = [
    [
        latitude - display_padding,
        longitude - display_padding
    ],
    [
        latitude + display_padding,
        longitude + display_padding
    ]
]


folium.Rectangle(
    bounds=guide_bounds,
    tooltip=(
        "Enlarged viewing guide — "
        "not the actual Overture bbox"
    ),
    popup=(
        "<b>Viewing guide only</b><br>"
        "This yellow dashed rectangle was enlarged "
        "to make the store location easier to see.<br>"
        "It is not part of Overture's bbox data."
    ),
    color="yellow",
    weight=3,
    dash_array="10, 7",
    fill=False
).add_to(store_map)


# ----------------------------------------------------
# 9. Add the actual red Overture bounding box
# ----------------------------------------------------
bbox_popup = (
    "<b>Actual Overture bounding box</b><br>"
    f"Width: approximately "
    f"{bbox_width_meters:.2f} meters<br>"
    f"Height: approximately "
    f"{bbox_height_meters:.2f} meters<br><br>"
    f"xmin: {xmin}<br>"
    f"ymin: {ymin}<br>"
    f"xmax: {xmax}<br>"
    f"ymax: {ymax}"
)


folium.Rectangle(
    bounds=[
        [
            ymin,
            xmin
        ],
        [
            ymax,
            xmax
        ]
    ],
    tooltip="Actual Overture bounding box",
    popup=bbox_popup,
    color="red",
    weight=8,
    fill=False
).add_to(store_map)


# ----------------------------------------------------
# 10. Add markers at the four bbox corners
# ----------------------------------------------------
bbox_corners = [
    (
        "Southwest corner",
        ymin,
        xmin
    ),
    (
        "Southeast corner",
        ymin,
        xmax
    ),
    (
        "Northwest corner",
        ymax,
        xmin
    ),
    (
        "Northeast corner",
        ymax,
        xmax
    )
]


for (
    corner_name,
    corner_latitude,
    corner_longitude
) in bbox_corners:

    folium.CircleMarker(
        location=[
            corner_latitude,
            corner_longitude
        ],
        radius=3,
        tooltip=corner_name,
        popup=(
            f"<b>{corner_name}</b><br>"
            f"Latitude: {corner_latitude}<br>"
            f"Longitude: {corner_longitude}"
        ),
        color="red",
        weight=2,
        fill=True,
        fill_color="red",
        fill_opacity=1
    ).add_to(store_map)


# ----------------------------------------------------
# 11. Create the blue point popup
# ----------------------------------------------------
point_popup = (
    f"<b>{location_value}</b><br>"
    f"Brand: {brand_value}<br>"
    f"Address: {address_value}<br>"
    f"Latitude: {latitude}<br>"
    f"Longitude: {longitude}"
)


# ----------------------------------------------------
# 12. Add a white halo behind the blue point
# ----------------------------------------------------
# This makes the blue point easy to see against both
# the red bbox and the satellite image.
folium.CircleMarker(
    location=[
        latitude,
        longitude
    ],
    radius=11,
    color="white",
    weight=3,
    fill=True,
    fill_color="white",
    fill_opacity=1,
    interactive=False
).add_to(store_map)


# ----------------------------------------------------
# 13. Add the visible blue point LAST
# ----------------------------------------------------
# It is drawn last so it stays above all other shapes.
folium.CircleMarker(
    location=[
        latitude,
        longitude
    ],
    radius=7,
    tooltip=(
        f"Overture place point: "
        f"{location_value}"
    ),
    popup=point_popup,
    color="white",
    weight=2,
    fill=True,
    fill_color="#0066ff",
    fill_opacity=1
).add_to(store_map)


# ----------------------------------------------------
# 14. Add title and explanation
# ----------------------------------------------------
title_html = f"""
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
    max-width: 410px;
">
    <b>Overture bbox check: {location_value}</b><br>

    <span style="
        display: inline-block;
        width: 13px;
        height: 13px;
        background-color: #0066ff;
        border: 2px solid white;
        border-radius: 50%;
        box-shadow: 0 0 0 1px gray;
        vertical-align: middle;
    "></span>
    Blue point: Overture place coordinate
    <br>

    <span style="
        color: red;
        font-size: 18px;
        vertical-align: middle;
    ">■</span>
    Red rectangle and corners: actual Overture bbox
    <br>

    <span style="
        color: #b59b00;
        font-size: 18px;
        vertical-align: middle;
    ">- - -</span>
    Yellow dashed rectangle: enlarged viewing guide only
    <br><br>

    Actual bbox size:
    {bbox_width_meters:.2f} m ×
    {bbox_height_meters:.2f} m
</div>
"""


store_map.get_root().html.add_child(
    folium.Element(title_html)
)


# ----------------------------------------------------
# 15. Add map layer control
# ----------------------------------------------------
folium.LayerControl().add_to(
    store_map
)


# ----------------------------------------------------
# 16. Set the map view around the yellow guide
# ----------------------------------------------------
store_map.fit_bounds(
    guide_bounds,
    padding=[
        40,
        40
    ]
)


# ----------------------------------------------------
# 17. Save the interactive HTML map
# ----------------------------------------------------
store_map.save(
    map_output_path
)


print("\nMap successfully saved to:")
print(map_output_path)

print(
    "\nOpen the HTML file in Chrome or Safari."
)

print(
    "The large blue point with a white border "
    "shows Overture's place coordinate."
)

print(
    "The red line and red corner points show "
    "the actual Overture bbox."
)

print(
    "The yellow dashed rectangle is only "
    "an enlarged viewing guide."
)