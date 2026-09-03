import duckdb
import pandas as pd
import re
import unicodedata


# ----------------------------------------------------
# 1. Connect to Overture Places
# ----------------------------------------------------
db = duckdb.connect()

db.execute("""
INSTALL spatial;
INSTALL httpfs;
""")

db.execute("""
LOAD spatial;
LOAD httpfs;
SET s3_region = 'us-west-2';
""")


# ----------------------------------------------------
# 2. Find the latest Overture release
# ----------------------------------------------------
latest = db.execute("""
    SELECT latest
    FROM read_json(
        'https://stac.overturemaps.org/catalog.json'
    )
""").fetchone()[0]

print("Latest Overture release:", latest)

path = (
    f"s3://overturemaps-us-west-2/release/{latest}/"
    "theme=places/type=place/*"
)

LIMIT_N = 3000


# ----------------------------------------------------
# 3. Extract a sample of 3,000 POI rows
# ----------------------------------------------------
df = db.execute(
    f"""
    SELECT
        brand.names.primary AS overture_brand_name,
        names.primary AS location_name,

        taxonomy.hierarchy AS taxonomy_hierarchy,

        addresses[1].freeform AS address,

        ST_Y(geometry) AS latitude,
        ST_X(geometry) AS longitude,

        bbox.xmin AS bbox_xmin,
        bbox.xmax AS bbox_xmax,
        bbox.ymin AS bbox_ymin,
        bbox.ymax AS bbox_ymax

    FROM read_parquet(
        ?,
        hive_partitioning = 1
    )

    WHERE addresses[1].country = ?
      AND confidence > 0.33
      AND names.primary IS NOT NULL

    LIMIT {LIMIT_N}
    """,
    [
        path,
        "US"
    ]
).df()

print("\nNumber of extracted rows:")
print(len(df))


# ----------------------------------------------------
# 4. Convert taxonomy hierarchy into a category path
# ----------------------------------------------------
def make_category_path(value):
    if value is None:
        return None

    if hasattr(value, "tolist"):
        value = value.tolist()

    if not isinstance(value, (list, tuple)):
        return None

    levels = [
        str(level).strip()
        for level in value
        if level is not None
        and str(level).strip() != ""
    ]

    if not levels:
        return None

    return " > ".join(levels)


df["category_path"] = (
    df["taxonomy_hierarchy"]
    .apply(make_category_path)
)


# ----------------------------------------------------
# 5. Normalize names before comparing them
# ----------------------------------------------------
def normalize_name(value):
    if pd.isna(value):
        return ""

    text = str(value)

    # Remove accent differences
    text = unicodedata.normalize("NFKD", text)

    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )

    # Convert to lowercase
    text = text.casefold()

    # McDonald's and McDonalds become equivalent
    text = text.replace("'", "")
    text = text.replace("’", "")

    # H&R and H and R become more comparable
    text = text.replace("&", " and ")

    # Replace punctuation with spaces
    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text
    )

    # Remove repeated spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ----------------------------------------------------
# 6. Build a reference list of known Overture brands
# ----------------------------------------------------
brand_reference = df[
    df["overture_brand_name"].notna()
][
    ["overture_brand_name"]
].copy()

brand_reference["brand_key"] = (
    brand_reference["overture_brand_name"]
    .apply(normalize_name)
)

brand_reference = brand_reference[
    brand_reference["brand_key"] != ""
].copy()


# Choose the most common original spelling for each brand
canonical_brand_names = (
    brand_reference
    .groupby("brand_key")["overture_brand_name"]
    .agg(
        lambda values:
        values.value_counts().index[0]
    )
    .to_dict()
)


# Check longer brand names before shorter brand names
brand_candidates = sorted(
    canonical_brand_names.items(),
    key=lambda item: len(item[0]),
    reverse=True
)


# ----------------------------------------------------
# 7. Assign the final brand
# ----------------------------------------------------
def assign_brand(row):
    overture_brand = row["overture_brand_name"]

    # Case 1: Overture already provides the brand
    if (
        pd.notna(overture_brand)
        and str(overture_brand).strip() != ""
    ):
        brand_key = normalize_name(
            overture_brand
        )

        canonical_name = (
            canonical_brand_names.get(
                brand_key,
                overture_brand
            )
        )

        return (
            canonical_name,
            "overture_brand"
        )

    # Case 2: Overture brand is blank
    location_key = normalize_name(
        row["location_name"]
    )

    if location_key == "":
        return (
            None,
            "unmatched"
        )

    # Compare the location name with known brands
    for brand_key, canonical_name in brand_candidates:

        # Exact match:
        # Walmart = Walmart
        exact_match = (
            location_key == brand_key
        )

        # Starts-with match:
        # Walmart Pharmacy begins with Walmart
        starts_with_match = (
            location_key.startswith(
                brand_key + " "
            )
        )

        if exact_match or starts_with_match:
            return (
                canonical_name,
                "name_match"
            )

    # No reliable brand match
    return (
        None,
        "unmatched"
    )


brand_results = df.apply(
    assign_brand,
    axis=1,
    result_type="expand"
)

brand_results.columns = [
    "brand_name",
    "brand_match_method"
]

df[
    [
        "brand_name",
        "brand_match_method"
    ]
] = brand_results


# ----------------------------------------------------
# 8. Create one bounding-box column
# Format: xmin, ymin, xmax, ymax
# ----------------------------------------------------
def make_bounding_box(row):
    bbox_values = [
        row["bbox_xmin"],
        row["bbox_ymin"],
        row["bbox_xmax"],
        row["bbox_ymax"]
    ]

    if any(
        pd.isna(value)
        for value in bbox_values
    ):
        return None

    return (
        f"[{row['bbox_xmin']}, "
        f"{row['bbox_ymin']}, "
        f"{row['bbox_xmax']}, "
        f"{row['bbox_ymax']}]"
    )


df["bounding_box"] = df.apply(
    make_bounding_box,
    axis=1
)


# ----------------------------------------------------
# 9. Create the location-level output
# ----------------------------------------------------
location_level = df[
    [
        "brand_name",
        "location_name",
        "address",
        "latitude",
        "longitude",
        "bounding_box"
    ]
].copy()


# Keep the original 3,000-row order
location_path = (
    "/Users/stephs/Desktop/"
    "overture_location_level_3000.csv"
)

location_level.to_csv(
    location_path,
    index=False
)

print("\nSaved location-level file:")
print(location_path)


# ----------------------------------------------------
# 10. Find the most common category for each brand
# ----------------------------------------------------
def most_common_category(series):
    series = series.dropna().astype(str)

    series = series[
        series.str.strip() != ""
    ]

    if series.empty:
        return None

    return (
        series
        .value_counts()
        .index[0]
    )


# ----------------------------------------------------
# 11. Create the brand-level output
# ----------------------------------------------------
matched_brands = df[
    df["brand_name"].notna()
].copy()

matched_brands = matched_brands[
    matched_brands["brand_name"]
    .astype(str)
    .str.strip()
    .ne("")
]


brand_level = (
    matched_brands

    .groupby(
        "brand_name"
    )

    .agg(
        number_of_locations=(
            "location_name",
            "size"
        ),

        categories=(
            "category_path",
            most_common_category
        )
    )

    .reset_index()

    .sort_values(
        "number_of_locations",
        ascending=False
    )
)


brand_path = (
    "/Users/stephs/Desktop/"
    "overture_brand_level_3000.csv"
)

brand_level.to_csv(
    brand_path,
    index=False
)

print("\nSaved brand-level file:")
print(brand_path)


# ----------------------------------------------------
# 12. Print summary information
# ----------------------------------------------------
overture_match_count = (
    df["brand_match_method"]
    .eq("overture_brand")
    .sum()
)

name_match_count = (
    df["brand_match_method"]
    .eq("name_match")
    .sum()
)

unmatched_count = (
    df["brand_match_method"]
    .eq("unmatched")
    .sum()
)


print("\nTotal location rows:")
print(len(location_level))

print("\nRows using Overture's original brand:")
print(overture_match_count)

print("\nAdditional rows assigned through name matching:")
print(name_match_count)

print("\nRows still without a brand:")
print(unmatched_count)

print("\nNumber of unique brands:")
print(len(brand_level))


# ----------------------------------------------------
# 13. Review all automatically inferred matches
# ----------------------------------------------------
name_match_review = df[
    df["brand_match_method"] == "name_match"
][
    [
        "brand_name",
        "location_name",
        "category_path"
    ]
].copy()

print("\nRows assigned through name matching:")
print(
    name_match_review.to_string(
        index=False
    )
)


print("\nTop 20 brands:")
print(
    brand_level
    .head(20)
    .to_string(index=False)
)

print("\nDone.")