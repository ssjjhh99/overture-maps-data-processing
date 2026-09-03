import pandas as pd
import duckdb

# ----------------------------------------------------
# Connect to DuckDB and load Overture Places data
# ----------------------------------------------------
db = duckdb.connect()

db.execute("""
INSTALL spatial;
INSTALL httpfs;
""")

db.execute("""
LOAD spatial;
LOAD httpfs;
SET s3_region='us-west-2';
""")

latest = db.execute(
    "SELECT latest FROM read_json('https://stac.overturemaps.org/catalog.json')"
).fetchone()[0]

print("Latest release:", latest)

path = f"s3://overturemaps-us-west-2/release/{latest}/theme=places/type=place/*"

LIMIT_N = 3000


# ----------------------------------------------------
# Export first 10 original rows with all original columns
# ----------------------------------------------------
first_10_original = db.execute("""
    SELECT *
    FROM read_parquet(?, filename=true, hive_partitioning=1)
    WHERE addresses[1].country = ?
    LIMIT 10
""", [
    path,
    "US"
]).df()

first_10_path = "/Users/stephs/Desktop/overture_first_10_original_all_columns.csv"
first_10_original.to_csv(first_10_path, index=False)

print(f"\nSaved first 10 original rows to: {first_10_path}")


# ----------------------------------------------------
# Count unique category values in the full U.S. dataset
# ----------------------------------------------------
primary_category_count = db.execute("""
    SELECT COUNT(DISTINCT categories.primary)
    FROM read_parquet(?, filename=true, hive_partitioning=1)
    WHERE addresses[1].country = ?
        AND confidence > 0.33
        AND categories.primary IS NOT NULL
""", [
    path,
    "US"
]).fetchone()[0]

basic_category_count = db.execute("""
    SELECT COUNT(DISTINCT basic_category)
    FROM read_parquet(?, filename=true, hive_partitioning=1)
    WHERE addresses[1].country = ?
        AND confidence > 0.33
        AND basic_category IS NOT NULL
""", [
    path,
    "US"
]).fetchone()[0]

print("\nNumber of unique primary_category values in full U.S. dataset:")
print(primary_category_count)

print("\nNumber of unique basic_category values in full U.S. dataset:")
print(basic_category_count)


# ----------------------------------------------------
# Export 3,000 original rows with all original columns
# ----------------------------------------------------
original_3000 = db.execute(f"""
    SELECT *
    FROM read_parquet(?, filename=true, hive_partitioning=1)
    WHERE addresses[1].country = ?
        AND confidence > 0.33
        AND names.primary IS NOT NULL
        AND categories.primary IS NOT NULL
        AND basic_category IS NOT NULL
    LIMIT {LIMIT_N}
""", [
    path,
    "US"
]).df()

original_3000_path = "/Users/stephs/Desktop/overture_original_3000_rows.csv"
original_3000.to_csv(original_3000_path, index=False)

print(f"\nSaved 3,000 original rows to: {original_3000_path}")


# ----------------------------------------------------
# Create grouped version by basic_category
# It only sorts the 3,000 rows so the same basic_category
# values appear together.
# ----------------------------------------------------
grouped_df = original_3000.copy()

grouped_df = grouped_df.sort_values(
    by=["basic_category"],
    ascending=True
)

grouped_path = "/Users/stephs/Desktop/overture_grouped_by_basic_category_3000.csv"
grouped_df.to_csv(grouped_path, index=False)

print(f"\nSaved grouped-by-basic-category output to: {grouped_path}")


# ----------------------------------------------------
# Print quick summary
# ----------------------------------------------------
print("\nNumber of rows in 3,000-row test file:")
print(len(original_3000))

print("\nNumber of unique basic_category groups in this 3,000-row test:")
print(original_3000["basic_category"].nunique())

print("\nTop 20 basic_category groups in this 3,000-row test:")
print(
    original_3000["basic_category"]
    .value_counts()
    .head(20)
    .to_string()
)

print("\nDone.")