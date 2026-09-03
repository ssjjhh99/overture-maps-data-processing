import pandas as pd
import duckdb

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

# ----------------------------------------------------
# 1. Print all column names and column types
# ----------------------------------------------------
schema_df = db.execute("""
    DESCRIBE SELECT *
    FROM read_parquet(?, filename=true, hive_partitioning=1)
    LIMIT 1
""", [path]).df()

print("\nNumber of top-level columns:")
print(schema_df.shape[0])

print("\nAll columns and types:")
print(schema_df.to_string(index=False))

schema_df.to_csv("/Users/stephs/Desktop/overture_all_columns.csv", index=False)
print("\nSaved overture_all_columns.csv to Desktop")


# ----------------------------------------------------
# 2. Print one real row with all top-level columns
# Geometry is converted to readable text.
# ----------------------------------------------------
one_row = db.execute("""
    SELECT
        ST_AsText(geometry) AS geometry_wkt,
        * EXCLUDE (geometry)
    FROM
        read_parquet(?, filename=true, hive_partitioning=1)
    WHERE addresses[1].country = ?
    LIMIT 1
""", [
    path,
    "US"
]).df()

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 1000)

print("\nOne real row with all columns:")
print(one_row.T.to_string(header=False))

one_row.to_csv("/Users/stephs/Desktop/overture_one_full_row.csv", index=False)
print("\nSaved overture_one_full_row.csv to Desktop")