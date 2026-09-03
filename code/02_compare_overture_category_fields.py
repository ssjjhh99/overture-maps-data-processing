import pandas as pd
import duckdb
import overturemaps

db = duckdb.connect()
df = db.execute(f"""
    SELECT
        ST_AsText(geometry) AS geometry,
        names.primary AS name,

        -- old category fields
        categories.primary AS old_categories_primary,
        categories.alternate AS old_categories_alternate,

        -- newer Overture category fields
        basic_category,
        taxonomy.primary AS taxonomy_primary,
        taxonomy.hierarchy AS taxonomy_hierarchy,

        -- split hierarchy into clearer levels
        taxonomy.hierarchy[1] AS top_level_category,
        taxonomy.hierarchy[2] AS second_level_category,
        taxonomy.hierarchy[3] AS third_level_category,
        taxonomy.hierarchy[4] AS fourth_level_category,

        -- address fields
        addresses[1].freeform AS address,
        addresses[1].locality AS city,
        addresses[1].postcode AS zip,
        addresses[1].region AS state
    FROM
        read_parquet(?, filename=true, hive_partitioning=1)
    WHERE addresses[1].country = ?
        AND confidence > 0.33
    LIMIT 100
""", [
    path,
    "US"
]).df()

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 300)

print("\nFirst 100 rows with old and new category fields:")
print(df.to_string())