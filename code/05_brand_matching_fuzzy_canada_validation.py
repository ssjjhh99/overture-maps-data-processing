import os
import re
import unicodedata
from collections import Counter, defaultdict

import duckdb
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein


# ============================================================
# 1. SETTINGS
# ============================================================

# Winston suggested Canada as a larger but more manageable test.
# Change to "US" when you are ready to process U.S. records.
COUNTRY = "CA"

MIN_OVERTURE_CONFIDENCE = 0.33

# Use a larger test than 3,000.
# Set this to None only when your computer has enough memory
# to load all matching country records into pandas.
LIMIT_N = 50_000

OUTPUT_DIRECTORY = "/Users/stephs/Desktop"

# Fuzzy matching thresholds. These are starting values.
# Final values should be selected using the validation file.
FUZZY_AUTO_SCORE = 92.0
FUZZY_REVIEW_SCORE = 85.0
SHORT_BRAND_AUTO_SCORE = 97.0
MIN_WINNING_MARGIN = 5.0

# Maximum brand/name segment length considered by the matcher.
MAX_SEGMENT_TOKENS = 8

# Candidate length buckets make fuzzy matching much faster.
LENGTH_BUCKET_SIZE = 3

# Short one-word brands such as Gap, Shell, Mango, and Ace
# are more ambiguous and therefore require stronger evidence.
SHORT_BRAND_MAX_CHARACTERS = 6

# Create validation candidates for Winston's suggested test and
# a few other ambiguous brand names.
VALIDATION_BRANDS = [
    "McDonald's",
    "Mango",
    "Shell",
]

# Optional aliases must be manually verified.
# The left side is an alternate spelling/name, and the right side
# is the canonical brand.
MANUAL_ALIASES = {
    # "boh": "Bank of Hawaii",
    # "mcdonald": "McDonald's",
}

# These terms do not automatically reject a match by themselves.
# They are used as a warning when a brand phrase appears in a
# professional/non-outlet context and the category is not compatible.
NON_OUTLET_CONTEXT_TERMS = {
    "llp",
    "law",
    "lawyer",
    "attorney",
    "construction",
    "engineering",
    "consulting",
    "accounting",
    "plumbing",
    "realty",
    "properties",
    "foundation",
}


# ============================================================
# 2. CONNECT TO OVERTURE PLACES
# ============================================================

db = duckdb.connect()

for extension in ("spatial", "httpfs"):
    try:
        db.execute(f"INSTALL {extension};")
    except Exception:
        # The extension may already be installed.
        pass

    db.execute(f"LOAD {extension};")

db.execute("SET s3_region = 'us-west-2';")


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


# ============================================================
# 3. GENERAL CLEANING FUNCTIONS
# ============================================================

def is_blank(value):
    """Return True for None, NaN, or empty text."""

    if value is None:
        return True

    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass

    return str(value).strip() == ""


def normalize_name(value):
    """
    Normalize names before comparison.

    Examples:
        McDonald's -> mcdonalds
        MCDONALD’S -> mcdonalds
        H&R Block  -> h and r block
    """

    if is_blank(value):
        return ""

    text = unicodedata.normalize(
        "NFKD",
        str(value)
    )

    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )

    text = text.casefold()

    text = text.replace("’", "")
    text = text.replace("'", "")
    text = text.replace("&", " and ")

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text
    )

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


def compact_name(value):
    """
    Remove spaces after normalization.

    This helps compare:
        Mc Donalds
        McDonalds
    """

    return normalize_name(value).replace(
        " ",
        ""
    )


def convert_to_list(value):
    """Convert list-like DuckDB values to a normal Python list."""

    if value is None:
        return []

    if hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, (list, tuple, set)):
        return [
            item
            for item in value
            if not is_blank(item)
        ]

    if is_blank(value):
        return []

    return [value]


def make_category_path(value):
    """Convert taxonomy hierarchy into a readable path."""

    levels = convert_to_list(value)

    cleaned_levels = [
        str(level).strip()
        for level in levels
        if not is_blank(level)
    ]

    if not cleaned_levels:
        return None

    return " > ".join(cleaned_levels)


def get_top_category(value):
    """Return the broadest category in taxonomy.hierarchy."""

    levels = convert_to_list(value)

    if not levels:
        return None

    return str(levels[0]).strip()


# ============================================================
# 4. BUILD A BRAND REFERENCE FROM THE FULL COUNTRY DATASET
# ============================================================

# Improvement 1:
# The original script learned brands only from the selected sample.
# This query learns explicit brands and their categories from all
# qualifying records in the selected country.
print(
    "\nBuilding the full-country brand reference..."
)


brand_stats = db.execute(
    """
    SELECT
        brand.names.primary AS overture_brand_name,
        basic_category,
        taxonomy.hierarchy[1] AS top_category,
        COUNT(*) AS occurrences

    FROM read_parquet(
        ?,
        hive_partitioning = 1
    )

    WHERE addresses[1].country = ?
      AND confidence > ?
      AND brand.names.primary IS NOT NULL
      AND TRIM(brand.names.primary) <> ''

    GROUP BY
        brand.names.primary,
        basic_category,
        taxonomy.hierarchy[1]
    """,
    [
        path,
        COUNTRY,
        MIN_OVERTURE_CONFIDENCE,
    ]
).df()


print(
    "Full-country brand/category rows:",
    len(brand_stats)
)


brand_stats["brand_key"] = (
    brand_stats["overture_brand_name"]
    .apply(normalize_name)
)

brand_stats = brand_stats[
    brand_stats["brand_key"] != ""
].copy()


# Choose the most common original spelling for each normalized brand.
spelling_counts = defaultdict(Counter)

for row in brand_stats.itertuples(
    index=False
):
    spelling_counts[
        row.brand_key
    ][
        str(row.overture_brand_name).strip()
    ] += int(row.occurrences)


canonical_brand_names = {
    brand_key: counts.most_common(1)[0][0]
    for brand_key, counts in spelling_counts.items()
}


# Learn the categories in which each explicitly labeled brand appears.
brand_top_categories = defaultdict(set)
brand_basic_categories = defaultdict(set)

for row in brand_stats.itertuples(
    index=False
):
    canonical_name = canonical_brand_names[
        row.brand_key
    ]

    if not is_blank(row.top_category):
        brand_top_categories[
            canonical_name
        ].add(
            str(row.top_category)
        )

    if not is_blank(row.basic_category):
        brand_basic_categories[
            canonical_name
        ].add(
            str(row.basic_category)
        )


# ============================================================
# 5. ADD MANUALLY VERIFIED ALIASES
# ============================================================

# Improvement 2:
# Exact aliases can handle known alternate forms without using a
# low-confidence fuzzy rule.
alias_to_canonical = {
    brand_key: canonical_name
    for brand_key, canonical_name
    in canonical_brand_names.items()
}


for alias, target_brand in MANUAL_ALIASES.items():
    alias_key = normalize_name(alias)
    target_key = normalize_name(target_brand)

    canonical_target = canonical_brand_names.get(
        target_key,
        target_brand
    )

    if alias_key:
        alias_to_canonical[
            alias_key
        ] = canonical_target


# ============================================================
# 6. BUILD CANDIDATE-BLOCKING INDEXES
# ============================================================

# Improvement 3:
# Candidate blocking avoids comparing every place name with every
# known brand. Candidates are grouped by first character and number
# of words. This improves speed and reduces random comparisons.
brand_entries = []
candidate_index = defaultdict(list)
prefix_index = defaultdict(list)


for alias_key, canonical_name in (
    alias_to_canonical.items()
):
    tokens = tuple(alias_key.split())

    if not tokens:
        continue

    entry = {
        "alias_key": alias_key,
        "alias_compact": alias_key.replace(" ", ""),
        "canonical_name": canonical_name,
        "tokens": tokens,
        "token_count": len(tokens),
        "first_character": tokens[0][0],
        "length_bucket": len(alias_key.replace(" ", "")) // LENGTH_BUCKET_SIZE,
    }

    brand_entries.append(entry)

    candidate_index[
        (
            entry["first_character"],
            entry["token_count"],
            entry["length_bucket"]
        )
    ].append(entry)

    prefix_index[
        tokens[0]
    ].append(entry)


# Check longer prefixes before shorter ones.
for first_token in prefix_index:
    prefix_index[first_token].sort(
        key=lambda item: (
            item["token_count"],
            len(item["alias_key"])
        ),
        reverse=True
    )


# ============================================================
# 7. CATEGORY COMPATIBILITY
# ============================================================

def get_category_status(
    canonical_brand,
    row_top_category,
    row_basic_category
):
    """
    Return:
        compatible
        conflict
        unknown

    Category compatibility is learned from records for which
    Overture already supplies the brand.
    """

    known_top = brand_top_categories.get(
        canonical_brand,
        set()
    )

    known_basic = brand_basic_categories.get(
        canonical_brand,
        set()
    )

    row_top = (
        None
        if is_blank(row_top_category)
        else str(row_top_category)
    )

    row_basic = (
        None
        if is_blank(row_basic_category)
        else str(row_basic_category)
    )

    if row_top is None and row_basic is None:
        return "unknown"

    if row_top and row_top in known_top:
        return "compatible"

    if row_basic and row_basic in known_basic:
        return "compatible"

    if not known_top and not known_basic:
        return "unknown"

    return "conflict"


def is_short_ambiguous_brand(brand_key):
    """Identify short one-word brand names that require extra care."""

    tokens = brand_key.split()

    return (
        len(tokens) == 1
        and len(tokens[0]) <= SHORT_BRAND_MAX_CHARACTERS
    )


# ============================================================
# 8. ORIGINAL BASELINE MATCHER
# ============================================================

# Improvement 4:
# Keep the original exact/prefix method as a baseline so the new
# results can be compared with it.
def baseline_match(
    overture_brand,
    location_name
):
    if not is_blank(overture_brand):
        key = normalize_name(
            overture_brand
        )

        return (
            canonical_brand_names.get(
                key,
                str(overture_brand).strip()
            ),
            "overture_brand"
        )

    location_key = normalize_name(
        location_name
    )

    if not location_key:
        return None, "unmatched"

    first_token = location_key.split()[0]

    for entry in prefix_index.get(
        first_token,
        []
    ):
        brand_key = entry["alias_key"]

        if (
            location_key == brand_key
            or location_key.startswith(
                brand_key + " "
            )
        ):
            return (
                entry["canonical_name"],
                "original_exact_or_prefix"
            )

    return None, "unmatched"


# ============================================================
# 9. CREATE FUZZY CANDIDATES
# ============================================================

def create_candidate_results(
    location_key,
    row_top_category,
    row_basic_category
):
    """
    Compare contiguous name segments with blocked brand candidates.

    Uses:
        - normalized Levenshtein similarity
        - token-sort similarity
        - compact-name similarity
        - category compatibility
        - match position
    """

    location_tokens = location_key.split()

    if not location_tokens:
        return []

    best_by_brand = {}

    maximum_segment_length = min(
        MAX_SEGMENT_TOKENS,
        len(location_tokens)
    )

    for start_position in range(
        len(location_tokens)
    ):
        for segment_length in range(
            1,
            maximum_segment_length + 1
        ):
            end_position = (
                start_position
                + segment_length
            )

            if end_position > len(
                location_tokens
            ):
                break

            segment_tokens = location_tokens[
                start_position:
                end_position
            ]

            segment = " ".join(
                segment_tokens
            )

            segment_compact = "".join(
                segment_tokens
            )

            first_character = (
                segment_tokens[0][0]
            )

            # Permit a one-token difference so that
            # "mc donalds" can be compared with "mcdonalds".
            possible_brand_token_counts = {
                max(1, segment_length - 1),
                segment_length,
                segment_length + 1,
            }

            candidate_entries = []

            segment_length_bucket = (
                len(segment_compact)
                // LENGTH_BUCKET_SIZE
            )

            possible_length_buckets = {
                max(0, segment_length_bucket - 1),
                segment_length_bucket,
                segment_length_bucket + 1,
            }

            for brand_token_count in (
                possible_brand_token_counts
            ):
                for length_bucket in (
                    possible_length_buckets
                ):
                    candidate_entries.extend(
                        candidate_index.get(
                            (
                                first_character,
                                brand_token_count,
                                length_bucket
                            ),
                            []
                        )
                    )

            for entry in candidate_entries:
                alias_key = entry["alias_key"]
                alias_compact = entry[
                    "alias_compact"
                ]

                # Avoid comparing strings with extremely
                # different lengths.
                shorter_length = min(
                    len(segment_compact),
                    len(alias_compact)
                )

                longer_length = max(
                    len(segment_compact),
                    len(alias_compact)
                )

                if longer_length == 0:
                    continue

                if (
                    shorter_length
                    / longer_length
                    < 0.60
                ):
                    continue

                levenshtein_score = (
                    Levenshtein
                    .normalized_similarity(
                        segment,
                        alias_key
                    )
                    * 100
                )

                compact_levenshtein_score = (
                    Levenshtein
                    .normalized_similarity(
                        segment_compact,
                        alias_compact
                    )
                    * 100
                )

                token_score = fuzz.token_sort_ratio(
                    segment,
                    alias_key
                )

                # Improvement 5:
                # Combine character-edit similarity with token
                # similarity instead of relying on one score.
                score = (
                    0.70
                    * max(
                        levenshtein_score,
                        compact_levenshtein_score
                    )
                    + 0.30
                    * token_score
                )

                exact_segment = (
                    segment == alias_key
                    or segment_compact
                    == alias_compact
                )

                whole_name_exact = (
                    location_key == alias_key
                    or compact_name(location_key)
                    == alias_compact
                )

                if whole_name_exact:
                    method = "exact_name"
                    score = 100.0

                elif (
                    exact_segment
                    and start_position == 0
                ):
                    method = "prefix_name"
                    score = max(
                        score,
                        99.0
                    )

                elif exact_segment:
                    method = "contained_brand_phrase"
                    score = max(
                        score,
                        97.0
                    )

                else:
                    method = "fuzzy_name"

                    # Beginning-of-name fuzzy matches are more
                    # trustworthy than middle-of-name matches.
                    if start_position == 0:
                        score = min(
                            100.0,
                            score + 1.5
                        )
                    else:
                        score = max(
                            0.0,
                            score - 1.5
                        )

                canonical_name = entry[
                    "canonical_name"
                ]

                category_status = (
                    get_category_status(
                        canonical_name,
                        row_top_category,
                        row_basic_category
                    )
                )

                candidate = {
                    "candidate_brand": canonical_name,
                    "candidate_brand_key": normalize_name(
                        canonical_name
                    ),
                    "matched_segment": segment,
                    "start_position": start_position,
                    "name_score": round(
                        score,
                        2
                    ),
                    "levenshtein_score": round(
                        levenshtein_score,
                        2
                    ),
                    "token_score": round(
                        token_score,
                        2
                    ),
                    "category_status": category_status,
                    "match_method": method,
                }

                previous = best_by_brand.get(
                    canonical_name
                )

                if (
                    previous is None
                    or candidate["name_score"]
                    > previous["name_score"]
                ):
                    best_by_brand[
                        canonical_name
                    ] = candidate

    return sorted(
        best_by_brand.values(),
        key=lambda item: item["name_score"],
        reverse=True
    )


# ============================================================
# 10. IMPROVED BRAND DECISION
# ============================================================

def improved_match(row):
    """
    Return a detailed matching decision.

    High-confidence rows receive brand_name.
    Medium-confidence rows receive candidate_brand but brand_name
    remains blank so they can be manually reviewed.
    """

    overture_brand = row[
        "overture_brand_name"
    ]

    # Improvement 6:
    # Explicit Overture brands always take priority.
    if not is_blank(overture_brand):
        brand_key = normalize_name(
            overture_brand
        )

        canonical_name = (
            canonical_brand_names.get(
                brand_key,
                str(overture_brand).strip()
            )
        )

        return pd.Series({
            "brand_name": canonical_name,
            "candidate_brand": canonical_name,
            "brand_match_method": "overture_brand",
            "brand_decision": "automatic",
            "name_score": 100.0,
            "levenshtein_score": 100.0,
            "token_score": 100.0,
            "second_best_score": None,
            "winning_margin": None,
            "category_status": "overture_provided",
            "matched_segment": normalize_name(
                overture_brand
            ),
            "negative_context": False,
        })

    location_key = normalize_name(
        row["location_name"]
    )

    if not location_key:
        return pd.Series({
            "brand_name": None,
            "candidate_brand": None,
            "brand_match_method": "unmatched",
            "brand_decision": "unmatched",
            "name_score": 0.0,
            "levenshtein_score": None,
            "token_score": None,
            "second_best_score": None,
            "winning_margin": None,
            "category_status": "unknown",
            "matched_segment": None,
            "negative_context": False,
        })

    candidates = create_candidate_results(
        location_key,
        row["top_category"],
        row["basic_category"]
    )

    if not candidates:
        return pd.Series({
            "brand_name": None,
            "candidate_brand": None,
            "brand_match_method": "unmatched",
            "brand_decision": "unmatched",
            "name_score": 0.0,
            "levenshtein_score": None,
            "token_score": None,
            "second_best_score": None,
            "winning_margin": None,
            "category_status": "unknown",
            "matched_segment": None,
            "negative_context": False,
        })

    best = candidates[0]

    second_best_score = (
        candidates[1]["name_score"]
        if len(candidates) > 1
        else 0.0
    )

    winning_margin = (
        best["name_score"]
        - second_best_score
    )

    location_token_set = set(
        location_key.split()
    )

    negative_context = bool(
        location_token_set
        & NON_OUTLET_CONTEXT_TERMS
    )

    ambiguous_short_brand = (
        is_short_ambiguous_brand(
            best["candidate_brand_key"]
        )
    )

    score_required = (
        SHORT_BRAND_AUTO_SCORE
        if ambiguous_short_brand
        else FUZZY_AUTO_SCORE
    )

    category_status = best[
        "category_status"
    ]

    method = best[
        "match_method"
    ]

    score = best[
        "name_score"
    ]

    # Improvement 7:
    # A category conflict prevents automatic assignment.
    category_allows_automatic = (
        category_status == "compatible"
    )

    # A long exact/prefix name may be accepted when category is
    # unavailable, but short/ambiguous brands still require a
    # compatible category.
    if (
        category_status == "unknown"
        and not ambiguous_short_brand
        and method in {
            "exact_name",
            "prefix_name",
        }
        and best["start_position"] == 0
    ):
        category_allows_automatic = True

    # Improvement 8:
    # Suspicious professional/non-outlet terms stop automatic
    # assignment unless the category is compatible.
    context_allows_automatic = not (
        negative_context
        and category_status
        != "compatible"
    )

    # Improvement 9:
    # Require a clear lead over the second-best brand.
    margin_allows_automatic = (
        winning_margin
        >= MIN_WINNING_MARGIN
    )

    automatic_match = (
        score >= score_required
        and category_allows_automatic
        and context_allows_automatic
        and margin_allows_automatic
    )

    if automatic_match:
        decision = "automatic"
        final_brand = best[
            "candidate_brand"
        ]

    elif score >= FUZZY_REVIEW_SCORE:
        # Improvement 10:
        # Do not force uncertain rows into a brand.
        decision = "manual_review"
        final_brand = None

    else:
        decision = "unmatched"
        final_brand = None

    return pd.Series({
        "brand_name": final_brand,
        "candidate_brand": best[
            "candidate_brand"
        ],
        "brand_match_method": method,
        "brand_decision": decision,
        "name_score": score,
        "levenshtein_score": best["levenshtein_score"],
        "token_score": best["token_score"],
        "second_best_score": round(
            second_best_score,
            2
        ),
        "winning_margin": round(
            winning_margin,
            2
        ),
        "category_status": category_status,
        "matched_segment": best[
            "matched_segment"
        ],
        "negative_context": negative_context,
    })


# ============================================================
# 11. EXTRACT THE TARGET TEST DATA
# ============================================================

limit_clause = (
    ""
    if LIMIT_N is None
    else f"LIMIT {int(LIMIT_N)}"
)


print(
    f"\nExtracting records for country={COUNTRY}, "
    f"limit={LIMIT_N}..."
)


df = db.execute(
    f"""
    SELECT
        brand.names.primary AS overture_brand_name,
        names.primary AS location_name,

        taxonomy.hierarchy AS taxonomy_hierarchy,
        basic_category,

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
      AND confidence > ?
      AND names.primary IS NOT NULL

    {limit_clause}
    """,
    [
        path,
        COUNTRY,
        MIN_OVERTURE_CONFIDENCE,
    ]
).df()


df.insert(
    0,
    "source_row_number",
    range(len(df))
)


df["category_path"] = (
    df["taxonomy_hierarchy"]
    .apply(make_category_path)
)

df["top_category"] = (
    df["taxonomy_hierarchy"]
    .apply(get_top_category)
)


print(
    "Number of extracted rows:",
    len(df)
)


# ============================================================
# 12. RUN BASELINE AND IMPROVED MATCHERS
# ============================================================

baseline_results = df.apply(
    lambda row: baseline_match(
        row["overture_brand_name"],
        row["location_name"]
    ),
    axis=1,
    result_type="expand"
)

baseline_results.columns = [
    "baseline_brand_name",
    "baseline_match_method",
]

df[
    [
        "baseline_brand_name",
        "baseline_match_method",
    ]
] = baseline_results


improved_results = df.apply(
    improved_match,
    axis=1
)

df[
    improved_results.columns
] = improved_results


# ============================================================
# 13. CREATE BOUNDING-BOX COLUMN
# ============================================================

def make_bounding_box(row):
    bbox_values = [
        row["bbox_xmin"],
        row["bbox_ymin"],
        row["bbox_xmax"],
        row["bbox_ymax"],
    ]

    if any(
        is_blank(value)
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


# ============================================================
# 14. OUTPUT FILE NAMES
# ============================================================

sample_label = (
    "all"
    if LIMIT_N is None
    else str(LIMIT_N)
)

file_prefix = (
    f"overture_{COUNTRY.lower()}_"
    f"{sample_label}"
)

location_path = os.path.join(
    OUTPUT_DIRECTORY,
    f"{file_prefix}_location_level.csv"
)

brand_path = os.path.join(
    OUTPUT_DIRECTORY,
    f"{file_prefix}_brand_level.csv"
)

audit_path = os.path.join(
    OUTPUT_DIRECTORY,
    f"{file_prefix}_brand_match_audit.csv"
)

validation_path = os.path.join(
    OUTPUT_DIRECTORY,
    f"{file_prefix}_validation_candidates.csv"
)

summary_path = os.path.join(
    OUTPUT_DIRECTORY,
    f"{file_prefix}_matching_summary.csv"
)


# ============================================================
# 15. LOCATION-LEVEL OUTPUT
# ============================================================

# Keep the six requested output columns and preserve row order.
location_level = df[
    [
        "brand_name",
        "location_name",
        "address",
        "latitude",
        "longitude",
        "bounding_box",
    ]
].copy()


location_level.to_csv(
    location_path,
    index=False
)


# ============================================================
# 16. BRAND-LEVEL OUTPUT
# ============================================================

def join_distinct_categories(series):
    values = sorted({
        str(value).strip()
        for value in series
        if not is_blank(value)
    })

    if not values:
        return None

    return " | ".join(values)


matched_brands = df[
    df["brand_name"].notna()
].copy()


brand_level = (
    matched_brands
    .groupby("brand_name")
    .agg(
        number_of_locations=(
            "location_name",
            "size"
        ),
        categories=(
            "category_path",
            join_distinct_categories
        ),
    )
    .reset_index()
    .sort_values(
        [
            "number_of_locations",
            "brand_name",
        ],
        ascending=[
            False,
            True,
        ]
    )
)


brand_level.to_csv(
    brand_path,
    index=False
)


# ============================================================
# 17. AUDIT OUTPUT
# ============================================================

# Improvement 11:
# The audit file explains every automatic, review, and unmatched
# decision instead of showing only the final brand.
audit_columns = [
    "source_row_number",
    "location_name",
    "overture_brand_name",

    "baseline_brand_name",
    "baseline_match_method",

    "brand_name",
    "candidate_brand",
    "brand_match_method",
    "brand_decision",

    "name_score",
    "levenshtein_score",
    "token_score",
    "second_best_score",
    "winning_margin",

    "category_status",
    "top_category",
    "basic_category",
    "category_path",

    "negative_context",
    "matched_segment",

    "address",
    "latitude",
    "longitude",
    "bounding_box",
]


df[
    audit_columns
].to_csv(
    audit_path,
    index=False
)


# ============================================================
# 18. VALIDATION-CANDIDATE OUTPUT
# ============================================================

# Improvement 12:
# Create a file that can be manually labeled as Winston requested.
validation_brand_keys = {
    normalize_name(brand)
    for brand in VALIDATION_BRANDS
}


def is_validation_candidate(row):
    possible_names = [
        row["brand_name"],
        row["candidate_brand"],
        row["baseline_brand_name"],
    ]

    return any(
        normalize_name(value)
        in validation_brand_keys
        for value in possible_names
        if not is_blank(value)
    )


validation_candidates = df[
    df.apply(
        is_validation_candidate,
        axis=1
    )
][
    [
        "location_name",
        "overture_brand_name",
        "candidate_brand",
        "brand_name",
        "brand_decision",
        "brand_match_method",
        "name_score",
        "levenshtein_score",
        "token_score",
        "category_status",
        "category_path",
        "address",
    ]
].copy()


# Fill this column manually:
# 1 = should match the candidate brand
# 0 = should not match the candidate brand
validation_candidates.insert(
    0,
    "should_match",
    ""
)

validation_candidates.insert(
    1,
    "review_notes",
    ""
)


validation_candidates.to_csv(
    validation_path,
    index=False
)


# ============================================================
# 19. METHOD SUMMARY
# ============================================================

matching_summary = (
    df
    .groupby(
        [
            "brand_decision",
            "brand_match_method",
        ],
        dropna=False
    )
    .size()
    .reset_index(
        name="number_of_rows"
    )
    .sort_values(
        [
            "brand_decision",
            "number_of_rows",
        ],
        ascending=[
            True,
            False,
        ]
    )
)


matching_summary.to_csv(
    summary_path,
    index=False
)


# ============================================================
# 20. PRINT RESULTS
# ============================================================

print("\nSaved files:")

print(
    "Location-level:",
    location_path
)

print(
    "Brand-level:",
    brand_path
)

print(
    "Audit:",
    audit_path
)

print(
    "Validation candidates:",
    validation_path
)

print(
    "Matching summary:",
    summary_path
)


print("\nDecision counts:")

print(
    df["brand_decision"]
    .value_counts(
        dropna=False
    )
    .to_string()
)


print(
    "\nRows whose improved result differs "
    "from the original baseline:"
)

different_results = (
    df["baseline_brand_name"]
    .fillna("")
    .ne(
        df["brand_name"]
        .fillna("")
    )
)

print(
    int(
        different_results.sum()
    )
)


print(
    "\nManual-review sample:"
)

manual_review_sample = df[
    df["brand_decision"]
    == "manual_review"
][
    [
        "location_name",
        "candidate_brand",
        "name_score",
        "winning_margin",
        "category_status",
        "category_path",
    ]
].head(30)

print(
    manual_review_sample.to_string(
        index=False
    )
)


print("\nDone.")
