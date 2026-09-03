import os
import re
import unicodedata
from collections import Counter, defaultdict

import duckdb
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein


# ============================================================
# SETTINGS
# ============================================================
COUNTRY = "CA"
SAMPLE_LIMIT = 50_000
MIN_CONFIDENCE = 0.33

# Conservative starting thresholds.
# These are matching rules, NOT accuracy claims.
NORMAL_THRESHOLD = 90.0
SHORT_BRAND_THRESHOLD = 95.0
REVIEW_THRESHOLD = 82.0
MIN_WINNING_MARGIN = 5.0
TOP_FUZZY_CANDIDATES = 25
MAX_SEGMENT_TOKENS = 6

OUTPUT_DIR = "/Users/stephs/Desktop"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. CONNECT TO OVERTURE
# ============================================================
db = duckdb.connect()

db.execute("""
INSTALL spatial;
INSTALL httpfs;
LOAD spatial;
LOAD httpfs;
SET s3_region = 'us-west-2';
""")

latest = db.execute("""
    SELECT latest
    FROM read_json(
        'https://stac.overturemaps.org/catalog.json'
    )
""").fetchone()[0]

places_path = (
    f"s3://overturemaps-us-west-2/release/{latest}/"
    "theme=places/type=place/*"
)

print("Latest Overture release:", latest)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================
def is_blank(value):
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
    Normalize place/brand names for matching.
    """
    if is_blank(value):
        return ""

    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )

    text = text.casefold()
    text = text.replace("'", "").replace("’", "")
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def hierarchy_levels(value):
    """
    Convert taxonomy.hierarchy into a clean Python list.
    """
    if value is None:
        return []

    if hasattr(value, "tolist"):
        value = value.tolist()

    if not isinstance(value, (list, tuple)):
        return []

    return [
        str(level).strip()
        for level in value
        if level is not None
        and str(level).strip() != ""
    ]


def make_category_path(row):
    """
    Prefer full taxonomy hierarchy.
    Fall back to taxonomy.primary, then basic_category.
    """
    levels = hierarchy_levels(row["taxonomy_hierarchy"])

    if levels:
        return " > ".join(levels)

    if not is_blank(row["taxonomy_primary"]):
        return str(row["taxonomy_primary"]).strip()

    if not is_blank(row["basic_category"]):
        return str(row["basic_category"]).strip()

    return None


def make_bbox(row):
    values = [
        row["bbox_xmin"],
        row["bbox_ymin"],
        row["bbox_xmax"],
        row["bbox_ymax"],
    ]

    if any(pd.isna(v) for v in values):
        return None

    return [
        float(row["bbox_xmin"]),
        float(row["bbox_ymin"]),
        float(row["bbox_xmax"]),
        float(row["bbox_ymax"]),
    ]


def make_segments(location_key):
    """
    Create contiguous parts of a normalized name.

    Example:
        black and mcdonalds llp

    creates:
        black
        and
        mcdonalds
        black and
        and mcdonalds
        ...
    """
    tokens = location_key.split()

    segments = []

    max_size = min(
        MAX_SEGMENT_TOKENS,
        len(tokens)
    )

    for size in range(1, max_size + 1):
        for start in range(
            0,
            len(tokens) - size + 1
        ):
            segment = " ".join(
                tokens[start:start + size]
            )

            segments.append(
                {
                    "segment": segment,
                    "start": start,
                    "size": size,
                }
            )

    return segments


# ============================================================
# 3. BUILD BRAND/CATEGORY REFERENCE
# ============================================================
print("\nBuilding Canada brand/category reference...")

brand_reference = db.execute(
    """
    SELECT
        brand.names.primary AS overture_brand_name,
        basic_category,
        taxonomy.primary AS taxonomy_primary,
        taxonomy.hierarchy AS taxonomy_hierarchy,
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
        taxonomy.primary,
        taxonomy.hierarchy
    """,
    [
        places_path,
        COUNTRY,
        MIN_CONFIDENCE,
    ]
).df()

print(
    "Grouped branded reference rows:",
    len(brand_reference)
)


# ============================================================
# 4. CANONICAL BRAND NAMES
# ============================================================
# Different spellings that normalize to the same value are
# grouped together. The most frequent original spelling becomes
# the displayed brand name.
spelling_counts = defaultdict(Counter)

for _, row in brand_reference.iterrows():
    brand = row["overture_brand_name"]
    brand_key = normalize_name(brand)

    if not brand_key:
        continue

    spelling_counts[brand_key][str(brand).strip()] += int(
        row["occurrences"]
    )

canonical_brand_names = {}

for brand_key, counts in spelling_counts.items():
    canonical_brand_names[brand_key] = (
        counts.most_common(1)[0][0]
    )

brand_keys = list(canonical_brand_names.keys())
brand_key_set = set(brand_keys)

print(
    "Unique normalized Canadian brands:",
    len(brand_keys)
)


# ============================================================
# 5. LEARN BRAND <-> CATEGORY RELATIONSHIPS
# ============================================================
# brand -> categories
brand_basic_categories = defaultdict(set)
brand_primary_categories = defaultdict(set)
brand_hierarchy_categories = defaultdict(set)

# category -> brands
basic_category_to_brand_keys = defaultdict(set)
primary_category_to_brand_keys = defaultdict(set)
hierarchy_category_to_brand_keys = defaultdict(set)
top_category_to_brand_keys = defaultdict(set)

for _, row in brand_reference.iterrows():
    raw_brand = row["overture_brand_name"]
    brand_key = normalize_name(raw_brand)

    if not brand_key:
        continue

    brand_name = canonical_brand_names[brand_key]

    # Basic category
    if not is_blank(row["basic_category"]):
        basic = str(
            row["basic_category"]
        ).strip()

        brand_basic_categories[
            brand_name
        ].add(basic)

        basic_category_to_brand_keys[
            basic
        ].add(brand_key)

    # Primary category
    if not is_blank(row["taxonomy_primary"]):
        primary = str(
            row["taxonomy_primary"]
        ).strip()

        brand_primary_categories[
            brand_name
        ].add(primary)

        primary_category_to_brand_keys[
            primary
        ].add(brand_key)

    # Hierarchy
    levels = hierarchy_levels(
        row["taxonomy_hierarchy"]
    )

    if levels:
        top_category_to_brand_keys[
            levels[0]
        ].add(brand_key)

        for level in levels[1:]:
            brand_hierarchy_categories[
                brand_name
            ].add(level)

            hierarchy_category_to_brand_keys[
                level
            ].add(brand_key)


# ============================================================
# 6. CATEGORY COMPATIBILITY
# ============================================================
def category_compatibility(
    brand_name,
    row_basic,
    row_primary,
    row_hierarchy,
):
    known_basic = brand_basic_categories.get(
        brand_name,
        set()
    )

    known_primary = brand_primary_categories.get(
        brand_name,
        set()
    )

    known_hierarchy = (
        brand_hierarchy_categories.get(
            brand_name,
            set()
        )
    )

    basic = (
        None
        if is_blank(row_basic)
        else str(row_basic).strip()
    )

    primary = (
        None
        if is_blank(row_primary)
        else str(row_primary).strip()
    )

    levels = hierarchy_levels(
        row_hierarchy
    )

    deeper_levels = (
        set(levels[1:])
        if len(levels) > 1
        else set()
    )

    if (
        basic is not None
        and basic in known_basic
    ):
        return "compatible", "basic"

    if (
        primary is not None
        and primary in known_primary
    ):
        return "compatible", "primary"

    if (
        deeper_levels
        and (
            deeper_levels
            & known_hierarchy
        )
    ):
        return "compatible", "hierarchy"

    row_has_category_evidence = (
        basic is not None
        or primary is not None
        or bool(deeper_levels)
    )

    brand_has_category_evidence = (
        bool(known_basic)
        or bool(known_primary)
        or bool(known_hierarchy)
    )

    if (
        row_has_category_evidence
        and brand_has_category_evidence
    ):
        return "conflict", "none"

    return "unknown", "none"


def category_candidate_pool(
    row_basic,
    row_primary,
    row_hierarchy,
):
    """
    Use category evidence to narrow which known brands
    are plausible candidates.
    """
    strong_pool = set()

    if not is_blank(row_basic):
        basic = str(row_basic).strip()

        strong_pool.update(
            basic_category_to_brand_keys.get(
                basic,
                set()
            )
        )

    if not is_blank(row_primary):
        primary = str(
            row_primary
        ).strip()

        strong_pool.update(
            primary_category_to_brand_keys.get(
                primary,
                set()
            )
        )

    levels = hierarchy_levels(
        row_hierarchy
    )

    for level in levels[1:]:
        strong_pool.update(
            hierarchy_category_to_brand_keys.get(
                level,
                set()
            )
        )

    if strong_pool:
        return (
            list(strong_pool),
            "strong_category_pool",
        )

    if levels:
        top_pool = (
            top_category_to_brand_keys.get(
                levels[0],
                set()
            )
        )

        if top_pool:
            return (
                list(top_pool),
                "top_category_pool",
            )

    return (
        brand_keys,
        "all_brands_pool",
    )


# ============================================================
# 7. FUZZY NAME SCORES
# ============================================================
def detailed_name_scores(
    segment,
    brand_key,
):
    """
    Calculate several string similarity scores.

    We intentionally DO NOT use partial_ratio as the final
    decision score because the McDonald experiment showed that
    substring matching can score unrelated names very highly.
    """
    segment_compact = (
        segment.replace(" ", "")
    )

    brand_compact = (
        brand_key.replace(" ", "")
    )

    levenshtein_score = (
        100.0
        * Levenshtein.normalized_similarity(
            segment_compact,
            brand_compact,
        )
    )

    ratio_score = fuzz.ratio(
        segment,
        brand_key,
    )

    token_sort_score = (
        fuzz.token_sort_ratio(
            segment,
            brand_key,
        )
    )

    wratio_score = fuzz.WRatio(
        segment,
        brand_key,
    )

    # Conservative consensus:
    # median of the three full-match style scores.
    #
    # This prevents one unusually high score from dominating.
    consensus_score = sorted(
        [
            levenshtein_score,
            ratio_score,
            token_sort_score,
        ]
    )[1]

    return {
        "levenshtein_score": float(
            levenshtein_score
        ),
        "ratio_score": float(
            ratio_score
        ),
        "token_sort_score": float(
            token_sort_score
        ),
        "wratio_score": float(
            wratio_score
        ),
        "consensus_score": float(
            consensus_score
        ),
    }


def best_segment_for_brand(
    segments,
    brand_key,
):
    best = None

    brand_length = max(
        len(brand_key),
        1
    )

    for item in segments:
        segment = item["segment"]

        # Avoid comparing extremely different-length segments.
        length_ratio = (
            len(segment)
            / brand_length
        )

        if (
            length_ratio < 0.55
            or length_ratio > 1.45
        ):
            continue

        scores = detailed_name_scores(
            segment,
            brand_key,
        )

        # Tiny tie-break only:
        # if the matching phrase starts at the beginning,
        # prefer it slightly.
        tie_break = (
            0.25
            if item["start"] == 0
            else 0.0
        )

        ranking_score = (
            scores["consensus_score"]
            + tie_break
        )

        candidate = {
            **scores,
            "ranking_score": ranking_score,
            "segment": segment,
        }

        if (
            best is None
            or candidate["ranking_score"]
            > best["ranking_score"]
        ):
            best = candidate

    return best


# ============================================================
# 8. EXTRACT 50,000 TARGET ROWS
# ============================================================
print("\nExtracting 50,000 Canadian target rows...")

df = db.execute(
    """
    SELECT
        brand.names.primary AS overture_brand_name,
        names.primary AS location_name,

        basic_category,
        taxonomy.primary AS taxonomy_primary,
        taxonomy.hierarchy AS taxonomy_hierarchy,

        addresses[1].freeform AS address,

        ST_Y(geometry) AS latitude,
        ST_X(geometry) AS longitude,

        bbox.xmin AS bbox_xmin,
        bbox.ymin AS bbox_ymin,
        bbox.xmax AS bbox_xmax,
        bbox.ymax AS bbox_ymax

    FROM read_parquet(
        ?,
        hive_partitioning = 1
    )

    WHERE addresses[1].country = ?
      AND confidence > ?
      AND names.primary IS NOT NULL

    LIMIT ?
    """,
    [
        places_path,
        COUNTRY,
        MIN_CONFIDENCE,
        SAMPLE_LIMIT,
    ]
).df()

df["category_path"] = df.apply(
    make_category_path,
    axis=1,
)

df["bounding_box"] = df.apply(
    make_bbox,
    axis=1,
)


# ============================================================
# 9. MATCH ONE ROW
# ============================================================
def match_blank_brand(row):
    location_key = normalize_name(
        row["location_name"]
    )

    if not location_key:
        return {
            "brand_name": None,
            "decision": "unmatched",
            "method": "empty_location_name",
            "candidate_brand": None,
            "matched_segment": None,
            "category_status": "unknown",
            "category_match_level": "none",
            "candidate_pool_method": "none",
            "levenshtein_score": None,
            "ratio_score": None,
            "token_sort_score": None,
            "wratio_score": None,
            "consensus_score": None,
            "second_best_score": None,
            "winning_margin": None,
        }

    segments = make_segments(
        location_key
    )

    # --------------------------------------------------------
    # A. Exact known-brand phrase + compatible category
    # --------------------------------------------------------
    exact_candidates = []

    for item in segments:
        segment = item["segment"]

        if segment not in brand_key_set:
            continue

        brand_name = (
            canonical_brand_names[
                segment
            ]
        )

        category_status, match_level = (
            category_compatibility(
                brand_name,
                row["basic_category"],
                row["taxonomy_primary"],
                row["taxonomy_hierarchy"],
            )
        )

        exact_candidates.append(
            {
                "brand_key": segment,
                "brand_name": brand_name,
                "segment": segment,
                "segment_size": item["size"],
                "category_status": category_status,
                "category_match_level": match_level,
            }
        )

    compatible_exact = [
        item
        for item in exact_candidates
        if item["category_status"]
        == "compatible"
    ]

    if compatible_exact:
        # Prefer the longest exact brand phrase.
        compatible_exact.sort(
            key=lambda item:
            item["segment_size"],
            reverse=True,
        )

        best_exact = (
            compatible_exact[0]
        )

        return {
            "brand_name": best_exact[
                "brand_name"
            ],
            "decision": "automatic",
            "method": (
                "exact_brand_phrase_and_category"
            ),
            "candidate_brand": best_exact[
                "brand_name"
            ],
            "matched_segment": best_exact[
                "segment"
            ],
            "category_status": (
                "compatible"
            ),
            "category_match_level": (
                best_exact[
                    "category_match_level"
                ]
            ),
            "candidate_pool_method": (
                "exact_phrase"
            ),
            "levenshtein_score": 100.0,
            "ratio_score": 100.0,
            "token_sort_score": 100.0,
            "wratio_score": 100.0,
            "consensus_score": 100.0,
            "second_best_score": None,
            "winning_margin": None,
        }

    # --------------------------------------------------------
    # B. Fuzzy candidate generation
    # --------------------------------------------------------
    pool, pool_method = (
        category_candidate_pool(
            row["basic_category"],
            row["taxonomy_primary"],
            row["taxonomy_hierarchy"],
        )
    )

    if not pool:
        return {
            "brand_name": None,
            "decision": "unmatched",
            "method": "no_candidate_pool",
            "candidate_brand": None,
            "matched_segment": None,
            "category_status": "unknown",
            "category_match_level": "none",
            "candidate_pool_method": pool_method,
            "levenshtein_score": None,
            "ratio_score": None,
            "token_sort_score": None,
            "wratio_score": None,
            "consensus_score": None,
            "second_best_score": None,
            "winning_margin": None,
        }

    shortlist = process.extract(
        location_key,
        pool,
        scorer=fuzz.WRatio,
        limit=TOP_FUZZY_CANDIDATES,
        score_cutoff=55,
    )

    candidate_results = []

    for brand_key, _, _ in shortlist:
        best_segment = (
            best_segment_for_brand(
                segments,
                brand_key,
            )
        )

        if best_segment is None:
            continue

        brand_name = (
            canonical_brand_names[
                brand_key
            ]
        )

        category_status, match_level = (
            category_compatibility(
                brand_name,
                row["basic_category"],
                row["taxonomy_primary"],
                row["taxonomy_hierarchy"],
            )
        )

        candidate_results.append(
            {
                "brand_key": brand_key,
                "brand_name": brand_name,
                "category_status": category_status,
                "category_match_level": match_level,
                **best_segment,
            }
        )

    # Keep exact phrase candidates visible even if their
    # category conflicted, so they can be reviewed.
    for exact in exact_candidates:
        scores = detailed_name_scores(
            exact["segment"],
            exact["brand_key"],
        )

        candidate_results.append(
            {
                "brand_key": exact[
                    "brand_key"
                ],
                "brand_name": exact[
                    "brand_name"
                ],
                "category_status": exact[
                    "category_status"
                ],
                "category_match_level": exact[
                    "category_match_level"
                ],
                "segment": exact[
                    "segment"
                ],
                "ranking_score": (
                    scores[
                        "consensus_score"
                    ]
                ),
                **scores,
            }
        )

    if not candidate_results:
        return {
            "brand_name": None,
            "decision": "unmatched",
            "method": "no_fuzzy_candidate",
            "candidate_brand": None,
            "matched_segment": None,
            "category_status": "unknown",
            "category_match_level": "none",
            "candidate_pool_method": pool_method,
            "levenshtein_score": None,
            "ratio_score": None,
            "token_sort_score": None,
            "wratio_score": None,
            "consensus_score": None,
            "second_best_score": None,
            "winning_margin": None,
        }

    # Keep only the best score for each brand.
    best_by_brand = {}

    for item in candidate_results:
        brand_name = item["brand_name"]

        current = best_by_brand.get(
            brand_name
        )

        if (
            current is None
            or item["consensus_score"]
            > current["consensus_score"]
        ):
            best_by_brand[
                brand_name
            ] = item

    ranked = sorted(
        best_by_brand.values(),
        key=lambda item:
        item["consensus_score"],
        reverse=True,
    )

    best = ranked[0]

    second_best_score = (
        ranked[1]["consensus_score"]
        if len(ranked) > 1
        else None
    )

    winning_margin = (
        best["consensus_score"]
        - second_best_score
        if second_best_score
        is not None
        else None
    )

    brand_key_tokens = (
        best["brand_key"].split()
    )

    is_short_brand = (
        len(brand_key_tokens) == 1
        and len(best["brand_key"]) <= 6
    )

    required_threshold = (
        SHORT_BRAND_THRESHOLD
        if is_short_brand
        else NORMAL_THRESHOLD
    )

    margin_ok = (
        winning_margin is None
        or winning_margin
        >= MIN_WINNING_MARGIN
    )

    if (
        best["consensus_score"]
        >= required_threshold
        and best["category_status"]
        == "compatible"
        and margin_ok
    ):
        return {
            "brand_name": best[
                "brand_name"
            ],
            "decision": "automatic",
            "method": (
                "fuzzy_name_and_category"
            ),
            "candidate_brand": best[
                "brand_name"
            ],
            "matched_segment": best[
                "segment"
            ],
            "category_status": best[
                "category_status"
            ],
            "category_match_level": best[
                "category_match_level"
            ],
            "candidate_pool_method": (
                pool_method
            ),
            "levenshtein_score": round(
                best[
                    "levenshtein_score"
                ],
                2,
            ),
            "ratio_score": round(
                best["ratio_score"],
                2,
            ),
            "token_sort_score": round(
                best[
                    "token_sort_score"
                ],
                2,
            ),
            "wratio_score": round(
                best["wratio_score"],
                2,
            ),
            "consensus_score": round(
                best[
                    "consensus_score"
                ],
                2,
            ),
            "second_best_score": (
                None
                if second_best_score
                is None
                else round(
                    second_best_score,
                    2,
                )
            ),
            "winning_margin": (
                None
                if winning_margin is None
                else round(
                    winning_margin,
                    2,
                )
            ),
        }

    if (
        best["consensus_score"]
        >= REVIEW_THRESHOLD
    ):
        return {
            "brand_name": None,
            "decision": "manual_review",
            "method": (
                "fuzzy_candidate_review"
            ),
            "candidate_brand": best[
                "brand_name"
            ],
            "matched_segment": best[
                "segment"
            ],
            "category_status": best[
                "category_status"
            ],
            "category_match_level": best[
                "category_match_level"
            ],
            "candidate_pool_method": (
                pool_method
            ),
            "levenshtein_score": round(
                best[
                    "levenshtein_score"
                ],
                2,
            ),
            "ratio_score": round(
                best["ratio_score"],
                2,
            ),
            "token_sort_score": round(
                best[
                    "token_sort_score"
                ],
                2,
            ),
            "wratio_score": round(
                best["wratio_score"],
                2,
            ),
            "consensus_score": round(
                best[
                    "consensus_score"
                ],
                2,
            ),
            "second_best_score": (
                None
                if second_best_score
                is None
                else round(
                    second_best_score,
                    2,
                )
            ),
            "winning_margin": (
                None
                if winning_margin is None
                else round(
                    winning_margin,
                    2,
                )
            ),
        }

    return {
        "brand_name": None,
        "decision": "unmatched",
        "method": "low_name_similarity",
        "candidate_brand": best[
            "brand_name"
        ],
        "matched_segment": best[
            "segment"
        ],
        "category_status": best[
            "category_status"
        ],
        "category_match_level": best[
            "category_match_level"
        ],
        "candidate_pool_method": (
            pool_method
        ),
        "levenshtein_score": round(
            best[
                "levenshtein_score"
            ],
            2,
        ),
        "ratio_score": round(
            best["ratio_score"],
            2,
        ),
        "token_sort_score": round(
            best[
                "token_sort_score"
            ],
            2,
        ),
        "wratio_score": round(
            best["wratio_score"],
            2,
        ),
        "consensus_score": round(
            best[
                "consensus_score"
            ],
            2,
        ),
        "second_best_score": (
            None
            if second_best_score is None
            else round(
                second_best_score,
                2,
            )
        ),
        "winning_margin": (
            None
            if winning_margin is None
            else round(
                winning_margin,
                2,
            )
        ),
    }


def process_row(row):
    overture_brand = (
        row["overture_brand_name"]
    )

    # --------------------------------------------------------
    # Brand already provided by Overture:
    # keep it directly.
    # --------------------------------------------------------
    if not is_blank(
        overture_brand
    ):
        key = normalize_name(
            overture_brand
        )

        brand_name = (
            canonical_brand_names.get(
                key,
                str(
                    overture_brand
                ).strip(),
            )
        )

        return {
            "brand_name": brand_name,
            "decision": "overture_provided",
            "method": "overture_brand",
            "candidate_brand": brand_name,
            "matched_segment": None,
            "category_status": (
                "provided_by_overture"
            ),
            "category_match_level": (
                "provided_by_overture"
            ),
            "candidate_pool_method": (
                "not_needed"
            ),
            "levenshtein_score": None,
            "ratio_score": None,
            "token_sort_score": None,
            "wratio_score": None,
            "consensus_score": None,
            "second_best_score": None,
            "winning_margin": None,
        }

    # --------------------------------------------------------
    # Blank brand:
    # only infer if evidence is strong enough.
    # --------------------------------------------------------
    return match_blank_brand(row)


# ============================================================
# 10. PROCESS ALL 50,000 ROWS
# ============================================================
print("\nMatching brands...")

match_results = df.apply(
    process_row,
    axis=1,
    result_type="expand",
)

for column in match_results.columns:
    df[column] = match_results[column]


# ============================================================
# 11. OUTPUT 1: LOCATION LEVEL
# ============================================================
# Same columns as your previous required output:
#
# brand_name
# location_name
# address
# latitude
# longitude
# bounding_box
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

location_level_path = os.path.join(
    OUTPUT_DIR,
    "overture_ca_50000_location_level.csv",
)

location_level.to_csv(
    location_level_path,
    index=False,
)

print(
    "Saved:",
    location_level_path,
)


# ============================================================
# 12. OUTPUT 2: BRAND LEVEL
# ============================================================
branded_rows = df[
    df["brand_name"].notna()
    & (
        df["brand_name"]
        .astype(str)
        .str.strip()
        != ""
    )
].copy()


def most_common_category(series):
    clean = [
        str(value).strip()
        for value in series
        if not is_blank(value)
    ]

    if not clean:
        return None

    return Counter(
        clean
    ).most_common(1)[0][0]


brand_level = (
    branded_rows
    .groupby(
        "brand_name",
        as_index=False,
    )
    .agg(
        number_of_locations=(
            "location_name",
            "size",
        ),
        categories=(
            "category_path",
            most_common_category,
        ),
    )
    .sort_values(
        [
            "number_of_locations",
            "brand_name",
        ],
        ascending=[
            False,
            True,
        ],
    )
    .reset_index(
        drop=True
    )
)

brand_level_path = os.path.join(
    OUTPUT_DIR,
    "overture_ca_50000_brand_level.csv",
)

brand_level.to_csv(
    brand_level_path,
    index=False,
)

print(
    "Saved:",
    brand_level_path,
)


# ============================================================
# 13. OUTPUT 3: INFERRED CONCLUSIONS
# ============================================================
# This file contains ONLY rows where:
#
#   Overture brand was blank
#   AND
#   this program concluded a brand automatically.
#
# It is an inspection/audit file.
# It is NOT an accuracy report.
conclusions = df[
    df["decision"].eq(
        "automatic"
    )
].copy()

conclusions = conclusions[
    [
        "brand_name",
        "candidate_brand",
        "location_name",
        "address",
        "basic_category",
        "taxonomy_primary",
        "category_path",
        "latitude",
        "longitude",
        "bounding_box",
        "method",
        "matched_segment",
        "category_status",
        "category_match_level",
        "candidate_pool_method",
        "levenshtein_score",
        "ratio_score",
        "token_sort_score",
        "wratio_score",
        "consensus_score",
        "second_best_score",
        "winning_margin",
    ]
].copy()

conclusions_path = os.path.join(
    OUTPUT_DIR,
    "overture_ca_50000_inferred_conclusions.csv",
)

conclusions.to_csv(
    conclusions_path,
    index=False,
)

print(
    "Saved:",
    conclusions_path,
)


# ============================================================
# 14. TERMINAL SUMMARY
# ============================================================
total_rows = len(df)

provided_count = int(
    df["decision"]
    .eq("overture_provided")
    .sum()
)

inferred_count = int(
    df["decision"]
    .eq("automatic")
    .sum()
)

review_count = int(
    df["decision"]
    .eq("manual_review")
    .sum()
)

unmatched_count = int(
    df["decision"]
    .eq("unmatched")
    .sum()
)

final_brand_count = int(
    df["brand_name"]
    .notna()
    .sum()
)

mcdonald_final = int(
    df["brand_name"]
    .fillna("")
    .apply(normalize_name)
    .eq("mcdonalds")
    .sum()
)

mcdonald_inferred = int(
    (
        df["decision"].eq(
            "automatic"
        )
        & df["brand_name"]
        .fillna("")
        .apply(normalize_name)
        .eq("mcdonalds")
    ).sum()
)

print("\n" + "=" * 65)
print("FINAL SUMMARY")
print("=" * 65)

print("\n50,000-row Canada sample")
print("------------------------")
print(
    "Total rows:",
    total_rows,
)
print(
    "Brand provided by Overture:",
    provided_count,
)
print(
    "Brand newly inferred automatically:",
    inferred_count,
)
print(
    "Possible match but left for review:",
    review_count,
)
print(
    "Still unmatched / blank:",
    unmatched_count,
)
print(
    "Final rows with a brand:",
    final_brand_count,
)

print("\nMcDonald's")
print("----------")
print(
    "Final McDonald's locations:",
    mcdonald_final,
)
print(
    "McDonald's locations newly inferred:",
    mcdonald_inferred,
)

print("\nIMPORTANT")
print("---------")
print(
    "These are matching conclusions, not an accuracy score."
)
print(
    "Rows are only auto-filled when name evidence is strong, "
    "category evidence is compatible, and the best candidate "
    "is sufficiently separated from the second-best candidate."
)

print("\nFiles created")
print("-------------")
print(
    location_level_path
)
print(
    brand_level_path
)
print(
    conclusions_path
)

db.close()
