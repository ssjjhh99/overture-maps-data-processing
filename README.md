# overture-maps-data-processing
Processing and cleaning Overture Maps Places data through category grouping, brand matching, and building-based bounding box refinement.

# Overture Maps Data Processing

This project processes and evaluates Overture Maps Places data, with a focus on category grouping, brand matching, and bounding box refinement using the Overture Buildings dataset.

## Project Overview

The project was developed in several stages:

1. Explore the Overture Places schema and available category fields.
2. Evaluate Overture's category hierarchy and broader category groupings.
3. Develop a simple brand-matching method.
4. Improve brand matching using fuzzy string matching and category compatibility.
5. Evaluate the original bounding boxes in the Places dataset.
6. Match Places with Overture Buildings and use Building bounding boxes as improved bounding box candidates.

## Repository Structure

```text
overture-maps-data-processing/
│
├── README.md
├── .gitignore
│
├── code/
│   ├── 01_inspect_overture_places_schema.py
│   ├── 02_compare_overture_category_fields.py
│   ├── 03_test_basic_category_grouping.py
│   ├── 04_brand_matching_simple_3000.py
│   ├── 05_brand_matching_fuzzy_canada_validation.py
│   ├── 06_brand_matching_fuzzy_canada_final.py
│   ├── 07_visualize_original_place_bbox.py
│   └── 08_match_places_to_buildings_final_bbox.py
│
└── outputs/
    ├── 01_sample_original_overture_places_10_rows.csv
    ├── 02_places_grouped_by_basic_category_3000.csv
    ├── 03_final_brand_matching_location_level_canada_50000.csv
    ├── 04_final_brand_matching_brand_level_canada_50000.csv
    ├── 05_final_brand_matching_inferred_matches_canada_50000.csv
    ├── 06_final_bbox_building_matching_canada_1000.csv
    ├── 07_original_place_bbox_visualization.html
    └── 08_building_bbox_validation_visualization.html
