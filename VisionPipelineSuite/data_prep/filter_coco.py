#!/usr/bin/env python3

import json
import logging
import argparse
from pathlib import Path

def filter_coco_annotations(input_json_path: Path, output_json_path: Path, classes_to_keep: list):
    """
    Reads a COCO annotations JSON from `input_json_path`, filters out annotations
    to keep only the specified `classes_to_keep`, reassigns category and annotation
    IDs sequentially, and saves the filtered dataset to `output_json_path`.
    """

    logging.info(f"Reading input annotations from: {input_json_path}")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # Build a mapping of category_name -> category_info
    # category_info is the entire category dict (id, name, supercategory, etc.)
    name_to_cat = {cat['name']: cat for cat in coco_data['categories']}

    # Determine which category IDs we should keep
    keep_category_ids = []
    for cls in classes_to_keep:
        if cls not in name_to_cat:
            logging.warning(f"Class '{cls}' not found in 'categories'. Skipping.")
        else:
            keep_category_ids.append(name_to_cat[cls]['id'])

    keep_category_ids = set(keep_category_ids)
    logging.debug(f"Category IDs to keep: {keep_category_ids}")

    # Filter annotations
    original_anns_count = len(coco_data['annotations'])
    filtered_annotations = [
        ann for ann in coco_data['annotations'] if ann['category_id'] in keep_category_ids
    ]
    logging.info(
        f"Number of original annotations: {original_anns_count}. "
        f"Number after filtering: {len(filtered_annotations)}"
    )

    # Filter categories based on keep_category_ids
    filtered_categories = [
        cat for cat in coco_data['categories'] if cat['id'] in keep_category_ids
    ]

    # Filter images to keep only those referenced by the filtered annotations
    keep_image_ids = {ann['image_id'] for ann in filtered_annotations}
    filtered_images = [
        img for img in coco_data['images'] if img['id'] in keep_image_ids
    ]
    logging.info(
        f"Number of original images: {len(coco_data['images'])}. "
        f"Number after filtering: {len(filtered_images)}"
    )

    # --- Reassign Category IDs Sequentially ---
    # We map old_category_id -> new_category_id in ascending order (1..N).
    # The order is based on the appearance in `filtered_categories`.
    cat_id_map = {}
    for i, cat in enumerate(filtered_categories, start=1):
        old_id = cat['id']
        cat_id_map[old_id] = i

    # Update category dicts with new IDs
    for cat in filtered_categories:
        old_id = cat['id']
        cat['id'] = cat_id_map[old_id]

    # --- Reassign Annotation IDs and Category IDs ---
    for i, ann in enumerate(filtered_annotations, start=1):
        # Reassign annotation ID
        ann['id'] = i
        # Update category ID based on the cat_id_map
        old_cat_id = ann['category_id']
        ann['category_id'] = cat_id_map[old_cat_id]

    # Build the new COCO dataset
    filtered_coco = {
        "info": coco_data.get("info", {}),
        "licenses": coco_data.get("licenses", []),
        "images": filtered_images,
        "annotations": filtered_annotations,
        "categories": filtered_categories
    }

    logging.info(f"Writing filtered annotations to: {output_json_path}")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_coco, f, ensure_ascii=False, indent=2)

def main():
    """Main function for parsing arguments, setting up logging, and calling filter function."""
    parser = argparse.ArgumentParser(description="Filter and reassign IDs in a COCO-style JSON.")
    parser.add_argument("--input-json", type=str, required=True, help="Path to the input COCO JSON.")
    parser.add_argument("--output-json", type=str, required=True, help="Path to save the filtered JSON.")
    parser.add_argument("--classes-to-keep", nargs="+", required=True, help="List of category names to retain.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (e.g., DEBUG, INFO).")

    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    logging.info("Starting COCO filtering script...")
    input_path = Path(args.input_json).resolve()
    output_path = Path(args.output_json).resolve()

    if not input_path.exists():
        logging.error(f"Input file does not exist: {input_path}")
        return

    filter_coco_annotations(input_json_path=input_path, output_json_path=output_path, classes_to_keep=args.classes_to_keep)
    logging.info("Filtering completed successfully.")

if __name__ == "__main__":
    main()
