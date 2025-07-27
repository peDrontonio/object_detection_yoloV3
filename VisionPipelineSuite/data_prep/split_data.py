import json
import shutil
import random
import argparse
import logging
from pathlib import Path
from collections import Counter

import pandas as pd
from sklearn.model_selection import KFold

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def split_data(images_dir, coco_json_path, output_dir, 
               train_ratio=0.75, val_ratio=0.1, ablation=0, k=0, pose_estimation=False, 
               rename_images=False, classes=[]):
    """
    Splits a COCO dataset into training, validation, and testing sets or creates k-fold splits.
    """
    logger.info("Loading COCO annotations...")
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    images_dir = Path(images_dir)
    output_dir = Path(output_dir)

    if not images_dir.exists():
        logger.error(f"Images directory does not exist: {images_dir}")
        return
    
    if classes:
        coco_data = filter_coco_by_classes(coco_data, classes)

    images = coco_data.get('images', [])
    annotations = coco_data.get('annotations', [])
    categories = coco_data.get('categories', [])

    logger.info(f"Total annotated images: {len(images)}")
    logger.info(f"Total annotated objects: {len(annotations)}")

    if k > 0:
        create_kfold_splits_with_dataframe(images, annotations, categories, images_dir, output_dir, k, rename_images, pose_estimation)
    elif ablation > 0:
        splits = generate_splits(images, train_ratio, val_ratio, ablation)
        process_splits(splits, images_dir, annotations, categories, output_dir, rename_images, pose_estimation)
    else:
        splits = generate_splits(images, train_ratio, val_ratio, ablation=0)
        process_splits(splits, images_dir, annotations, categories, output_dir, rename_images, pose_estimation)

def create_kfold_splits_with_dataframe(images, annotations, categories, images_dir, output_dir, k, rename_images, pose_estimation):
    """
    Create k-fold splits for the dataset using a pandas DataFrame of class counts per image.
    """
    logger.info("Creating class count DataFrame...")

    # Create a mapping from image IDs to their file names
    image_id_to_filename = {image['id']: image['file_name'] for image in images}

    # Create a mapping from category IDs to their names
    category_id_to_name = {category['id']: category['name'] for category in categories}

    # Initialize a DataFrame
    index = [image['file_name'] for image in images]  # Use image file names as index
    labels_df = pd.DataFrame(0, columns=category_id_to_name.values(), index=index)

    # Populate the DataFrame with object counts per class for each image
    for annotation in annotations:
        image_id = annotation['image_id']
        category_id = annotation['category_id']

        if image_id in image_id_to_filename:
            image_name = image_id_to_filename[image_id]
            class_name = category_id_to_name[category_id]
            labels_df.loc[image_name, class_name] += 1

    labels_df = labels_df.fillna(0)  # Replace NaN values with 0
    logger.info("Class count DataFrame created.")

    # Perform k-fold splitting
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(labels_df)):
        fold_name = f"fold_{fold+1}"
        logger.info(f"Processing {fold_name}...")

        # Get the training and validation file names
        train_files = labels_df.index[train_idx].tolist()
        val_files = labels_df.index[val_idx].tolist()

        # Filter images and annotations for each split
        train_images = [img for img in images if img['file_name'] in train_files]
        val_images = [img for img in images if img['file_name'] in val_files]

        train_annotations = filter_annotations(train_images, annotations, pose_estimation)
        val_annotations = filter_annotations(val_images, annotations, pose_estimation)

        # Prepare directories
        # train_images_path = output_dir / fold_name / "images" 
        # val_images_path = output_dir / fold_name / "images" 

        train_images_path = output_dir / fold_name / "images" / "train"
        train_labels_path = output_dir / fold_name / "labels" / "train"
        val_images_path = output_dir / fold_name / "images" / "val"
        val_labels_path = output_dir / fold_name / "labels" / "val"
        
        train_images_path.mkdir(parents=True, exist_ok=True)
        train_labels_path.mkdir(parents=True, exist_ok=True)
        val_images_path.mkdir(parents=True, exist_ok=True)
        val_labels_path.mkdir(parents=True, exist_ok=True)

        # Copy images and create subsets
        train_updated_images = copy_images(train_images, images_dir, train_images_path, rename_images)
        val_updated_images = copy_images(val_images, images_dir, val_images_path, rename_images)

        train_coco = create_coco_subset(train_updated_images, train_annotations, categories)
        val_coco = create_coco_subset(val_updated_images, val_annotations, categories)

        # Save JSON files
        with open(train_labels_path / "coco.json", 'w') as f:
            json.dump(train_coco, f, indent=4)
        with open(val_labels_path / "coco.json", 'w') as f:
            json.dump(val_coco, f, indent=4)

        logger.info(f"{fold_name} split completed.")

def process_splits(splits, images_dir, annotations, categories, output_dir, rename_images, pose_estimation):
    """
    Process and save splits for training, validation, and testing.
    """
    for split_name, images_set in splits.items():
        try:
            # Create directories
            images_output_path = output_dir / "images" / split_name
            labels_output_path = output_dir / "labels" / split_name
            images_output_path.mkdir(parents=True, exist_ok=True)
            labels_output_path.mkdir(parents=True, exist_ok=True)

            # Copy images and create COCO subset
            updated_images = copy_images(images_set, images_dir, images_output_path, rename_images)
            filtered_annotations = filter_annotations(updated_images, annotations, pose_estimation)
            coco_subset = create_coco_subset(updated_images, filtered_annotations, categories)

            # Save COCO JSON
            with open(labels_output_path / "coco.json", 'w') as file:
                json.dump(coco_subset, file, indent=4)

            # Save metadata
            save_metadata(output_dir, split_name, coco_subset)

            logger.info(f"Dataset for {split_name} saved successfully.")
        except Exception as e:
            logger.error(f"Failed to process data for {split_name}: {e}")

def generate_splits(images, train_ratio, val_ratio, ablation):
    """
    Generates a dictionary mapping split names to image lists based on given ratios and ablation settings.
    """
    random.shuffle(images)
    total_images = len(images)

    if ablation > 0:
        val_size = int(total_images * val_ratio)
        val_images = images[:val_size]
        ablation_images = images[val_size:]

        ablation_chunks = [int(len(ablation_images) * (i + 1) / ablation) for i in range(ablation)]
        splits = {"val": val_images}
        for i, chunk_size in enumerate(ablation_chunks):
            percentage = f"{int((chunk_size / len(images)) * 100)}%"
            splits[percentage] = ablation_images[:chunk_size]
    else:
        train_end = int(total_images * train_ratio)
        val_end = train_end + int(total_images * val_ratio)

        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:],
        }

    return splits

def copy_images(images, src_dir, dest_dir, rename_images, name_padding=5):
    """
    Copy selected images to a specified directory, and optionally rename them with new numerical IDs.
    Update the images' filenames in the dataset metadata if renamed.
    """

    id_format = f"{{:0{str(name_padding)}d}}"

    updated_images = []
    for image in images:
        try:
            image_path = Path(image['file_name'])

            if rename_images:
                if image_path.suffix in ['.jpg', '.jpeg']:
                    image_name = image_path.stem + '.jpeg'
                else:
                    image_name = image_path.name
               
                # If rename_images is True, rename files using numerical IDs
                new_file_name = id_format.format(image["id"]) + "." + image_name.split(".")[-1]

                # Update the image metadata to reflect the new filename
                updated_image = image.copy()
                updated_image['file_name'] = new_file_name
                updated_images.append(updated_image)
           
            else:
                updated_images = images
            
            src_path = Path(src_dir) / image['file_name']
            dest_path = Path(dest_dir) / new_file_name
            shutil.copy(src_path, dest_path)
            logger.debug(f"Successfully copied {src_path} to {dest_path}")
        
        except Exception as e:
            logger.error(f"Failed to copy {src_path} to {dest_path}: {e}")

    return updated_images

def filter_annotations(images_set, annotations, pose_estimation):
    """
    Filter annotations to include only those for the provided image set.
    """
    image_ids = {image['id'] for image in images_set}
    if pose_estimation:
        return [annotation for annotation in annotations if annotation['image_id'] in image_ids and 'keypoints' in annotation]
    else:
        return [annotation for annotation in annotations if annotation['image_id'] in image_ids]

def filter_coco_by_classes(coco_data, classes):
    """
    Filters the COCO data to only include specified classes.
    """
    category_name_to_id = {category['name']: category['id'] for category in coco_data['categories']}

    selected_category_ids = [category_name_to_id[name] for name in classes if name in category_name_to_id]

    if not selected_category_ids:
        logger.error(f"No matching categories found for classes: {classes}")
        raise ValueError(f"No matching categories found for classes: {classes}")

    filtered_annotations = [annotation for annotation in coco_data['annotations'] if annotation['category_id'] in selected_category_ids]

    image_ids = {annotation['image_id'] for annotation in filtered_annotations}

    filtered_images = [image for image in coco_data['images'] if image['id'] in image_ids]

    filtered_categories = [category for category in coco_data['categories'] if category['id'] in selected_category_ids]

    filtered_coco_data = {
        'images': filtered_images,
        'annotations': filtered_annotations,
        'categories': filtered_categories
    }

    return filtered_coco_data

def create_coco_subset(images, annotations, categories):
    """
    Create a COCO-formatted subset from images, annotations, and categories.
    """
    return {
        'images': images,
        'annotations': annotations,
        'categories': categories
    }

def log_object_count_per_class(coco_data):
    """
    Logs the total number of objects for each class in the COCO dataset.
    """
    category_counts = {category['name']: 0 for category in coco_data['categories']}
    for annotation in coco_data['annotations']:
        category_id = annotation['category_id']
        category_name = next((cat['name'] for cat in coco_data['categories'] if cat['id'] == category_id), None)
        if category_name:
            category_counts[category_name] += 1

    logger.info("Object counts per class:")
    for category, count in category_counts.items():
        logger.info(f"  {category}: {count}")
    return category_counts

def save_metadata(output_dir, split_name, coco_subset):
    """
    Save metadata, such as class-wise object counts, to a text file.
    """
    chunk_category_counts = log_object_count_per_class(coco_subset)
    meta_file_path = output_dir / f".{split_name}_meta.txt"
    with open(meta_file_path, 'w') as meta_file:
        meta_file.write("Class-wise Object Counts:\n")
        for category, count in chunk_category_counts.items():
            meta_file.write(f"{category}: {count}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split COCO dataset into training, validation, testing sets or k-fold splits.")
    parser.add_argument("images_dir", help="Path to the input directory containing images.")
    parser.add_argument("coco_json_path", help="Path to the COCO JSON file containing annotations.")
    parser.add_argument("output_dir", help="Path to the root output directory for splits.")
    parser.add_argument("--k", type=int, default=0, help="Number of folds for k-fold cross-validation (default: 0, i.e., no k-fold split).")
    parser.add_argument("--train_ratio", type=float, default=0.75, help="Proportion of images for training")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Proportion of images for validation")
    parser.add_argument("--ablation", type=int, default=0, help="Number of dataset chunks for ablation testing")
    parser.add_argument("--rename_images", action="store_true", default=True, help="Assign new numerical IDs to image file names")
    parser.add_argument("--classes", nargs='+', default=[], help="List of class names to process (default: all classes)")

    args = parser.parse_args()

    split_data(
        images_dir=args.images_dir,
        coco_json_path=args.coco_json_path,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        ablation=args.ablation,
        k=args.k,
        rename_images=args.rename_images,
        classes=args.classes
    )
