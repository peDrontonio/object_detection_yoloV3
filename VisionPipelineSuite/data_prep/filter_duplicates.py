import hashlib
import json
from collections import defaultdict
from pathlib import Path

def hash_image(file_path):
    """
    Generate a hash for an image file to check for duplicates.
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def scan_for_duplicates(folder_path):
    """
    Scan a folder for image duplicates and return a dictionary
    mapping unique IDs to lists of duplicate filenames.
    """
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder path {folder_path} does not exist.")

    image_hashes = defaultdict(list)

    for file_path in folder_path.rglob("*"):
        if file_path.is_file():
            try:
                file_hash = hash_image(file_path)
                image_hashes[file_hash].append(file_path)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    duplicate_dict = {}
    for idx, (hash_key, file_paths) in enumerate(image_hashes.items()):
        if len(file_paths) > 1:
            unique_id = f"{idx:04d}"
            duplicate_dict[unique_id] = file_paths

    return duplicate_dict

def get_annotation_count(coco_data, image_id):
    """
    Get the number of annotations for a given image ID in the COCO JSON data.
    """
    return sum(1 for ann in coco_data["annotations"] if ann["image_id"] == image_id)

def _update_coco_json(coco_data, files_to_keep):
    """
    Update the COCO JSON file by removing references to deleted images and reassigning image IDs.
    """

    image_info = {img["file_name"]: img for img in coco_data["images"]}
    updated_images = []
    updated_annotations = []
    # used_file_names = set()

    # # Add images without duplicates to used_file_names
    # duplicate_files = {path.name for paths in duplicates.values() for path in paths}
    # for image in coco_data["images"]:
    #     if image["file_name"] not in duplicate_files:
    #         used_file_names.add(image["file_name"])

    # # Find images to keep from duplicates
    # for unique_id, file_paths in duplicates.items():
    #     annotations_count = [
    #         (path, get_annotation_count(coco_data, image_info[path.name]["id"]))
    #         for path in file_paths
    #     ]
    #     annotations_count.sort(key=lambda x: x[1], reverse=True)
    #     file_to_keep = annotations_count[0][0]
    #     used_file_names.add(file_to_keep.name)

    # Update images and annotations with reordered IDs
    new_image_id_map = {}
    for new_id, file_name in enumerate(sorted(files_to_keep), start=1):
        image = image_info[file_name]
        new_image_id_map[image["id"]] = new_id
        image["id"] = new_id
        updated_images.append(image)

    for annotation in coco_data["annotations"]:
        if annotation["image_id"] in new_image_id_map:
            annotation["image_id"] = new_image_id_map[annotation["image_id"]]
            updated_annotations.append(annotation)

    # Save the updated COCO JSON
    coco_data["images"] = updated_images
    coco_data["annotations"] = updated_annotations

    return coco_data

def delete_duplicates(duplicates, coco_file):
    """
    Delete duplicate images and update the COCO JSON file.
    """
    
    files_to_keep = set()

    with open(coco_file, "r") as f:
        coco_data = json.load(f)

    # Add images without duplicates to used_file_names
    duplicate_files = {path.name for paths in duplicates.values() for path in paths}
    for image in coco_data["images"]:
        if image["file_name"] not in duplicate_files:
            files_to_keep.add(image["file_name"])

    # Find images to keep from duplicates
    # The criteria is to keep the image with most annotations
    for unique_id, file_paths in duplicates.items():
        if len(file_paths) < 2:
            continue

        annotations_count = [
            (path, get_annotation_count(coco_data, path)) for path in file_paths
        ]
        annotations_count.sort(key=lambda x: x[1], reverse=True)
        file_to_keep = annotations_count[0][0]
        files_to_keep.add(file_to_keep.name)

        for file_path, _ in annotations_count[1:]:
            file_path.unlink()

    new_coco_data = _update_coco_json(coco_data, files_to_keep)

    with open(coco_file, "w") as f:
        json.dump(new_coco_data, f, indent=4)

if __name__ == "__main__":
    folder_path = input("Enter the path to the folder containing images: ").strip()
    coco_file = input("Enter the path to the COCO JSON file: ").strip()

    duplicates = scan_for_duplicates(folder_path)

    if duplicates:
        print("Found duplicate images. Processing...")
        delete_duplicates(duplicates, coco_file)
        print("Duplicates removed and COCO JSON updated.")
    else:
        print("No duplicate images found.")
