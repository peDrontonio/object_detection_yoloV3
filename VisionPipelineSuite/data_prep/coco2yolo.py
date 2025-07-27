import json
import argparse
import logging
import cv2
import numpy as np

from pathlib import Path
from pycocotools import mask as maskUtils
from joblib import Parallel, delayed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def coco2yolo(dataset_path, mode, custom_yaml_data_path=None):
    """Process COCO annotations and generate YOLO or KITTI dataset files.

    Args:
        dataset_path (str): Path to the root directory of the dataset.
        dataset_splits (list): List of dataset splits to process (e.g., ['train', 'val', 'test']).
        mode (str): Processing mode ('detection', 'segmentation', 'od_kitti', or 'pose_detection').
    """
    dataset_root = Path(dataset_path)

    labels_dir = dataset_root / "labels"
    splits = [f.name for f in labels_dir.iterdir() if f.is_dir()]

    for split in splits:

        coco_path = labels_dir / split / "coco.json"

        if not coco_path.exists():
            logger.error(f"File not found: {coco_path}")
            continue

        with open(coco_path) as f:
            data = json.load(f)
        
        image_info = {img['id']: img for img in data['images']}
        
        annotations = process_annotations_parallel(image_info, data, mode)

        create_annotation_files(annotations, coco_path.parent)

        if split not in ["val", "test"]:
            create_yaml_file(dataset_root, custom_yaml_data_path, data, mode, split)

def convert_bounding_boxes(size, box, category_id):
    """Convert COCO bounding box format to YOLO format.

    Args:
        size (tuple): Image dimensions (width, height).
        box (list): COCO bounding box [x_min, y_min, width, height].
        category_id (int): Category ID for the object.

    Returns:
        str: YOLO formatted bounding box.
    """
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2] / 2.0) * dw
    y = (box[1] + box[3] / 2.0) * dh
    w = box[2] * dw
    h = box[3] * dh
    return f"{category_id} {x} {y} {w} {h}"

def convert_pose_keypoints(size, box, keypoints, category_id):
    """Convert COCO pose keypoints to YOLO format.

    Args:
        size (tuple): Image dimensions (width, height).
        box (list): COCO bounding box [x_min, y_min, width, height].
        keypoints (list): Keypoints in COCO format [x1, y1, v1, x2, y2, v2, ...].
        category_id (int): Category ID for the object.

    Returns:
        str: YOLO formatted bounding box and keypoints.
    """
    yolo_bbox = convert_bounding_boxes(size, box, category_id)
    dw = 1. / size[0]
    dh = 1. / size[1]
    yolo_keypoints = [(keypoints[i] * dw, keypoints[i + 1] * dh, keypoints[i + 2]) for i in range(0, len(keypoints), 3)]
    keypoints_str = ' '.join([f"{kp[0]} {kp[1]} {kp[2]}" for kp in yolo_keypoints])
    return f"{yolo_bbox} {keypoints_str}"

def convert_segmentation_masks(size, segmentation_mask, category_id, min_pixels=0):
    """Convert COCO segmentation masks (RLE or polygon) to YOLO format with filtering by pixel area.

    Args:
        size (tuple): Image dimensions (width, height).
        segmentation_mask (dict or list): COCO segmentation mask (RLE or polygon).
        category_id (int): Category ID for the object.
        min_pixels (int): Minimum number of pixels required for a mask region to be included.

    Returns:
        str: The mask's annotation string line.
    """
    width, height = size
    annotation_line = f"{category_id}"

    if isinstance(segmentation_mask, dict) and 'counts' in segmentation_mask:  # RLE mask
        rle = maskUtils.frPyObjects(segmentation_mask, segmentation_mask['size'][0], segmentation_mask['size'][1])
        binary_mask = maskUtils.decode(rle)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) <= min_pixels:  # Filter by the area of the contour
                continue
            norm_coords = contour.flatten().astype(np.float32)
            norm_coords[0::2] = np.round(norm_coords[0::2] / width, 5)
            norm_coords[1::2] = np.round(norm_coords[1::2] / height, 5)
            annotation_line += ' ' + ' '.join(map(str, norm_coords))

    elif isinstance(segmentation_mask, list):  # Polygon format
        for polygon in segmentation_mask:
            poly_array = np.array(polygon).reshape(-1, 2)  # Reshape into a 2D array
            if cv2.contourArea(poly_array) <= min_pixels:  # Filter by the area of the polygon
                continue
            norm_coords = np.array(polygon).astype(np.float32)
            norm_coords[0::2] = np.round(norm_coords[0::2] / width, 5)
            norm_coords[1::2] = np.round(norm_coords[1::2] / height, 5)
            annotation_line += ' ' + ' '.join(map(str, norm_coords))

    return annotation_line

def convert_segmentation_masks_direct(size, segmentation_mask, category_id, min_pixels=20):
    """Convert COCO segmentation masks (RLE or polygon) to YOLO format string, optimized for speed.

    Args:
        size (tuple): Image dimensions (width, height).
        segmentation_mask (dict or list): COCO segmentation mask (RLE or polygon).
        category_id (int): Category ID for the object.
        min_pixels (int): Minimum number of pixels required for a mask region to be included.

    Returns:
        str: YOLO formatted segmentation mask as a single line.
    """
    width, height = size
    annotation_line = f"{category_id}"

    if isinstance(segmentation_mask, dict) and 'counts' in segmentation_mask:  # RLE mask
        # Decode RLE into binary mask
        rle = maskUtils.frPyObjects(segmentation_mask, segmentation_mask['size'][0], segmentation_mask['size'][1])
        binary_mask = maskUtils.decode(rle)
        binary_mask = filter_small_regions(binary_mask, 50)

        # Find all non-zero pixels in the mask
        rows, cols = np.nonzero(binary_mask)  # Faster than np.argwhere

        norm_coords = np.vstack((cols / width, rows / height)).T.flatten()
        annotation_line += ' ' + ' '.join(map(str, norm_coords))

    elif isinstance(segmentation_mask, list):  # Polygon format
        for polygon in segmentation_mask:
            poly_array = np.array(polygon).reshape(-1, 2)  # Reshape into a 2D array
            if len(poly_array) <= min_pixels:  # Filter small polygons
                continue

            # Normalize polygon coordinates directly
            norm_coords = []
            for x, y in poly_array:
                norm_coords.append(round(x / width, 5))  # Normalize x
                norm_coords.append(round(y / height, 5))  # Normalize y

            # Add normalized coordinates to the annotation line
            annotation_line += ' ' + ' '.join(map(str, norm_coords))

    return annotation_line
    

def convert_coco_to_kitti(size, box, category_name):
    """Convert COCO bounding box format to KITTI format.

    Args:
        size (tuple): Image dimensions (width, height).
        box (list): COCO bounding box [x_min, y_min, width, height].
        category_name (str): Category name for the object.

    Returns:
        str: KITTI formatted bounding box.
    """
    x1, y1 = box[0], box[1]
    x2, y2 = box[0] + box[2], box[1] + box[3]
    return f"{category_name} 0 0 0 {x1} {y1} {x2} {y2} 0 0 0 0 0 0 0"

def process_annotations(image_info, data, mode):
    """Process COCO annotations into the desired format based on mode.

    Args:
        image_info (dict): Mapping of image IDs to image metadata.
        data (dict): COCO dataset JSON data.
        mode (str): Processing mode ('detection', 'segmentation', 'od_kitti', or 'pose_detection').

    Returns:
        dict: Mapping of image filenames to annotation lines.
    """
    annotations_by_image = {}
    is_pose_estimation = mode.startswith("pose")
    for ann in data['annotations']:
        img_id = ann['image_id']
        coco_bbox = ann['bbox']
        category_id = ann['category_id'] - 1
        category_name = next((cat['name'] for cat in data['categories'] if cat['id'] == ann['category_id']), "unknown")
        img_filename = Path(image_info[img_id]['file_name'])
        img_size = (image_info[img_id]['width'], image_info[img_id]['height'])

        if img_filename not in annotations_by_image:
            annotations_by_image[img_filename] = []

        match mode:
            case "detection" | "pose_detection":
                if is_pose_estimation and 'keypoints' in ann:
                    annotation_line = convert_pose_keypoints(img_size, coco_bbox, ann['keypoints'], category_id)
                else:
                    annotation_line = convert_bounding_boxes(img_size, coco_bbox, category_id)        

            case "segmentation":
                annotation_line = convert_segmentation_masks_direct(img_size, ann["segmentation"], category_id)

            case "od_kitti":
                annotation_line = convert_coco_to_kitti(img_size, coco_bbox, category_name)
        
        annotations_by_image[img_filename].append(annotation_line)
    
    return annotations_by_image

def process_annotations_parallel(image_info, data, mode, n_jobs=-1):
    """Process COCO annotations into the desired format based on mode, with parallelization.

    Args:
        image_info (dict): Mapping of image IDs to image metadata.
        data (dict): COCO dataset JSON data.
        mode (str): Processing mode ('detection', 'segmentation', 'od_kitti', or 'pose_detection').
        n_jobs (int): Number of parallel jobs to run. Default is -1 (use all available processors).

    Returns:
        dict: Mapping of image filenames to annotation lines.
    """
    # Process all annotations in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_annotation)(ann, image_info, data, mode) for ann in data['annotations']
    )

    # Group annotations by image filename
    annotations_by_image = {}
    for img_filename, annotation_line in results:
        if img_filename not in annotations_by_image:
            annotations_by_image[img_filename] = []
        annotations_by_image[img_filename].append(annotation_line)
    
    return annotations_by_image

def process_single_annotation(ann, image_info, data, mode):
    """Process a single COCO annotation based on mode."""
    img_id = ann['image_id']
    coco_bbox = ann['bbox']
    category_id = ann['category_id'] - 1
    category_name = next((cat['name'] for cat in data['categories'] if cat['id'] == ann['category_id']), "unknown")
    img_filename = Path(image_info[img_id]['file_name'])
    img_size = (image_info[img_id]['width'], image_info[img_id]['height'])

    match mode:
        case "detection" | "pose_detection":
            if mode.startswith("pose") and 'keypoints' in ann:
                annotation_line = convert_pose_keypoints(img_size, coco_bbox, ann['keypoints'], category_id)
            else:
                annotation_line = convert_bounding_boxes(img_size, coco_bbox, category_id)
        
        case "segmentation":
            annotation_line = convert_segmentation_masks_direct(img_size, ann["segmentation"], category_id)
        
        case "od_kitti":
            annotation_line = convert_coco_to_kitti(img_size, coco_bbox, category_name)
    
    return img_filename, annotation_line

def filter_small_regions(binary_mask, min_pixels):
    """Remove small blob regions from a binary mask.

    Args:
        binary_mask (numpy.ndarray): Binary mask with regions to filter.
        min_pixels (int): Minimum number of pixels required to retain a region.

    Returns:
        numpy.ndarray: Filtered binary mask with small regions removed.
    """
    # Perform connected components analysis
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    # Create an empty mask to store the filtered result
    filtered_mask = np.zeros_like(binary_mask, dtype=np.uint8)

    # Iterate over each region, skipping the background (label 0)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]  # Get the area of the region
        if area > min_pixels:  # Retain regions larger than the threshold
            filtered_mask[labels == i] = 1

    return filtered_mask
def create_annotation_files(annotations_by_image, output_dir):
    """Write annotation files for each image.

    Args:
        annotations_by_image (dict): Mapping of image filenames to annotation lines.
        output_dir (Path): Directory where annotation files will be written.
    """
    for img_filename, annotations in annotations_by_image.items():
        txt_path = output_dir / (img_filename.stem + '.txt')
        try:
            with open(txt_path, 'w') as file:
                file.write("\n".join(annotations) + "\n")
            logger.debug(f"Processed annotation for image: {img_filename}")
        except IOError as e:
            logger.error(f"Error writing to file {txt_path}: {e}")

def create_yaml_file(dataset_path, custom_yaml_data_path, data, mode, split=None):
    """Generate a YAML configuration file for the dataset.

    Args:
        dataset_path (Path): Path to the root directory of the dataset.
        data_train_split (str): Dataset split used for training (e.g., 'train').
        mode (str): Processing mode ('detection', 'segmentation', 'od_kitti', or 'pose_detection').
    """

    yaml_path = dataset_path / (split + ".yaml")
    train_path = "images/" + split
    val_path = "images/val"

    class_names = {category['id'] - 1: category['name'] for category in data['categories']}
    sorted_class_names = sorted(class_names.items())
    class_entries = "\n".join([f"  {id}: {name}" for id, name in sorted_class_names])

    if custom_yaml_data_path:
        dataset_path = Path(custom_yaml_data_path)
    
    yaml_content = f"""path: {dataset_path.absolute()}  # dataset root dir
train: {train_path}  # train images (relative to 'path')
val: {val_path}  # val images (relative to 'path')
test:  # test images (optional)

# Classes
names:
{class_entries}
    """

    if mode.startswith("pose"):
        categories = data['categories']
        keypoints_info = categories[0].get('keypoints', [])
        kpt_shape = [len(keypoints_info), 3]

        yaml_content += f"\n\n# Keypoints\nkpt_shape: {kpt_shape}"

    try:
        with open(yaml_path, 'w') as file:
            file.write(yaml_content.strip())
        logger.info(f"YAML file created at {yaml_path}")
    except IOError as e:
        logger.error(f"Error writing to file {yaml_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process COCO annotations and create YOLO or KITTI dataset.")
    parser.add_argument("dataset_path", type=str, help="Path to the root directory of the dataset.")
    parser.add_argument("--mode", choices=["detection", "segmentation", "od_kitti", "pose_detection"], default="detection",
                        help="Choose processing mode: 'detection' for bounding boxes, 'segmentation' for segmentation masks, 'pose_detection' for pose estimation.")
    parser.add_argument("--custom_yaml_data_path", type=str, help=" A custom data path string to overwrite the yolo's data yaml file. Useful for running when training on different machines")

    args = parser.parse_args()

    coco2yolo(**vars(args))
