import os
import json
import argparse
import base64
import zlib
import cv2
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(labels, images):

    labels_path = labels
    images_path = images

    output_dir = os.path.join(labels_path, 'yolo_annotations')
    os.makedirs(output_dir, exist_ok=True)

    class_mapping = {}

    label_files = [f for f in os.listdir(labels_path) if f.endswith('.json')]

    if not label_files:
        logger.error("No JSON label files found in the labels directory.")
        return

    for label_file in label_files:
        label_path = os.path.join(labels_path, label_file)
        process_annotation_file(label_path, output_dir, class_mapping)

    if class_mapping:
        save_class_mapping(class_mapping, output_dir)
        create_yaml_file(os.path.dirname(labels_path), class_mapping)

    else:
        logger.warning("No classes found in annotations.")

def decode_bitmap(data):
    """
    Decodes the bitmap data from the JSON annotation.

    Parameters:
        data (str): Base64-encoded, zlib-compressed image data.

    Returns:
        np.ndarray: The decoded binary mask as a NumPy array.
    """
    try:
        compressed_data = base64.b64decode(data)

        decompressed_data = zlib.decompress(compressed_data)

        image_array = np.frombuffer(decompressed_data, dtype=np.uint8)

        mask = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)

        if mask is None:
            logger.error("Failed to decode bitmap to image.")
            return None

        # Ensure mask is single-channel
        if len(mask.shape) == 3 and mask.shape[2] == 4:
            # Extract the alpha channel
            mask = mask[:, :, 3]
        elif len(mask.shape) == 3:
            # Convert RGB to grayscale
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        elif len(mask.shape) == 2:
            # Mask is already single-channel
            pass
        else:
            logger.error(f"Unexpected mask shape: {mask.shape}")
            return None

        # Threshold the mask to make sure it's binary
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        return mask

    except Exception as e:
        logger.error(f"Error decoding bitmap: {e}")
        return None


def get_segmentation_points(mask):
    """
    Extracts the segmentation points from the binary mask.

    Parameters:
        mask (np.ndarray): Binary mask of the object.

    Returns:
        list: A list of (x, y) coordinates representing the segmentation polygon.
    """
    try:
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            logger.warning("No contours found in mask.")
            return []

        # Assume the largest contour is the object
        largest_contour = max(contours, key=cv2.contourArea)

        # Flatten the contour array and convert to a list
        segmentation = largest_contour.flatten().tolist()

        return segmentation

    except Exception as e:
        logger.error(f"Error extracting segmentation points: {e}")
        return []

def process_annotation_file(label_path, output_dir, class_mapping):
    """
    Processes a single annotation JSON file and writes the YOLO segmentation annotation.

    Parameters:
        label_path (str): Path to the JSON label file.
        output_dir (str): Directory where the YOLO annotation files will be saved.
        class_mapping (dict): Dictionary mapping class titles to class IDs.

    Returns:
        None
    """
    try:
        with open(label_path, 'r') as f:
            data = json.load(f)

        image_width = data['size']['width']
        image_height = data['size']['height']
        objects = data.get('objects', [])

        if not objects:
            logger.info(f"No objects found in {label_path}.")
            return

        yolo_annotations = []
        for obj in objects:
            class_title = obj['classTitle']

            # Map class_title to class_id
            class_id = class_mapping.setdefault(class_title, len(class_mapping))

            if obj['geometryType'] == 'bitmap':
                bitmap = obj['bitmap']
                bitmap_data = bitmap['data']
                origin = bitmap['origin']  # [x, y]
                mask = decode_bitmap(bitmap_data)

                if mask is None:
                    continue

                # Get segmentation points
                segmentation = get_segmentation_points(mask)
                if not segmentation:
                    continue

                # Adjust segmentation points with the origin and normalize
                adjusted_segmentation = []
                for i in range(0, len(segmentation), 2):
                    x = (segmentation[i] + origin[0]) / image_width
                    y = (segmentation[i+1] + origin[1]) / image_height
                    adjusted_segmentation.extend([x, y])

                # Create the annotation line
                annotation_line = f"{class_id} " + ' '.join(map(str, adjusted_segmentation))
                yolo_annotations.append(annotation_line)
            else:
                logger.warning(f"Unsupported geometryType '{obj['geometryType']}' in {label_path}.")

        if yolo_annotations:
            # Write the annotations to a txt file
            image_filename = os.path.splitext(os.path.basename(label_path))[0]
            annotation_filename = image_filename + '.txt'
            annotation_path = os.path.join(output_dir, annotation_filename)

            with open(annotation_path, 'w') as f:
                for annotation in yolo_annotations:
                    f.write(annotation + '\n')
            logger.info(f"Processed annotations for {image_filename}.")
        else:
            logger.info(f"No valid annotations for {label_path}.")

    except Exception as e:
        logger.error(f"Error processing file {label_path}: {e}")

def save_class_mapping(class_mapping, output_dir):
    """
    Saves the class mapping to a 'classes.txt' file.

    Parameters:
        class_mapping (dict): Dictionary mapping class titles to class IDs.
        output_dir (str): Directory where the 'classes.txt' will be saved.

    Returns:
        None
    """
    try:
        class_mapping_path = os.path.join(output_dir, 'classes.txt')
        with open(class_mapping_path, 'w') as f:
            for class_title, class_id in sorted(class_mapping.items(), key=lambda x: x[1]):
                f.write(f"{class_title}\n")
        logger.info(f"Saved class mapping to {class_mapping_path}.")
    except Exception as e:
        logger.error(f"Error saving class mapping: {e}")

def create_yaml_file(dataset_path, class_mapping):
    """
    Creates a data.yaml file for YOLO training.

    Parameters:
        dataset_path (str): Path to the root of the dataset.
        class_mapping (dict): Dictionary mapping class titles to class IDs.

    Returns:
        None
    """
    try:
        # Ensure class names are in order of class IDs
        class_names = [None] * len(class_mapping)
        for class_title, class_id in class_mapping.items():
            class_names[class_id] = class_title

        yaml_content = f"""# YOLOv5 PyTorch Ultralytics format
path: {os.path.abspath(dataset_path)}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')
test:  # test images (optional)

# Classes
names:
"""

        for name in class_names:
            yaml_content += f"  - {name}\n"

        yaml_path = os.path.join(dataset_path, 'data.yaml')
        with open(yaml_path, 'w') as file:
            file.write(yaml_content.strip())
        logger.info(f"YAML file created at {yaml_path}")
    except Exception as e:
        logger.error(f"Error creating YAML file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert instance segmentation annotations to YOLO segmentation format.")
    parser.add_argument('--labels', type=str, required=True, help='Path to the labels folder.')
    parser.add_argument('--images', type=str, required=True, help='Path to the images folder.')
    args = parser.parse_args()
    
    main(*vars(args))
