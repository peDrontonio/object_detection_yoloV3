
#Import librarys
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import json
import cv2
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

print(f"Available GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True  # Otimiza para tamanhos fixos
    torch.backends.cudnn.enabled = True  # Ativa otimizações da NVIDIA
    torch.autocast("cuda", dtype=torch.float16).__enter__()
 
import gc
torch.cuda.empty_cache()
gc.collect()
 
#sam2 video_predictor    
from sam2.build_sam import build_sam2_video_predictor
print(os.getcwd())  # Prints the current working directory
print("step 1 complete")
sam2_checkpoint = "/home/nexus/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "//home/nexus/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
print("load hydra")

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))




# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
video_dir = "/home/nexus/sam2/sam2/sam_test/images/default"

# Load COCO annotations from a JSON file
coco_json_path ="/home/nexus/sam2/sam2/sam_test/annotations/instances_default.json"
# Update this with the actual path

# Load COCO annotations from JSON
with open(coco_json_path, "r") as f:
    coco_data = json.load(f)

# Create a dictionary mapping category IDs to their actual names
category_id_to_name = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

# Debugging: Print the category mapping
print("Loaded category mapping:", category_id_to_name)
print("step 2 complete json files and annotations up")

coco = COCO(coco_json_path)
image_ids = coco.getImgIds()
# Get image IDs from annotations
total_annotated_images = len(image_ids)
print(f"Total annotated images: {total_annotated_images}")


# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))


inference_state = predictor.init_state(video_path=video_dir)
predictor.reset_state(inference_state)
print("step 3 inferente_state start")

prompts = {}  # hold all the clicks we add for visualization

coco = COCO(coco_json_path)
image_ids = coco.getImgIds()
# Get image IDs from annotations
total_annotated_images = len(image_ids)
print(f"Total annotated images: {total_annotated_images}")

# Verify how many images are available
print(f"Total images in dataset: {len(image_ids)}")

# Select and process 3 different images
for i, img_id in enumerate(image_ids[:3]):  # Pick 3 different images
    image_data = coco.loadImgs(img_id)[0]
    image_name = image_data["file_name"].replace(".png", ".jpg")  # Troca a extensão
    image_path = os.path.join(video_dir, image_name)


    # Debugging print
    print(f"Processing Image {i+1}: ID={img_id}, Path={image_path}")

    # Ensure the file exists before loading
    if not os.path.exists(image_path):
        print(f"WARNING: Image file not found at {image_path}")
        continue

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Get annotations for the current image
    ann_ids = coco.getAnnIds(imgIds=img_id)
    annotations = coco.loadAnns(ann_ids)

    # Extract bounding boxes and segmentation masks
    masks = []
    bboxes = []
    for ann in annotations:
        if "segmentation" in ann and ann["segmentation"]:
            rle = mask_utils.frPyObjects(ann["segmentation"], image_data["height"], image_data["width"])
            mask = mask_utils.decode(rle)
            if mask is not None:
                masks.append(mask)
        if "bbox" in ann:
            bboxes.append(ann["bbox"])

    masks = np.array(masks) if masks else None  # Ensure valid format

    # Process full-frame annotations with SAM2
    ann_frame_idx = 0  # Frame index
    prompts = {}  # Reset prompts for each image
    for ann, bbox in zip(annotations, bboxes):
        ann_obj_id = ann["id"]  # Unique object ID
        points = np.array([[bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2]], dtype=np.float32)  # Center of bbox
        labels = np.array([1], np.int32)  # Positive click
        prompts[ann_obj_id] = points, labels
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )
# Directory paths
output_dir = "/home/nexus/sam2/sam2/fotos_mn/output/output_im"
mask_output_dir = "/home/nexus/sam2/sam2/fotos_mn/output/mask_output"
coco_annotations_path = "/home/nexus/sam2/sam2/fotos_mn/mn_data/output/coco_annotations.json"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(mask_output_dir, exist_ok=True)
print("teste")
# Criar diretório de saída se não existir
os.makedirs(os.path.dirname(coco_annotations_path), exist_ok=True)
#cache
torch.cuda.empty_cache()
gc.collect()
 
# Run propagation throughout the video and collect the results in a dictionary
video_segments = {}  # Contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# Load COCO annotations from JSON
with open(coco_json_path, "r") as f:
    coco_data = json.load(f)

# Create a mapping from category ID to category name
category_id_to_name = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

# Create a mapping from object ID to category ID using annotations
obj_to_category = {}
for ann in coco_data["annotations"]:
    obj_to_category[ann["id"]] = ann["category_id"]

# Initialize COCO annotation dictionary
coco_annotations = {
    "images": [],
    "annotations": [],
    "categories": []
}

category_map = {}
category_id = 1
annotation_id = 1

# Render and save every frame with segmentation results
plt.close("all")
for out_frame_idx in range(len(frame_names)):
    frame_path = os.path.join(video_dir, frame_names[out_frame_idx])
    frame_image = Image.open(frame_path)
    width, height = frame_image.size

    # Save original frame with segmentation overlay
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(frame_image)

    # Initialize a blank mask image
    mask_image = np.zeros((height, width), dtype=np.uint8)

    # Add image metadata to COCO format
    image_info = {
        "id": out_frame_idx,
        "file_name": frame_names[out_frame_idx],
        "width": width,
        "height": height
    }
    coco_annotations["images"].append(image_info)

    # Overlay each object's mask
    for i, (out_obj_id, out_mask) in enumerate(video_segments.get(out_frame_idx, {}).items()):
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

        # Ensure out_mask is 2D
        if out_mask.ndim == 3:
            out_mask = out_mask.squeeze(axis=0)

        # Assign unique grayscale intensity for each object
        mask_image[out_mask > 0] = (i + 1) * 30  # Avoid 0 to distinguish background

        # Convert mask to COCO RLE format
        rle = mask_utils.encode(np.asfortranarray(out_mask.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("utf-8")

        # Extract segmentation contours as polygons
        contours, _ = cv2.findContours(out_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        segmentation = [contour.flatten().tolist() for contour in contours if len(contour) >= 3]

        # Get the correct category ID from obj_to_category mapping
        category_id = obj_to_category.get(out_obj_id, None)

        if category_id is not None:
            # Retrieve category name from COCO dataset
            category_name = category_id_to_name.get(category_id, f"Unknown_{category_id}")

            # Assign category if not already mapped
            if category_id not in category_map:
                category_map[category_id] = category_id  # Ensure the correct mapping

                # Store category in COCO format
                coco_annotations["categories"].append({
                    "id": category_id,  # Ensure it's the actual category ID
                    "name": category_name,  # Correct object name from JSON
                    "supercategory": "object"
                })

            # Debugging: Print assigned category name
            print(f"Assigned category for obj_id {out_obj_id}: {category_name} (ID {category_id})")

            # Add annotation entry
            annotation = {
                "id": annotation_id,
                "image_id": out_frame_idx,
                "category_id": category_id,
                "segmentation": segmentation,
                "area": int(mask_utils.area(rle)),
                "bbox": mask_utils.toBbox(rle).tolist(),
                "iscrowd": 0
            }
            coco_annotations["annotations"].append(annotation)
            annotation_id += 1

    # Save the figure with overlays
    output_path = os.path.join(output_dir, f"frame_{out_frame_idx}.png")
    plt.savefig(output_path)
    plt.close()

    # Save the separate mask image
    mask_output_path = os.path.join(mask_output_dir, f"mask_{out_frame_idx}.png")
    Image.fromarray(mask_image).save(mask_output_path)

    print(f"Saved: {output_path} and {mask_output_path}")

# Save COCO annotations to JSON file
with open(coco_annotations_path, "w") as f:
    json.dump(coco_annotations, f, indent=4)

print("Processing complete. Check", output_dir, mask_output_dir, "and", coco_annotations_path)


        
