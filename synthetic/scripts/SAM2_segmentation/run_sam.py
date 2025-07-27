#%%
import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import cv2
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from PIL import Image

from pathlib import Path
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

#%%
### FOLDERS ###
annotation_file = "/home/nexus/sam2/sam2/datasets/giovanna/annotations/instances_default.json"
video_dir = Path("/home/nexus/sam2/sam2/datasets/giovanna/images/default")

sam2_checkpoint = "/home/nexus/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "//home/nexus/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"

#%%
# scan all the JPEG frame names in this directory
frame_names = [ p for p in video_dir.iterdir()
    if p.suffix in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: p.stem.split("_")[-1])

if frame_names[0].stem != "00000":
    new_names = [p.with_name("0" + p.name.split("_")[-1]) for p in frame_names]
    for i, n in enumerate(new_names): frame_names[i].rename(n) 
else:
    new_names = frame_names

#%%
# select the device for computationA primeira coisa a fazer é garantir que o caminho do arquivo de configuração sam2.1_hiera_l.yaml esteja correto. O erro menciona que ele está procurando o arquivo em:
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )
# %%
from sam2.build_sam import build_sam2_video_predictor


predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

#%%
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


#%%
# take a look the first video frame
frame_idx = 0
plt.figure(figsize=(9, 6))
plt.title(f"frame {frame_idx}")
plt.imshow(Image.open(new_names[frame_idx]))
#%%
inference_state = predictor.init_state(video_path=str(video_dir.absolute()))
#%%
predictor.reset_state(inference_state)
#%%


def run_prediction_on_coco(annotation_file: Path, images_dir: Path, predictor, inference_state):
    """
    Loads COCO-style annotations, decodes RLE masks, and runs predictor.add_new_mask() on each mask.

    :param annotation_file: Path to COCO JSON annotation file.
    :param images_dir: Path to directory containing images.
    :param predictor: The predictor object with an add_new_mask() method.
    :param inference_state: Predictor's required inference state.
    """

    # Ensure paths are Path objects
    annotation_file = Path(annotation_file)
    images_dir = Path(images_dir)

    # Initialize COCO API
    coco = COCO(str(annotation_file))

    # Get all annotation IDs
    ann_ids = coco.getAnnIds()
    annotations = coco.loadAnns(ann_ids)
    cat_ids = coco.getCatIds()
    categories = coco.loadCats(cat_ids)

    category_mapping = {}
    for cat in categories:
        cat_name = cat['name']
        cat_id = cat['id']
        
        # Extract the last part of the name split by "_"
        try:
            object_id = int(cat_name.split("_")[-1])  # Convert last part to int
            category_mapping[cat_id] = object_id
        except ValueError:
            print(f"Warning: Could not extract integer from category name '{cat_name}'")
            continue

    # Loop through each annotation
    for ann in annotations:
        segm = ann.get('segmentation', None)
        
        if isinstance(segm, dict) and 'counts' in segm:  # RLE mask
            rle = maskUtils.frPyObjects(segm, segm['size'][0], segm['size'][1])
            binary_mask = maskUtils.decode(rle)
        else:
            print("Mask not in RLE encoding")
            continue

        # Use category_id as object_id (object class ID)
        category_id = ann['category_id']
        frame_idx = ann['image_id'] - 1 # Image ID as frame index

        object_id = category_mapping[category_id]
        # Run predictor on the mask
        labels = np.array([1], np.int32)
        _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=object_id,
            mask=binary_mask
        )

        # Post-processing (if needed)
        # Example: print object_id and frame_idx
        print(f"Processed mask for class {object_id} in frame {frame_idx}")


run_prediction_on_coco(annotation_file, video_dir, predictor, inference_state)

# show the results on the current (interacted) frame
# plt.figure(figsize=(9, 6))
# plt.title(f"frame {ann_frame_idx}")
# plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
# show_points(points, labels, plt.gca())
# show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

#%%
# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }
#%%
frames = [p for p in video_dir.iterdir()]
frames.sort(key=lambda p: p.stem)

#%%
def print_results(video_segments, frames, initial_frame=0, stride=10):
    plt.close("all")
    for out_frame_idx in range(initial_frame, initial_frame + stride):
        plt.figure(figsize=(6, 4))
        plt.title(f"frame {out_frame_idx}")
        plt.imshow(Image.open(frames[out_frame_idx]))
        if out_frame_idx in video_segments.keys():
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
# %%
print_results(video_segments, frames, initial_frame=44)
#%%
for k, v in video_segments.items():
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {k}")
    plt.imshow(Image.open(frames[k]))
    for out_obj_id, out_mask in v.items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

# %%
#Salva os frames + mascaras + anotacoes coco
output_dir = Path("/home/nexus/sam2/sam2/datasets/giovanna/output/im_output")
mask_output_dir = Path("/home/nexus/sam2/sam2/datasets/giovanna/output/mask_output")
coco_annotations_path = Path("/home/nexus/sam2/sam2/datasets/giovanna/output/instances_coco_output.json")

#verifica se os diretorios existem, sen ele cria
output_dir.mkdir(parents=True, exist_ok=True)
mask_output_dir.mkdir(parents=True, exist_ok=True)

with open(annotation_file, "r") as f:
    coco_data =json.load(f)

category_id_to_name = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
obj_to_category = {ann["id"]: ann["category_id"] for ann in coco_data["annotations"]}

coco_annotations = {"images": [], "annotations": [], "categories": []}
category_map = {}
annotation_id = 1

for out_frame_idx, frame_path in enumerate(frame_names):
    full_frame_path= os.path.join(video_dir, frame_path.name)
    frame_image = Image.open(full_frame_path)
    width, height = frame_image.size

    image_info = {
        "id": out_frame_idx,
        "file_name": frame_path.name,
        "width": width,
        "height": height
    }
    coco_annotations["images"].append(image_info)

    mask_image = np.zeros((height, width), dtype=np.uint8)
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(frame_image)

    for i, (out_obj_id, out_mask) in enumerate(video_segments.get(out_frame_idx, {}).items()):
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

        if out_mask.ndim == 3:
            out_mask = out_mask.squeeze(0)

        mask_image[out_mask > 0] = (i + 1) * 30

        rle = maskUtils.encode(np.asfortranarray(out_mask.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("utf-8")
        contours, _ = cv2.findContours(out_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        segmentation = [contour.flatten().tolist() for contour in contours if len(contour) >= 3]

        category_id = obj_to_category.get(out_obj_id)
        if category_id is not None:
            category_name = category_id_to_name.get(category_id, f"Unknown_{category_id}")
            if category_id not in category_map:
                category_map[category_id] = category_id
                coco_annotations["categories"].append({
                    "id": category_id,
                    "name": category_name,
                    "supercategory": "object"
                })

            coco_annotations["annotations"].append({
                "id": annotation_id,
                "image_id": out_frame_idx,
                "category_id": category_id,
                "segmentation": segmentation,
                "area": int(maskUtils.area(rle)),
                "bbox": maskUtils.toBbox(rle).tolist(),
                "iscrowd": 0
            })
            annotation_id += 1

    # Salva frame com máscara sobreposta
    plt.savefig(output_dir / f"frame_{out_frame_idx}.png")
    plt.close()

    # Salva imagem da máscara
    Image.fromarray(mask_image).save(mask_output_dir / f"mask_{out_frame_idx}.png")

# Salva anotações COCO
with open(coco_annotations_path, "w") as f:
    json.dump(coco_annotations, f, indent=4)

print("Everthing save")
#%%




