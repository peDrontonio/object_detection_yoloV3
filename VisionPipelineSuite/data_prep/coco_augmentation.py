import os
import json
import cv2
import random
from tqdm import tqdm
from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, RandomBrightnessContrast,
    HueSaturationValue, ShiftScaleRotate, PadIfNeeded
)

def load_coco_annotation(json_path, img_dir, start_img_id, start_ann_id):
    with open(json_path, 'r') as f:
        data = json.load(f)

    images = []
    annotations = []

    img_id_map = {}
    for img in data["images"]:
        new_id = start_img_id
        img_id_map[img["id"]] = new_id
        img["id"] = new_id
        img["file_name"] = os.path.join(img_dir, img["file_name"])
        images.append(img)
        start_img_id += 1

    for ann in data["annotations"]:
        ann["image_id"] = img_id_map[ann["image_id"]]
        ann["id"] = start_ann_id
        annotations.append(ann)
        start_ann_id += 1

    return images, annotations, data["categories"], start_img_id, start_ann_id

def augment_image(img, anns, transform):
    valid = [ann for ann in anns if ann["bbox"][2] > 0 and ann["bbox"][3] > 0]
    if not valid:
        return img, []  # skip se não tiver bbox válida

    bboxes = [ann["bbox"] for ann in valid]
    labels = [ann["category_id"] for ann in valid]
    aug = transform(image=img, bboxes=bboxes, category_id=labels)
    new_img = aug["image"]

    new_anns = []
    for (x, y, w, h), cat_id in zip(aug["bboxes"], aug["category_id"]):
        new_anns.append({
        "bbox": [x, y, w, h],
        "category_id": int(cat_id),
         "iscrowd": 0,
        "area": w * h,
    })
    return new_img, new_anns

def main():
    json_dirs = [
        ("/home/nexus/IC_Petrobras-Repositorio_4/synthetic/dataset/eletroquad_dataprep/inside_00/00_inside/images/default", "/home/nexus/IC_Petrobras-Repositorio_4/synthetic/dataset/eletroquad_dataprep/inside_00/00_inside/Output/instances_coco_output.json"),
        ("/home/nexus/IC_Petrobras-Repositorio_4/synthetic/dataset/eletroquad_dataprep/inside_01/images/default", "/home/nexus/IC_Petrobras-Repositorio_4/synthetic/dataset/eletroquad_dataprep/inside_01/Output/instances_coco_output.json"),
        ("/home/nexus/IC_Petrobras-Repositorio_4/synthetic/dataset/eletroquad_dataprep/inside_02/images/default", "/home/nexus/IC_Petrobras-Repositorio_4/synthetic/dataset/eletroquad_dataprep/inside_02/Output/instances_coco_output.json"),
        ("/home/nexus/IC_Petrobras-Repositorio_4/synthetic/dataset/eletroquad_dataprep/inside_03/images/default", "/home/nexus/IC_Petrobras-Repositorio_4/synthetic/dataset/eletroquad_dataprep/inside_03/Output/instances_coco_output.json"),
        ("/home/nexus/IC_Petrobras-Repositorio_4/synthetic/dataset/eletroquad_dataprep/outside_04/images/default", "/home/nexus/IC_Petrobras-Repositorio_4/synthetic/dataset/eletroquad_dataprep/outside_04/Output/instances_coco_output.json"),
        ("/home/nexus/IC_Petrobras-Repositorio_4/synthetic/dataset/eletroquad_dataprep/outside_05/images/default", "/home/nexus/IC_Petrobras-Repositorio_4/synthetic/dataset/eletroquad_dataprep/outside_05/Output/instances_coco_output.json"),
    ]

    output_img_dir = "/home/nexus/IC_Petrobras-Repositorio_4/synthetic/dataset/eletroquad_dataprep/Output/augmented_img"
    output_json_path = "/home/nexus/IC_Petrobras-Repositorio_4/synthetic/dataset/eletroquad_dataprep/Output/instaces_aug_coco.json"
    os.makedirs(output_img_dir, exist_ok=True)

    all_images, all_annotations = [], []
    categories = []
    img_id, ann_id = 1, 1

    transform = Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.2),
        RandomBrightnessContrast(p=0.3),
        HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.3),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
        PadIfNeeded(min_height=512, min_width=512, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0)
    ], bbox_params={'format': 'coco', 'label_fields': ['category_id']})

    for img_dir, ann_path in json_dirs:
        imgs, anns, cats, img_id, ann_id = load_coco_annotation(ann_path, img_dir, img_id, ann_id)
        if not categories:
            categories = cats

        for img_info in tqdm(imgs, desc=f"Augmenting {ann_path}"):
            img = cv2.imread(img_info["file_name"])
            if img is None:
                print(f"⚠️ Erro ao ler {img_info['file_name']}")
                continue

            img_anns = [ann for ann in anns if ann["image_id"] == img_info["id"]]

            for i in range(3):  # N versões por imagem
                aug_img, aug_anns = augment_image(img, img_anns, transform)
                new_file_name = f"{os.path.splitext(os.path.basename(img_info['file_name']))[0]}_aug_{i}.jpg"
                new_path = os.path.join(output_img_dir, new_file_name)
                cv2.imwrite(new_path, aug_img)

                new_img_id = img_id
                img_id += 1
                all_images.append({
                    "id": new_img_id,
                    "file_name": new_file_name,
                    "width": aug_img.shape[1],
                    "height": aug_img.shape[0]
                })

                for ann in aug_anns:
                    ann.update({"id": ann_id, "image_id": new_img_id})
                    all_annotations.append(ann)
                    ann_id += 1

    coco_dict = {
        "images": all_images,
        "annotations": all_annotations,
        "categories": categories
    }

    with open(output_json_path, 'w') as f:
        json.dump(coco_dict, f, indent=2)
    print("✅ Augmentação finalizada e COCO salvo!")

if __name__ == "__main__":
    main()
