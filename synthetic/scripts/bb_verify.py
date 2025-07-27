#!/usr/bin/env python3
"""
Batch: desenha bounding boxes COCO e salva todas as imagens anotadas,
ignorando warnings de ICC profile e buscando imagens recursivamente.
"""

import os
import sys
import json
import cv2
import glob
import argparse
import random

# Suprime os libpng warnings do OpenCV
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

def parse_args():
   home = os.path.expanduser("~")
   default_images = os.path.join(home, "IC_Petrobras-Repositorio_4", "synthetic", "dataset", "output", "coco_data", "images")
   default_annotations = os.path.join(home, "IC_Petrobras-Repositorio_4", "synthetic", "dataset", "output", "coco_data", "instances_coco_output.json")
   default_output = os.path.join(home, "IC_Petrobras-Repositorio_4", "synthetic", "dataset", "output", "BB")
   parser = argparse.ArgumentParser(
       description="Batch: desenha bounding boxes COCO e salva todas as imagens anotadas"
   )
   parser.add_argument(
       "--images", "-i", type=str, default=default_images,
       help="Diretório base com as imagens COCO (procura recursivamente)"
   )
   parser.add_argument(
       "--annotations", "-a", type=str, default=default_annotations,
       help="Arquivo JSON COCO (ex: instances_coco_output.json)"
   )
   parser.add_argument(
       "--output", "-o", type=str, default=default_output,
       help="Diretório de saída para imagens anotadas"
   )
   return parser.parse_args()

def find_annotation_file(path):
   if os.path.isfile(path):
       return path
   dir_ = os.path.dirname(path)
   candidates = glob.glob(os.path.join(dir_, "*.json"))
   if candidates:
       print(f"[!] JSON não encontrado em '{path}', usando '{candidates[0]}'")
       return candidates[0]
   print(f"[ERROR] Nenhum JSON de anotações em '{dir_}'.")
   sys.exit(1)

def load_coco_annotations(json_path):
   with open(json_path, 'r') as f:
       coco = json.load(f)
   images_info = {img['id']: img for img in coco.get('images', [])}
   annos_per_image = {}
   for ann in coco.get('annotations', []):
       annos_per_image.setdefault(ann['image_id'], []).append(ann)
   return images_info, annos_per_image

def find_image_path(img_dir, fname):
   """
   Busca recursivamente pelo basename(fname) dentro de img_dir.
   """
   bname = os.path.basename(fname)
   for root, _, files in os.walk(img_dir):
       if bname in files:
           return os.path.join(root, bname)
   return None

def main():
   args = parse_args()
   img_dir   = args.images
   json_path = find_annotation_file(args.annotations)
   out_dir   = args.output

   os.makedirs(out_dir, exist_ok=True)

   images_info, annos_per_image = load_coco_annotations(json_path)
   category_colors = {}
   total = len(images_info)
   saved = 0

   print(f"[+] {total} imagens no JSON COCO.")
   print(f"[+] Salvando anotadas em '{out_dir}'\n")

   for img_id, info in images_info.items():
       fname = info['file_name']
       img_path = find_image_path(img_dir, fname)
       if not img_path:
           print(f"[!] Não achei '{fname}', pulando.")
           continue

       img = cv2.imread(img_path)
       if img is None:
           print(f"[!] Falha ao ler '{img_path}', pulando.")
           continue

       annotated = img.copy()
       for ann in annos_per_image.get(img_id, []):
           x, y, bw, bh = ann['bbox']
           x1, y1 = int(x), int(y)
           x2, y2 = int(x + bw), int(y + bh)
           cat = ann['category_id']
           if cat not in category_colors:
               random.seed(cat)
               category_colors[cat] = tuple(random.randint(0,255) for _ in range(3))
           color = category_colors[cat]
           cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
           cv2.putText(
               annotated, str(cat), (x1, y1-5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
           )

       out_name = os.path.basename(fname)
       save_path = os.path.join(out_dir, out_name)
       cv2.imwrite(save_path, annotated)
       saved += 1

   print(f"\n[✔] Concluído: {saved}/{total} imagens salvas em '{out_dir}'")

if __name__ == "__main__":
   main()

""""
    python3 bb_verify.py \
  --images ~/IC_Petrobras-Repositorio_4/synthetic/dataset/output/coco_data/images \
  --annotations ~/IC_Petrobras-Repositorio_4/synthetic/dataset/output/coco_data/instances_coco_output.json \
  --output   ~/IC_Petrobras-Repositorio_4/synthetic/dataset/output/BB
"""""