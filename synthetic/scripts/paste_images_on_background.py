#!/usr/bin/env python3
"""This script automatically pastes generated object images onto random backgrounds,
with visual progress feedback and default argument paths."""

import os
import random
import argparse
from PIL import Image

# Optional: visual progress bar; install via `pip install tqdm`
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--images",
        default="/home/pedrinho/IC_Petrobras-Repositorio_4/synthetic/dataset/output/coco_data/images",
        type=str,
        help="/home/pedrinho/IC_Petrobras-Repositorio_4/synthetic/dataset/output/coco_data/images"
    )
    parser.add_argument(
        "-b", "--backgrounds",
        default="/home/pedrinho/IC_Petrobras-Repositorio_4/synthetic/dataset/backgrounds2paste",
        type=str,
        help="Path to background images to paste on. Default: ./backgrounds"
    )
    parser.add_argument(
        "-t", "--types",
        default=("jpg", "jpeg", "png"),
        type=str,
        nargs='+',
        help="File types to consider. Default: jpg, jpeg, png."
    )
    parser.add_argument(
        "-w", "--overwrite",
        action="store_true",
        help="Overwrites original images. Default: False."
    )
    parser.add_argument(
        "-o", "--output",
        default="/home/pedrinho/IC_Petrobras-Repositorio_4/synthetic/dataset/output/coco_data/images",
        type=str,
        help="Output directory. Default: ./output"
    )
    args = parser.parse_args()

    # Determine and create output directory
    if args.overwrite:
        output_dir = args.images
    else:
        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)

    # Collect image files
    image_files = [f for f in os.listdir(args.images)
                   if f.lower().endswith(tuple(args.types))]
    total = len(image_files)

    # Paste each object image onto a random background
    for idx, file_name in enumerate(tqdm(image_files, desc="Processing images")):
        img_path = os.path.join(args.images, file_name)
        with Image.open(img_path) as img:
            img_w, img_h = img.size

            # Select a random background
            bg_candidates = [p for p in os.listdir(args.backgrounds)
                             if p.lower().endswith(tuple(args.types))]
            background_path = os.path.join(
                args.backgrounds,
                random.choice(bg_candidates)
            )
            with Image.open(background_path) as bg_img:
                background = bg_img.resize((img_w, img_h))

            # Paste object using its alpha channel
            background.paste(img, mask=img.convert('RGBA'))

            # Save merged image
            save_path = img_path if args.overwrite else os.path.join(output_dir, file_name)
            background.save(save_path)

    # Final summary
    print(f"Done! {total} images processed and saved to {output_dir}.")


if __name__ == "__main__":
    main()
