import argparse
import cv2

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def bbox_maker(images_folder, labels_folder, output_folder, resize, gt):
    images_folder = Path(images_folder)
    labels_folder = Path(labels_folder)
    output_folder = Path(output_folder)

    if not images_folder.exists():
        print(f"Images folder {images_folder} does not exist.")
        return
    if not labels_folder.exists():
        print(f"Labels folder {labels_folder} does not exist.")
        return

    output_folder.mkdir(parents=True, exist_ok=True)

    resize_dims = parse_resize_arg(resize)
    process_images(images_folder, labels_folder, output_folder, resize_dims, gt)

def process_images(images_folder, labels_folder, output_folder, resize_dims, gt):

    # TODO: Generalize color/class mapping
    # This is hardcoded color map for only two classes
    class_map = {
        '0': {'name': 'soy', 'color': (252, 236, 3)},       
        '1': {'name': 'cotton', 'color': (201, 14, 230)} 
    }

    image_extensions = ['.jpg', '.jpeg', '.png']
    images = [f for f in images_folder.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]

    if not images:
        print(f"No images found in {images_folder}.")
        return

    for image_path in images:

        label_file = labels_folder / (image_path.stem + '.txt')
        if not label_file.exists():
            print(f"Label file {label_file} does not exist for image {image_path.name}. Skipping this image.")
            continue

        labels = read_labels(label_file, gt)
        # img = cv2.imread(str(image_path))
        img = Image.open(image_path)

        if resize_dims is not None:
            w, h = resize_dims
            # PIL's resize expects (width, height)
            # For high-quality downsampling, use Image.LANCZOS (similar to cv2.INTER_AREA)
            img = img.resize((w, h), Image.LANCZOS)

        # with Image.open(image_path) as img:

        img_with_boxes = draw_bounding_boxes(img, labels, class_map, gt)

        output_path = output_folder / (image_path.name)
        img_with_boxes.save(output_path)
        print(f"Saved image with bounding boxes to {output_path}")

def parse_resize_arg(resize_arg):
    """
    Parse the '--resize' argument.
    If the argument is 'None', return None.
    Otherwise, expect the format 'widthxheight' and return (width, height) as integers.
    """
    if resize_arg is None or resize_arg.lower() == "none":
        return None
    
    try:
        w, h = resize_arg.lower().split("x")
        return (int(w), int(h))
    except ValueError:
        print(f"Invalid format for --resize: '{resize_arg}'. Use 'widthxheight' or 'None'.")
        return None


def read_labels(label_file, gt):
    labels = []
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 6 and not gt:

                cls, x_center, y_center, width, height, confidence = parts
                labels.append({
                    'class_id': cls,
                    'x_center': float(x_center),
                    'y_center': float(y_center),
                    'width': float(width),
                    'height': float(height),
                    'confidence': float(confidence)
                })
            
            elif len(parts) == 5 and gt:
                cls, x_center, y_center, width, height = parts
                labels.append({
                    'class_id': cls,
                    'x_center': float(x_center),
                    'y_center': float(y_center),
                    'width': float(width),
                    'height': float(height),
                })

            else:
                print(f"Invalid label format in {label_file}: {line}")


    return labels

def draw_bounding_boxes(img, labels, class_map, gt):
    # Convert the image to RGBA to support transparency
    img = img.convert('RGBA')
    img_width, img_height = img.size
    # overlay = img.copy()

    overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    colors = []
    for _, cls_info in class_map.items():
        colors.append(cls_info['color'][::-1] + (255,))
    print(colors)
    
    # Load text font
    try:
        font = ImageFont.truetype("DroidSerif-Regular.ttf", size=32)
        legend_font = ImageFont.truetype("DroidSerif-Regular.ttf", size=36)

    except IOError as e:
        font = ImageFont.load_default(size=30)
        legend_font = ImageFont.load_default(size=36)
        # print(f"Could not load custom font: {e}")

    for label in labels:
        cls_id = label['class_id']
        if cls_id not in class_map:
            continue
        class_info = class_map[cls_id]
        class_name = class_info['name']
        color = class_info['color']
        # Convert normalized coordinates to absolute pixel coordinates
        x_center = label['x_center'] * img_width
        y_center = label['y_center'] * img_height
        width = label['width'] * img_width
        height = label['height'] * img_height

        # Calculate bounding box coordinates
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2

        # Ensure bounding box is within image bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img_width, x_max)
        y_max = min(img_height, y_max)

        # Create a semi-transparent fill color
        fill_opacity = 0.161  # Opacity level (10%)
        alpha = int(255 * fill_opacity)
        fill_color = color + (alpha,)          # Semi-transparent fill color
        outline_color = color + (255,)         # Fully opaque outline color

        # Draw rounded rectangle on the overlay
        draw_rounded_rectangle(
            draw,
            [x_min, y_min, x_max, y_max],
            radius=10,
            fill=fill_color,
            outline=outline_color,
            width=3
        )

        # Confidence values rendering
        if not gt:
            confidence = label['confidence']
            text = f"{confidence:.2f}"

            # Use font.getbbox() to get the size of the text
            x_left, y_top, x_right, y_bottom = font.getbbox(text)
            text_width = abs(x_right - x_left)
            text_height = abs(y_bottom - y_top)

            bbox_xmid = (x_max - x_min)/2

            # text_position = (x_min + bbox_xmid - text_width/2, y_min - text_height*1.161)
            text_position = (x_min + bbox_xmid - text_width//2, y_min - 2*text_height)
            text_position_back = (x_min, y_min - text_height - 6)

            # Ensure text is within image bounds
            if text_position[1] < 0:
                text_position = (x_min + bbox_xmid - text_width//2, y_max)

            # Define shadow offset and color
            shadow_offset = (1, 1)  # (x_offset, y_offset)
            shadow_color = (0, 0, 0, 128)  # Semi-transparent black

            # Draw shadow text on the overlay
            shadow_position = (text_position[0] + shadow_offset[0], text_position[1] + shadow_offset[1])
            draw.text(
                shadow_position,
                text,
                font=font,
                fill=colors[1]
            )

            # Draw text on the overlay
            draw.text(
                text_position,
                text,
                font=font,
                fill=colors[0],
            )

    # Draw the legend
    draw_legend(draw, class_map, legend_font, img_width, img_height)

    # Composite the overlay onto the original image
    img = Image.alpha_composite(img, overlay)

    # Convert back to RGB mode if desired
    return img.convert('RGB')

def draw_rounded_rectangle(draw, xy, radius=5, fill=None, outline=None, width=1):
    
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)

def draw_legend(draw, class_map, font, img_width, img_height, radius=10):

    legend_x = 10  # Padding from the left edge
    legend_y = 10  # Padding from the top edge

    y_text_offset = 5
    x_text_offset = 5

    max_text_width = 0
    total_text_height = 0
    entries = []

    for cls_id, class_info in class_map.items():
        class_name = class_info['name'].capitalize()
        color = class_info['color']
        text = class_name

        # Get text size
        x_left, y_top, x_right, y_bottom = font.getbbox(text)
        text_width = abs(x_right - x_left)
        text_height = abs(y_bottom - y_top)

        max_text_width = max(max_text_width, text_width)
        total_text_height += text_height + 5  # Adding spacing between entries

        entries.append({
            'text': text,
            'text_width': text_width,
            'text_height': text_height,
            'color': color
        })

    # Square size is 80% of text heightimages_folder/
    square_size = int(0.6 * entries[-1]["text_height"])

    # Background for the legend (optional)
    legend_width =  square_size + max_text_width + 5*x_text_offset  
    legend_height = total_text_height + 3*y_text_offset 
    legend_background = [
        (legend_x, legend_y),
        (legend_x + legend_width, legend_y + legend_height)
    ]
    draw.rounded_rectangle(legend_background, radius=radius, fill=(50, 50, 50, 180))

    # Draw each legend entry
    current_y = legend_y + y_text_offset  # Starting y position with padding
    for entry in entries:
        text = entry['text']
        text_width = entry['text_width']
        text_height = entry['text_height']
        color = entry['color']

        square_offset = abs(text_height + 2*y_text_offset - square_size)/2
        square_y = current_y + square_offset

        # Draw the color square
        square_coords = [
            legend_x + x_text_offset,
            square_y,
            legend_x + x_text_offset + square_size,
            square_y + square_size
        ]
        draw.rounded_rectangle(square_coords, radius=radius*0.1, fill=color + (255,), outline=None)

        text_position = (legend_x + x_text_offset*3 + square_size, current_y)
        draw.text(
            text_position,
            text,
            fill=(255, 255, 255, 255), 
            font=font
        )

        current_y += text_height + y_text_offset 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render bounding boxes on images.")
    parser.add_argument("images_folder", help="Path to the folder containing images.")
    parser.add_argument("labels_folder", help="Path to the folder containing label files.")
    parser.add_argument("output_folder", help="Path to the folder where output images will be saved.")
    parser.add_argument("--gt", action='store_true', default=False, help="Ground truth masks flag.")
    parser.add_argument(
        "--resize", 
        default="None", 
        help="Resize images to 'widthxheight'. With 'None' the option to not resize"
    )

    args = parser.parse_args()
    
    bbox_maker(**vars(args))
