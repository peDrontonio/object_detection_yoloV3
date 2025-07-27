import os
from PIL import Image
from PIL.ExifTags import TAGS
from collections import Counter

def extract_metadata(image_path):
    """Extract metadata from an image."""
    metadata = {}
    try:
        with Image.open(image_path) as img:
            exif_data = img._getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag_name = TAGS.get(tag_id, tag_id)
                    metadata[tag_name] = value
    except Exception as e:
        metadata["Error"] = str(e)
    return metadata

def write_metadata_to_file(metadata, output_file, image_name):
    """Write metadata to a text file."""
    with open(output_file, "a") as file:
        file.write(f"Metadata for {image_name}:\n")
        for key, value in metadata.items():
            file.write(f"  {key}: {value}\n")
        file.write("\n")

def collect_resolutions_with_counts(folder_path):
    """Collect resolutions and their counts from images in the folder."""
    resolutions = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            try:
                with Image.open(file_path) as img:
                    resolutions.append(img.size)
            except Exception as e:
                print(f"Failed to get resolution for {file_name}: {e}")
    return Counter(resolutions)

def process_images_in_folder(folder_path, output_file):
    """Process all images in a folder and write their metadata to a file."""
    if not os.path.exists(folder_path):
        print("Error: The specified folder does not exist.")
        return

    # Clear or create the output file
    open(output_file, "w").close()

    resolutions = collect_resolutions_with_counts(folder_path)

    # Write resolution counts to the first line of the output file
    with open(output_file, "a") as file:
        file.write("Unique resolutions and their counts:\n")
        for resolution, count in resolutions.items():
            file.write(f"  {resolution}: {count} images\n")
        file.write("\n")

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            try:
                metadata = extract_metadata(file_path)
                write_metadata_to_file(metadata, output_file, file_name)
                print(f"Metadata for {file_name} has been written.")
            except Exception as e:
                print(f"Failed to process {file_name}: {e}")

if __name__ == "__main__":
    folder_path = input("Enter the folder path containing images: ").strip()
    output_file = "image_metadata.txt"
    process_images_in_folder(folder_path, output_file)
    print(f"Metadata extraction completed. Check the file: {output_file}")
