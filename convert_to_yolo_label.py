import os
import json
import cv2
from typing import List, Dict

def convert_json_to_yolo(json_data: Dict, image_width: int, image_height: int) -> List[str]:
    """
    Convert one annotation JSON (custom format) to YOLO format lines.
    Each line: <class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
    """
    classes = json_data.get("classes", [])
    labels = json_data.get("labels", [])

    yolo_lines = []
    for label in labels:
        class_name = label["class"]
        if class_name not in classes:
            continue

        class_id = classes.index(class_name)
        x, y, w, h = label["x"], label["y"], label["width"], label["height"]

        # Convert to YOLO normalized format
        x_center = (x + w / 2) / image_width
        y_center = (y + h / 2) / image_height
        w_norm = w / image_width
        h_norm = h / image_height

        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

    return yolo_lines


def process_json_folder(json_folder: str, image_folder: str, output_folder: str):
    """
    Reads all .json files in json_folder, finds corresponding image in image_folder,
    reads its dimensions, and saves YOLO .txt files to output_folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    json_files = [f for f in os.listdir(json_folder) if f.endswith(".json")]

    if not json_files:
        print(f"⚠️ No JSON files found in folder: {json_folder}")
        return

    for i, json_file in enumerate(sorted(json_files)):
        json_path = os.path.join(json_folder, json_file)
        base_name = os.path.splitext(json_file)[0]

        # --- Find corresponding image ---
        image_path = None
        for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
            candidate = os.path.join(image_folder, base_name + ext)
            if os.path.exists(candidate):
                image_path = candidate
                break

        if image_path is None:
            print(f"⚠️ Image not found for {json_file}. Skipping...")
            continue

        # --- Read image size ---
        img = cv2.imread(image_path)
        if img is None:
            print(f"⚠️ Failed to read image: {image_path}")
            continue

        image_height, image_width = img.shape[:2]

        # --- Load JSON and convert ---
        with open(json_path, "r") as f:
            json_data = json.load(f)

        yolo_lines = convert_json_to_yolo(json_data, image_width, image_height)

        # --- Save .txt output ---
        txt_path = os.path.join(output_folder, f"{base_name}.txt")
        with open(txt_path, "w") as f:
            f.write("\n".join(yolo_lines))

        print(f"{i+1} : ✅ Saved: {txt_path} (size: {image_width}x{image_height})")

    print(f"\n🎯 Conversion complete! Processed {len(json_files)} JSON files.")


if __name__ == "__main__":
    # Example usage:
    json_folder = "all_jsons"   # Folder containing .json files
    image_folder = "all_data"       # Folder containing corresponding images
    output_folder = "all_labels"        # Where .txt files will be saved

    process_json_folder(json_folder, image_folder, output_folder)
