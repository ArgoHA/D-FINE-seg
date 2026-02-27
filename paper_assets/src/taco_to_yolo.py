"""
Convert TACO dataset annotations to YOLO polygon format.

TACO JSON format -> YOLO polygon format (.txt files)
Each line in YOLO format: class_index x1 y1 x2 y2 x3 y3 ... (normalized coordinates)

Usage:
    python taco_to_yolo.py --input_dir /path/to/taco/annotations --output_dir /path/to/output --classes_file classes.txt
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Convert TACO annotations to YOLO polygon format")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing TACO JSON annotation files",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for YOLO format labels"
    )
    parser.add_argument(
        "--classes_file",
        type=str,
        default=None,
        help="Path to save/load classes.txt file. If exists, uses existing mapping.",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default=None,
        help="Optional: Directory containing images (for verification)",
    )
    return parser.parse_args()


def load_class_mapping(classes_file: str) -> dict:
    """Load existing class mapping from classes.txt file."""
    class_to_idx = {}
    if os.path.exists(classes_file):
        with open(classes_file, "r") as f:
            for idx, line in enumerate(f):
                class_name = line.strip()
                if class_name:
                    class_to_idx[class_name] = idx
        print(f"Loaded {len(class_to_idx)} classes from {classes_file}")
    return class_to_idx


def save_class_mapping(class_to_idx: dict, classes_file: str):
    """Save class mapping to classes.txt file."""
    # Sort by index to maintain order
    sorted_classes = sorted(class_to_idx.items(), key=lambda x: x[1])
    with open(classes_file, "w") as f:
        for class_name, _ in sorted_classes:
            f.write(f"{class_name}\n")
    print(f"Saved {len(class_to_idx)} classes to {classes_file}")


def discover_classes(input_dir: str) -> dict:
    """Discover all unique classes from annotation files."""
    class_names = set()
    json_files = list(Path(input_dir).glob("*.json"))

    print("Discovering classes from annotations...")
    for json_file in tqdm(json_files):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            for obj in data.get("objects", []):
                class_title = obj.get("classTitle", "")
                if class_title:
                    class_names.add(class_title)
        except Exception as e:
            print(f"Error reading {json_file}: {e}")

    # Create mapping (sorted alphabetically for consistency)
    class_to_idx = {name: idx for idx, name in enumerate(sorted(class_names))}
    print(f"Discovered {len(class_to_idx)} unique classes: {list(class_to_idx.keys())}")
    return class_to_idx


def rectangle_to_polygon(points: list) -> list:
    """Convert rectangle (2 points) to polygon (4 points)."""
    # Rectangle is defined by top-left and bottom-right corners
    x1, y1 = points[0]
    x2, y2 = points[1]

    # Return 4 corners: top-left, top-right, bottom-right, bottom-left
    return [
        [x1, y1],  # top-left
        [x2, y1],  # top-right
        [x2, y2],  # bottom-right
        [x1, y2],  # bottom-left
    ]


def normalize_polygon(points: list, img_width: int, img_height: int) -> list:
    """Normalize polygon points to 0-1 range."""
    normalized = []
    for x, y in points:
        nx = x / img_width
        ny = y / img_height
        # Clamp to valid range
        nx = max(0.0, min(1.0, nx))
        ny = max(0.0, min(1.0, ny))
        normalized.extend([nx, ny])
    return normalized


def convert_annotation(json_file: str, output_dir: str, class_to_idx: dict) -> dict:
    """Convert a single TACO annotation file to YOLO format."""
    stats = {"polygons": 0, "rectangles": 0, "skipped": 0}

    with open(json_file, "r") as f:
        data = json.load(f)

    # Get image dimensions
    img_width = data["size"]["width"]
    img_height = data["size"]["height"]

    # Get image filename (remove .json extension)
    json_name = os.path.basename(json_file)
    # Handle both "image.jpg.json" and "image.json" naming
    if json_name.endswith(".json"):
        base_name = json_name[:-5]  # Remove .json
        # If there's still an image extension, get the base name for the label
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            if base_name.endswith(ext):
                label_name = base_name[: -len(ext)] + ".txt"
                break
        else:
            label_name = base_name + ".txt"
    else:
        label_name = Path(json_name).stem + ".txt"

    # Process objects
    yolo_lines = []
    for obj in data.get("objects", []):
        class_title = obj.get("classTitle", "")
        geometry_type = obj.get("geometryType", "")
        points_data = obj.get("points", {})
        exterior = points_data.get("exterior", [])

        # Skip if class not in mapping
        if class_title not in class_to_idx:
            stats["skipped"] += 1
            continue

        class_idx = class_to_idx[class_title]

        # Only process polygon geometry (skip rectangles/bboxes)
        if geometry_type == "polygon":
            if len(exterior) < 3:
                stats["skipped"] += 1
                continue
            polygon_points = exterior
            stats["polygons"] += 1
        else:
            # Skip rectangles and other geometry types (we only want masks)
            stats["skipped"] += 1
            continue

        # Normalize coordinates
        normalized = normalize_polygon(polygon_points, img_width, img_height)

        # Format YOLO line: class_idx x1 y1 x2 y2 ...
        coords_str = " ".join([f"{c:.6f}" for c in normalized])
        yolo_lines.append(f"{class_idx} {coords_str}")

    # Write YOLO label file
    output_path = os.path.join(output_dir, label_name)
    with open(output_path, "w") as f:
        f.write("\n".join(yolo_lines))

    return stats


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Handle class mapping
    if args.classes_file and os.path.exists(args.classes_file):
        class_to_idx = load_class_mapping(args.classes_file)
    else:
        class_to_idx = discover_classes(args.input_dir)
        if args.classes_file:
            save_class_mapping(class_to_idx, args.classes_file)

    if not class_to_idx:
        print("Error: No classes found!")
        return

    # Get all JSON files
    json_files = list(Path(args.input_dir).glob("*.json"))
    print(f"\nFound {len(json_files)} annotation files")

    # Convert annotations
    total_stats = defaultdict(int)
    print("\nConverting annotations...")
    for json_file in tqdm(json_files):
        try:
            stats = convert_annotation(str(json_file), args.output_dir, class_to_idx)
            for key, value in stats.items():
                total_stats[key] += value
        except Exception as e:
            print(f"\nError processing {json_file}: {e}")

    # Always save labels.txt in output directory
    labels_file = os.path.join(args.output_dir, "labels.txt")
    save_class_mapping(class_to_idx, labels_file)

    # Print summary
    print(f"\n{'=' * 50}")
    print("Conversion Summary:")
    print(f"  Total files processed: {len(json_files)}")
    print(f"  Polygons converted: {total_stats['polygons']}")
    print(f"  Rectangles converted: {total_stats['rectangles']}")
    print(f"  Objects skipped: {total_stats['skipped']}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Labels file: {labels_file}")
    if args.classes_file:
        print(f"  Classes file: {args.classes_file}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
