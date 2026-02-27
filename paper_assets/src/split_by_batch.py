"""
Split TACO dataset by batch numbers for YOLO training.

Ensures that images from the same batch stay together (either all in train or all in val).
This prevents data leakage between train and validation sets.

Usage:
    python split_by_batch.py --images_dir /path/to/images --labels_dir /path/to/labels --output_dir /path/to/output --val_ratio 0.15
"""

import argparse
import os
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Split TACO dataset by batch numbers")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument(
        "--labels_dir", type=str, required=True, help="Directory containing YOLO label files"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for split dataset"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Ratio of data for validation set (default: 0.15)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of moving them")
    parser.add_argument(
        "--dry_run", action="store_true", help="Show what would be done without actually doing it"
    )
    return parser.parse_args()


def extract_batch_number(filename: str) -> str:
    """
    Extract batch identifier from filename.

    Handles patterns like:
    - batch_2_000012.JPG -> batch_2
    - batch_10_00001.jpg -> batch_10
    - batch2_image.png -> batch2
    """
    # Try pattern: batch_N_... or batch_N-...
    match = re.match(r"(batch[_-]?\d+)", filename, re.IGNORECASE)
    if match:
        return match.group(1).lower().replace("-", "_")

    # Try extracting from any "batch" followed by number
    match = re.search(r"(batch\s*\d+)", filename, re.IGNORECASE)
    if match:
        return match.group(1).lower().replace(" ", "_")

    # If no batch pattern found, return 'unknown'
    return "unknown"


def get_image_extensions():
    """Return common image file extensions."""
    return {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tiff",
        ".tif",
        ".webp",
        ".JPG",
        ".JPEG",
        ".PNG",
        ".BMP",
        ".TIFF",
        ".TIF",
        ".WEBP",
    }


def find_matching_label(image_path: Path, labels_dir: Path) -> Path | None:
    """Find the matching label file for an image."""
    # Get image name without extension
    image_stem = image_path.stem

    # Try common label patterns
    label_path = labels_dir / f"{image_stem}.txt"
    if label_path.exists():
        return label_path

    return None


def find_matching_image(label_path: Path, images_dir: Path) -> Path | None:
    """Find the matching image file for a label."""
    label_stem = label_path.stem

    for ext in get_image_extensions():
        image_path = images_dir / f"{label_stem}{ext}"
        if image_path.exists():
            return image_path

    return None


def group_by_batch(images_dir: str, labels_dir: str) -> dict:
    """
    Group image-label pairs by batch number.

    Returns:
        dict: {batch_id: [(image_path, label_path), ...]}
    """
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)

    batches = defaultdict(list)
    orphan_images = []
    orphan_labels = []

    # Find all images with matching labels
    image_extensions = get_image_extensions()

    for image_file in images_path.iterdir():
        if image_file.suffix not in image_extensions:
            continue

        label_file = find_matching_label(image_file, labels_path)
        if label_file:
            batch_id = extract_batch_number(image_file.name)
            batches[batch_id].append((image_file, label_file))
        else:
            orphan_images.append(image_file)

    # Check for orphan labels (labels without images)
    for label_file in labels_path.glob("*.txt"):
        image_file = find_matching_image(label_file, images_path)
        if not image_file:
            orphan_labels.append(label_file)

    return dict(batches), orphan_images, orphan_labels


def split_batches(batches: dict, val_ratio: float, seed: int) -> tuple:
    """
    Split batches into train and val sets.

    Ensures entire batches stay together to prevent data leakage.
    """
    random.seed(seed)

    batch_ids = list(batches.keys())
    random.shuffle(batch_ids)

    # Calculate number of images per batch
    batch_sizes = {bid: len(batches[bid]) for bid in batch_ids}
    total_images = sum(batch_sizes.values())
    target_val_images = int(total_images * val_ratio)

    # Greedily assign batches to val until we reach target
    val_batches = []
    train_batches = []
    val_count = 0

    for batch_id in batch_ids:
        batch_size = batch_sizes[batch_id]

        # If adding this batch keeps us closer to target, add to val
        if val_count + batch_size <= target_val_images * 1.2:  # Allow 20% overshoot
            val_batches.append(batch_id)
            val_count += batch_size
        else:
            train_batches.append(batch_id)

    # If val is empty, force at least one batch
    if not val_batches and batch_ids:
        # Pick the smallest batch for val
        smallest_batch = min(batch_ids, key=lambda b: batch_sizes[b])
        val_batches.append(smallest_batch)
        train_batches.remove(smallest_batch)
        val_count = batch_sizes[smallest_batch]

    return train_batches, val_batches, val_count, total_images


def create_split_structure(output_dir: str, dry_run: bool = False):
    """Create YOLO directory structure."""
    dirs = [
        os.path.join(output_dir, "images", "train"),
        os.path.join(output_dir, "images", "val"),
        os.path.join(output_dir, "labels", "train"),
        os.path.join(output_dir, "labels", "val"),
    ]

    if not dry_run:
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    return dirs


def transfer_files(
    batches: dict,
    batch_ids: list,
    output_dir: str,
    split: str,
    copy: bool = False,
    dry_run: bool = False,
):
    """Transfer files to output directory."""
    images_out = os.path.join(output_dir, "images", split)
    labels_out = os.path.join(output_dir, "labels", split)

    transfer_func = shutil.copy2 if copy else shutil.move
    action = "Would copy" if dry_run else ("Copying" if copy else "Moving")

    file_count = 0
    for batch_id in batch_ids:
        for image_path, label_path in batches[batch_id]:
            if dry_run:
                print(f"  {action}: {image_path.name} -> {split}/")
            else:
                transfer_func(str(image_path), os.path.join(images_out, image_path.name))
                transfer_func(str(label_path), os.path.join(labels_out, label_path.name))
            file_count += 1

    return file_count


def create_dataset_yaml(output_dir: str, classes_file: str = None):
    """Create dataset.yaml file for YOLO training."""
    yaml_content = f"""# TACO Dataset - Split by batch
path: {os.path.abspath(output_dir)}
train: images/train
val: images/val

# Classes
names:
"""

    # Try to read classes from file
    if classes_file and os.path.exists(classes_file):
        with open(classes_file, "r") as f:
            for idx, line in enumerate(f):
                class_name = line.strip()
                if class_name:
                    yaml_content += f"  {idx}: {class_name}\n"
    else:
        yaml_content += "  # Add your class names here\n"
        yaml_content += "  # 0: class_name_1\n"
        yaml_content += "  # 1: class_name_2\n"

    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    return yaml_path


def main():
    args = parse_args()

    print("=" * 60)
    print("TACO Dataset Splitter (by batch)")
    print("=" * 60)

    # Group files by batch
    print("\nScanning files and grouping by batch...")
    batches, orphan_images, orphan_labels = group_by_batch(args.images_dir, args.labels_dir)

    if not batches:
        print("Error: No valid image-label pairs found!")
        return

    # Print batch statistics
    print(f"\nFound {len(batches)} batches:")
    for batch_id, files in sorted(batches.items()):
        print(f"  {batch_id}: {len(files)} images")

    if orphan_images:
        print(f"\n⚠️  Found {len(orphan_images)} images without labels")
    if orphan_labels:
        print(f"⚠️  Found {len(orphan_labels)} labels without images")

    # Split batches
    train_batches, val_batches, val_count, total_images = split_batches(
        batches, args.val_ratio, args.seed
    )

    train_count = total_images - val_count
    actual_val_ratio = val_count / total_images if total_images > 0 else 0

    print(f"\nSplit Summary:")
    print(
        f"  Train: {len(train_batches)} batches, {train_count} images ({100 - actual_val_ratio * 100:.1f}%)"
    )
    print(
        f"  Val:   {len(val_batches)} batches, {val_count} images ({actual_val_ratio * 100:.1f}%)"
    )
    print(f"\n  Train batches: {sorted(train_batches)}")
    print(f"  Val batches:   {sorted(val_batches)}")

    if args.dry_run:
        print("\n[DRY RUN] No files will be moved/copied")

    # Create output structure
    print(f"\nCreating output directory structure in: {args.output_dir}")
    create_split_structure(args.output_dir, args.dry_run)

    # Transfer files
    action = "Copying" if args.copy else "Moving"
    print(f"\n{action} files...")

    if not args.dry_run:
        print("  Processing train set...")
        transfer_files(batches, train_batches, args.output_dir, "train", args.copy, args.dry_run)

        print("  Processing val set...")
        transfer_files(batches, val_batches, args.output_dir, "val", args.copy, args.dry_run)
    else:
        transfer_files(batches, train_batches, args.output_dir, "train", args.copy, args.dry_run)
        transfer_files(batches, val_batches, args.output_dir, "val", args.copy, args.dry_run)

    # Create dataset.yaml
    if not args.dry_run:
        # Try to find classes.txt in common locations
        classes_file = None
        for path in [
            os.path.join(args.labels_dir, "classes.txt"),
            os.path.join(os.path.dirname(args.labels_dir), "classes.txt"),
            os.path.join(args.output_dir, "classes.txt"),
        ]:
            if os.path.exists(path):
                classes_file = path
                break

        yaml_path = create_dataset_yaml(args.output_dir, classes_file)
        print(f"\nCreated dataset.yaml: {yaml_path}")

    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
