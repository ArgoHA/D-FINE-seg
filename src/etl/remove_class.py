"""
This script takes a path to labels and label id to remove.
It delets all objects with that label id and if there are no other labels on the image - removes image too

"""

from pathlib import Path


def remove_label_objects(label_dir, target_label_id):
    label_dir = Path(label_dir)
    images_dir = label_dir.parent / "images"
    image_extensions = [".jpg", ".jpeg", ".png"]

    # Iterate over each text file in the labels directory.
    for label_file in label_dir.glob("*.txt"):
        with label_file.open("r") as f:
            lines = f.readlines()

        # Filter out lines that have the target label id.
        new_lines = []
        for line in lines:
            tokens = line.strip().split()
            if not tokens:
                continue  # skip empty lines
            if tokens[0] != str(target_label_id):
                new_lines.append(line)

        if new_lines:
            # Write the filtered labels back to the file.
            with label_file.open("w") as f:
                f.writelines(new_lines)
        else:
            # If no labels remain, remove the label file.
            label_file.unlink()
            # Construct a potential image file name using the same stem.
            base_name = label_file.stem
            for ext in image_extensions:
                image_file = images_dir.with_name(base_name + ext)
                if image_file.exists():
                    image_file.unlink()
                    break


labels_path = ""
for class_to_remove in [2]:
    remove_label_objects(labels_path, class_to_remove)
