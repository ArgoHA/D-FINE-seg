import os
from pathlib import Path


def update_annotations(directory, class_to_remove):
    for filename in os.listdir(directory):
        if filename == "labels.txt":
            continue
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r") as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.split()
                    class_id = int(parts[0])
                    if class_id == class_to_remove:
                        print(filename)
                        (Path(directory) / filename).unlink()
                        (Path(directory).parent / "images" / f"{Path(filename).stem}.jpg").unlink()



# Update the directory path to your annotations folder
annotations_directory = "/Users/argosaakyan/Downloads/labels"
class_to_remove = 2

update_annotations(annotations_directory, class_to_remove)
# a7b9a4c2-3c3c-496a-bfa4-f2c8c1ec34f8
