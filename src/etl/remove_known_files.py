from pathlib import Path

path1 = Path("/Users/argosaakyan/Data/Veryfi/HW/hw_labels_raw/Rydoo/auto_annotate_1")
path2 = Path("/Users/argosaakyan/Data/Veryfi/HW/hw_labels_raw/Rydoo/auto_annotate")

imgs_1 = [img.name for img in path1.iterdir() if not str(img).startswith(".")]
imgs_2 = [img.name for img in path2.iterdir() if not str(img).startswith(".")]

known = set(imgs_1) & set(imgs_2)
counter = 0
for img in known:
    (path2 / Path(img)).unlink()
    counter += 1

print(f"Removed {counter} known files")
