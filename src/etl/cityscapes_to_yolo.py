"""Convert Cityscapes gtFine polygons to YOLO instance labels (polygons).

gtFine/<split>/<city>/<stem>_gtFine_polygons.json holds per-object polygons as
absolute pixel coords. The 8 "thing" classes (person, rider, car, truck, bus,
train, motorcycle, bicycle) become YOLO instance labels (0-7), normalized.

Output: <stem>.txt with one line per instance: `cls x1 y1 x2 y2 ... xn yn`.
Run polys2bbox afterwards for detection bboxes.

Usage:
    python -m src.etl.cityscapes_to_yolo \
        --gtfine /path/to/raw/gtFine \
        --images /path/to/dataset/images \
        --out   /path/to/dataset_seg/labels
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm

THING_CLASSES = {
    "person": 0,
    "rider": 1,
    "car": 2,
    "truck": 3,
    "bus": 4,
    "train": 5,
    "motorcycle": 6,
    "bicycle": 7,
}


def build_json_index(gtfine_root: Path) -> dict[str, Path]:
    index = {}
    for jf in gtfine_root.rglob("*_gtFine_polygons.json"):
        stem = jf.name.replace("_gtFine_polygons.json", "")
        index[stem] = jf
    return index


def convert(gtfine_root: Path, images_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_index = build_json_index(gtfine_root)

    images = [p for p in images_dir.iterdir() if not p.name.startswith(".")]
    missing = converted = instances = 0

    for img in tqdm(images, desc="Converting"):
        stem = img.stem
        jf = json_index.get(stem)
        if jf is None:
            missing += 1
            continue

        with open(jf) as f:
            data = json.load(f)
        w, h = data["imgWidth"], data["imgHeight"]

        lines = []
        for obj in data["objects"]:
            cls = THING_CLASSES.get(obj["label"])
            if cls is None:
                continue
            poly = obj["polygon"]
            pts = [(float(x) / w, float(y) / h) for x, y in poly]
            if len(pts) < 3:
                continue
            flat = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
            lines.append(f"{cls} {flat}")

        with open(out_dir / f"{stem}.txt", "w") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")

        converted += 1
        instances += len(lines)

    print(f"Converted {converted} images ({instances} instances), {missing} missing jsons")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--gtfine", type=Path, required=True, help="Path to raw/gtFine")
    p.add_argument("--images", type=Path, required=True, help="Path to dataset/images")
    p.add_argument("--out", type=Path, required=True, help="Output labels dir")
    args = p.parse_args()
    convert(args.gtfine, args.images, args.out)
