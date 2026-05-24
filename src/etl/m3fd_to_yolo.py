"""Convert the M3FD multi-modal detection dataset to this repo's YOLO + multi-channel
``.npy`` layout.

M3FD ships as:
    M3FD.zip
      Annotation/*.xml   PASCAL VOC bboxes (one per Vis/Ir pair)
      Vis/*.png          visible-light, 3-channel uint8
      Ir/*.png           thermal, stored as a grayscale-replicated 3-channel uint8 PNG

We collapse the Ir channels (they are identical) to a single thermal plane and
stack with the visible image to produce 4-channel uint8 arrays saved as ``.npy``.
On-disk channel order is RGBT — ``np.load`` is byte-faithful, so the dataset
reader can consume the file directly with no swap. (TIFF was rejected because
``cv2.imread(IMREAD_UNCHANGED)`` mangles 4-channel TIFFs from non-cv2 producers:
see ``scripts/test_tiff_channel_order.py``.)

Usage:
    uv run python -m src.etl.m3fd_to_yolo \\
        --src /path/to/M3FD.zip \\
        --dst /path/to/dataset

``--src`` accepts either the zip or an already-extracted directory containing
Annotation/, Vis/, Ir/.
"""

import argparse
import shutil
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# Canonical M3FD class ordering used by the dataset paper.
CLASSES: Tuple[str, ...] = ("People", "Car", "Bus", "Lamp", "Motorcycle", "Truck")
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}


def _parse_voc_xml(xml_path: Path) -> Tuple[int, int, List[Tuple[int, float, float, float, float]]]:
    """Returns (width, height, [(cls, xc, yc, w, h), ...]) with normalized YOLO boxes."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = int(size.findtext("width"))
    height = int(size.findtext("height"))

    boxes = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name not in CLASS_TO_ID:
            raise ValueError(f"Unknown class '{name}' in {xml_path}")
        b = obj.find("bndbox")
        xmin = float(b.findtext("xmin"))
        ymin = float(b.findtext("ymin"))
        xmax = float(b.findtext("xmax"))
        ymax = float(b.findtext("ymax"))

        # Clamp + drop degenerate boxes (M3FD has a few that touch image edges).
        xmin = max(0.0, min(xmin, width))
        xmax = max(0.0, min(xmax, width))
        ymin = max(0.0, min(ymin, height))
        ymax = max(0.0, min(ymax, height))
        if xmax - xmin < 1 or ymax - ymin < 1:
            continue

        xc = (xmin + xmax) / 2.0 / width
        yc = (ymin + ymax) / 2.0 / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height
        boxes.append((CLASS_TO_ID[name], xc, yc, w, h))
    return width, height, boxes


def _process_one(args) -> Tuple[str, str]:
    """Worker: convert a single sample. Returns (stem, status)."""
    stem, src_dir, dst_dir = args
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    xml_path = src_dir / "Annotation" / f"{stem}.xml"
    vis_path = src_dir / "Vis" / f"{stem}.png"
    ir_path = src_dir / "Ir" / f"{stem}.png"

    if not (xml_path.exists() and vis_path.exists() and ir_path.exists()):
        return stem, "missing"

    width, height, boxes = _parse_voc_xml(xml_path)

    vis = cv2.imread(str(vis_path), cv2.IMREAD_UNCHANGED)  # BGR uint8
    ir = cv2.imread(str(ir_path), cv2.IMREAD_UNCHANGED)
    if vis is None or ir is None:
        return stem, "decode_fail"

    if vis.shape[:2] != ir.shape[:2]:
        return stem, f"shape_mismatch vis={vis.shape} ir={ir.shape}"
    if vis.shape[:2] != (height, width):
        return stem, f"xml_size_mismatch xml=({height},{width}) img={vis.shape[:2]}"

    # Swap vis BGR -> RGB and collapse the replicated IR to one channel.
    # On-disk order is [R, G, B, T]; np.load is byte-faithful so the reader needs no swap.
    rgb = vis[..., ::-1]
    thermal = ir[..., 0] if ir.ndim == 3 else ir
    stacked = np.dstack([rgb, thermal]).astype(np.uint8, copy=False)  # H,W,4

    out_img = dst_dir / "images" / f"{stem}.npy"
    out_lbl = dst_dir / "labels" / f"{stem}.txt"
    np.save(str(out_img), stacked)

    with open(out_lbl, "w") as f:
        for cls, xc, yc, w, h in boxes:
            f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
    return stem, "ok"


def _resolve_src(src: Path) -> Tuple[Path, Path]:
    """Returns (src_root, temp_dir). temp_dir is None when src was already a directory."""
    if src.is_dir():
        return src, None
    if src.suffix.lower() != ".zip":
        raise SystemExit(f"--src must be a zip or directory, got {src}")
    tmp = Path(tempfile.mkdtemp(prefix="m3fd_"))
    print(f"Extracting {src} -> {tmp}")
    with zipfile.ZipFile(src) as zf:
        zf.extractall(tmp)
    return tmp, tmp


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--src", type=Path, required=True, help="M3FD.zip or extracted root dir")
    parser.add_argument(
        "--dst", type=Path, required=True, help="Output dataset dir (gets images/ and labels/)"
    )
    parser.add_argument("--workers", type=int, default=0, help="0 = os.cpu_count()")
    parser.add_argument(
        "--keep-tmp", action="store_true", help="Don't delete extracted zip on exit"
    )
    args = parser.parse_args()

    src_root, tmp_dir = _resolve_src(args.src)
    try:
        ann_dir = src_root / "Annotation"
        if not ann_dir.exists():
            raise SystemExit(f"Annotation/ not found under {src_root}")

        (args.dst / "images").mkdir(parents=True, exist_ok=True)
        (args.dst / "labels").mkdir(parents=True, exist_ok=True)
        with open(args.dst / "labels.txt", "w") as f:
            f.write("\n".join(CLASSES) + "\n")

        stems = sorted(p.stem for p in ann_dir.glob("*.xml"))
        print(f"Found {len(stems)} samples")

        workers = args.workers or None
        jobs = [(s, str(src_root), str(args.dst)) for s in stems]

        counts = {
            "ok": 0,
            "missing": 0,
            "decode_fail": 0,
            "shape_mismatch": 0,
            "xml_size_mismatch": 0,
        }
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_process_one, j) for j in jobs]
            for fut in tqdm(as_completed(futs), total=len(futs)):
                stem, status = fut.result()
                key = status.split(" ", 1)[0]
                counts[key] = counts.get(key, 0) + 1
                if key != "ok":
                    tqdm.write(f"  {stem}: {status}")

        print("\nDone:")
        for k, v in counts.items():
            if v:
                print(f"  {k}: {v}")
        print(f"\nWrote {counts['ok']} pairs to {args.dst}")
        print(f"Classes (label_to_name): {dict(enumerate(CLASSES))}")
    finally:
        if tmp_dir is not None and not args.keep_tmp:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
