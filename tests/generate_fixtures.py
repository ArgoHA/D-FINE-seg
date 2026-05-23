"""Bootstrap the accuracy-regression fixtures.

Run once after pulling fresh pretrained weights, or after a deliberate model
change you want to lock in as the new baseline:

    uv run python -m tests.generate_fixtures

What it does, for every image in `tests/assets/`:
1. Downscales (if needed) to <= MAX_LONG_SIDE pixels on the long edge and
   re-encodes as JPEG. Non-`.jpg` sources are normalized to `.jpg` and the
   original is removed so the folder stays lean. Big enough to keep visual
   detail, small enough to be cheap in git.
2. Runs the CPU `dfine_s_coco.pt` model on the downscaled image.
3. Writes the high-confidence predictions next to it as a YOLO-format
   `<stem>.txt` file.
4. Pins a baseline mAP_50 in `tests/assets/baseline.json`. Because the labels
   are exactly the model's own predictions, mAP_50 on a re-run is 1.0 by
   construction; the baseline is set lower (0.95) to leave headroom for tiny
   numerical drift across torch / numpy / opencv minor versions.

The accuracy test then asserts mAP_50 >= baseline. Anything that silently
changes the model's CPU output (arch, postprocess, letterbox, NMS, loader)
will trip it.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from src.d_fine.utils import ensure_pretrained
from src.dl.utils import abs_xyxy_to_norm_xywh
from src.infer.torch_model import Torch_model

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
BASELINE_PATH = ASSETS_DIR / "baseline.json"

CONF_THRESH = 0.5
MAX_LONG_SIDE = 1024
JPEG_QUALITY = 92
IMG_GLOBS = ("*.png", "*.jpg", "*.jpeg")


def _source_images() -> list[Path]:
    images: list[Path] = []
    for pattern in IMG_GLOBS:
        images.extend(sorted(ASSETS_DIR.glob(pattern)))
    return images


def _downscale_to_jpg(src: Path, max_long_side: int, jpeg_quality: int) -> Path | None:
    """Re-encode `src` as a downscaled `.jpg` in the same folder.

    Returns the path of the resulting JPEG, or None if `src` was unreadable.
    If `src` was a non-jpg source (e.g. .png), it's removed after a successful
    write so the asset folder stays lean and the test has one obvious input
    per stem.
    """
    img = cv2.imread(str(src))
    if img is None:
        return None
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side > max_long_side:
        scale = max_long_side / long_side
        img = cv2.resize(
            img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA
        )
    out_path = src.with_suffix(".jpg")
    cv2.imwrite(str(out_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if src.suffix.lower() != ".jpg":
        src.unlink(missing_ok=True)
    return out_path


def _wipe_stale_labels(image_stems: set[str]) -> None:
    # Drop label files whose source image is gone; leave baseline.json alone
    # so a partial re-run doesn't desync.
    for p in ASSETS_DIR.glob("*.txt"):
        if p.stem not in image_stems:
            p.unlink()


def _write_labels(image_path: Path, boxes_xyxy: np.ndarray, labels: np.ndarray,
                  img_h: int, img_w: int) -> None:
    label_path = image_path.with_suffix(".txt")
    if len(boxes_xyxy) == 0:
        label_path.write_text("")
        return
    yolo = abs_xyxy_to_norm_xywh(boxes_xyxy, height=img_h, width=img_w)
    lines = [
        f"{int(c)} {row[0]:.6f} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f}"
        for c, row in zip(labels, yolo)
    ]
    label_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    weights_path = REPO_ROOT / "pretrained" / "dfine_s_coco.pt"
    weights_path = Path(ensure_pretrained(weights_path))
    print(f"weights: {weights_path}")

    sources = _source_images()
    if not sources:
        raise SystemExit(
            f"no source images in {ASSETS_DIR}; drop one or more .png/.jpg files there."
        )

    # Downscale + normalize extensions first so the rest of the pipeline only
    # ever sees `.jpg` inputs and labels are written against the final file.
    images: list[Path] = []
    for src in sources:
        out = _downscale_to_jpg(src, MAX_LONG_SIDE, JPEG_QUALITY)
        if out is None:
            print(f"  {src.name}: unreadable, skipped")
            continue
        images.append(out)
    _wipe_stale_labels(image_stems={p.stem for p in images})

    # Use the production inference wrapper so we exercise the same preprocess /
    # postprocess path the real bench uses, not a hand-rolled mini-driver.
    model = Torch_model(
        model_name="s",
        model_path=str(weights_path),
        n_outputs=80,
        input_width=640,
        input_height=640,
        conf_thresh=CONF_THRESH,
        keep_ratio=False,
        apply_nms=True,
        device="cpu",
        enable_mask_head=False,
    )

    total = 0
    kept = 0
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  {img_path.name}: unreadable, skipped")
            continue
        h, w = img.shape[:2]
        preds = model(img)[0]
        keep = preds["scores"] >= CONF_THRESH
        n_det = int(keep.sum().item())
        if n_det == 0:
            # An empty label file would still be loaded by the test as zero GT,
            # which is unhelpful. Skip the image and clear any stale label.
            img_path.with_suffix(".txt").unlink(missing_ok=True)
            print(f"  {img_path.name}: 0 detections, no label written")
            continue
        boxes = preds["boxes"][keep].cpu().numpy()
        labels = preds["labels"][keep].cpu().numpy()
        _write_labels(img_path, boxes, labels, img_h=h, img_w=w)
        total += n_det
        kept += 1
        print(f"  {img_path.name}: {n_det} detections ({w}x{h})")

    BASELINE_PATH.write_text(json.dumps({
        "mAP_50_min": 0.95,
        "mAP_50_95_min": 0.70,
        "n_images": kept,
        "n_detections": total,
        "conf_thresh": CONF_THRESH,
        "max_long_side": MAX_LONG_SIDE,
    }, indent=2) + "\n")
    print(f"wrote baseline -> {BASELINE_PATH}")


if __name__ == "__main__":
    torch.manual_seed(42)
    main()
