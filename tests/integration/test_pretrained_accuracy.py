"""End-to-end CPU accuracy regression.

Loads `dfine_s_coco.pt`, runs it through the production `Torch_model` wrapper on
a tiny fixture set, and asserts `mAP_50` matches what was pinned by
`tests/generate_fixtures.py`. Anything that silently changes the model's CPU
output (arch, weights loader, postprocess, letterbox, NMS, Validator math)
trips this test.

Marked `slow` because it builds the `s` model on CPU and runs forward on each
fixture (~10–20s on a Mac, faster on the lab box).
"""

import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.dl.utils import norm_xywh_to_abs_xyxy
from src.dl.validator import Validator
from src.infer.torch_model import Torch_model

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
BASELINE_PATH = ASSETS_DIR / "baseline.json"

# COCO class index -> short name. Validator only needs an entry for every
# class id that appears in preds OR gt; using a small COCO dict is sufficient.
COCO80 = {i: str(i) for i in range(80)}


def _require_fixtures() -> dict:
    if not BASELINE_PATH.exists():
        pytest.skip(
            "fixtures not generated; run `uv run python -m tests.generate_fixtures`"
        )
    return json.loads(BASELINE_PATH.read_text())


def _load_yolo_labels(label_path: Path, img_h: int, img_w: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Parse a YOLO-format .txt into (labels, xyxy boxes in absolute pixels)."""
    if not label_path.exists() or label_path.stat().st_size == 0:
        return torch.empty(0, dtype=torch.long), torch.empty((0, 4))
    rows = []
    classes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        classes.append(int(parts[0]))
        rows.append([float(p) for p in parts[1:]])
    if not rows:
        return torch.empty(0, dtype=torch.long), torch.empty((0, 4))
    norm = np.array(rows, dtype=np.float32)
    xyxy = norm_xywh_to_abs_xyxy(norm, height=img_h, width=img_w, to_round=False)
    return torch.tensor(classes, dtype=torch.long), torch.from_numpy(xyxy).float()


@pytest.mark.slow
def test_pretrained_s_cpu_mAP_holds_baseline(coco_pretrained_path):
    baseline = _require_fixtures()
    # Only images with a matching label file count as fixture inputs — that's
    # what the bootstrap writes, and it lets stray source images sit in the
    # folder without breaking the test.
    image_paths = [
        p for p in sorted(ASSETS_DIR.iterdir())
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.with_suffix(".txt").exists()
    ]
    assert image_paths, f"no fixture images with labels found in {ASSETS_DIR}"

    # Use the production wrapper — the same path used by infer.py and bench.py.
    model = Torch_model(
        model_name="s",
        model_path=str(coco_pretrained_path),
        n_outputs=80,
        input_width=640,
        input_height=640,
        conf_thresh=baseline["conf_thresh"],
        keep_ratio=False,
        apply_nms=True,
        device="cpu",
        enable_mask_head=False,
    )

    gt, preds = [], []
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        pred = model(img)[0]
        # Drop sub-threshold detections from the eval set so the test stays
        # apples-to-apples with how the fixture labels were captured.
        keep = pred["scores"] >= baseline["conf_thresh"]
        preds.append({
            "labels": pred["labels"][keep].cpu(),
            "boxes": pred["boxes"][keep].cpu(),
            "scores": pred["scores"][keep].cpu(),
        })
        gt_labels, gt_boxes = _load_yolo_labels(img_path.with_suffix(".txt"), img_h=h, img_w=w)
        gt.append({"labels": gt_labels, "boxes": gt_boxes})

    v = Validator(
        gt=copy.deepcopy(gt),
        preds=copy.deepcopy(preds),
        label_to_name=COCO80,
        conf_thresh=baseline["conf_thresh"],
    )
    metrics = v.compute_metrics()

    msg = (
        f"mAP_50={metrics['mAP_50']:.3f} (baseline_min={baseline['mAP_50_min']}), "
        f"mAP_50_95={metrics['mAP_50_95']:.3f} (baseline_min={baseline['mAP_50_95_min']}), "
        f"TPs={metrics['TPs']}, FPs={metrics['FPs']}, FNs={metrics['FNs']}"
    )
    assert metrics["mAP_50"] >= baseline["mAP_50_min"], msg
    assert metrics["mAP_50_95"] >= baseline["mAP_50_95_min"], msg
