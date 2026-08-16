"""Bench must read GT through the same path as training.

BenchLoader used to build its datasets directly, without `coco_annotations`, so a
`coco_dataset: True` run silently fell back to the YOLO `labels/` dir — GT was empty
(or a stale bbox-only leftover) while predictions were real.
"""

import json

import cv2
import numpy as np
import pytest
from omegaconf import OmegaConf

from dfine_seg.dl.bench import BenchLoader

H, W = 30, 40


def _cfg(tmp_path, coco_dataset=True):
    return OmegaConf.create(
        {
            "task": "segment",
            "train": {
                "root": str(tmp_path),
                "coco_dataset": coco_dataset,
                "in_channels": 3,
                "keep_ratio": False,
                "use_one_class": False,
                "debug_img_path": str(tmp_path / "debug"),
                "label_to_name": {0: "person"},
                "mosaic_augs": {
                    "mosaic_prob": 0.0,
                    "mosaic_scale": [0.5, 1.5],
                    "degrees": 0.0,
                    "translate": 0.2,
                    "shear": 2.0,
                },
                "augs": {"multiscale_prob": 0.0},
            },
        }
    )


def _dataset(tmp_path, segmentation=True):
    (tmp_path / "images").mkdir()
    cv2.imwrite(str(tmp_path / "images" / "a.jpg"), np.zeros((H, W, 3), np.uint8))
    ann = {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 10, 10]}
    if segmentation:
        ann["segmentation"] = [[10, 10, 20, 10, 20, 20, 10, 20]]
    coco = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": W, "height": H}],
        "categories": [{"id": 1, "name": "person"}],
        "annotations": [ann],
    }
    for split in ("train", "val"):
        (tmp_path / f"{split}.json").write_text(json.dumps(coco))


def test_bench_loader_reads_coco_gt(tmp_path):
    _dataset(tmp_path)
    val_loader, test_loader = BenchLoader(
        root_path=tmp_path, img_size=(64, 64), batch_size=1, num_workers=0, cfg=_cfg(tmp_path)
    ).build_dataloaders()

    assert test_loader is None
    ds = val_loader.dataset
    assert ds.coco_mode and ds.mode == "bench"

    _, targets, _ = next(iter(val_loader))
    target = targets[0]
    assert len(target["labels"]) == 1
    assert target["masks"].sum() > 0
    assert len(target["polys"][0]) == 1  # original-res GT polygon, used for mask mAP


def test_bbox_only_coco_raises_for_segment(tmp_path):
    """Same guard the YOLO path has: bbox-only anns would make every GT mask all-zero."""
    _dataset(tmp_path, segmentation=False)
    with pytest.raises(ValueError, match="no polygon annotations"):
        BenchLoader(
            root_path=tmp_path, img_size=(64, 64), batch_size=1, num_workers=0, cfg=_cfg(tmp_path)
        ).build_dataloaders()
