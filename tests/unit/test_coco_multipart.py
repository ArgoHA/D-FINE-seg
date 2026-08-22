"""Pin COCO ingest of multi-part instances.

A COCO instance occluded into several islands carries one `segmentation` entry per
island. Keeping only one (the old `max(seg, key=len)`) silently dropped mask area
from both the training target and the val/test mask-mAP GT.
"""

import json

import numpy as np
import pytest
from loguru import logger

from dfine_seg.dl.dataset import load_coco_split
from dfine_seg.dl.utils import poly_abs_to_mask


def _square(x, y, side):
    return [x, y, x + side, y, x + side, y + side, x, y + side]


def _write_coco(tmp_path, anns, width=200, height=100):
    coco = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": width, "height": height}],
        "categories": [{"id": 1, "name": "person"}],
        "annotations": anns,
    }
    path = tmp_path / "train.json"
    path.write_text(json.dumps(coco))
    return path


def test_all_islands_kept_and_rasterized(tmp_path):
    # one instance, three disjoint islands spanning x 10..90
    seg = [_square(10, 10, 20), _square(40, 10, 20), _square(70, 10, 20)]
    path = _write_coco(
        tmp_path,
        [{"id": 7, "image_id": 1, "category_id": 1, "bbox": [10, 10, 80, 20], "segmentation": seg}],
    )

    entries, _ = load_coco_split(path)
    parts = entries[0]["polys_abs"][0]
    assert entries[0]["targets"].shape[0] == 1  # still ONE instance
    assert len(parts) == 3

    mask = poly_abs_to_mask(parts, h=100, w=200)
    for x in (20, 50, 80):
        assert mask[20, x] == 1  # every island present
    assert mask[20, 35] == 0  # gap between islands stays background


def test_box_comes_from_ann_bbox(tmp_path):
    """ann['bbox'] is authoritative, not the polygon extent."""
    seg = [_square(10, 10, 20), _square(70, 10, 20)]
    path = _write_coco(
        tmp_path,
        [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [8, 9, 84, 23], "segmentation": seg}],
    )
    entries, _ = load_coco_split(path)
    np.testing.assert_allclose(entries[0]["targets"][0, 1:], [8, 9, 92, 32])


def test_mask_outside_box_warns(tmp_path):
    seg = [_square(10, 10, 20), _square(150, 10, 20)]  # second island outside the box
    path = _write_coco(
        tmp_path,
        [{"id": 3, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "segmentation": seg}],
    )
    msgs = []  # loguru doesn't route through caplog
    sink = logger.add(msgs.append, level="WARNING")
    try:
        load_coco_split(path)
    finally:
        logger.remove(sink)
    assert any("outside ann['bbox']" in m for m in msgs)


def test_short_parts_dropped_but_instance_kept(tmp_path):
    seg = [_square(10, 10, 20), [1.0, 2.0, 3.0, 4.0]]  # 2 points: not a polygon
    path = _write_coco(
        tmp_path,
        [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "segmentation": seg}],
    )
    entries, _ = load_coco_split(path)
    assert len(entries[0]["polys_abs"][0]) == 1


def test_rle_segmentation_raises(tmp_path):
    ann = {
        "id": 5,
        "image_id": 1,
        "category_id": 1,
        "bbox": [10, 10, 20, 20],
        "segmentation": {"counts": "abc", "size": [100, 200]},
    }
    path = _write_coco(tmp_path, [ann])
    with pytest.raises(ValueError, match="RLE"):
        load_coco_split(path)


def test_iscrowd_skipped_before_rle_check(tmp_path):
    """Crowd anns are dropped by design, so their RLE must not raise."""
    ann = {
        "id": 6,
        "image_id": 1,
        "category_id": 1,
        "iscrowd": 1,
        "bbox": [10, 10, 20, 20],
        "segmentation": {"counts": "abc", "size": [100, 200]},
    }
    entries, _ = load_coco_split(_write_coco(tmp_path, [ann]))
    assert entries[0]["targets"].shape[0] == 0
