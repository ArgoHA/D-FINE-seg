"""Pin IoU / GIoU / cxcywh<->xyxy in `src/d_fine/arch/utils.py`.

These are inside the matcher and the GIoU loss — silent regressions here break
training. Identity / disjoint / round-trip checks pin the invariants.
"""

import torch

from src.d_fine.arch.utils import (
    box_cxcywh_to_xyxy,
    box_iou,
    box_xyxy_to_cxcywh,
    generalized_box_iou,
)


def test_box_iou_identical_boxes_is_one():
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0]])
    iou, union = box_iou(boxes, boxes)
    diag = torch.diag(iou)
    assert torch.allclose(diag, torch.ones_like(diag))


def test_box_iou_disjoint_boxes_is_zero():
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b = torch.tensor([[100.0, 100.0, 200.0, 200.0]])
    iou, _ = box_iou(a, b)
    assert iou.item() == 0.0


def test_box_iou_half_overlap():
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b = torch.tensor([[5.0, 0.0, 15.0, 10.0]])
    iou, _ = box_iou(a, b)
    # intersection 50, union 150 -> 1/3
    assert torch.allclose(iou, torch.tensor([[1 / 3]]), atol=1e-6)


def test_generalized_box_iou_equals_iou_for_overlapping_boxes():
    # GIoU == IoU when the enclosing box equals the union (no extra area).
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    iou, _ = box_iou(a, a)
    giou = generalized_box_iou(a, a)
    assert torch.allclose(giou, iou)


def test_generalized_box_iou_disjoint_is_negative():
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b = torch.tensor([[100.0, 100.0, 110.0, 110.0]])
    giou = generalized_box_iou(a, b)
    assert giou.item() < 0  # always negative when boxes don't overlap


def test_cxcywh_xyxy_round_trip():
    rng = torch.Generator().manual_seed(7)
    cx = torch.rand((16,), generator=rng) * 100
    cy = torch.rand((16,), generator=rng) * 100
    w = torch.rand((16,), generator=rng) * 20 + 1
    h = torch.rand((16,), generator=rng) * 20 + 1
    boxes = torch.stack([cx, cy, w, h], dim=-1)
    back = box_xyxy_to_cxcywh(box_cxcywh_to_xyxy(boxes))
    assert torch.allclose(back, boxes, atol=1e-5)


def test_cxcywh_to_xyxy_clamps_negative_dims():
    # Negative w/h should not produce inverted boxes (matters for early-training stability).
    boxes = torch.tensor([[10.0, 10.0, -5.0, -5.0]])
    xyxy = box_cxcywh_to_xyxy(boxes)
    assert xyxy[0, 2] >= xyxy[0, 0]
    assert xyxy[0, 3] >= xyxy[0, 1]
