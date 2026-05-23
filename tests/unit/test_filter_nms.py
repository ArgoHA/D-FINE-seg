"""Pin post-NMS prediction filters."""

import pytest
import torch

from src.infer.torch_model import cleanup_masks, filter_preds


def _pred(scores, labels, boxes):
    return {
        "scores": torch.tensor(scores),
        "labels": torch.tensor(labels, dtype=torch.long),
        "boxes": torch.tensor(boxes, dtype=torch.float32),
    }


def test_filter_preds_per_class_threshold():
    # threshold[0] = 0.5, threshold[1] = 0.8
    preds = [
        _pred(
            scores=[0.9, 0.4, 0.85, 0.75],
            labels=[0, 0, 1, 1],
            boxes=[[0, 0, 1, 1]] * 4,
        )
    ]
    out = filter_preds(preds, conf_threshs=[0.5, 0.8])
    assert out[0]["scores"].tolist() == pytest.approx([0.9, 0.85])
    assert out[0]["labels"].tolist() == [0, 1]


def test_filter_preds_drops_everything_under_threshold():
    preds = [_pred(scores=[0.1, 0.2], labels=[0, 0], boxes=[[0, 0, 1, 1]] * 2)]
    out = filter_preds(preds, conf_threshs=[0.99])
    assert out[0]["scores"].numel() == 0
    assert out[0]["boxes"].numel() == 0


def test_cleanup_masks_zeros_outside_box():
    masks = torch.ones((1, 10, 10), dtype=torch.float32)
    boxes = torch.tensor([[2, 3, 7, 8]], dtype=torch.float32)
    out = cleanup_masks(masks, boxes)
    # Pixels inside the box (rows 3..7, cols 2..6) should be 1, rest 0.
    assert out[0, 5, 5].item() == 1.0
    assert out[0, 0, 0].item() == 0.0
    assert out[0, 9, 9].item() == 0.0
    # Sum equals box area (5 * 5 = 25 pixels).
    assert out.sum().item() == 25.0


def test_cleanup_masks_no_overlap_zeros_all():
    masks = torch.ones((1, 5, 5))
    boxes = torch.tensor([[10, 10, 20, 20]], dtype=torch.float32)
    out = cleanup_masks(masks, boxes)
    assert out.sum().item() == 0.0
