"""Pin matcher cost functions and Hungarian assignment.

`HungarianMatcher` is the centerpiece of DETR-style training. The identity case
(perfect preds for a single target) must always yield the identity assignment.
"""

import torch

from src.d_fine.matcher import HungarianMatcher, dice_cost, sigmoid_focal_cost


def _matcher(use_focal=True, weights=None):
    weights = weights or {"cost_class": 2, "cost_bbox": 5, "cost_giou": 2}
    return HungarianMatcher(
        weight_dict=weights, use_focal_loss=use_focal, alpha=0.25, gamma=2.0
    )


def test_dice_cost_zero_when_pred_equals_target():
    masks = torch.ones((2, 8, 8))  # binary
    cost = dice_cost(pred_masks=masks, gt_masks=masks)
    assert torch.allclose(cost, torch.zeros_like(cost), atol=1e-3)


def test_dice_cost_one_when_disjoint():
    pred = torch.zeros((1, 8, 8))
    gt = torch.ones((1, 8, 8))
    cost = dice_cost(pred_masks=pred, gt_masks=gt)
    assert cost.item() > 0.99


def test_sigmoid_focal_cost_shape():
    pred = torch.zeros((3, 16))  # logits
    gt = torch.zeros((2, 16))
    cost = sigmoid_focal_cost(pred, gt)
    assert cost.shape == (3, 2)


def test_matcher_identity_assignment_for_perfect_pred():
    # 4 queries, 2 targets. The first 2 queries are deliberately near-perfect
    # matches for the targets (correct class + correct box). The matcher must
    # pick them, not the random later queries.
    num_queries, num_classes = 4, 5

    target_labels = torch.tensor([1, 3], dtype=torch.long)
    target_boxes_cxcywh = torch.tensor([[0.25, 0.25, 0.2, 0.2], [0.75, 0.75, 0.2, 0.2]])

    pred_logits = torch.full((1, num_queries, num_classes), -10.0)  # near-zero sigmoid
    pred_logits[0, 0, 1] = 10.0
    pred_logits[0, 1, 3] = 10.0

    pred_boxes = torch.tensor(
        [
            [
                [0.25, 0.25, 0.2, 0.2],  # matches target 0
                [0.75, 0.75, 0.2, 0.2],  # matches target 1
                [0.1, 0.1, 0.05, 0.05],
                [0.9, 0.9, 0.05, 0.05],
            ]
        ]
    )

    outputs = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
    targets = [{"labels": target_labels, "boxes": target_boxes_cxcywh}]

    matcher = _matcher(use_focal=True)
    indices = matcher(outputs, targets)["indices"]

    pred_idx, tgt_idx = indices[0]
    # Each target picked the correct pred query.
    pairs = sorted(zip(pred_idx.tolist(), tgt_idx.tolist()))
    assert pairs == [(0, 0), (1, 1)]
