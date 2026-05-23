"""Pin loss-function building blocks.

Tests synthesize tiny prediction / target tensors and assert basic invariants:
losses are finite, non-negative scalars, and shrink toward zero as the prediction
approaches the target.
"""

import torch

from src.d_fine.dfine_criterion import DFINECriterion
from src.d_fine.matcher import HungarianMatcher


def _criterion(num_classes=5):
    matcher = HungarianMatcher(
        weight_dict={"cost_class": 2, "cost_bbox": 5, "cost_giou": 2}, use_focal_loss=True
    )
    return DFINECriterion(
        matcher,
        weight_dict={"loss_vfl": 1, "loss_bbox": 5, "loss_giou": 2},
        losses=["vfl", "boxes"],
        num_classes=num_classes,
    )


def _indices_identity(n):
    return [(torch.arange(n), torch.arange(n))]


def test_loss_boxes_zero_for_perfect_match():
    crit = _criterion()
    pred_boxes = torch.tensor([[[0.25, 0.25, 0.2, 0.2], [0.7, 0.7, 0.1, 0.1]]])
    targets = [{"boxes": pred_boxes[0].clone(), "labels": torch.tensor([0, 1])}]
    outputs = {"pred_boxes": pred_boxes, "pred_logits": torch.zeros(1, 2, 5)}
    losses = crit.loss_boxes(outputs, targets, indices=_indices_identity(2), num_boxes=2)
    assert losses["loss_bbox"].item() == 0.0
    # GIoU loss is 1 - GIoU; identical boxes -> 0.
    assert losses["loss_giou"].item() < 1e-5


def test_loss_boxes_positive_for_offset_pred():
    crit = _criterion()
    pred = torch.tensor([[[0.50, 0.50, 0.1, 0.1]]])
    tgt = torch.tensor([[0.25, 0.25, 0.1, 0.1]])
    targets = [{"boxes": tgt, "labels": torch.tensor([0])}]
    outputs = {"pred_boxes": pred, "pred_logits": torch.zeros(1, 1, 5)}
    losses = crit.loss_boxes(outputs, targets, indices=_indices_identity(1), num_boxes=1)
    assert losses["loss_bbox"].item() > 0
    assert losses["loss_giou"].item() > 0


def test_loss_labels_focal_finite_and_nonnegative():
    crit = _criterion(num_classes=5)
    outputs = {"pred_logits": torch.randn(2, 4, 5)}
    targets = [
        {"labels": torch.tensor([1]), "boxes": torch.zeros(1, 4)},
        {"labels": torch.tensor([0, 3]), "boxes": torch.zeros(2, 4)},
    ]
    indices = [(torch.tensor([0]), torch.tensor([0])),
               (torch.tensor([0, 1]), torch.tensor([0, 1]))]
    losses = crit.loss_labels_focal(outputs, targets, indices=indices, num_boxes=3)
    val = losses["loss_focal"].item()
    assert val >= 0 and torch.isfinite(losses["loss_focal"])


def test_cropped_bce_zero_for_perfect_mask():
    # Logit large positive inside box, large negative outside -> sigmoid ~ tgt.
    M, H, W = 1, 16, 16
    boxes = torch.tensor([[4.0, 4.0, 12.0, 12.0]])
    tgt = torch.zeros(M, H, W)
    tgt[0, 4:12, 4:12] = 1.0
    logits = torch.full((M, H, W), -10.0)
    logits[0, 4:12, 4:12] = 10.0
    loss = DFINECriterion._cropped_bce_loss(logits, tgt, boxes)
    assert loss.item() < 1e-3


def test_cropped_dice_zero_for_perfect_mask():
    M, H, W = 1, 16, 16
    boxes = torch.tensor([[4.0, 4.0, 12.0, 12.0]])
    tgt = torch.zeros(M, H, W)
    tgt[0, 4:12, 4:12] = 1.0
    logits = torch.full((M, H, W), -10.0)
    logits[0, 4:12, 4:12] = 10.0
    loss = DFINECriterion._cropped_dice_loss(logits, tgt, boxes)
    assert loss.item() < 1e-2


def test_cropped_bce_empty_returns_zero():
    # No instances -> guarded zero return; must stay differentiable / scalar.
    logits = torch.zeros(0, 8, 8)
    tgt = torch.zeros(0, 8, 8)
    boxes = torch.zeros(0, 4)
    loss = DFINECriterion._cropped_bce_loss(logits, tgt, boxes)
    assert loss.item() == 0.0
