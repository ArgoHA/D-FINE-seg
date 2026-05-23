"""Pin Validator's metric surface.

`Validator.compute_metrics()` is the accuracy oracle for the whole framework —
it has to return a dict containing every metric key that train.py and bench.py
consume. The "perfect preds" case must yield mAP_50 = 1.0; a near-perfect case
with one extra false positive must still keep precision below 1.
"""

import copy

import pytest
import torch

from src.dl.validator import Validator

LABEL_TO_NAME = {0: "cat", 1: "dog"}

REQUIRED_KEYS_BBOX = {"f1", "precision", "recall", "iou", "TPs", "FPs", "FNs", "mAP_50", "mAP_50_95"}


def test_validator_returns_required_keys(synthetic_preds_gt):
    v = Validator(
        gt=copy.deepcopy(synthetic_preds_gt["gt"]),
        preds=copy.deepcopy(synthetic_preds_gt["preds"]),
        label_to_name=LABEL_TO_NAME,
    )
    metrics = v.compute_metrics()
    assert REQUIRED_KEYS_BBOX.issubset(metrics.keys()), \
        f"missing keys: {REQUIRED_KEYS_BBOX - metrics.keys()}"


def test_validator_perfect_preds_score_one():
    gt = [
        {
            "labels": torch.tensor([0, 1], dtype=torch.long),
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0],
                                   [60.0, 60.0, 100.0, 100.0]]),
        }
    ]
    preds = [
        {
            "labels": torch.tensor([0, 1], dtype=torch.long),
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0],
                                   [60.0, 60.0, 100.0, 100.0]]),
            "scores": torch.tensor([0.99, 0.99]),
        }
    ]
    v = Validator(gt=copy.deepcopy(gt), preds=copy.deepcopy(preds), label_to_name=LABEL_TO_NAME)
    metrics = v.compute_metrics()
    assert metrics["mAP_50"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["TPs"] == 2
    assert metrics["FPs"] == 0
    assert metrics["FNs"] == 0


def test_validator_extra_fp_drags_precision_below_one(synthetic_preds_gt):
    # synthetic_preds_gt has an extra FP in image 2.
    v = Validator(
        gt=copy.deepcopy(synthetic_preds_gt["gt"]),
        preds=copy.deepcopy(synthetic_preds_gt["preds"]),
        label_to_name=LABEL_TO_NAME,
    )
    metrics = v.compute_metrics()
    assert metrics["FPs"] >= 1
    assert metrics["precision"] < 1.0
    # Recall is still 1.0 — we didn't miss any GT box.
    assert metrics["recall"] == pytest.approx(1.0)


def test_validator_with_masks_emits_mask_keys():
    # Single image, one object: a 100x100 mask with a 20x20 filled patch.
    mask = torch.zeros((1, 100, 100), dtype=torch.uint8)
    mask[0, 20:40, 20:40] = 1
    gt = [{
        "labels": torch.tensor([0], dtype=torch.long),
        "boxes": torch.tensor([[20.0, 20.0, 40.0, 40.0]]),
        "masks": mask,
    }]
    preds = [{
        "labels": torch.tensor([0], dtype=torch.long),
        "boxes": torch.tensor([[20.0, 20.0, 40.0, 40.0]]),
        "scores": torch.tensor([0.99]),
        "masks": mask.clone(),
    }]
    v = Validator(gt=copy.deepcopy(gt), preds=copy.deepcopy(preds), label_to_name=LABEL_TO_NAME)
    metrics = v.compute_metrics()
    assert "mAP_50_mask" in metrics
    assert metrics["mAP_50_mask"] == pytest.approx(1.0)
    assert metrics["mAP_50"] == pytest.approx(1.0)


def test_validator_compute_maps_false_skips_torchmetrics(synthetic_preds_gt):
    v = Validator(
        gt=copy.deepcopy(synthetic_preds_gt["gt"]),
        preds=copy.deepcopy(synthetic_preds_gt["preds"]),
        label_to_name=LABEL_TO_NAME,
        compute_maps=False,
    )
    metrics = v.compute_metrics()
    # mAP keys are skipped, but the simple metrics still come back.
    assert "mAP_50" not in metrics
    assert {"f1", "precision", "recall", "iou", "TPs", "FPs", "FNs"} <= metrics.keys()
