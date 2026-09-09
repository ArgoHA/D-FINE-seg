"""Exercise the optional backend through TorchMetrics and the real Validator."""

import copy
import pickle

import pytest
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from dfine_seg.dl.coco_metric import coco_metric
from dfine_seg.dl.utils import encode_sample_masks_to_rle
from dfine_seg.dl.validator import Validator


def samples():
    gt, preds = [], []
    for index in range(5):
        boxes = torch.tensor([[2.0, 3.0, 18.0, 20.0], [12.0, 10.0, 28.0, 30.0]])
        masks = torch.zeros((2, 32, 32), dtype=torch.bool)
        masks[0, 3:20, 2:18] = True
        masks[1, 10:30, 12:28] = True
        n_gt = 0 if index in (3, 4) else 2
        n_dt = 0 if index in (2, 4) else 2
        gt.append(
            {
                "boxes": boxes[:n_gt].clone(),
                "labels": torch.tensor([0, 1])[:n_gt],
                "masks": masks[:n_gt].clone(),
                "iscrowd": torch.tensor([0, index % 2])[:n_gt],
                "area": torch.tensor([272.0, 320.0])[:n_gt],
            }
        )
        preds.append(
            {
                "boxes": boxes[:n_dt].clone() + index,
                "labels": torch.tensor([0, 1])[:n_dt],
                "scores": torch.tensor([0.8, 0.8])[:n_dt],
                "masks": torch.roll(masks[:n_dt], shifts=index, dims=-1),
            }
        )
    return gt, preds


@pytest.mark.parametrize("iou_type", ["bbox", "segm"])
def test_full_arrays_and_metric_lifecycle(iou_type):
    pytest.importorskip("ultrafast_pycocotools")
    pytest.importorskip("pycocotools")
    gt, preds = samples()
    reference = MeanAveragePrecision(
        box_format="xyxy", iou_type=iou_type, sync_on_compute=False, backend="pycocotools"
    )
    fast = coco_metric(iou_type, "ultrafast")
    existing = coco_metric(iou_type)
    assert existing._coco_backend.coco.__module__.startswith("faster_coco_eval")
    assert fast._coco_backend.coco.__module__.startswith("ultrafast_pycocotools")
    for metric in (reference, fast, existing):
        metric.extended_summary = True
        metric.class_metrics = True
        # Incremental updates exercise the mask-batch path used by Validator.
        metric.update(copy.deepcopy(preds[:2]), copy.deepcopy(gt[:2]))
        metric.update(copy.deepcopy(preds[2:]), copy.deepcopy(gt[2:]))
    expected = reference.compute()
    actual = fast.compute()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    for key in ("precision", "recall", "scores"):
        assert actual[key].numpy().tobytes() == expected[key].numpy().tobytes()
    torch.testing.assert_close(existing.compute(), expected, rtol=0, atol=1e-12, equal_nan=True)
    restored = pickle.loads(pickle.dumps(fast))
    torch.testing.assert_close(restored.compute(), expected, rtol=0, atol=0, equal_nan=True)
    for metric in (fast, restored):
        metric.reset()
        metric.update(copy.deepcopy(preds), copy.deepcopy(gt))
        torch.testing.assert_close(metric.compute(), expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("encoding", ["bbox", "dense", "rle"])
def test_validator_metrics_match_existing_backend(encoding):
    pytest.importorskip("ultrafast_pycocotools")
    gt, preds = samples()
    # Validator's F1 code expects binary uint8 masks, matching training outputs.
    for item in gt + preds:
        item["masks"] = item["masks"].to(torch.uint8)
        if encoding == "bbox":
            del item["masks"]
        elif encoding == "rle":
            encode_sample_masks_to_rle(item)
    outputs = []
    for backend in ("faster_coco_eval", "ultrafast"):
        validator = Validator(
            copy.deepcopy(gt),
            copy.deepcopy(preds),
            {0: "cat", 1: "dog"},
            coco_backend=backend,
            mask_batch_size=2,
        )
        outputs.append(validator.compute_metrics(extended=True))
        assert validator.torchmetrics_preds is None
        assert len(validator.torch_metric.detection_labels) == 0
    assert outputs[0].keys() == outputs[1].keys()
    for key in outputs[0]:
        assert outputs[1][key] == pytest.approx(outputs[0][key]), key


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match="train.coco_backend"):
        coco_metric("bbox", "typo")


def test_invalid_backend_fails_before_training_setup():
    from omegaconf import OmegaConf

    from dfine_seg.dl.train import Trainer

    cfg = OmegaConf.create({"task": "detect", "train": {"coco_backend": "typo"}})
    with pytest.raises(ValueError, match="train.coco_backend"):
        Trainer(cfg)


def test_missing_optional_dependency_is_actionable(monkeypatch):
    from dfine_seg.dl import coco_metric as module

    def missing(name):
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(module, "import_module", missing)
    with pytest.raises(ModuleNotFoundError, match=r"dfine-seg\[ultrafast\]"):
        coco_metric("bbox", "ultrafast")
    # The optional package must not be required by the default backend.
    assert coco_metric("bbox")._coco_backend.coco.__module__.startswith("faster_coco_eval")


def test_no_maps_still_returns_f1(synthetic_preds_gt):
    pytest.importorskip("ultrafast_pycocotools")
    validator = Validator(
        copy.deepcopy(synthetic_preds_gt["gt"]),
        copy.deepcopy(synthetic_preds_gt["preds"]),
        {0: "cat", 1: "dog"},
        coco_backend="ultrafast",
        compute_maps=False,
    )
    result = validator.compute_metrics()
    assert "f1" in result and "mAP_50" not in result
