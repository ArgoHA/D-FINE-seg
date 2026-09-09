"""Per-metric COCO backend selection; no global import replacement."""

from importlib import import_module

import numpy as np
from torchmetrics.detection.helpers import CocoBackend
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class _MaskTools:
    @staticmethod
    def encode(value):
        from ultrafast_pycocotools import mask

        # TorchMetrics stores boolean masks; COCO's mask API expects uint8.
        if value.dtype == np.bool_:
            value = np.asfortranarray(value, dtype=np.uint8)
        return mask.encode(value)

    def __getattr__(self, name):
        from ultrafast_pycocotools import mask

        return getattr(mask, name)


class _UltrafastBackend(CocoBackend):
    @property
    def coco(self):
        from ultrafast_pycocotools import COCO

        return COCO

    @property
    def cocoeval(self):
        from ultrafast_pycocotools import COCOeval

        return COCOeval

    @property
    def mask_utils(self):
        return _MaskTools()


def validate_coco_backend(backend: str) -> None:
    """Reject unavailable backends before starting a training epoch."""
    if backend not in ("faster_coco_eval", "ultrafast"):
        raise ValueError("train.coco_backend must be 'faster_coco_eval' or 'ultrafast'")
    if backend == "ultrafast":
        try:
            import_module("ultrafast_pycocotools")
        except ModuleNotFoundError as exc:
            if exc.name != "ultrafast_pycocotools":
                raise
            raise ModuleNotFoundError(
                "The ultrafast backend requires: pip install 'dfine-seg[ultrafast]'"
            ) from exc


def coco_metric(iou_type: str, backend: str = "faster_coco_eval") -> MeanAveragePrecision:
    """Create a fresh bbox/segm metric with TorchMetrics' normal state lifecycle."""
    validate_coco_backend(backend)
    metric = MeanAveragePrecision(
        box_format="xyxy", iou_type=iou_type, sync_on_compute=False, backend="faster_coco_eval"
    )
    if backend == "ultrafast":
        # TorchMetrics 1.9 exposes no public custom-backend constructor argument.
        if not isinstance(getattr(metric, "_coco_backend", None), CocoBackend):
            raise RuntimeError("Unsupported TorchMetrics COCO backend layout")
        metric._coco_backend = _UltrafastBackend("faster_coco_eval")
    metric.warn_on_many_detections = False
    return metric
