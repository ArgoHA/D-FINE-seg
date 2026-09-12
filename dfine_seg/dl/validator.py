import copy
import gc
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou

from dfine_seg.dl.utils import filter_preds, rle_to_masks

# Suppress verbose output from faster_coco_eval
logging.getLogger("faster_coco_eval").setLevel(logging.WARNING)


def match_instances(ious, pred_labels, gt_labels, iou_thresh):
    """Greedily match prediction/GT pairs by descending IoU."""
    matched_preds, matched_gts = set(), set()
    matches = []
    pred_idx, gt_idx = torch.nonzero(ious >= iou_thresh, as_tuple=True)
    if pred_idx.numel():
        values = ious[pred_idx, gt_idx]
        order = torch.argsort(values, descending=True)
        for pi, gi, iou in zip(pred_idx[order], gt_idx[order], values[order]):
            pi, gi = pi.item(), gi.item()
            if pi in matched_preds or gi in matched_gts:
                continue
            matched_preds.add(pi)
            matched_gts.add(gi)
            matches.append((pi, gi, float(iou)))

    unmatched_preds = sorted(set(range(len(pred_labels))) - matched_preds)
    unmatched_gts = sorted(set(range(len(gt_labels))) - matched_gts)
    return matches, unmatched_preds, unmatched_gts


class Validator:
    def __init__(
        self,
        gt: List[Dict[str, torch.Tensor]],
        preds: List[Dict[str, torch.Tensor]],
        label_to_name: Dict[int, str],
        conf_thresh=0.5,
        iou_thresh=0.5,
        mask_batch_size=1000,
        compute_maps=True,
    ) -> None:
        """
        Format example:
        gt = [{'labels': tensor([0]), 'boxes': tensor([[561.0, 297.0, 661.0, 359.0]])}, ...]
        len(gt) is the number of images
        bboxes are in format [x1, y1, x2, y2], absolute values

        mask_batch_size - Number of images to process at once when computing mask metrics.
            Lower values use less RAM but may be slower. Default 500.
        """
        self.gt = gt
        self.preds = preds
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        # floor below conf_thresh so the f1 sweep can find a real optimum (preds for the
        # sweep come from the unfiltered all_* arrays, see save_plots)
        self.thresholds = np.arange(0.05, 1.0, 0.05)
        self.label_to_name = label_to_name
        self.conf_matrix = None
        self.mask_batch_size = mask_batch_size
        self.compute_maps = compute_maps

        # Use faster_coco_eval backend for numpy 2.x compatibility
        self.torch_metric = MeanAveragePrecision(
            box_format="xyxy", iou_type="bbox", sync_on_compute=False, backend="faster_coco_eval"
        )
        self.torch_metric.warn_on_many_detections = False

        # get raw preds for torchmetrics (only needed when computing mAPs; the deepcopy
        # duplicates every dense mask, so skip it otherwise to avoid an OOM spike)
        if self.compute_maps:
            self.torchmetrics_preds = copy.deepcopy(preds)

            if len(self.torchmetrics_preds) > 0 and "all_boxes" in self.torchmetrics_preds[0]:
                for torchmetrics_pred in self.torchmetrics_preds:
                    for key in ["boxes", "labels", "scores"]:
                        torchmetrics_pred[key] = torchmetrics_pred[f"all_{key}"]
                        del torchmetrics_pred[f"all_{key}"]

            self.torch_metric.update(self.torchmetrics_preds, gt)

        # Check if masks available (either dense or RLE-encoded)
        def _has_masks(sample):
            if "masks" in sample and sample["masks"] is not None:
                if hasattr(sample["masks"], "numel"):
                    return sample["masks"].numel() > 0
                return True
            if "masks_rle" in sample and sample["masks_rle"]:
                return True
            return False

        self.use_masks = any(_has_masks(p) for p in preds) and any(_has_masks(g) for g in gt)
        if self.use_masks and self.compute_maps:
            self.torch_metric_mask = MeanAveragePrecision(
                box_format="xyxy",
                iou_type="segm",
                sync_on_compute=False,
                backend="faster_coco_eval",
            )
            self.torch_metric_mask.warn_on_many_detections = False
            # Decode RLE masks for torchmetrics in batches to avoid OOM
            # torchmetrics supports incremental .update() calls
            batch_size = self.mask_batch_size
            n_samples = len(preds)
            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)

                # Deep copy only this batch
                preds_batch = [copy.deepcopy(preds[i]) for i in range(batch_start, batch_end)]
                gt_batch = [copy.deepcopy(gt[i]) for i in range(batch_start, batch_end)]

                # Decode RLE to dense for this batch only
                preds_batch = self._prepare_masks_for_torchmetrics(preds_batch)
                gt_batch = self._prepare_masks_for_torchmetrics(gt_batch)

                # Update metrics incrementally
                self.torch_metric_mask.update(preds_batch, gt_batch)

                # Explicitly free memory
                del preds_batch, gt_batch

    def compute_metrics(self, extended=False, ignore_masks=False, cleanup=True) -> Dict[str, float]:
        if self.compute_maps:
            self.torch_metrics = self.torch_metric.compute()

        metrics = self._compute_main_metrics(self.preds, ignore_masks=ignore_masks)
        if self.compute_maps:
            metrics["mAP_50"] = self.torch_metrics["map_50"].item()
            metrics["mAP_50_95"] = self.torch_metrics["map"].item()
            if self.use_masks and not ignore_masks:
                tm_mask = self.torch_metric_mask.compute()
                metrics["mAP_50_mask"] = tm_mask["map_50"].item()
                metrics["mAP_50_95_mask"] = tm_mask["map"].item()
                metrics["extended_metrics"].update(
                    {
                        "mAP_50_95_mask": metrics["mAP_50_95_mask"],
                        "mAP_50_95": metrics["mAP_50_95"],
                    }
                )
                del tm_mask

        if not extended:
            metrics.pop("extended_metrics", None)
        # Clean up large data structures to free RAM
        if cleanup and self.compute_maps:
            self._cleanup_torchmetrics()
        return metrics

    def _cleanup_torchmetrics(self):
        """Reset torchmetrics internal state to free memory."""

        # Reset torchmetrics - this clears their internal detection/groundtruth lists
        if hasattr(self, "torch_metric"):
            self.torch_metric.reset()
        if hasattr(self, "torch_metric_mask"):
            self.torch_metric_mask.reset()

        # Clear the deep copy of predictions used for torchmetrics
        if hasattr(self, "torchmetrics_preds"):
            del self.torchmetrics_preds
            self.torchmetrics_preds = None

        # Clear stored torch_metrics results
        if hasattr(self, "torch_metrics"):
            del self.torch_metrics
            self.torch_metrics = None

        # Force garbage collection
        gc.collect()

    def _prepare_masks_for_torchmetrics(self, samples: List[Dict]) -> List[Dict]:
        """Decode and binarize masks for torchmetrics."""
        for s in samples:
            s["masks"] = self._sample_masks(s, allow_probs=True)
            s.pop("masks_rle", None)
            s.pop("masks_size", None)
        return samples

    def _to_nhw_uint8(self, m: torch.Tensor) -> torch.Tensor:
        if m is None or m.numel() == 0:
            return torch.zeros((0, 1, 1), dtype=torch.uint8)
        if m.dtype != torch.uint8:
            m = (m > float(self.conf_thresh)).to(torch.uint8)
        if m.ndim == 4 and m.shape[1] == 1:
            m = m[:, 0]
        elif m.ndim != 3:
            m = m.reshape(m.shape[0], -1, m.shape[-2], m.shape[-1])
            m = m[:, 0]
        return m

    def _sample_masks(self, sample: Dict, allow_probs: bool = False) -> torch.Tensor:
        if sample.get("masks_rle"):
            return rle_to_masks(sample["masks_rle"])
        masks = sample.get("masks")
        if masks is not None and masks.numel():
            return self._to_nhw_uint8(masks)
        probs = sample.get("mask_probs") if allow_probs else None
        if probs is not None and probs.numel():
            return self._to_nhw_uint8(probs)
        return torch.zeros((0, 1, 1), dtype=torch.uint8)

    def _pairwise_mask_iou(self, pm: torch.Tensor, gm: torch.Tensor) -> torch.Tensor:
        # pm: [Np,H,W] uint8, gm: [Ng,H,W] uint8
        if pm.numel() == 0 or gm.numel() == 0:
            return torch.zeros((pm.shape[0], gm.shape[0]))
        pmf = pm.to(dtype=torch.float32).flatten(1)  # [Np,HW]
        gmf = gm.to(dtype=torch.float32).flatten(1)  # [Ng,HW]
        inter = pmf @ gmf.T  # [Np,Ng]
        area_p = pmf.sum(dim=1, keepdim=True)  # [Np,1]
        area_g = gmf.sum(dim=1, keepdim=True).T  # [1,Ng]
        union = area_p + area_g - inter
        return torch.where(union > 0, inter / union, torch.zeros_like(union))

    def _compute_main_metrics(self, preds, ignore_masks=False):
        (
            self.metrics_per_class,
            self.conf_matrix,
            self.class_to_idx,
        ) = self._compute_metrics_and_confusion_matrix(preds, ignore_masks=ignore_masks)
        tps, fps, fns = 0, 0, 0
        ious = []
        extended_metrics = {}
        for key, value in self.metrics_per_class.items():
            tps += value["TPs"]
            fps += value["FPs"]
            fns += value["FNs"]
            ious.extend(value["IoUs"])

            extended_metrics[f"precision_{self.label_to_name[key]}"] = (
                value["TPs"] / (value["TPs"] + value["FPs"])
                if value["TPs"] + value["FPs"] > 0
                else 0
            )
            extended_metrics[f"recall_{self.label_to_name[key]}"] = (
                value["TPs"] / (value["TPs"] + value["FNs"])
                if value["TPs"] + value["FNs"] > 0
                else 0
            )

            extended_metrics[f"iou_{self.label_to_name[key]}"] = np.mean(value["IoUs"])
            extended_metrics[f"f1_{self.label_to_name[key]}"] = (
                2
                * (
                    extended_metrics[f"precision_{self.label_to_name[key]}"]
                    * extended_metrics[f"recall_{self.label_to_name[key]}"]
                )
                / (
                    extended_metrics[f"precision_{self.label_to_name[key]}"]
                    + extended_metrics[f"recall_{self.label_to_name[key]}"]
                )
                if (
                    extended_metrics[f"precision_{self.label_to_name[key]}"]
                    + extended_metrics[f"recall_{self.label_to_name[key]}"]
                )
                > 0
                else 0
            )

        precision = tps / (tps + fps) if (tps + fps) > 0 else 0
        recall = tps / (tps + fns) if (tps + fns) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "iou": np.mean(ious) if ious else 0,
            "TPs": tps,
            "FPs": fps,
            "FNs": fns,
            "extended_metrics": extended_metrics,
        }

    def _compute_metrics_and_confusion_matrix(self, preds, ignore_masks):
        if self.use_masks and not ignore_masks:
            return self._instance_metrics(preds, self._mask_ious)
        return self._instance_metrics(preds, self._box_ious)

    @staticmethod
    def _box_ious(pred, gt):
        if len(pred["boxes"]) and len(gt["boxes"]):
            return box_iou(pred["boxes"], gt["boxes"])
        return torch.zeros((len(pred["labels"]), len(gt["labels"])))

    def _mask_ious(self, pred, gt):
        pred_masks = self._sample_masks(pred, allow_probs=True)
        gt_masks = self._sample_masks(gt)
        if len(pred_masks) and len(gt_masks) and pred_masks.shape[-2:] != gt_masks.shape[-2:]:
            pred_masks = torch.nn.functional.interpolate(
                pred_masks[:, None].float(),
                size=gt_masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[:, 0]
            pred_masks = (pred_masks > 0.5).to(torch.uint8)
        return self._pairwise_mask_iou(pred_masks, gt_masks)

    def _instance_metrics(self, preds, iou_fn):
        metrics_per_class = defaultdict(lambda: {"TPs": 0, "FPs": 0, "FNs": 0, "IoUs": []})
        all_classes = sorted(
            {label for sample in [*preds, *self.gt] for label in sample["labels"].tolist()}
        )
        class_to_idx = {cls_id: idx for idx, cls_id in enumerate(all_classes)}
        n_classes = len(all_classes)
        conf_matrix = np.zeros((n_classes + 1, n_classes + 1), dtype=int)

        for pred, gt in zip(preds, self.gt):
            pred_labels = pred["labels"]
            gt_labels = gt["labels"]
            matches, unmatched_preds, unmatched_gts = match_instances(
                iou_fn(pred, gt), pred_labels, gt_labels, self.iou_thresh
            )
            for pred_idx, gt_idx, iou in matches:
                pred_label = pred_labels[pred_idx].item()
                gt_label = gt_labels[gt_idx].item()
                conf_matrix[class_to_idx[gt_label], class_to_idx[pred_label]] += 1
                if pred_label == gt_label:
                    metrics_per_class[gt_label]["TPs"] += 1
                    metrics_per_class[gt_label]["IoUs"].append(iou)
                else:
                    metrics_per_class[gt_label]["FNs"] += 1
                    metrics_per_class[pred_label]["FPs"] += 1
                    metrics_per_class[gt_label]["IoUs"].append(0)
                    metrics_per_class[pred_label]["IoUs"].append(0)

            for pred_idx in unmatched_preds:
                pred_label = pred_labels[pred_idx].item()
                conf_matrix[n_classes, class_to_idx[pred_label]] += 1
                metrics_per_class[pred_label]["FPs"] += 1
                metrics_per_class[pred_label]["IoUs"].append(0)

            for gt_idx in unmatched_gts:
                gt_label = gt_labels[gt_idx].item()
                conf_matrix[class_to_idx[gt_label], n_classes] += 1
                metrics_per_class[gt_label]["FNs"] += 1
                metrics_per_class[gt_label]["IoUs"].append(0)

        return metrics_per_class, conf_matrix, class_to_idx

    def save_plots(self, path_to_save) -> float:
        path_to_save = Path(path_to_save)
        path_to_save.mkdir(parents=True, exist_ok=True)

        if self.conf_matrix is not None:
            class_labels = [str(cls_id) for cls_id in self.class_to_idx.keys()] + ["background"]

            plt.figure(figsize=(10, 8))
            plt.imshow(self.conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
            plt.title("Confusion Matrix")
            plt.colorbar()
            tick_marks = np.arange(len(class_labels))
            plt.xticks(tick_marks, class_labels, rotation=45)
            plt.yticks(tick_marks, class_labels)

            # Add labels to each cell
            thresh = self.conf_matrix.max() / 2.0
            for i in range(self.conf_matrix.shape[0]):
                for j in range(self.conf_matrix.shape[1]):
                    plt.text(
                        j,
                        i,
                        format(self.conf_matrix[i, j], "d"),
                        horizontalalignment="center",
                        color="white" if self.conf_matrix[i, j] > thresh else "black",
                    )

            plt.ylabel("True label")
            plt.xlabel("Predicted label")
            plt.tight_layout()
            plt.savefig(path_to_save / "confusion_matrix.png")
            plt.close()

        thresholds = self.thresholds
        precisions, recalls, f1_scores = [], [], []

        if not self.preds:
            return None

        # Sweep over the UNFILTERED preds (all_* arrays, the same mAP uses): self.preds["scores"]
        # are already filtered at conf_thresh, so sweeping them can never go below it.
        base_preds = []
        for p in self.preds:
            if "all_scores" in p:
                base_preds.append(
                    {"scores": p["all_scores"], "boxes": p["all_boxes"], "labels": p["all_labels"]}
                )
            else:  # bench-style preds have no all_*; fall back to the (already-thresholded) scores
                base_preds.append(
                    {"scores": p["scores"], "boxes": p["boxes"], "labels": p["labels"]}
                )

        for threshold in thresholds:
            filtered_preds = filter_preds(copy.deepcopy(base_preds), threshold, mask_source="masks")
            # Compute metrics with the filtered predictions (using boxes only)
            metrics = self._compute_main_metrics(filtered_preds, ignore_masks=True)
            precisions.append(metrics["precision"])
            recalls.append(metrics["recall"])
            f1_scores.append(metrics["f1"])

        # Plot Precision and Recall vs Threshold
        plt.figure()
        plt.plot(thresholds, precisions, label="Precision", marker="o")
        plt.plot(thresholds, recalls, label="Recall", marker="o")
        plt.xlabel("Threshold")
        plt.ylabel("Value")
        plt.title("Precision and Recall vs Threshold")
        plt.legend()
        plt.grid(True)
        plt.savefig(path_to_save / "precision_recall_vs_threshold.png")
        plt.close()

        # Plot F1 Score vs Threshold
        plt.figure()
        plt.plot(thresholds, f1_scores, label="F1 Score", marker="o")
        plt.xlabel("Threshold")
        plt.ylabel("F1 Score")
        plt.title("F1 Score vs Threshold")
        plt.grid(True)
        plt.savefig(path_to_save / "f1_score_vs_threshold.png")
        plt.close()

        # Find the best threshold based on F1 Score (last occurence)
        best_idx = len(f1_scores) - np.argmax(f1_scores[::-1]) - 1
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]

        logger.info(
            f"Best Threshold for object detection: {round(best_threshold, 2)} with F1 Score: {round(best_f1, 3)}"
        )
        self.optimal_thresh = round(float(best_threshold), 2)
        return self.optimal_thresh


class SemSegValidator:
    """Streaming validator for task=sem_seg: a [C, C] pixel confusion matrix
    (rows = GT, cols = pred) accumulated at ORIGINAL image resolution - nothing
    dense is stored. GT pixels equal to ignore_index never enter the matrix.
    """

    def __init__(self, num_classes: int, label_to_name: Dict[int, str], ignore_index: int = 255):
        self.num_classes = num_classes
        self.label_to_name = label_to_name
        self.ignore_index = ignore_index
        self.cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, gt: torch.Tensor) -> None:
        """pred/gt: (H, W) integer tensors at the same (original) resolution."""
        valid = gt != self.ignore_index
        gt_v = gt[valid].long()
        if gt_v.numel() and int(gt_v.max()) >= self.num_classes:
            raise ValueError(
                f"GT mask contains class id {int(gt_v.max())} >= num_classes="
                f"{self.num_classes} (ignore_index={self.ignore_index}); "
                "masks must use contiguous label_to_name ids"
            )
        idx = gt_v * self.num_classes + pred[valid].long()
        cm = torch.bincount(idx, minlength=self.num_classes**2)
        self.cm += cm.reshape(self.num_classes, self.num_classes).cpu()

    def compute_metrics(self, extended: bool = False) -> Dict[str, float]:
        cm = self.cm.double()
        diag = cm.diag()
        union = cm.sum(1) + cm.sum(0) - diag
        present = cm.sum(1) > 0  # classes with GT pixels
        iou = diag / union.clamp(min=1)
        miou = iou[present].mean().item() if present.any() else 0.0
        acc = (diag.sum() / cm.sum().clamp(min=1)).item()
        metrics = {"mIoU": round(miou, 4), "pixel_acc": round(acc, 4)}
        if extended:
            metrics["extended_metrics"] = {
                f"iou_{self.label_to_name[c]}": round(iou[c].item(), 4)
                for c in range(self.num_classes)
                if present[c]
            }
        return metrics

    def save_plots(self, path_to_save) -> None:
        """Row-normalized pixel confusion matrix."""
        path_to_save = Path(path_to_save)
        path_to_save.mkdir(parents=True, exist_ok=True)

        cm = self.cm.double()
        cm_norm = (cm / cm.sum(1, keepdim=True).clamp(min=1)).numpy()
        class_labels = [str(self.label_to_name[c]) for c in range(self.num_classes)]

        plt.figure(figsize=(max(8, self.num_classes * 0.5), max(6, self.num_classes * 0.45)))
        plt.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
        plt.title("Pixel Confusion Matrix (row-normalized)")
        plt.colorbar()
        tick_marks = np.arange(self.num_classes)
        plt.xticks(tick_marks, class_labels, rotation=90)
        plt.yticks(tick_marks, class_labels)
        plt.ylabel("True class")
        plt.xlabel("Predicted class")
        plt.tight_layout()
        plt.savefig(path_to_save / "confusion_matrix.png")
        plt.close()
