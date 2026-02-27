"""
COCO-Style AP Computation Script for Paper

Computes proper COCO-style metrics:
- Detection: AP@[.50:.95], AP50, AP75, AR (average recall)
- Instance segmentation: mask AP@[.50:.95], mask AP50, mask AP75

Key design decisions for memory efficiency:
1. Process one image at a time (no batching)
2. Use very low confidence threshold (0.001) for proper mAP
3. Use RLE encoding for masks (10-100x smaller than dense)
4. Limit max detections per image to prevent OOM
5. Batch torchmetrics updates with immediate cleanup
6. Aggressive garbage collection

Usage:
    python -m paper_assets.compute_coco_ap
    python -m paper_assets.compute_coco_ap model_name=s exp_name=1_2_seg_s

Author: Generated for paper evaluation
"""

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import hydra
import numpy as np
import pandas as pd
import torch
from faster_coco_eval.core import mask as mask_utils
from loguru import logger
from omegaconf import DictConfig
from src.infer.yolo_trt_model import YOLO_TRT_model
from tabulate import tabulate
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from src.dl.dataset import CustomDataset, Loader
from src.dl.utils import get_latest_experiment_name, process_boxes, process_masks, seed_worker
from src.infer.trt_model import TRT_model

torch.multiprocessing.set_sharing_strategy("file_system")


@dataclass
class EvalConfig:
    """Configuration for evaluation"""

    # Confidence thresholds for COCO mAP
    # Note: With 0.001, models output 100s of detections causing OOM during mask processing
    conf_thresh_dfine: float = 0.001
    conf_thresh_yolo: float = 0.001

    # Max detections per image (COCO standard is 100)
    # Lower = less memory usage
    max_detections_per_image: int = 100

    # How many images to accumulate before flushing to torchmetrics
    # Lower = less peak memory but more overhead
    update_batch_size: int = 10

    # Whether to compute per-class metrics
    compute_per_class: bool = True


def masks_to_rle(masks: torch.Tensor) -> List[Dict]:
    """
    Convert dense masks to RLE format using faster_coco_eval.

    Args:
        masks: [N, H, W] uint8 binary masks

    Returns:
        List of RLE dicts with 'counts' and 'size' keys
    """
    if masks.numel() == 0:
        return []

    rles = []
    masks_np = masks.cpu().numpy().astype(np.uint8)

    for mask in masks_np:
        # Ensure Fortran order for RLE encoding
        mask_f = np.asfortranarray(mask)
        rle = mask_utils.encode(mask_f)
        # Convert bytes to string for JSON serialization if needed
        if isinstance(rle["counts"], bytes):
            rle["counts"] = rle["counts"].decode("utf-8")
        rles.append(rle)

    return rles


def rle_to_masks(rles: List[Dict], device: str = "cpu") -> torch.Tensor:
    """
    Convert RLE-encoded masks back to dense format.

    Args:
        rles: List of RLE dicts from masks_to_rle()
        device: Target device for output tensor

    Returns:
        [N, H, W] uint8 tensor of masks
    """
    if not rles:
        return torch.zeros((0, 1, 1), dtype=torch.uint8, device=device)

    masks = []
    for rle in rles:
        # Handle both bytes and string counts
        if isinstance(rle["counts"], str):
            rle = {"counts": rle["counts"].encode("utf-8"), "size": rle["size"]}
        mask = mask_utils.decode(rle)
        masks.append(mask)

    masks_np = np.stack(masks, axis=0)
    return torch.from_numpy(masks_np).to(device=device, dtype=torch.uint8)


class SingleImageLoader(Loader):
    """
    DataLoader that processes one image at a time for minimal memory footprint.
    """

    def build_dataloaders(self):
        val_ds = CustomDataset(
            self.img_size,
            self.root_path,
            self.splits["val"],
            self.debug_img_processing,
            mode="bench",
            cfg=self.cfg,
        )

        # Single image, no workers = minimal memory
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            num_workers=0,
            shuffle=False,
            collate_fn=self.val_collate_fn,
            worker_init_fn=seed_worker,
            prefetch_factor=None,
            pin_memory=False,
        )

        test_loader = None
        if len(self.splits["test"]):
            test_ds = CustomDataset(
                self.img_size,
                self.root_path,
                self.splits["test"],
                self.debug_img_processing,
                mode="bench",
                cfg=self.cfg,
            )
            test_loader = DataLoader(
                test_ds,
                batch_size=1,
                num_workers=0,
                shuffle=False,
                collate_fn=self.val_collate_fn,
                worker_init_fn=seed_worker,
                prefetch_factor=None,
                pin_memory=False,
            )

        return val_loader, test_loader


class COCOAPEvaluator:
    """
    Memory-efficient COCO-style AP evaluator.

    Stores predictions and ground truth as RLE-encoded masks,
    and batches updates to torchmetrics.
    """

    def __init__(
        self,
        n_classes: int,
        label_to_name: Dict[int, str],
        max_detections: int = 300,
        update_batch_size: int = 10,
    ):
        self.n_classes = n_classes
        self.label_to_name = label_to_name
        self.max_detections = max_detections
        self.update_batch_size = update_batch_size

        # Buffers for batched updates
        self.preds_buffer: List[Dict] = []
        self.gt_buffer: List[Dict] = []

        # Initialize torchmetrics with faster_coco_eval backend (numpy 2.x compatible)
        self.bbox_metric = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            sync_on_compute=False,
            backend="faster_coco_eval",
            class_metrics=True,  # For per-class mAP
        )
        self.bbox_metric.warn_on_many_detections = False

        self.mask_metric = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="segm",
            sync_on_compute=False,
            backend="faster_coco_eval",
            class_metrics=True,
        )
        self.mask_metric.warn_on_many_detections = False

        # Track if we have any masks (for detection-only vs segmentation)
        self.has_masks = False

        # Stats
        self.n_images = 0
        self.n_preds = 0
        self.n_gts = 0

    def add_sample(
        self,
        pred_boxes: torch.Tensor,  # [N, 4] xyxy absolute
        pred_scores: torch.Tensor,  # [N]
        pred_labels: torch.Tensor,  # [N]
        pred_masks: Optional[torch.Tensor],  # [N, H, W] float probabilities or None
        gt_boxes: torch.Tensor,  # [M, 4] xyxy absolute
        gt_labels: torch.Tensor,  # [M]
        gt_masks: Optional[torch.Tensor],  # [M, H, W] uint8 or None
        mask_binarize_thresh: float = 0.5,
    ):
        """
        Add one image's predictions and ground truth.
        Masks are converted to RLE immediately to save memory.
        """
        # Limit to max detections (keep highest scoring)
        if pred_scores.numel() > self.max_detections:
            topk = pred_scores.argsort(descending=True)[: self.max_detections]
            pred_boxes = pred_boxes[topk]
            pred_scores = pred_scores[topk]
            pred_labels = pred_labels[topk]
            if pred_masks is not None and pred_masks.numel() > 0:
                pred_masks = pred_masks[topk]

        # Binarize prediction masks
        has_pred_masks = pred_masks is not None and pred_masks.numel() > 0
        if has_pred_masks:
            if pred_masks.dtype != torch.uint8:
                pred_masks = (pred_masks > mask_binarize_thresh).to(torch.uint8)
            pred_masks_rle = masks_to_rle(pred_masks.cpu())
            pred_mask_size = tuple(pred_masks.shape[-2:])
        else:
            pred_masks_rle = []
            pred_mask_size = (0, 0)

        # Binarize GT masks
        has_gt_masks = gt_masks is not None and gt_masks.numel() > 0
        if has_gt_masks:
            if gt_masks.dtype != torch.uint8:
                gt_masks = (gt_masks > 0.5).to(torch.uint8)
            gt_masks_rle = masks_to_rle(gt_masks.cpu())
            gt_mask_size = tuple(gt_masks.shape[-2:])
        else:
            gt_masks_rle = []
            gt_mask_size = (0, 0)

        # Track if we have any masks in the dataset
        if has_pred_masks or has_gt_masks:
            self.has_masks = True

        # Store with RLE encoding
        self.preds_buffer.append(
            {
                "boxes": pred_boxes.cpu().to(torch.float32),
                "scores": pred_scores.cpu().to(torch.float32),
                "labels": pred_labels.cpu().to(torch.int64),
                "masks_rle": pred_masks_rle,
                "masks_size": pred_mask_size,
            }
        )

        self.gt_buffer.append(
            {
                "boxes": gt_boxes.cpu().to(torch.float32),
                "labels": gt_labels.cpu().to(torch.int64),
                "masks_rle": gt_masks_rle,
                "masks_size": gt_mask_size,
            }
        )

        self.n_images += 1
        self.n_preds += pred_boxes.shape[0]
        self.n_gts += gt_boxes.shape[0]

        # Flush when buffer is full
        if len(self.preds_buffer) >= self.update_batch_size:
            self._flush_buffer()

    def _decode_rle_to_dense_bbox(self, samples: List[Dict]) -> List[Dict]:
        """Convert samples to format for bbox metric (no masks needed)."""
        decoded = []
        for s in samples:
            d = {
                "boxes": s["boxes"],
                "labels": s["labels"],
            }
            if "scores" in s:
                d["scores"] = s["scores"]
            decoded.append(d)
        return decoded

    def _decode_rle_to_dense_mask(self, samples: List[Dict]) -> List[Dict]:
        """Convert RLE-encoded masks back to dense for mask metric."""
        decoded = []
        for s in samples:
            d = {
                "boxes": s["boxes"],
                "labels": s["labels"],
            }
            if "scores" in s:
                d["scores"] = s["scores"]

            n_items = len(s["labels"])
            if s.get("masks_rle") and len(s["masks_rle"]) == n_items:
                d["masks"] = rle_to_masks(s["masks_rle"], device="cpu")
            else:
                # Create placeholder masks matching the number of labels
                size = s.get("masks_size", (1, 1))
                if size[0] == 0:
                    size = (64, 64)  # Minimum size for empty masks
                d["masks"] = torch.zeros((n_items, size[0], size[1]), dtype=torch.uint8)

            decoded.append(d)
        return decoded

    def _flush_buffer(self):
        """Decode RLE, update torchmetrics, clear buffer."""
        if not self.preds_buffer:
            return

        # Always update bbox metric
        preds_bbox = self._decode_rle_to_dense_bbox(self.preds_buffer)
        gt_bbox = self._decode_rle_to_dense_bbox(self.gt_buffer)
        self.bbox_metric.update(preds_bbox, gt_bbox)
        del preds_bbox, gt_bbox

        # Only update mask metric if we have masks
        if self.has_masks:
            preds_mask = self._decode_rle_to_dense_mask(self.preds_buffer)
            gt_mask = self._decode_rle_to_dense_mask(self.gt_buffer)
            self.mask_metric.update(preds_mask, gt_mask)
            del preds_mask, gt_mask

        # Clear everything
        self.preds_buffer.clear()
        self.gt_buffer.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def compute(self) -> Dict[str, float]:
        """Compute all metrics and return results dict."""
        # Flush remaining samples
        self._flush_buffer()

        # Compute bbox metrics
        bbox_results = self.bbox_metric.compute()

        # Compute mask metrics only if we have masks
        if self.has_masks:
            mask_results = self.mask_metric.compute()
        else:
            mask_results = {}

        metrics = {
            # Detection metrics (COCO-style)
            "bbox_AP": bbox_results["map"].item(),  # AP@[.50:.95]
            "bbox_AP50": bbox_results["map_50"].item(),
            "bbox_AP75": bbox_results["map_75"].item(),
            "bbox_AR_1": bbox_results.get("mar_1", torch.tensor(0.0)).item(),  # AR maxDet=1
            "bbox_AR_10": bbox_results.get("mar_10", torch.tensor(0.0)).item(),  # AR maxDet=10
            "bbox_AR_100": bbox_results.get("mar_100", torch.tensor(0.0)).item(),  # AR maxDet=100
            "bbox_AR_large": bbox_results.get("mar_large", torch.tensor(0.0)).item(),
            "bbox_AR_medium": bbox_results.get("mar_medium", torch.tensor(0.0)).item(),
            "bbox_AR_small": bbox_results.get("mar_small", torch.tensor(0.0)).item(),
            # Size-based AP
            "bbox_AP_small": bbox_results.get("map_small", torch.tensor(0.0)).item(),
            "bbox_AP_medium": bbox_results.get("map_medium", torch.tensor(0.0)).item(),
            "bbox_AP_large": bbox_results.get("map_large", torch.tensor(0.0)).item(),
            # Mask metrics (Instance segmentation) - only if masks available
            "mask_AP": mask_results.get("map", torch.tensor(-1.0)).item() if mask_results else -1.0,
            "mask_AP50": mask_results.get("map_50", torch.tensor(-1.0)).item()
            if mask_results
            else -1.0,
            "mask_AP75": mask_results.get("map_75", torch.tensor(-1.0)).item()
            if mask_results
            else -1.0,
            "mask_AR_100": mask_results.get("mar_100", torch.tensor(-1.0)).item()
            if mask_results
            else -1.0,
            "mask_AP_small": mask_results.get("map_small", torch.tensor(-1.0)).item()
            if mask_results
            else -1.0,
            "mask_AP_medium": mask_results.get("map_medium", torch.tensor(-1.0)).item()
            if mask_results
            else -1.0,
            "mask_AP_large": mask_results.get("map_large", torch.tensor(-1.0)).item()
            if mask_results
            else -1.0,
            # Flag for whether masks were evaluated
            "has_masks": self.has_masks,
            # Stats
            "n_images": self.n_images,
            "n_predictions": self.n_preds,
            "n_ground_truths": self.n_gts,
        }

        # Per-class metrics if available
        if "map_per_class" in bbox_results and bbox_results["map_per_class"].numel() > 0:
            per_class_map = bbox_results["map_per_class"]
            for i, class_ap in enumerate(per_class_map):
                if i < len(self.label_to_name):
                    name = self.label_to_name.get(i, str(i))
                    metrics[f"bbox_AP_{name}"] = class_ap.item()

        if (
            mask_results
            and "map_per_class" in mask_results
            and mask_results["map_per_class"].numel() > 0
        ):
            per_class_map = mask_results["map_per_class"]
            for i, class_ap in enumerate(per_class_map):
                if i < len(self.label_to_name):
                    name = self.label_to_name.get(i, str(i))
                    metrics[f"mask_AP_{name}"] = class_ap.item()

        # Reset to free memory
        self.bbox_metric.reset()
        if self.has_masks:
            self.mask_metric.reset()
        gc.collect()

        return metrics

    def reset(self):
        """Reset all state."""
        self.preds_buffer.clear()
        self.gt_buffer.clear()
        self.bbox_metric.reset()
        self.mask_metric.reset()
        self.n_images = 0
        self.n_preds = 0
        self.n_gts = 0
        gc.collect()


def clear_gpu_memory():
    """Aggressively clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def warmup_model(model, data_path: Path, img_paths, n_warmup: int = 3):
    """Run warmup iterations on the model (reduced count for memory)."""
    warmup_img = cv2.imread(str(data_path / "images" / img_paths[0]))
    for _ in range(n_warmup):
        _ = model(warmup_img)
        clear_gpu_memory()  # Clear after each warmup
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def evaluate_single_model(
    model,
    model_name: str,
    data_loader: DataLoader,
    data_path: Path,
    processed_size: Tuple[int, int],
    keep_ratio: bool,
    device: str,
    label_to_name: Dict[int, str],
    eval_config: EvalConfig,
) -> Dict[str, float]:
    """
    Evaluate a model and return COCO-style AP metrics.

    Args:
        model: Inference model (TRT_model or YOLO_TRT_model)
        model_name: Name for logging
        data_loader: DataLoader yielding (_, targets, img_paths)
        data_path: Path to dataset root
        processed_size: (H, W) of model input
        keep_ratio: Whether letterboxing is used
        device: Device string
        label_to_name: Class ID to name mapping
        eval_config: Evaluation configuration

    Returns:
        Dict with all COCO-style metrics
    """
    logger.info(f"Evaluating {model_name}...")
    logger.info(f"  Max detections per image: {eval_config.max_detections_per_image}")
    logger.info(f"  Update batch size: {eval_config.update_batch_size}")

    evaluator = COCOAPEvaluator(
        n_classes=len(label_to_name),
        label_to_name=label_to_name,
        max_detections=eval_config.max_detections_per_image,
        update_batch_size=eval_config.update_batch_size,
    )

    batch_idx = 0  # We process single images, so always index 0

    # Collect image paths for warmup
    first_batch = next(iter(data_loader))
    warmup_model(model, data_path, first_batch[2])

    for _, targets_batch, img_paths in tqdm(data_loader, desc=f"{model_name}"):
        for img_path, targets in zip(img_paths, targets_batch):
            # Load image
            img = cv2.imread(str(data_path / "images" / img_path))
            if img is None:
                logger.warning(f"Could not load image: {img_path}")
                continue

            orig_h, orig_w = img.shape[:2]

            # Prepare ground truth
            gt_boxes = process_boxes(
                targets["boxes"][None],
                processed_size,
                targets["orig_size"][None],
                keep_ratio,
                device,
            )[batch_idx].cpu()

            gt_labels = targets["labels"].cpu()

            if "masks" in targets and targets["masks"].numel() > 0:
                gt_masks = process_masks(
                    targets["masks"][None],
                    processed_size,
                    targets["orig_size"][None],
                    keep_ratio,
                )[batch_idx].cpu()
            else:
                gt_masks = None

            # Run inference
            preds = model(img)

            # Extract predictions
            pred = preds[0]
            pred_boxes = pred["boxes"]
            pred_scores = pred["scores"]
            pred_labels = pred["labels"]

            # Get masks if available
            if "masks" in pred and pred["masks"].numel() > 0:
                pred_masks = pred["masks"]
            else:
                pred_masks = None

            # Add to evaluator
            evaluator.add_sample(
                pred_boxes=pred_boxes,
                pred_scores=pred_scores,
                pred_labels=pred_labels,
                pred_masks=pred_masks,
                gt_boxes=gt_boxes,
                gt_labels=gt_labels,
                gt_masks=gt_masks,
            )

            # Cleanup
            del pred, preds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Compute final metrics
    metrics = evaluator.compute()

    return metrics


def format_results(metrics: Dict[str, float], model_name: str) -> str:
    """Format metrics for display."""
    has_masks = metrics.get("has_masks", False)

    lines = [
        f"\n{'=' * 60}",
        f"Results for {model_name}",
        f"{'=' * 60}",
        "",
        "Detection Metrics (COCO-style):",
        f"  AP@[.50:.95]:     {metrics.get('bbox_AP', 0) * 100:.2f}%",
        f"  AP@.50:           {metrics.get('bbox_AP50', 0) * 100:.2f}%",
        f"  AP@.75:           {metrics.get('bbox_AP75', 0) * 100:.2f}%",
        f"  AR (maxDet=100):  {metrics.get('bbox_AR_100', 0) * 100:.2f}%",
        f"  AP (small):       {metrics.get('bbox_AP_small', 0) * 100:.2f}%",
        f"  AP (medium):      {metrics.get('bbox_AP_medium', 0) * 100:.2f}%",
        f"  AP (large):       {metrics.get('bbox_AP_large', 0) * 100:.2f}%",
        "",
    ]

    if has_masks:
        lines.extend(
            [
                "Instance Segmentation Metrics (COCO-style):",
                f"  mask AP@[.50:.95]: {metrics.get('mask_AP', 0) * 100:.2f}%",
                f"  mask AP@.50:       {metrics.get('mask_AP50', 0) * 100:.2f}%",
                f"  mask AP@.75:       {metrics.get('mask_AP75', 0) * 100:.2f}%",
                f"  mask AR (maxDet=100): {metrics.get('mask_AR_100', 0) * 100:.2f}%",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Instance Segmentation Metrics: N/A (detection-only model)",
                "",
            ]
        )

    lines.extend(
        [
            "Statistics:",
            f"  Images:          {metrics.get('n_images', 0)}",
            f"  Predictions:     {metrics.get('n_predictions', 0)}",
            f"  Ground Truths:   {metrics.get('n_ground_truths', 0)}",
            f"{'=' * 60}",
        ]
    )
    return "\n".join(lines)


def create_summary_table(all_metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Create a summary table of key metrics for all models."""
    summary_data = []
    for model_name, metrics in all_metrics.items():
        has_masks = metrics.get("has_masks", False)
        row = {
            "Model": model_name,
            "bbox AP": f"{metrics.get('bbox_AP', 0) * 100:.2f}",
            "bbox AP50": f"{metrics.get('bbox_AP50', 0) * 100:.2f}",
            "bbox AR": f"{metrics.get('bbox_AR_100', 0) * 100:.2f}",
        }
        if has_masks:
            row["mask AP"] = f"{metrics.get('mask_AP', 0) * 100:.2f}"
            row["mask AP50"] = f"{metrics.get('mask_AP50', 0) * 100:.2f}"
        else:
            row["mask AP"] = "N/A"
            row["mask AP50"] = "N/A"
        summary_data.append(row)
    return pd.DataFrame(summary_data)


@hydra.main(version_base=None, config_path="../", config_name="config")
def main(cfg: DictConfig):
    """Main evaluation entry point."""
    work_path = Path(cfg.train.root)
    # Evaluation configuration
    # Note: 0.01 threshold is low enough for accurate mAP while avoiding OOM
    eval_config = EvalConfig(
        conf_thresh_dfine=0.01,  # Low enough for mAP, high enough to avoid OOM
        conf_thresh_yolo=0.01,
        max_detections_per_image=100,  # COCO standard
        update_batch_size=5,  # Small batches for memory efficiency
    )

    # Clear GPU memory before starting
    clear_gpu_memory()

    # Get latest experiment
    cfg.exp = get_latest_experiment_name(cfg.exp, cfg.train.path_to_save)
    logger.info(f"Experiment: {cfg.exp}")
    logger.info(f"Task: {cfg.task}, Model size: {cfg.model_name}")

    # Define model configs (we'll load one at a time to save memory)
    model_configs = []

    # D-FINE-seg TensorRT model
    dfine_path = Path(cfg.train.path_to_save) / "model.engine"
    if dfine_path.exists():
        model_configs.append(
            {
                "name": "D-FINE-seg",
                "type": "dfine",
                "path": dfine_path,
                "conf_thresh": eval_config.conf_thresh_dfine,
            }
        )
    else:
        logger.warning(f"D-FINE-seg model not found at: {dfine_path}")

    # YOLO TensorRT model
    yolo_base_path = work_path / "runs" / cfg.task / cfg.model_name / "weights"
    yolo_path = yolo_base_path / "best.engine"
    if yolo_path.exists():
        model_configs.append(
            {
                "name": "YOLO26",
                "type": "yolo",
                "path": yolo_path,
                "conf_thresh": eval_config.conf_thresh_yolo,
            }
        )
    else:
        logger.warning(f"YOLO model not found at: {yolo_path}")

    if not model_configs:
        logger.error("No models found! Check paths in config.")
        return

    # Create data loader
    data_path = Path(cfg.train.data_path)
    logger.info(f"Dataset path: {data_path}")

    val_loader, test_loader = SingleImageLoader(
        root_path=data_path,
        img_size=tuple(cfg.train.img_size),
        batch_size=1,
        num_workers=0,
        cfg=cfg,
        debug_img_processing=False,
    ).build_dataloaders()

    # Evaluate each model one at a time (load, evaluate, unload)
    all_metrics = {}

    for model_cfg in model_configs:
        model_name = model_cfg["name"]
        model = None

        # Clear GPU before loading each model
        clear_gpu_memory()
        logger.info(f"GPU memory cleared before loading {model_name}")

        try:
            # Load model just-in-time
            logger.info(f"Loading {model_name} from: {model_cfg['path']}")
            if model_cfg["type"] == "dfine":
                model = TRT_model(
                    model_path=model_cfg["path"],
                    n_outputs=len(cfg.train.label_to_name),
                    input_width=cfg.train.img_size[1],
                    input_height=cfg.train.img_size[0],
                    conf_thresh=model_cfg["conf_thresh"],
                    rect=False,
                    half=cfg.export.half,
                    keep_ratio=cfg.train.keep_ratio,
                )
            elif model_cfg["type"] == "yolo":
                model = YOLO_TRT_model(
                    model_path=str(model_cfg["path"]),
                    conf_thresh=model_cfg["conf_thresh"],
                    imgsz=cfg.train.img_size[0],
                    half=True,
                )

            # Evaluate
            metrics = evaluate_single_model(
                model=model,
                model_name=model_name,
                data_loader=val_loader,
                data_path=data_path,
                processed_size=tuple(cfg.train.img_size),
                keep_ratio=cfg.train.keep_ratio,
                device=cfg.train.device,
                label_to_name=dict(cfg.train.label_to_name),
                eval_config=eval_config,
            )
            all_metrics[model_name] = metrics

            # Print detailed results
            print(format_results(metrics, model_name))

        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            import traceback

            traceback.print_exc()

        finally:
            # Always cleanup: delete model and clear GPU
            if model is not None:
                del model
            clear_gpu_memory()
            logger.info(f"Cleaned up after {model_name}")

    # Print summary table
    if all_metrics:
        summary_df = create_summary_table(all_metrics)
        print("\n" + "=" * 80)
        print("SUMMARY TABLE (for paper)")
        print("=" * 80)
        print(tabulate(summary_df, headers="keys", tablefmt="pretty", showindex=False))

        # Save results
        output_path = Path(cfg.train.path_to_save) / "coco_ap_results.csv"
        full_results_df = pd.DataFrame.from_dict(all_metrics, orient="index")
        full_results_df.to_csv(output_path)
        logger.info(f"Full results saved to: {output_path}")

        # Print LaTeX-friendly format
        print("\n" + "=" * 80)
        print("LaTeX Table Format:")
        print("=" * 80)
        for model_name, metrics in all_metrics.items():
            print(
                f"{model_name} & "
                f"{metrics.get('bbox_AP', 0) * 100:.1f} & "
                f"{metrics.get('bbox_AP50', 0) * 100:.1f} & "
                f"{metrics.get('bbox_AR_100', 0) * 100:.1f} & "
                f"{metrics.get('mask_AP', 0) * 100:.1f} & "
                f"{metrics.get('mask_AP50', 0) * 100:.1f} \\\\"
            )


if __name__ == "__main__":
    main()
