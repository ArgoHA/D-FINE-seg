import gc
import json
import logging
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
import torch
import wandb
from albumentations.core.transforms_interface import DualTransform
from faster_coco_eval.core import mask as mask_utils
from loguru import logger
from omegaconf import OmegaConf
from tabulate import tabulate

from dfine_seg.viz import overlay_sem_seg, sem_seg_palette

logging.getLogger("faster_coco_eval").setLevel(logging.WARNING)


def set_seeds(seed: int, cudnn_fixed: bool = False) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if cudnn_fixed:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["PYTHONHASHSEED"] = str(seed)


def seed_worker(worker_id):  # noqa
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def wandb_logger(loss, metrics: Dict[str, float], epoch, mode: str) -> None:
    log_data = {"epoch": epoch}
    if loss:
        log_data[f"{mode}/loss/"] = loss

    for metric_name, metric_value in metrics.items():
        if metric_name == "extended_metrics":
            for ext_metric_name, ext_metric_value in metric_value.items():
                log_data[f"{mode}_extended/{ext_metric_name}"] = ext_metric_value
        else:
            log_data[f"{mode}/metrics/{metric_name}"] = metric_value

    wandb.log(log_data)


def log_metrics_locally(
    all_metrics: Dict[str, Dict[str, float]], path_to_save: Path, epoch: int, extended=False
) -> None:
    metrics_df = pd.DataFrame.from_dict(all_metrics, orient="index")
    metrics_df = metrics_df.round(4)
    if extended:
        # keep only rows that actually carry extended metrics (e.g. skip an empty test row)
        ext_rows = metrics_df["extended_metrics"].dropna()
        extended_metrics = pd.DataFrame.from_records(ext_rows.tolist(), index=ext_rows.index).round(
            4
        )

    if "mIoU" in metrics_df.columns:  # sem_seg
        metrics_list = ["mIoU", "pixel_acc"]
    else:
        metrics_list = [
            "mAP_50",
            "f1",
            "precision",
            "recall",
            "iou",
            "mAP_50_95",
            "TPs",
            "FPs",
            "FNs",
        ]
        if "mAP_50_mask" in metrics_df.columns:
            metrics_list.insert(1, "mAP_50_mask")
            metrics_list.remove("mAP_50_95")
    metrics_df = metrics_df[metrics_list]

    tabulated_data = tabulate(metrics_df, headers="keys", tablefmt="pretty", showindex=True)
    if epoch:
        logger.info(f"Metrics on epoch {epoch}:\n{tabulated_data}\n")
    else:
        logger.info(f"Best epoch metrics:\n{tabulated_data}\n")

    if path_to_save:
        metrics_df.to_csv(path_to_save / "metrics.csv")

        if extended:
            extended_metrics.to_csv(path_to_save / "extended_metrics.csv")


def save_metrics(train_metrics, metrics, loss, epoch, path_to_save, use_wandb) -> None:
    log_metrics_locally(
        all_metrics={"train": train_metrics, "val": metrics}, path_to_save=path_to_save, epoch=epoch
    )
    if use_wandb:
        wandb_logger(loss, train_metrics, epoch, mode="train")
        wandb_logger(None, metrics, epoch, mode="val")


def calculate_remaining_time(
    one_epoch_time, epoch_start_time, epoch, epochs, cur_iter, all_iters
) -> str:
    if one_epoch_time is None:
        average_iter_time = (time.time() - epoch_start_time) / cur_iter
        remaining_iters = epochs * all_iters - cur_iter

        hours, remainder = divmod(average_iter_time * remaining_iters, 3600)
        minutes, _ = divmod(remainder, 60)
        return f"{int(hours):02}:{int(minutes):02}"

    time_for_remaining_epochs = max(one_epoch_time * (epochs + 1 - epoch), 0)
    current_epoch_progress = time.time() - epoch_start_time
    hours, remainder = divmod(time_for_remaining_epochs - current_epoch_progress, 3600)
    minutes, _ = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}"


def get_vram_usage():
    if not torch.cuda.is_available():
        return 0
    free, total = torch.cuda.mem_get_info()
    return round(100 * (total - free) / total)


def norm_xywh_to_abs_xyxy(boxes: np.ndarray, height: int, width: int) -> np.ndarray:
    # Convert normalized centers to absolute pixel coordinates
    x_center = boxes[:, 0] * width
    y_center = boxes[:, 1] * height
    box_width = boxes[:, 2] * width
    box_height = boxes[:, 3] * height

    # Compute the top-left and bottom-right coordinates
    x_min = x_center - (box_width / 2)
    y_min = y_center - (box_height / 2)
    x_max = x_center + (box_width / 2)
    y_max = y_center + (box_height / 2)

    x_min = np.clip(x_min, 0, width)
    y_min = np.clip(y_min, 0, height)
    x_max = np.clip(x_max, 0, width)
    y_max = np.clip(y_max, 0, height)
    return np.stack([x_min, y_min, x_max, y_max], axis=1)


def abs_xyxy_to_norm_xywh(boxes: np.ndarray, height: int, width: int) -> np.ndarray:
    x_center = (boxes[:, 0] + boxes[:, 2]) / 2 / width
    y_center = (boxes[:, 1] + boxes[:, 3]) / 2 / height
    box_width = (boxes[:, 2] - boxes[:, 0]) / width
    box_height = (boxes[:, 3] - boxes[:, 1]) / height
    return np.stack([x_center, y_center, box_width, box_height], axis=1)


def get_aug_params(value, center=0):
    if isinstance(value, float):
        return random.uniform(center - value, center + value)
    elif len(value) == 2:
        return random.uniform(value[0], value[1])
    else:
        raise ValueError(
            "Affine params should be either a sequence containing two values\
                          or single float values. Got {}".format(value)
        )


def clip_polygon_to_rect(poly: np.ndarray, width: float, height: float) -> np.ndarray:
    """
    Clip a polygon to a rectangle [0, width] x [0, height] using Sutherland-Hodgman algorithm.
    Returns the clipped polygon as (M, 2) array, or empty (0, 2) if fully outside.
    """
    if poly.size == 0:
        return np.empty((0, 2), dtype=np.float32)

    def inside(p, edge):
        x, y = p
        if edge == "left":
            return x >= 0
        elif edge == "right":
            return x <= width
        elif edge == "top":
            return y >= 0
        elif edge == "bottom":
            return y <= height

    def intersection(p1, p2, edge):
        x1, y1 = p1
        x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1
        if edge == "left":
            t = (0 - x1) / dx if dx != 0 else 0
            return np.array([0, y1 + t * dy])
        elif edge == "right":
            t = (width - x1) / dx if dx != 0 else 0
            return np.array([width, y1 + t * dy])
        elif edge == "top":
            t = (0 - y1) / dy if dy != 0 else 0
            return np.array([x1 + t * dx, 0])
        elif edge == "bottom":
            t = (height - y1) / dy if dy != 0 else 0
            return np.array([x1 + t * dx, height])

    output = poly.copy()
    for edge in ["left", "right", "top", "bottom"]:
        if len(output) == 0:
            return np.empty((0, 2), dtype=np.float32)
        input_list = output
        output = []
        for i in range(len(input_list)):
            current = input_list[i]
            prev = input_list[i - 1]
            if inside(current, edge):
                if not inside(prev, edge):
                    output.append(intersection(prev, current, edge))
                output.append(current)
            elif inside(prev, edge):
                output.append(intersection(prev, current, edge))
        output = np.array(output) if len(output) > 0 else np.empty((0, 2), dtype=np.float32)

    if len(output) < 3:
        return np.empty((0, 2), dtype=np.float32)
    return output.astype(np.float32)


def box_candidates(
    box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16
):  # box1(4,n), box2(4,n)
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (
        (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)
    )  # candidates


def get_transform_matrix(img_shape, new_shape, degrees, scale, shear, translate):
    new_width, new_height = new_shape
    # Center
    C = np.eye(3)
    C[0, 2] = -img_shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img_shape[0] / 2  # y translation (pixels)
    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = get_aug_params(scale, center=1.0)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * new_width  # x translation (pixels)
    T[1, 2] = (
        random.uniform(0.5 - translate, 0.5 + translate) * new_height
    )  # y transla ion (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ C  # order of operations (right to left) is IMPORTANT
    return M, s


def random_affine(img, targets, segments, target_size, degrees, translate, scales, shear):
    """
    Args:
      img: (Hbig, Wbig, 3)
      targets: (N, 5) -> [cls, x1, y1, x2, y2] ABS on the mosaic canvas
      segments: list of length N; entry i is that object's list of (K,2) ABS polygon parts
                (islands) on the mosaic canvas - empty list for a bbox-only annotation.
    Returns:
      img_aff: final (target_h, target_w, 3)
      targets_aff: (M, 5) filtered + transformed
      segments_aff: list length=M of transformed part lists
    """
    M, scale = get_transform_matrix(img.shape[:2], target_size, degrees, scales, shear, translate)

    # warp image - borderValue must match channel count (cv2 Scalar capped at 4)
    if (M != np.eye(3)).any():
        border = tuple([114] * img.shape[2]) if img.ndim == 3 else 114
        img = cv2.warpAffine(img, M[:2], dsize=target_size, borderValue=border)

    n = len(targets)
    if n:
        # transform boxes by corners
        xy = np.ones((n * 4, 3), dtype=np.float32)
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
        xy = xy @ M.T
        xy = xy[:, :2].reshape(n, 8)

        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = np.stack([x.min(1), y.min(1), x.max(1), y.max(1)], axis=1)

        # clip boxes into target frame
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, target_size[0])
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, target_size[1])

        # transform segments (if provided)
        segs_out = []
        # False only for rows that had parts and lost every one to clipping
        alive = np.ones(n, dtype=bool)
        if segments is None or len(segments) == 0:
            segs_out = [[] for _ in range(n)]
        else:
            # keep 1:1 with targets
            for idx, parts in enumerate(segments):
                if not len(parts):
                    segs_out.append([])  # detection-only annotation
                    continue
                kept = []
                for s in parts:
                    pts = np.concatenate([s, np.ones((len(s), 1), dtype=np.float32)], axis=1)
                    pts = (pts @ M.T)[:, :2]
                    # Properly clip polygon to the target frame
                    clipped = clip_polygon_to_rect(pts, target_size[0], target_size[1])
                    if clipped.size >= 6:  # At least 3 points for a valid polygon
                        kept.append(clipped)
                segs_out.append(kept)
                if kept:
                    # Update bounding box from the union of the clipped parts
                    all_pts = np.concatenate(kept, axis=0)
                    new[idx] = [*all_pts.min(axis=0), *all_pts.max(axis=0)]
                else:
                    alive[idx] = False  # else an all-zero mask would supervise a live box

        # filter candidates and keep segments in sync
        i = box_candidates(box1=targets[:, 1:5].T * scale, box2=new.T, area_thr=0.1) & alive
        targets = targets[i]
        targets[:, 1:5] = new[i]
        segs_out = [segs_out[k] for k, keep in enumerate(i) if keep]

    else:
        segs_out = []

    return img, targets, segs_out


def get_mosaic_coordinate(mosaic_image, mosaic_index, xc, yc, w, h, target_h, target_w):
    # TODO update doc
    # index0 to top left part of image
    if mosaic_index == 0:
        x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
        small_coord = w - (x2 - x1), h - (y2 - y1), w, h
    # index1 to top right part of image
    elif mosaic_index == 1:
        x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, target_w * 2), yc
        small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
    # index2 to bottom left part of image
    elif mosaic_index == 2:
        x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(target_h * 2, yc + h)
        small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
    # index2 to bottom right part of image
    elif mosaic_index == 3:
        x1, y1, x2, y2 = xc, yc, min(xc + w, target_w * 2), min(target_h * 2, yc + h)  # noqa
        small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
    return (x1, y1, x2, y2), small_coord


def filter_preds(preds, conf_thresh, mask_source="mask_probs"):
    """
    Filters predictions by score AND keeps masks in-sync with the kept indices.
    - If mask_source == "mask_probs" and present, also populates pred["masks"] as uint8 via conf_thresh.
    """
    for pred in preds:
        keep = pred["scores"] >= conf_thresh
        pred["scores"] = pred["scores"][keep]
        pred["boxes"] = pred["boxes"][keep]
        pred["labels"] = pred["labels"][keep]

        # Keep mask tensors aligned with kept queries
        if (
            mask_source in pred
            and pred[mask_source] is not None
            and getattr(pred[mask_source], "numel", lambda: 0)() > 0
        ):
            m = pred[mask_source][keep]
            pred[mask_source] = m
            # Ensure binary mask view exists (uint8)
            if mask_source == "mask_probs":
                pred["masks"] = (m > conf_thresh).to(torch.uint8)
        elif (
            "masks" in pred
            and pred["masks"] is not None
            and getattr(pred["masks"], "numel", lambda: 0)() > 0
        ):
            pred["masks"] = pred["masks"][keep].to(torch.uint8)

    return preds


def label_color(label: int):
    # deterministic color per class (BGR for OpenCV)
    palette = [
        (255, 56, 56),
        (255, 159, 56),
        (255, 255, 56),
        (56, 255, 56),
        (56, 255, 255),
        (56, 56, 255),
        (255, 56, 255),
        (180, 130, 70),
        (204, 153, 255),
        (80, 175, 76),
        (42, 157, 143),
        (233, 196, 106),
        (244, 162, 97),
        (231, 111, 81),
        (69, 123, 157),
        (29, 53, 87),
    ]
    return palette[int(label) % len(palette)]


def draw_mask(
    img: np.ndarray, mask: np.ndarray, color=(148, 70, 44), alpha: float = 0.4, outline: bool = True
):
    """
    img: BGR uint8 [H,W,3]
    mask: uint8/bool [H,W] (1=mask)
    color: (B,G,R)
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    if mask.ndim == 3:  # [1,H,W] -> [H,W]
        mask = mask.squeeze(0)

    if mask.max() == 0:
        return img

    # fast alpha blend on masked pixels
    m = mask.astype(bool)
    overlay = np.zeros_like(img, dtype=np.uint8)
    overlay[:] = color
    img[m] = cv2.addWeighted(img[m], 1 - alpha, overlay[m], alpha, 0)

    if outline:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, color, 2)
    return img


def visualize_sem_seg(
    img_path,
    gt_map: np.ndarray,
    pred_map: np.ndarray,
    dataset_path: Path,
    path_to_save: Path,
    n_classes: int,
    ignore_index: int = 255,
    max_side: int = 1280,
) -> None:
    """Save a GT | pred side-by-side overlay, downscaled to max_side."""
    from dfine_seg.dl.dataset import read_image_hwc  # local to avoid circular import

    img = read_image_hwc(dataset_path / img_path)
    if img is None:
        return
    if img.shape[2] > 3:
        img = img[..., :3]
    if Path(img_path).suffix.lower() == ".npy":
        img = np.ascontiguousarray(img[..., ::-1])

    scale = max_side / max(img.shape[:2])
    if scale < 1:
        new_wh = (round(img.shape[1] * scale), round(img.shape[0] * scale))
        img = cv2.resize(img, new_wh, interpolation=cv2.INTER_AREA)

    palette = sem_seg_palette(n_classes)
    panes = []
    for name, label_map in (("GT", gt_map), ("pred", pred_map)):
        pane = overlay_sem_seg(img, label_map, palette, ignore_index=ignore_index)
        cv2.putText(
            pane, name, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA
        )
        panes.append(pane)

    path_to_save.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path_to_save / f"{Path(img_path).stem}.jpg"), cv2.hconcat(panes))


def vis_one_box(img, box, label, mode, label_to_name, score=None):
    if mode == "gt":
        prefix = "GT: "
        color = (46, 153, 60)
        postfix = ""
    elif mode == "pred":
        prefix = ""
        color = (148, 70, 44)
        postfix = f" {score:.2f}"

    x1, y1, x2, y2 = map(int, box.tolist())
    cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        color=color,
        thickness=2,
    )
    y = y1 - 16 if mode == "gt" else y1 - 4
    cv2.putText(
        img,
        f"{prefix}{label_to_name[int(label)]}{postfix}",
        (x1, max(0, y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        thickness=2,
    )


def visualize(
    img_paths,
    gt,
    preds,
    dataset_path,
    path_to_save,
    label_to_name,
    mask_alpha_gt: float = 0.35,
    mask_alpha_pred: float = 0.40,
):
    """
    Saves images with:
      - GT: green boxes + optional green masks
      - Preds: brown boxes + colored masks per class
    Expects pred dicts possibly containing "masks" (uint8)
    """
    from dfine_seg.dl.dataset import read_image_hwc  # local to avoid circular import

    path_to_save.mkdir(parents=True, exist_ok=True)

    draw_gt_masks = "masks" in gt[0]
    draw_pred_masks = "masks" in preds[0]

    for gt_dict, pred_dict, img_path in zip(gt, preds, img_paths):
        img = read_image_hwc(dataset_path / img_path)
        if img is None:
            continue
        # cv2 draws/writes in BGR; .npy stacks are RGB(+extras) by convention.
        if img.shape[2] > 3:
            img = img[..., :3]
        if Path(img_path).suffix.lower() == ".npy":
            img = np.ascontiguousarray(img[..., ::-1])

        # Draw GT masks (green-ish)
        if (
            draw_gt_masks
            and "masks" in gt_dict
            and gt_dict["masks"] is not None
            and len(gt_dict["masks"]) > 0
            and gt_dict["masks"].shape[1] != 0
        ):
            for m in gt_dict["masks"]:
                img = draw_mask(
                    img, m.numpy(), color=(46, 153, 60), alpha=mask_alpha_gt, outline=True
                )

        # Draw GT boxes (green)
        for box, label in zip(gt_dict["boxes"], gt_dict["labels"]):
            vis_one_box(img, box, label, mode="gt", label_to_name=label_to_name)

        # Prepare predicted masks
        pred_masks_to_draw = None
        if draw_pred_masks:
            if (
                "masks" in pred_dict
                and pred_dict["masks"] is not None
                and len(pred_dict["masks"]) > 0
            ):
                pred_masks_to_draw = pred_dict["masks"]
        # Draw predicted masks (colored by class)
        if pred_masks_to_draw is not None:
            pm = pred_masks_to_draw.cpu().numpy()
            for m, lab in zip(pm, pred_dict["labels"]):
                color = label_color(int(lab))
                img = draw_mask(img, m, color=color, alpha=mask_alpha_pred, outline=True)

        # Draw predicted boxes (blue-ish)
        for box, label, score in zip(pred_dict["boxes"], pred_dict["labels"], pred_dict["scores"]):
            vis_one_box(
                img,
                box,
                label,
                mode="pred",
                label_to_name=label_to_name,
                score=score,
            )

        # cv2.imwrite picks the encoder from the extension - .npy isn't an image format.
        outpath = path_to_save / f"{img_path.stem}.jpg"
        cv2.imwrite(str(outpath), img)


def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def scale_boxes_ratio_kept(boxes, img0_shape, img1_shape, ratio_pad=None, padding=True):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
            img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
        )  # gain  = old / new
        pad = (
            round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
            round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        boxes[..., [0, 2]] -= pad[0]  # x padding
        boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def scale_boxes(boxes, orig_shape, resized_shape):
    """
    boxes in format: [x1, y1, x2, y2], absolute values
    orig_shape: [height, width]
    resized_shape: [height, width]
    """
    scale_x = orig_shape[1] / resized_shape[1]
    scale_y = orig_shape[0] / resized_shape[0]
    boxes[:, 0] *= scale_x
    boxes[:, 2] *= scale_x
    boxes[:, 1] *= scale_y
    boxes[:, 3] *= scale_y
    return boxes


def process_boxes(boxes, processed_size, orig_sizes, keep_ratio, device):
    """
    Inputs:
        boxes: Torch.tensor[batch_size, num_boxes, 4]
        processed_size: Torch.tensor[2] h, w
        orig_sizes: Torch.tensor[batch_size, 2] h, w
        keep_ratio: bool
        device: Torch.device

    Outputs:
        Torch.tensor[batch_size, num_boxes, 4]

    """
    bs = orig_sizes.shape[0]
    processed_sizes = np.repeat(
        np.array([processed_size[0], processed_size[1]])[None, :], bs, axis=0
    )
    orig_sizes = orig_sizes.cpu().numpy()
    boxes = boxes.cpu().numpy()

    final_boxes = np.zeros_like(boxes)
    for idx, box in enumerate(boxes):
        final_boxes[idx] = norm_xywh_to_abs_xyxy(
            box, processed_sizes[idx][0], processed_sizes[idx][1]
        )

    for i in range(bs):
        if keep_ratio:
            final_boxes[i] = scale_boxes_ratio_kept(
                final_boxes[i],
                orig_sizes[i],
                processed_sizes[i],
            )
        else:
            final_boxes[i] = scale_boxes(
                final_boxes[i],
                orig_sizes[i],
                processed_sizes[i],
            )
    return torch.tensor(final_boxes).to(device)


def process_masks(
    pred_masks,  # Tensor [B, Q, Hm, Wm] or [Q, Hm, Wm]
    processed_size,  # (H, W) of network input (after your A.Compose)
    orig_sizes,  # Tensor [B, 2] (H, W)
    keep_ratio: bool,
) -> List[torch.Tensor]:
    """
    Returns list of length B with masks resized to original image sizes:
    Each item: Float Tensor [Q, H_orig, W_orig] in [0,1] (no thresholding here).
    - Handles letterbox padding removal if keep_ratio=True.
    - Works for both batched and single-image inputs.
    """
    single = pred_masks.dim() == 3  # [Q,Hm,Wm]
    if single:
        pred_masks = pred_masks.unsqueeze(0)  # -> [1,Q,Hm,Wm]

    if pred_masks.shape[1] == 0:
        return [torch.zeros((0, int(orig_sizes[0, 0]), int(orig_sizes[0, 1])))]

    B, Q, Hm, Wm = pred_masks.shape
    device = pred_masks.device
    dtype = pred_masks.dtype

    # 1) Upsample masks to processed (input) size
    proc_h, proc_w = int(processed_size[0]), int(processed_size[1])
    masks_proc = torch.nn.functional.interpolate(
        pred_masks, size=(proc_h, proc_w), mode="bilinear", align_corners=False
    )  # [B,Q,Hp,Wp] with Hp=proc_h, Wp=proc_w

    out = []
    for b in range(B):
        H0, W0 = int(orig_sizes[b, 0].item()), int(orig_sizes[b, 1].item())
        m = masks_proc[b]  # [Q, Hp, Wp]
        if keep_ratio:
            # Compute same gain/pad as in scale_boxes_ratio_kept
            gain = min(proc_h / H0, proc_w / W0)
            padw = round((proc_w - W0 * gain) / 2 - 0.1)
            padh = round((proc_h - H0 * gain) / 2 - 0.1)

            # Remove padding before final resize
            y1 = max(padh, 0)
            y2 = proc_h - max(padh, 0)
            x1 = max(padw, 0)
            x2 = proc_w - max(padw, 0)
            m = m[:, y1:y2, x1:x2]  # [Q, cropped_h, cropped_w]

        # 2) Resize to original size
        m = torch.nn.functional.interpolate(
            m.unsqueeze(0), size=(H0, W0), mode="bilinear", align_corners=False
        ).squeeze(0)  # [Q, H0, W0]
        out.append(m.clamp_(0, 1).to(device=device, dtype=dtype))

    if single:
        return [out[0]]
    return out


def cleanup_masks(masks, boxes):
    # clean up masks outside of the corresponding bbox
    N, H, W = masks.shape
    ys = torch.arange(H)[None, :, None]  # (1, H, 1)
    xs = torch.arange(W)[None, None, :]  # (1, 1, W)

    x1, y1, x2, y2 = boxes.T
    inside = (
        (xs >= x1[:, None, None])
        & (xs < x2[:, None, None])
        & (ys >= y1[:, None, None])
        & (ys < y2[:, None, None])
    )  # (N, H, W), bool
    masks = masks * inside.to(dtype=masks.dtype)
    return masks


def get_latest_experiment_name(exp: str, output_dir: str):
    output_dir = Path(output_dir)
    if output_dir.exists():
        return exp

    target_exp_name = Path(exp).name.rsplit("_", 1)[0]
    runs_dir = output_dir.parent
    latest_exp = None

    # Everything here is best-effort: the dir may not exist yet, and it can hold anything.
    for exp_path in runs_dir.iterdir() if runs_dir.is_dir() else []:
        exp_name, _, exp_date = exp_path.name.rpartition("_")
        if target_exp_name != exp_name:
            continue
        try:
            exp_date = datetime.strptime(exp_date, "%Y-%m-%d")
        except ValueError:  # not a dated run directory
            continue
        if not latest_exp or exp_date > latest_exp:
            latest_exp = exp_date

    if latest_exp is None:
        raise FileNotFoundError(
            f"no run matching '{target_exp_name}_<date>' under {runs_dir}. Train first, or "
            "point train.path_to_save at an existing run."
        )
    final_exp_name = f"{target_exp_name}_{latest_exp.strftime('%Y-%m-%d')}"
    logger.info(f"Latest experiment: {final_exp_name}")
    return final_exp_name


class LetterboxRect(DualTransform):
    def __init__(
        self,
        height: int,
        width: int,
        color=(114, 114, 114),
        auto: bool = False,
        scale_fill: bool = False,
        scaleup: bool = True,
        stride: int = 32,
        dense_mask: bool = False,  # True: dense label map (NEAREST, pad mask_fill); False: binary
        mask_fill: int = 0,  # pad value for dense masks (e.g. sem_seg ignore_index)
        always_apply: bool = True,
        p: float = 1.0,
    ):
        super().__init__(always_apply, p)
        self.height = int(height)
        self.width = int(width)
        self.color = tuple(color)
        self.auto = bool(auto)
        self.scale_fill = bool(scale_fill)
        self.scaleup = bool(scaleup)
        self.stride = int(stride)
        self.dense_mask = bool(dense_mask)
        self.mask_fill = int(mask_fill)

    def get_transform_init_args_names(self):
        return (
            "height",
            "width",
            "color",
            "auto",
            "scale_fill",
            "scaleup",
            "stride",
            "dense_mask",
            "mask_fill",
        )

    @property
    def targets_as_params(self):
        return ["image"]

    # Generate all deterministic params needed by apply/apply_to_bboxes
    # (computed once per call, then reused for image and bboxes)
    def get_params_dependent_on_data(self, params, data):
        img = data["image"]
        h, w = img.shape[:2]

        if self.scale_fill:
            # stretch to exact size
            new_unpad_w, new_unpad_h = self.width, self.height
            ratio_x = self.width / w
            ratio_y = self.height / h
            dw, dh = 0.0, 0.0
        else:
            # keep aspect ratio
            r = min(self.height / h, self.width / w)
            if not self.scaleup:
                r = min(r, 1.0)

            new_unpad_w = int(round(w * r))
            new_unpad_h = int(round(h * r))

            dw = self.width - new_unpad_w
            dh = self.height - new_unpad_h

            if self.auto:
                # pad to stride multiple, like inference `auto=True`
                dw = np.mod(dw, self.stride)
                dh = np.mod(dh, self.stride)

            ratio_x = r
            ratio_y = r

        # split padding equally to both sides
        dw *= 0.5
        dh *= 0.5

        # match inference border rounding
        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))
        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))

        return {
            # original size
            "orig_h": h,
            "orig_w": w,
            # resized (pre-pad) size
            "new_w": new_unpad_w,
            "new_h": new_unpad_h,
            # scale ratios
            "ratio_x": float(ratio_x),
            "ratio_y": float(ratio_y),
            # padding to apply
            "pad_left": left,
            "pad_top": top,
            "pad_right": right,
            "pad_bottom": bottom,
            # final canvas target (sanity)
            "target_h": self.height,
            "target_w": self.width,
        }

    # Image transform
    def apply(
        self, img, new_w=0, new_h=0, pad_left=0, pad_top=0, pad_right=0, pad_bottom=0, **kwargs
    ):
        # resize if needed
        if img.shape[1] != new_w or img.shape[0] != new_h:
            img = cv2.resize(img, (int(new_w), int(new_h)), interpolation=cv2.INTER_LINEAR)

        # pad if needed
        if pad_top or pad_bottom or pad_left or pad_right:
            img = cv2.copyMakeBorder(
                img,
                int(pad_top),
                int(pad_bottom),
                int(pad_left),
                int(pad_right),
                cv2.BORDER_CONSTANT,
                value=self.color,
            )
        return img

    # Mask transform - binary: LINEAR+re-threshold, pad 0; dense: NEAREST, pad mask_fill
    def apply_to_mask(
        self, mask, new_w=0, new_h=0, pad_left=0, pad_top=0, pad_right=0, pad_bottom=0, **kwargs
    ):
        needs_resize = mask.shape[1] != new_w or mask.shape[0] != new_h
        if self.dense_mask:
            # dense label map: NEAREST keeps integer class ids intact, pad with mask_fill
            if needs_resize:
                mask = cv2.resize(mask, (int(new_w), int(new_h)), interpolation=cv2.INTER_NEAREST)
            pad_value = self.mask_fill
        else:
            # binary mask: LINEAR for smooth edges, then re-threshold (NEAREST -> ladder edges!)
            if needs_resize:
                mask_float = mask.astype(np.float32)
                mask_float = cv2.resize(
                    mask_float, (int(new_w), int(new_h)), interpolation=cv2.INTER_LINEAR
                )
                mask = (mask_float > 0.5).astype(mask.dtype)
            pad_value = 0  # padding is not part of any object

        if pad_top or pad_bottom or pad_left or pad_right:
            mask = cv2.copyMakeBorder(
                mask,
                int(pad_top),
                int(pad_bottom),
                int(pad_left),
                int(pad_right),
                cv2.BORDER_CONSTANT,
                value=pad_value,
            )
        return mask

    # Bboxes transform (Pascal VOC: abs xyxy)
    def apply_to_bboxes(
        self,
        bboxes,
        ratio_x=1.0,
        ratio_y=1.0,
        pad_left=0,
        pad_top=0,
        orig_w=0,
        orig_h=0,
        target_w=0,
        target_h=0,
        **kwargs,
    ):
        # Albumentations passes bboxes in its INTERNAL NORMALIZED format [0..1]
        # We must return normalized bboxes for the transformed image.
        if bboxes is None or len(bboxes) == 0:
            return bboxes

        b = np.asarray(bboxes, dtype=np.float32)

        has_extra = b.shape[1] > 4
        extra = None
        if has_extra:
            extra = b[:, 4:].copy()
            b = b[:, :4]

        # to absolute coordinates (original image)
        b[:, [0, 2]] *= float(orig_w)
        b[:, [1, 3]] *= float(orig_h)

        # resize
        b[:, [0, 2]] *= float(ratio_x)
        b[:, [1, 3]] *= float(ratio_y)

        # pad
        b[:, [0, 2]] += float(pad_left)
        b[:, [1, 3]] += float(pad_top)

        # back to normalized (final canvas)
        b[:, [0, 2]] /= max(float(target_w), 1e-6)
        b[:, [1, 3]] /= max(float(target_h), 1e-6)

        # clip to [0,1] to avoid filtering
        b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0.0, 1.0)
        b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0.0, 1.0)

        # ensure x2>=x1, y2>=y1 numerically
        b[:, 2] = np.maximum(b[:, 2], b[:, 0])
        b[:, 3] = np.maximum(b[:, 3], b[:, 1])

        if has_extra:
            b = np.concatenate([b, extra], axis=1)

        return b


def norm_poly_to_abs(poly_norm_flat: np.ndarray, H: int, W: int) -> np.ndarray:
    """poly_norm_flat: [x1,y1,x2,y2,...] normalized -> (K,2) absolute"""
    if poly_norm_flat.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    pts = poly_norm_flat.reshape(-1, 2).copy()
    pts[:, 0] *= W
    pts[:, 1] *= H
    return pts.astype(np.float32)


def poly_abs_to_mask(parts, h: int, w: int) -> np.ndarray:
    """Rasterize one instance's polygon parts (islands) into a single (h,w) uint8 mask.

    Accepts a (K,2) array or a list of them. Parts are filled one per call: a single
    multi-contour fillPoly applies the even-odd rule and punches a hole where parts
    nest, while pycocotools merges an instance's parts by OR.
    """
    if isinstance(parts, np.ndarray):
        parts = [parts]
    m = np.zeros((h, w), dtype=np.uint8)
    for poly_abs in parts:
        pts = np.round(poly_abs).astype(np.int32)
        if len(pts) >= 3:  # cv2.fillPoly segfaults on empty/degenerate contours
            cv2.fillPoly(m, [pts], 1)
    return m


# def poly_abs_to_mask(poly_abs: np.ndarray, h: int, w: int, supersample: int = 4) -> np.ndarray:
#     # Rasterize at higher resolution for anti-aliasing
#     pts = poly_abs.copy() * supersample
#     pts = np.round(pts).astype(np.int32)
#     m = np.zeros((h * supersample, w * supersample), dtype=np.uint8)
#     cv2.fillPoly(m, [pts], 1)
#     # Downsample with area averaging for anti-aliased edges
#     m = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA)
#     return (m > 0.5).astype(np.uint8)


# ============================================================================
# RLE Encoding/Decoding for memory-efficient mask storage
# ============================================================================


def masks_to_rle(masks: torch.Tensor) -> List[Dict]:
    """
    Encode binary masks to COCO RLE format for memory-efficient storage.

    Args:
        masks: [N, H, W] uint8 tensor with binary masks (0/1)

    Returns:
        List of RLE dicts, each with 'size' and 'counts' keys
    """
    if masks is None or masks.numel() == 0:
        return []

    # Ensure proper format: [N, H, W], uint8, on CPU
    if masks.dim() == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    masks = masks.to(torch.uint8).cpu().numpy()

    rles = []
    for m in masks:
        # pycocotools expects Fortran-order (column-major) array
        rle = mask_utils.encode(np.asfortranarray(m))
        # Convert bytes to string for JSON serialization compatibility
        rle["counts"] = (
            rle["counts"].decode("utf-8") if isinstance(rle["counts"], bytes) else rle["counts"]
        )
        rles.append(rle)

    return rles


def rle_to_masks(rles: List[Dict], device: str = "cpu") -> torch.Tensor:
    """
    Decode RLE-encoded masks back to dense tensor format.

    Args:
        rles: List of RLE dicts from masks_to_rle()
        device: Target device for output tensor

    Returns:
        [N, H, W] uint8 tensor with binary masks
    """
    if not rles:
        return torch.zeros((0, 1, 1), dtype=torch.uint8)

    # Ensure counts are bytes for pycocotools
    rles_bytes = []
    for rle in rles:
        rle_copy = rle.copy()
        if isinstance(rle_copy["counts"], str):
            rle_copy["counts"] = rle_copy["counts"].encode("utf-8")
        rles_bytes.append(rle_copy)

    # Decode all masks at once (more efficient)
    masks_np = mask_utils.decode(rles_bytes)  # [H, W, N]
    if masks_np.ndim == 2:
        # Single mask case: [H, W] -> [1, H, W]
        masks_np = masks_np[np.newaxis, ...]
    else:
        masks_np = np.transpose(masks_np, (2, 0, 1))  # [N, H, W]

    return torch.from_numpy(masks_np.copy()).to(dtype=torch.uint8, device=device)


def encode_sample_masks_to_rle(sample: Dict) -> Dict:
    """
    Convert a prediction/GT sample dict to use RLE-encoded masks.
    Replaces 'masks' tensor with 'masks_rle' list and stores original size.

    Args:
        sample: Dict with 'masks' key containing [N, H, W] tensor

    Returns:
        Same dict with 'masks' replaced by 'masks_rle' and 'masks_size'
    """
    if "masks" not in sample or sample["masks"] is None:
        return sample

    masks = sample["masks"]
    if masks.numel() == 0:
        sample["masks_rle"] = []
        sample["masks_size"] = (0, 0)
        del sample["masks"]
        return sample

    # Store size for later decoding
    if masks.dim() == 3:
        _, H, W = masks.shape
    else:
        H, W = masks.shape[-2], masks.shape[-1]

    sample["masks_rle"] = masks_to_rle(masks)
    sample["masks_size"] = (H, W)
    del sample["masks"]

    return sample


def search_batch_size(try_batch, maximum=1024):
    """Return a tested batch, or zero if even batch 1 exceeds the budget."""
    best, upper = 0, 1
    while upper <= maximum and try_batch(upper):
        best, upper = upper, upper * 2
    if best == 0:
        return 0
    lo, hi = best + 1, min(upper - 1, maximum)
    while lo <= hi:
        mid = (lo + hi) // 2
        if try_batch(mid):
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best


# The CUDA probe runs in a child process to isolate weights, RNG and DDP collectives.
def auto_batch_size(cfg, device, target_fraction=0.7):
    if not 0 < target_fraction <= 1:
        raise ValueError("Auto batch target_fraction must be in (0, 1]")
    if device.type != "cuda":
        logger.warning("Auto batch size only works on CUDA devices, defaulting to batch_size=4")
        return 4
    logger.info("Searching for the optimal batch size...")
    with tempfile.TemporaryDirectory(prefix="dfine-autobatch-") as directory:
        config = Path(directory) / "config.yaml"
        result_path = Path(directory) / "result.json"
        OmegaConf.save(OmegaConf.to_container(cfg, resolve=True), config)
        env = os.environ.copy()
        # Use this checkout even when the installed console script belongs to another worktree.
        root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (root, env.get("PYTHONPATH"))))
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "dfine_seg.dl.utils",
                str(config),
                str(device),
                str(target_fraction),
                str(result_path),
            ],
            env=env,
            check=False,
        )
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
    batch = int(result.get("batch", 0)) if process.returncode == 0 else 0
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        agreed = torch.tensor(batch, device=device, dtype=torch.int64)
        torch.distributed.all_reduce(agreed, op=torch.distributed.ReduceOp.MIN)
        batch = int(agreed.item())
    if not batch:
        detail = result.get("error", "Batch 1 failed on this or another rank; see probe output")
        raise RuntimeError(f"Auto batch size failed: {detail}")
    total_mem = torch.cuda.get_device_properties(device).total_memory
    logger.info(
        f"Optimal batch size: {batch} "
        f"(target {target_fraction:.0%} of {total_mem / 1024**3:.1f} GB VRAM)"
    )
    return batch


def _probe(cfg, device, target_fraction):
    from torch.amp import GradScaler
    from torch.utils.data import DataLoader

    from dfine_seg.dl.dataset import Loader, sem_seg_collate_fn
    from dfine_seg.dl.train import (
        KDTeacher,
        ModelEMA,
        amp_dtype,
        training_forward,
        training_optimizer,
    )
    from dfine_seg.dl.utils import set_seeds
    from dfine_seg.model.dfine import build_loss, build_model, freeze_except_mask

    torch.cuda.set_device(device.index if device.index is not None else 0)
    set_seeds(cfg.train.seed, cfg.train.cudnn_fixed)
    dtype = amp_dtype(cfg)
    if cfg.train.amp_enabled and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError("bf16 AMP not supported; set train.amp_dtype=float16")
    model = build_model(
        cfg.model_name,
        len(cfg.train.label_to_name),
        cfg.task == "segment",
        str(device),
        img_size=cfg.train.img_size,
        in_channels=cfg.train.in_channels,
        pretrained_model_path=cfg.train.pretrained_model_path,
        pretrained_backbone=cfg.train.get("imagenet_backbone", False),
        task=cfg.task,
    ).train()
    if cfg.train.get("freeze_except_mask", False):
        if cfg.task != "segment":
            raise ValueError("train.freeze_except_mask requires task=segment")
        for module in freeze_except_mask(model):
            module.eval()
    kd = cfg.train.get("kd")
    teacher = KDTeacher(cfg, device) if kd and kd.get("teacher") else None
    if teacher is not None and cfg.task == "sem_seg":
        model.decoder.return_quarter_logits = True
    loss_fn = build_loss(
        cfg.model_name,
        len(cfg.train.label_to_name),
        cfg.train.label_smoothing,
        cfg.task == "segment",
        task=cfg.task,
        ignore_index=int(cfg.train.sem_seg.ignore_index) if cfg.task == "sem_seg" else 255,
        class_weights=cfg.train.sem_seg.class_weights if cfg.task == "sem_seg" else None,
        kd=OmegaConf.to_container(kd) if teacher is not None else None,
    ).train()
    optimizer = training_optimizer(model, cfg)
    ema = ModelEMA(model, cfg.train.ema_momentum) if cfg.train.use_ema else None
    scaler = GradScaler(enabled=cfg.train.amp_enabled and dtype == torch.float16)
    accum = max(1, cfg.train.b_accum_steps)

    base = Loader(
        Path(cfg.train.data_path), tuple(cfg.train.img_size), 1, 0, cfg, debug_img_processing=False
    )
    dataset = base.build_dataloaders(distributed=False)[0].dataset
    if cfg.train.ignore_background_epochs and cfg.task != "sem_seg":
        dataset.ignore_background = True
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        collate_fn=sem_seg_collate_fn if cfg.task == "sem_seg" else base.val_collate_fn,
    )
    sample_img, sample_targets, most = None, None, -1
    for index, (img, targets, _) in enumerate(loader):
        if img is not None:
            count = 0 if cfg.task == "sem_seg" else sum(t["labels"].numel() for t in targets)
            if count > most:
                sample_img, sample_targets, most = img, targets, count
            if cfg.task == "sem_seg":
                break
        if index >= 99:
            break
    if sample_img is None:
        raise ValueError("Could not load a training sample for auto batch size")
    if cfg.task != "sem_seg" and cfg.train.augs.multiscale_prob:
        size = tuple(v + 64 for v in cfg.train.img_size)
        sample_img = torch.nn.functional.interpolate(
            sample_img, size=size, mode="bilinear", align_corners=False
        )
        for target in sample_targets:
            if target["masks"].numel():
                target["masks"] = (
                    torch.nn.functional.interpolate(
                        target["masks"][:, None].float(),
                        size=size,
                        mode="bilinear",
                        align_corners=False,
                    )[:, 0]
                    > 0.5
                ).to(torch.uint8)

    # Allocate lazy optimizer states before measuring any candidate, even if fp16 scaling
    # skips the first real step. Zero LR keeps the disposable pretrained weights intact.
    rates = [group["lr"] for group in optimizer.param_groups]
    for group in optimizer.param_groups:
        group["lr"] = 0.0
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    for group, rate in zip(optimizer.param_groups, rates):
        group["lr"] = rate
    optimizer.zero_grad(set_to_none=accum == 1)
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(device)
    budget = int(min(total, free + torch.cuda.memory_reserved(device)) * target_fraction)
    measurements = {}

    def run_batch(batch, repeats):
        images = sample_img.to(device).repeat(batch, 1, 1, 1)
        # Distinct target storage matches a real batch (list multiplication alone aliases masks).
        targets = [
            {
                key: value.to(device, copy=True) if torch.is_tensor(value) else value
                for key, value in target.items()
            }
            for _ in range(batch)
            for target in sample_targets
        ]
        for step in range(repeats):
            losses = training_forward(
                model, loss_fn, teacher, images, targets, enabled=cfg.train.amp_enabled, dtype=dtype
            )
            loss = sum(losses.values()) / accum
            scaler.scale(loss).backward()
            if cfg.train.clip_max_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip_max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=accum == 1)
            if ema is not None:
                ema.update(step + 1, model)
            del loss, losses
        torch.cuda.synchronize(device)

    def try_batch(batch, repeats=1):
        nonlocal scaler
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        fits = False
        try:
            run_batch(batch, repeats)
            peak = torch.cuda.max_memory_reserved(device)
            measurements[batch] = peak
            fits = peak <= budget
        except torch.cuda.OutOfMemoryError:
            scaler = GradScaler(enabled=cfg.train.amp_enabled and dtype == torch.float16)
        finally:
            optimizer.zero_grad(set_to_none=accum == 1)
            if hasattr(loss_fn, "_clear_cache"):
                loss_fn._clear_cache()
        # Release traceback/temporary references before clearing the CUDA allocator.
        gc.collect()
        torch.cuda.empty_cache()
        return fits

    best = search_batch_size(try_batch)
    while best and not try_batch(best, repeats=3):
        best -= 1
    if not best:
        raise RuntimeError(
            "Batch 1 does not fit the VRAM budget; reduce resolution/teacher chunk size"
        )
    return {"batch": best, "peak_reserved": measurements[best], "budget": budget}


if __name__ == "__main__":
    config_path, device_name, fraction, output_path = sys.argv[1:]
    logger.disable("dfine_seg")
    try:
        result = _probe(OmegaConf.load(config_path), torch.device(device_name), float(fraction))
    except Exception as exc:
        Path(output_path).write_text(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        raise
    Path(output_path).write_text(json.dumps(result))
