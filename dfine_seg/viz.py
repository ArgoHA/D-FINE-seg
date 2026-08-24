"""Rendering shared by the public API, the training pipeline, the CLI and the demo.

Lives outside `dl/` so `dfine predict` and the Gradio demo can draw without importing
the training stack (wandb, pandas, albumentations). Single source of the class colors:
a second copy of this in demo.py had already drifted to a different hue formula.
"""

from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch


def classes_from_model(model) -> Tuple[int, Dict[int, str]]:
    """(class count, names) read off an inference wrapper; count is 0 when neither is known.

    Fused-postprocess graphs (.onnx/.engine/.mlpackage) carry no class count at all, so
    names may be the only source.
    """
    names = getattr(model, "names", None) or {}
    n_outputs = getattr(model, "n_outputs", 0) or 0
    return max(n_outputs, max(names) + 1 if names else 0), names


class Visualizer:
    """Draws detection / segmentation results with consistent per-class colors for inference."""

    def __init__(
        self,
        model=None,
        *,
        n_classes: int = None,
        class_names: Dict[int, str] = None,
        ignore_index: int = 255,
    ):
        """*model*: a wrapper from `load_model` (class count and names read off it), or an int
        class count. Falls back to 80 (COCO) when a graph artifact carries neither."""
        if isinstance(model, int):
            n_classes = model
        elif model is not None:
            known, names = classes_from_model(model)
            n_classes = n_classes or known or 80
            class_names = class_names or names or None
        self.n_classes = n_classes = max(n_classes or 80, 1)
        self.class_names = class_names or {i: str(i) for i in range(n_classes)}
        self.colors = self.generate_colors(n_classes)
        self.ignore_index = ignore_index
        self._palette = None

    @staticmethod
    def generate_colors(n: int) -> List[tuple]:
        """Evenly spaced hues on a violet→red arc -> BGR tuples. Class 0 = deep purple, last = red."""
        colors = []
        n = max(n, 1)
        hue_start = 135  # deep violet in OpenCV's [0, 179] hue range
        denom = max(n - 1, 1)
        for i in range(n):
            hue = int(hue_start * (n - 1 - i) / denom)
            hsv = np.array([[[hue, 230, 200]]], dtype=np.uint8)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(int(c) for c in bgr))
        return colors

    # ── public API ──────────────────────────────────────────────────────
    @property
    def palette(self) -> np.ndarray:
        """sem_seg lookup table, built once per instance - deriving it from the labels present
        in one frame would repaint every class as the scene changes."""
        if self._palette is None:
            self._palette = sem_seg_palette(self.n_classes)
        return self._palette

    def __call__(self, img: np.ndarray, results: dict, **kwargs) -> np.ndarray:
        """Draw whatever the wrapper returned: boxes/masks, or a palette overlay for sem_seg.

        Args:
            img: BGR, HWC, uint8 - 3 channels (slice a 4-channel `.npy` stack yourself).
            results: one image's result dict, i.e. `model(img)[0]`.
            kwargs: forwarded to `draw` (`alpha`, `minimize`) / `overlay_sem_seg` (`alpha`).
        """
        if isinstance(results, (list, tuple)):
            raise TypeError("pass one image's result dict, e.g. model(img)[0]")
        if img.ndim == 3 and img.shape[2] > 3:
            raise ValueError(
                f"{img.shape[2]}-channel input; pass 3-channel BGR, e.g. img[:, :, :3]"
            )
        if "sem_seg" in results:
            label_map = results["sem_seg"]
            if isinstance(label_map, torch.Tensor):
                label_map = label_map.cpu().numpy()
            kwargs.pop("minimize", None)  # a dense overlay draws no text to hide
            return overlay_sem_seg(
                img, label_map, self.palette, ignore_index=self.ignore_index, **kwargs
            )
        return self.draw(img, results, **kwargs)

    def draw(
        self, img: np.ndarray, results: dict, alpha: float = 0.6, minimize: bool = False
    ) -> np.ndarray:
        """Draw prediction results on *img* (BGR, uint8).

        Args:
            results: dict with keys ``labels``, ``boxes``, ``scores``,
                     and optionally ``masks`` and ``track_ids``. When
                     ``track_ids`` is present the label is prefixed with
                     ``id=N``.
            alpha: opacity of bbox outlines and label background rectangles
                in [0, 1]. Lower values make annotations less likely to fully
                occlude small neighboring objects. Text stays fully opaque.
            minimize: draw boxes and masks only, no text labels.
        Returns:
            Annotated copy of *img*.
        """
        img = img.copy()
        labels = results["labels"]
        boxes = results["boxes"]
        scores = results["scores"]
        track_ids = results.get("track_ids")
        has_masks = "masks" in results and results["masks"] is not None

        if len(labels) == 0:
            return img

        # Adaptive sizes based on image resolution
        ref = max(img.shape[:2])
        box_thick = max(1, int(ref / 400))
        font_scale = max(0.35, ref / 1800)
        font_thick = max(1, int(ref / 600))
        edge_thick = max(1, int(ref / 350))

        # Masks first (underneath boxes)
        if has_masks:
            masks = results["masks"]
            if isinstance(masks, torch.Tensor):
                masks = masks.cpu().numpy()
            for i in range(min(len(labels), len(masks))):
                label_id = self._as_int(labels[i])
                color = self.colors[label_id % len(self.colors)]
                self._draw_mask(img, masks[i], color, edge_thickness=edge_thick)

        # Boxes - draw onto an overlay, then alpha-blend back into img
        overlay = img.copy()
        for i in range(len(labels)):
            label_id = self._as_int(labels[i])
            color = self.colors[label_id % len(self.colors)]
            x1, y1, x2, y2 = self._box_coords(boxes[i])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, box_thick)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)

        if minimize:
            return img

        # Labels (background blended internally, text opaque)
        for i in range(len(labels)):
            label_id = self._as_int(labels[i])
            name = self.class_names.get(label_id, str(label_id))
            color = self.colors[label_id % len(self.colors)]
            score = float(scores[i].item()) if hasattr(scores[i], "item") else float(scores[i])
            text = f"{name} {score:.2f}"
            if track_ids is not None:
                text = f"id={self._as_int(track_ids[i])} {text}"

            x1, y1, _, _ = self._box_coords(boxes[i])
            self._draw_label(img, text, x1, y1, color, font_scale, font_thick, alpha)

        return img

    # ── private helpers ─────────────────────────────────────────────────
    @staticmethod
    def _as_int(value) -> int:
        return int(value.item()) if hasattr(value, "item") else int(value)

    @staticmethod
    def _box_coords(box) -> tuple:
        if hasattr(box, "tolist"):
            return tuple(map(int, box.tolist()))
        return tuple(map(int, box))

    @staticmethod
    def _draw_label(
        img: np.ndarray,
        text: str,
        x: int,
        y: int,
        bg_color: tuple,
        font_scale: float,
        font_thick: int,
        alpha: float = 0.6,
    ):
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, font_thick)
        pad = 4

        # Try placing above the box; fall back to below
        if y - th - 2 * pad >= 0:
            bg_y1, bg_y2, text_y = y - th - 2 * pad, y, y - pad
        else:
            bg_y1, bg_y2, text_y = y, y + th + 2 * pad, y + th + pad

        bg_x1, bg_x2 = x, x + tw + 2 * pad
        h_img, w_img = img.shape[:2]
        x1c, x2c = max(0, bg_x1), min(w_img, bg_x2)
        y1c, y2c = max(0, bg_y1), min(h_img, bg_y2)
        if x2c > x1c and y2c > y1c:
            roi = img[y1c:y2c, x1c:x2c]
            overlay = np.full_like(roi, bg_color, dtype=np.uint8)
            cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)

        # White or black text depending on background brightness
        lum = 0.299 * bg_color[2] + 0.587 * bg_color[1] + 0.114 * bg_color[0]
        txt_col = (0, 0, 0) if lum > 140 else (255, 255, 255)
        cv2.putText(img, text, (x + pad, text_y), font, font_scale, txt_col, font_thick)

    @staticmethod
    def _draw_mask(
        img: np.ndarray,
        mask: np.ndarray,
        color: tuple,
        body_alpha: float = 0.45,
        edge_alpha: float = 0.70,
        edge_thickness: int = 2,
    ):
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        if mask.dtype != np.uint8:
            mask = (mask > 0.5).astype(np.uint8)
        if mask.ndim == 3:
            mask = mask.squeeze(0)
        if mask.max() == 0:
            return

        color_arr = np.array(color, dtype=np.float32)

        # Semi-transparent body fill
        m = mask.astype(bool)
        img[m] = (img[m] * (1 - body_alpha) + color_arr * body_alpha).astype(np.uint8)

        # More opaque edge
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            edge_mask = np.zeros_like(mask)
            cv2.drawContours(edge_mask, contours, -1, 1, edge_thickness)
            e = edge_mask.astype(bool)
            img[e] = (img[e] * (1 - edge_alpha) + color_arr * edge_alpha).astype(np.uint8)


def sem_seg_palette(n_classes: int) -> np.ndarray:
    """[256, 3] BGR uint8 lookup table; ids >= n_classes (incl. ignore=255) stay black."""
    palette = np.zeros((256, 3), dtype=np.uint8)
    palette[:n_classes] = np.array(Visualizer.generate_colors(n_classes), dtype=np.uint8)
    return palette


def overlay_sem_seg(
    img: np.ndarray,
    label_map: np.ndarray,
    palette: np.ndarray,
    alpha: float = 0.5,
    ignore_index: int = 255,
) -> np.ndarray:
    """Blend a colorized label map over a BGR image; ignore pixels stay unblended."""
    if label_map.shape != img.shape[:2]:
        label_map = cv2.resize(
            label_map, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    out = cv2.addWeighted(img, 1 - alpha, palette[label_map], alpha, 0)
    keep = label_map == ignore_index
    out[keep] = img[keep]
    return out
