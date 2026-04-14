from typing import Dict, List, Tuple

import coremltools as ct
import cv2
import numpy as np
import torch
from loguru import logger
from numpy.typing import NDArray
from torchvision.ops import nms


class CoreML_model:
    def __init__(
        self,
        model_path: str,
        n_outputs: int,
        conf_thresh: float | List[float] = 0.5,
        binarize_masks: bool = True,
        mask_threshold: float = 0.5,
        rect: bool = False,
        keep_ratio: bool = False,
        apply_nms: bool = True,
        nms_iou_thresh: float = 0.7,
    ) -> None:
        self.model_path = model_path
        self.n_outputs = n_outputs
        self.rect = rect
        self.keep_ratio = keep_ratio
        self.channels = 3
        self.binarize_masks = binarize_masks
        self.mask_threshold = mask_threshold
        self.np_dtype = np.float32
        self.apply_nms = apply_nms
        self.nms_iou_thresh = nms_iou_thresh

        self._load_model()
        self._read_model_metadata()

        # Per-class confidence thresholds
        if isinstance(conf_thresh, float):
            self.conf_threshs = [conf_thresh] * self.n_outputs
        elif isinstance(conf_thresh, list):
            self.conf_threshs = conf_thresh

        self._test_pred()

    def _load_model(self):
        self.model = ct.models.MLModel(str(self.model_path))

        spec = self.model.get_spec()
        self._input_name = spec.description.input[0].name
        self._output_names = [out.name for out in spec.description.output]
        logger.info(f"CoreML model loaded from {self.model_path}")

    def _read_model_metadata(self):
        """Auto-read input_size and detect mask presence from the model spec."""
        spec = self.model.get_spec()
        inp = spec.description.input[0]

        shape = inp.type.multiArrayType.shape  # [B, C, H, W] or [C, H, W]
        if len(shape) == 4:
            self.input_size = (int(shape[2]), int(shape[3]))
        elif len(shape) == 3:
            self.input_size = (int(shape[1]), int(shape[2]))
        else:
            raise ValueError(f"Unexpected input shape: {list(shape)}")

        # Fused model outputs: labels, boxes, scores, [masks]
        self.has_masks = len(self._output_names) > 3

    def _test_pred(self) -> None:
        random_image = np.random.randint(0, 255, size=(1100, 1000, self.channels), dtype=np.uint8)
        processed_inputs, processed_sizes, original_sizes = self._prepare_inputs(random_image)
        preds = self._predict(processed_inputs)
        self._postprocess(preds, processed_sizes, original_sizes)

    @staticmethod
    def rescale_boxes(boxes, processed_sizes, orig_sizes, keep_ratio):
        """Rescale absolute xyxy boxes from input-size space to original image size."""
        out = boxes.clone()
        for i in range(boxes.shape[0]):
            if keep_ratio:
                out[i] = scale_boxes_ratio_kept(out[i], processed_sizes[i], orig_sizes[i])
            else:
                out[i] = scale_boxes(out[i], orig_sizes[i], processed_sizes[i])
        return out

    @staticmethod
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

        B, Q, Hm, Wm = pred_masks.shape
        proc_h, proc_w = int(processed_size[0]), int(processed_size[1])

        out = []
        for b in range(B):
            H0, W0 = int(orig_sizes[b, 0].item()), int(orig_sizes[b, 1].item())
            m = pred_masks[b]  # [Q, Hm, Wm]

            if keep_ratio:
                gain = min(proc_h / H0, proc_w / W0)
                padw = round((proc_w - W0 * gain) / 2 - 0.1)
                padh = round((proc_h - H0 * gain) / 2 - 0.1)

                scale_h, scale_w = Hm / proc_h, Wm / proc_w
                y1 = int(max(padh, 0) * scale_h)
                y2 = int((proc_h - max(padh, 0)) * scale_h)
                x1 = int(max(padw, 0) * scale_w)
                x2 = int((proc_w - max(padw, 0)) * scale_w)
                m = m[:, y1:y2, x1:x2]

            m = torch.nn.functional.interpolate(
                m.unsqueeze(0), size=(H0, W0), mode="bilinear", align_corners=False
            ).squeeze(0)
            out.append(m.clamp_(0, 1))

        if single:
            return [out[0]]
        return out

    def _compute_nearest_size(self, shape, target_size, stride=32) -> Tuple[int, int]:
        """Get nearest size that is divisible by 32"""
        scale = target_size / max(shape)
        new_shape = [int(round(dim * scale)) for dim in shape]
        return [max(stride, int(np.ceil(dim / stride) * stride)) for dim in new_shape]

    def _preprocess(self, img: NDArray, stride: int = 32) -> NDArray:
        """Resize, RGB, CHW"""
        if not self.keep_ratio:
            img = cv2.resize(
                img, (self.input_size[1], self.input_size[0]), interpolation=cv2.INTER_LINEAR
            )
        elif self.rect:
            target_height, target_width = self._compute_nearest_size(
                img.shape[:2], max(*self.input_size)
            )
            img = letterbox(img, (target_height, target_width), stride=stride, auto=False)[0]
        else:
            img = letterbox(
                img, (self.input_size[0], self.input_size[1]), stride=stride, auto=False
            )[0]

        img = img[..., ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img, dtype=self.np_dtype)
        img /= 255.0
        return img

    def _prepare_inputs(self, inputs):
        original_sizes = []
        processed_sizes = []

        if isinstance(inputs, np.ndarray) and inputs.ndim == 3:  # single image
            processed_inputs = self._preprocess(inputs)[None]
            original_sizes.append((inputs.shape[0], inputs.shape[1]))
            processed_sizes.append((processed_inputs[0].shape[1], processed_inputs[0].shape[2]))

        elif isinstance(inputs, np.ndarray) and inputs.ndim == 4:  # batch of images
            processed_inputs = np.zeros(
                (inputs.shape[0], self.channels, self.input_size[0], self.input_size[1]),
                dtype=self.np_dtype,
            )
            for idx, image in enumerate(inputs):
                processed_inputs[idx] = self._preprocess(image)
                original_sizes.append((image.shape[0], image.shape[1]))
                processed_sizes.append(
                    (processed_inputs[idx].shape[1], processed_inputs[idx].shape[2])
                )
        return processed_inputs, processed_sizes, original_sizes

    def _predict(self, img: NDArray) -> List[np.ndarray]:
        result = self.model.predict({self._input_name: img})
        return [result[name] for name in self._output_names]

    def _postprocess(
        self,
        outputs: List[np.ndarray],
        processed_sizes: List[Tuple[int, int]],
        original_sizes: List[Tuple[int, int]],
    ) -> List[Dict[str, torch.Tensor]]:
        """
        returns List with BS length. Each element is a dict {"labels", "boxes", "scores"}
        """
        labels = torch.from_numpy(outputs[0]).long()  # [B, K]
        boxes = torch.from_numpy(outputs[1]).float()  # [B, K, 4], absolute xyxy in input_size space
        scores = torch.from_numpy(outputs[2]).float()  # [B, K]
        pred_masks = (
            torch.from_numpy(outputs[3]).float() if self.has_masks else None
        )  # [B, K, Hm, Wm]
        B = labels.shape[0]

        boxes = self.rescale_boxes(boxes, processed_sizes, original_sizes, self.keep_ratio)

        results = []
        for b in range(B):
            sb, lb, bb = scores[b], labels[b], boxes[b]
            # Apply per-class confidence thresholds
            if self.conf_threshs is not None:
                conf_t = torch.tensor(self.conf_threshs, device=sb.device)
                conf_keep = sb >= conf_t[lb]
            else:
                conf_keep = sb >= self.conf_thresh
            keep_indices = torch.where(conf_keep)[0]
            sb, lb, bb = sb[conf_keep], lb[conf_keep], bb[conf_keep]

            if self.apply_nms and bb.numel() > 0:
                nms_keep = nms(bb, sb, self.nms_iou_thresh)
                sb, lb, bb = sb[nms_keep], lb[nms_keep], bb[nms_keep]
                keep_indices = keep_indices[nms_keep]

            out = {"labels": lb, "boxes": bb, "scores": sb}

            if pred_masks is not None and lb.numel() > 0:
                mb = pred_masks[b][keep_indices]  # [K, Hm, Wm]
                orig_sizes_tensor = torch.tensor([original_sizes[b]], device=mb.device)
                masks_list = self.process_masks(
                    mb.unsqueeze(0),
                    processed_size=processed_sizes[b],
                    orig_sizes=orig_sizes_tensor,
                    keep_ratio=self.keep_ratio,
                )
                out["masks"] = masks_list[0]
                if self.binarize_masks:
                    out["masks"] = (out["masks"] >= self.mask_threshold).to(torch.uint8)
                out["masks"] = cleanup_masks(out["masks"], out["boxes"])

            results.append(out)
        return results

    def __call__(self, inputs: NDArray[np.uint8]) -> List[Dict[str, torch.Tensor]]:
        """
        Input image as ndarray (BGR, HWC) or BHWC
        Output:
            List of batch size length. Each element is a dict {"labels", "boxes", "scores"}
            labels: torch.Tensor of shape (N,), dtype int64
            boxes: torch.Tensor of shape (N, 4), dtype float32, abs values
            scores: torch.Tensor of shape (N,), dtype float32
            masks: torch.Tensor of shape (N, H, W), dtype float32. N = number of objects
        """
        processed_inputs, processed_sizes, original_sizes = self._prepare_inputs(inputs)
        preds = self._predict(processed_inputs)
        return self._postprocess(preds, processed_sizes, original_sizes)

    @staticmethod
    def mask2poly(masks: np.ndarray, img_shape: Tuple[int, int]) -> List[np.ndarray]:
        """
        Convert binary masks to normalized polygon coordinates for YOLO segmentation format.

        Args:
            masks: Binary masks array of shape [N, H, W] where N is number of instances
            img_shape: Tuple of (height, width) of the original image

        Returns:
            List of normalized polygon coordinates, each as array of shape [num_points, 2]
            with values in range [0, 1]. Returns empty array for invalid masks.
        """
        h, w = img_shape[:2]
        polys = []

        for mask in masks:
            if not isinstance(mask, np.ndarray):
                mask = mask.numpy()
            mask = (mask > 0.5).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                contour = contour.reshape(-1, 2)
                if len(contour) >= 3:
                    norm_contour = contour.astype(np.float32)
                    norm_contour[:, 0] /= w
                    norm_contour[:, 1] /= h
                    polys.append(norm_contour)
                else:
                    polys.append(np.array([]))
            else:
                polys.append(np.array([]))

        return polys


def letterbox(
    im,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scale_fill=False,
    scaleup=True,
    stride=32,
):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scale_fill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(np.floor(dh)), int(np.ceil(dh))
    left, right = int(np.floor(dw)), int(np.ceil(dw))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)


def clip_boxes(boxes, shape):
    if isinstance(boxes, torch.Tensor):
        boxes[..., 0].clamp_(0, shape[1])
        boxes[..., 1].clamp_(0, shape[0])
        boxes[..., 2].clamp_(0, shape[1])
        boxes[..., 3].clamp_(0, shape[0])
    else:
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])


def scale_boxes_ratio_kept(boxes, img1_shape, img0_shape, ratio_pad=None, padding=True):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (
            round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
            round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
        )
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        boxes[..., [0, 2]] -= pad[0]
        boxes[..., [1, 3]] -= pad[1]
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def scale_boxes(boxes, orig_shape, resized_shape):
    scale_x = orig_shape[1] / resized_shape[1]
    scale_y = orig_shape[0] / resized_shape[0]
    boxes[:, 0] *= scale_x
    boxes[:, 2] *= scale_x
    boxes[:, 1] *= scale_y
    boxes[:, 3] *= scale_y
    return boxes


def cleanup_masks(masks: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    N, H, W = masks.shape
    device = masks.device
    dtype = masks.dtype

    ys = torch.arange(H, device=device)[None, :, None]
    xs = torch.arange(W, device=device)[None, None, :]

    x1, y1, x2, y2 = boxes.T
    inside = (
        (xs >= x1[:, None, None])
        & (xs < x2[:, None, None])
        & (ys >= y1[:, None, None])
        & (ys < y2[:, None, None])
    )
    masks = masks * inside.to(dtype)
    return masks
