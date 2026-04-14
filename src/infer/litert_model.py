from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from loguru import logger
from numpy.typing import NDArray
from torchvision.ops import nms


class LiteRT_model:
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
    ):
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

        if isinstance(conf_thresh, float):
            self.conf_threshs = [conf_thresh] * self.n_outputs
        elif isinstance(conf_thresh, list):
            self.conf_threshs = conf_thresh

        self._test_pred()

    def _load_model(self):
        try:
            from ai_edge_litert import interpreter as litert

            self.interpreter = litert.Interpreter(model_path=str(self.model_path))
        except ImportError:
            import tensorflow as tf

            self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        logger.info(f"LiteRT model loaded: {self.model_path}")

    def _read_model_metadata(self):
        inp_shape = self.input_details[0]["shape"]  # [1, 3, H, W]
        self.input_size = (int(inp_shape[2]), int(inp_shape[3]))  # (H, W)
        self.has_masks = len(self.output_details) > 2

    def _test_pred(self):
        random_image = np.random.randint(0, 255, size=(1000, 1110, self.channels), dtype=np.uint8)
        processed_inputs, processed_sizes, original_sizes = self._prepare_inputs(random_image)
        preds = self._predict(processed_inputs)
        self._postprocess(preds, processed_sizes, original_sizes)

    @staticmethod
    def process_boxes(boxes, processed_sizes, orig_sizes, keep_ratio):
        final_boxes = torch.zeros_like(boxes)
        for idx in range(boxes.shape[0]):
            final_boxes[idx] = norm_xywh_to_abs_xyxy(
                boxes[idx], processed_sizes[idx][0], processed_sizes[idx][1]
            )
        for i in range(len(orig_sizes)):
            if keep_ratio:
                final_boxes[i] = scale_boxes_ratio_kept(
                    final_boxes[i], processed_sizes[i], orig_sizes[i]
                )
            else:
                final_boxes[i] = scale_boxes(final_boxes[i], orig_sizes[i], processed_sizes[i])
        return final_boxes

    @staticmethod
    def process_masks(
        pred_masks,
        processed_size,
        orig_sizes,
        keep_ratio,
    ) -> List[torch.Tensor]:
        single = pred_masks.dim() == 3
        if single:
            pred_masks = pred_masks.unsqueeze(0)

        B, Q, Hm, Wm = pred_masks.shape
        proc_h, proc_w = int(processed_size[0]), int(processed_size[1])

        out = []
        for b in range(B):
            H0, W0 = int(orig_sizes[b, 0].item()), int(orig_sizes[b, 1].item())
            m = pred_masks[b]

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

        return [out[0]] if single else out

    def _compute_nearest_size(self, shape, target_size, stride=32) -> Tuple[int, int]:
        scale = target_size / max(shape)
        new_shape = [int(round(dim * scale)) for dim in shape]
        return [max(stride, int(np.ceil(dim / stride) * stride)) for dim in new_shape]

    def _preprocess(self, img: NDArray, stride: int = 32) -> NDArray:
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

        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, then HWC to CHW
        img = np.ascontiguousarray(img, dtype=self.np_dtype)
        img /= 255.0
        return img

    def _prepare_inputs(self, inputs):
        original_sizes = []
        processed_sizes = []

        if isinstance(inputs, np.ndarray) and inputs.ndim == 3:
            processed_inputs = self._preprocess(inputs)[None]
            original_sizes.append((inputs.shape[0], inputs.shape[1]))
            processed_sizes.append((processed_inputs[0].shape[1], processed_inputs[0].shape[2]))
        elif isinstance(inputs, np.ndarray) and inputs.ndim == 4:
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

    def _predict(self, img: NDArray) -> List[NDArray]:
        self.interpreter.set_tensor(self.input_details[0]["index"], img)
        self.interpreter.invoke()
        outputs = []
        for detail in self.output_details:
            outputs.append(self.interpreter.get_tensor(detail["index"]))
        return outputs

    def _postprocess(
        self,
        outputs: List[NDArray],
        processed_sizes: List[Tuple[int, int]],
        original_sizes: List[Tuple[int, int]],
        num_top_queries=300,
        use_focal_loss=True,
    ) -> List[Dict[str, torch.Tensor]]:
        # Output order from _LiteRTRawAdapter: logits, boxes, [masks]
        # TFLite output order may differ from the export order, match by shape
        tensors = [torch.from_numpy(o) for o in outputs]

        logits, boxes, pred_masks = None, None, None
        for t in tensors:
            if t.dim() == 3 and t.shape[-1] == 4:
                boxes = t
            elif t.dim() == 3 and t.shape[-1] != 4:
                logits = t
            elif t.dim() == 4:
                pred_masks = t

        has_masks = pred_masks is not None
        B, Q = logits.shape[:2]

        boxes = self.process_boxes(boxes, processed_sizes, original_sizes, self.keep_ratio)

        if use_focal_loss:
            scores_all = torch.sigmoid(logits)
            flat = scores_all.flatten(1)
            K = min(num_top_queries, flat.shape[1])
            topk_scores, topk_idx = torch.topk(flat, K, dim=-1)
            topk_labels = topk_idx % self.n_outputs
            topk_qidx = topk_idx // self.n_outputs
        else:
            probs = torch.softmax(logits, dim=-1)[:, :, :-1]
            topk_scores, topk_labels = probs.max(dim=-1)
            K = min(num_top_queries, Q)
            topk_scores, order = torch.topk(topk_scores, K, dim=-1)
            topk_labels = topk_labels.gather(1, order)
            topk_qidx = order

        results = []
        for b in range(B):
            sb = topk_scores[b]
            lb = topk_labels[b]
            qb = topk_qidx[b]

            conf_threshs_tensor = torch.tensor(self.conf_threshs, device=sb.device)
            keep = sb >= conf_threshs_tensor[lb]

            sb = sb[keep]
            lb = lb[keep]
            qb = qb[keep]
            bb = boxes[b].gather(0, qb.unsqueeze(-1).repeat(1, 4))

            if self.apply_nms and bb.numel() > 0:
                nms_keep = nms(bb, sb, self.nms_iou_thresh)
                sb, lb, bb = sb[nms_keep], lb[nms_keep], bb[nms_keep]
                qb = qb[nms_keep]

            out = {"labels": lb, "boxes": bb, "scores": sb}

            if has_masks and qb.numel() > 0:
                mb = pred_masks[b, qb]
                orig_sizes_tensor = torch.tensor([original_sizes[b]])
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

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scale_fill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(np.floor(dh)), int(np.ceil(dh))
    left, right = int(np.floor(dw)), int(np.ceil(dw))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, (r, r), (dw, dh)


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


def norm_xywh_to_abs_xyxy(
    boxes: torch.Tensor, height: int, width: int, to_round=True
) -> torch.Tensor:
    x_center = boxes[:, 0] * width
    y_center = boxes[:, 1] * height
    box_width = boxes[:, 2] * width
    box_height = boxes[:, 3] * height

    x_min = x_center - (box_width / 2)
    y_min = y_center - (box_height / 2)
    x_max = x_center + (box_width / 2)
    y_max = y_center + (box_height / 2)

    if to_round:
        x_min = torch.clamp(torch.floor(x_min), min=1)
        y_min = torch.clamp(torch.floor(y_min), min=1)
        x_max = torch.clamp(torch.ceil(x_max), max=width - 1)
        y_max = torch.clamp(torch.ceil(y_max), max=height - 1)
    else:
        x_min = torch.clamp(x_min, min=0)
        y_min = torch.clamp(y_min, min=0)
        x_max = torch.clamp(x_max, max=width)
        y_max = torch.clamp(y_max, max=height)
    return torch.stack([x_min, y_min, x_max, y_max], dim=1)


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
