from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from loguru import logger
from numpy.typing import NDArray
from openvino import Core
from torchvision.ops import nms


class OVModel:
    def __init__(
        self,
        model_path: str,
        conf_thresh: float | List[float] = 0.5,
        binarize_masks: bool = True,
        mask_threshold: float = 0.5,
        rect: bool = False,
        half: bool = False,
        keep_ratio: bool = False,
        max_batch_size: int = 1,
        device: str = None,
        apply_nms: bool = True,
        nms_iou_thresh: float = 0.7,
        labels_to_use: List[int] = None,  # empty -> keep all classes; else keep only these ids
        input_width: int = None,  # for dynamic-H/W graphs; None -> IR rt_info img_size
        input_height: int = None,
    ):
        self.model_path = model_path
        self.input_width = input_width
        self.input_height = input_height
        self.device = device.upper() if device else device
        self.rect = rect
        self.half = half
        self.keep_ratio = keep_ratio
        self.max_batch_size = max_batch_size
        self.torch_device = "cpu"
        self.binarize_masks = binarize_masks
        self.mask_threshold = mask_threshold
        self.np_dtype = np.float32
        self.apply_nms = apply_nms
        self.nms_iou_thresh = nms_iou_thresh
        self.labels_to_use = labels_to_use or []

        self._load_model()
        self._read_model_metadata()

        # Per-class confidence thresholds
        self.conf_thresh = self._prep_conf(conf_thresh)

        self._test_pred()

    @staticmethod
    def _prep_conf(conf_thresh: float | List[float]) -> float | torch.Tensor:
        """One float, or a per-class tensor on the CPU, where the numpy-backed scores live."""
        if np.ndim(conf_thresh):
            return torch.as_tensor(conf_thresh, dtype=torch.float32, device="cpu")
        return float(conf_thresh)

    def _load_model(self):
        core = Core()
        det_ov_model = core.read_model(self.model_path)

        if not self.device:
            # The GPU plugin targets Intel GPUs. OpenVINO >= 2025.4 also lists NVIDIA cards
            # (via OpenCL), and then fails to compile on them.
            intel_gpus = [
                d
                for d in core.get_available_devices()
                if d.startswith("GPU") and "Intel" in core.get_property(d, "FULL_DEVICE_NAME")
            ]
            self.device = intel_gpus[0] if intel_gpus and not self.rect else "CPU"

        # Read channel count + input_size. partial_shape (not .shape) so a dynamic
        # batch dim (max_batch_size > 1 exports) doesn't raise; C stays static.
        inp_shape = det_ov_model.inputs[0].partial_shape  # [B, C, H, W]
        self.channels = int(inp_shape[1].get_length())
        dynamic_hw = inp_shape[2].is_dynamic or inp_shape[3].is_dynamic
        explicit = (self.input_height, self.input_width)
        if (explicit[0] is None) != (explicit[1] is None):
            raise ValueError("Pass both input_height and input_width, or neither.")
        if dynamic_hw:
            # explicit args, else the img_size our export writes into the IR rt_info
            if explicit[0] is not None:
                self.input_size = explicit
            else:
                h, w = str(det_ov_model.get_rt_info(["model_info", "img_size"]).get()).split("x")
                self.input_size = (int(h), int(w))
        else:
            self.input_size = (int(inp_shape[2].get_length()), int(inp_shape[3].get_length()))
            if explicit[0] is not None and explicit != self.input_size:
                raise ValueError(
                    f"{Path(self.model_path).name} is a static {self.input_size[0]}x"
                    f"{self.input_size[1]} IR, it cannot run at {explicit[0]}x{explicit[1]}. "
                    "Re-export with export.dynamic_input=true."
                )

        # Only rect + keep_ratio feeds per-image sizes; otherwise every input is resized to
        # input_size, so pin H/W static (~10% faster on CPU than leaving them dynamic).
        keep_dynamic_hw = dynamic_hw and self.rect and self.keep_ratio
        if self.device != "CPU" or (dynamic_hw and not keep_dynamic_hw):
            batch = -1 if self.max_batch_size > 1 else 1
            hw = [-1, -1] if keep_dynamic_hw else list(self.input_size)
            det_ov_model.reshape({"input": [batch, self.channels, *hw]})

        inference_hint = "f16" if self.half else "f32"
        inference_mode = "CUMULATIVE_THROUGHPUT" if self.max_batch_size > 1 else "LATENCY"
        self.model = core.compile_model(
            det_ov_model,
            self.device,
            config={"PERFORMANCE_HINT": inference_mode, "INFERENCE_PRECISION_HINT": inference_hint},
        )
        logger.info(f"OpenVino running on {self.device}")

    def _read_model_metadata(self):
        """Detect graph type, mask presence and num classes from the compiled model."""
        # sem_seg graph = label map [+ probs]; the raw detection head has 2-4 outputs too
        # (logits, boxes, [masks], [keypoints]), so match by name.
        # an output can carry several names (the mask one is {"einsum", "masks"}): match any
        names = [o.get_names() for o in self.model.outputs]
        self.output_names = [
            next((n for n in ("masks", "kpts") if n in ns), o.get_any_name())
            for o, ns in zip(self.model.outputs, names)
        ]
        self.sem_seg = "sem_seg" in names[0]
        self.has_masks = "masks" in self.output_names
        self.has_kpts = "kpts" in self.output_names
        if self.sem_seg:
            self.n_outputs = 1  # classes are baked into the fused argmax
        else:
            # partial_shape: output batch dim is dynamic for max_batch_size > 1 exports
            self.n_outputs = int(self.model.outputs[0].partial_shape[2].get_length())  # [B, Q, C]

    def _test_pred(self):
        random_image = np.random.randint(0, 255, size=(1000, 1110, self.channels), dtype=np.uint8)
        self.model(self._prepare_inputs(random_image)[0])

    @staticmethod
    def process_boxes(boxes, processed_sizes, orig_sizes, keep_ratio):
        """All operations stay on torch tensors."""
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
    def process_kpts(kpts, processed_sizes, orig_sizes, keep_ratio):
        """[B, Q, K, 4] (x, y normalized) -> x, y absolute in the original image, like the boxes.
        Not clipped: a point may lie outside the image."""
        kpts = kpts.float().clone()
        for i in range(len(orig_sizes)):
            proc_h, proc_w = processed_sizes[i]
            h0, w0 = orig_sizes[i]
            x, y = kpts[i, ..., 0] * proc_w, kpts[i, ..., 1] * proc_h
            if keep_ratio:
                gain = min(proc_h / h0, proc_w / w0)
                x = (x - round((proc_w - w0 * gain) / 2 - 0.1)) / gain
                y = (y - round((proc_h - h0 * gain) / 2 - 0.1)) / gain
            else:
                x, y = x * w0 / proc_w, y * h0 / proc_h
            kpts[i, ..., 0], kpts[i, ..., 1] = x, y
        return kpts

    @staticmethod
    def process_masks(
        pred_masks,  # Tensor [B, Q, Hm, Wm] or [Q, Hm, Wm]
        processed_size,  # (H, W) of network input (after your A.Compose)
        orig_sizes,  # sequence of B (H, W) pairs - keep it on the host, it is only read there
        keep_ratio: bool,
    ) -> List[torch.Tensor]:
        """
        Returns list of length B with masks resized to original image sizes:
        Each item: [Q, H_orig, W_orig] in [0,1], half on CUDA / float on CPU (not thresholded).
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
            H0, W0 = int(orig_sizes[b][0]), int(orig_sizes[b][1])
            m = pred_masks[b]  # [Q, Hm, Wm]

            if keep_ratio:
                # Compute same gain/pad as in scale_boxes_ratio_kept
                gain = min(proc_h / H0, proc_w / W0)
                padw = round((proc_w - W0 * gain) / 2 - 0.1)
                padh = round((proc_h - H0 * gain) / 2 - 0.1)

                # Calculate crop region in mask space (scaled from processed_size to mask_size)
                scale_h, scale_w = Hm / proc_h, Wm / proc_w
                y1 = int(max(padh, 0) * scale_h)
                y2 = int((proc_h - max(padh, 0)) * scale_h)
                x1 = int(max(padw, 0) * scale_w)
                x2 = int((proc_w - max(padw, 0)) * scale_w)
                m = m[:, y1:y2, x1:x2]  # [Q, cropped_h, cropped_w]

            # Single resize directly to original size, in fp16 on GPU: this is the largest
            # tensor in postprocess ([Q, H0, W0]) and it is only compared against a
            # threshold afterwards. No clamp - bilinear of values already in [0,1] is a
            # convex combination, so it cannot leave [0,1].
            if m.is_cuda:  # on CPU, half interpolate is ~1.7x slower than float
                m = m.half()
            m = torch.nn.functional.interpolate(
                m.unsqueeze(0), size=(H0, W0), mode="bilinear", align_corners=False
            ).squeeze(0)  # [Q, H0, W0]
            out.append(m)

        return [out[0]] if single else out

    def _compute_nearest_size(self, shape, target_size, stride=32) -> Tuple[int, int]:
        """Get nearest size that is divisible by 32"""
        scale = target_size / max(shape)
        new_shape = [int(round(dim * scale)) for dim in shape]
        return [max(stride, int(np.ceil(dim / stride) * stride)) for dim in new_shape]

    def _preprocess(self, img: NDArray, stride: int = 32, bgr: bool = True) -> torch.tensor:
        if not self.keep_ratio:  # simple resize
            img = cv2.resize(
                img, (self.input_size[1], self.input_size[0]), interpolation=cv2.INTER_LINEAR
            )
        elif self.rect:  # keep ratio and cut paddings
            target_height, target_width = self._compute_nearest_size(
                img.shape[:2], max(*self.input_size)
            )
            img = letterbox(img, (target_height, target_width), stride=stride, auto=False)[0]
        else:  # keep ratio adding paddings
            img = letterbox(
                img, (self.input_size[0], self.input_size[1]), stride=stride, auto=False
            )[0]

        # 3ch BGR (cv2.imread) needs a swap; 3ch .npy is RGB already; >3ch is RGB+extras.
        if self.channels == 3 and bgr:
            img = img[:, :, ::-1].transpose(2, 0, 1)
        else:
            img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=self.np_dtype)
        img /= 255.0
        return img

    def _prepare_inputs(self, inputs, bgr: bool = True):
        original_sizes = []
        processed_sizes = []

        if isinstance(inputs, np.ndarray) and inputs.ndim == 3:  # single image
            processed_inputs = self._preprocess(inputs, bgr=bgr)[None]
            original_sizes.append((inputs.shape[0], inputs.shape[1]))
            processed_sizes.append((processed_inputs[0].shape[1], processed_inputs[0].shape[2]))

        elif isinstance(inputs, np.ndarray) and inputs.ndim == 4:  # batch of images
            processed_inputs = np.zeros(
                (inputs.shape[0], self.channels, self.input_size[0], self.input_size[1]),
                dtype=self.np_dtype,
            )
            for idx, image in enumerate(inputs):
                processed_inputs[idx] = self._preprocess(image, bgr=bgr)
                original_sizes.append((image.shape[0], image.shape[1]))
                processed_sizes.append(
                    (processed_inputs[idx].shape[1], processed_inputs[idx].shape[2])
                )
        return processed_inputs, processed_sizes, original_sizes

    def _predict(self, img: NDArray) -> List[NDArray]:
        return list(self.model(img).values())

    def _postprocess_sem_seg(
        self, outputs, processed_sizes, original_sizes
    ) -> List[Dict[str, torch.Tensor]]:
        """Engine (fused graph): 1/4 logits -> bilinear upsample to model input size -> argmax.
        Here: per-image NEAREST resize of the label map to original size (letterbox pads cropped
        first).
        """
        maps = np.asarray(outputs[0])  # [B, H, W] int
        # export.sem_seg_probs: [B, C, H/4, W/4] softmax
        probs = np.asarray(outputs[1], dtype=np.float32) if len(outputs) > 1 else None
        if self.labels_to_use:  # ids not requested -> 255 (ignore/void, not class 0)
            maps = np.where(np.isin(maps, self.labels_to_use), maps, 255)
        results = []
        for b in range(maps.shape[0]):
            m = maps[b].astype(np.uint8)
            H0, W0 = int(original_sizes[b][0]), int(original_sizes[b][1])
            if self.keep_ratio:
                proc_h, proc_w = int(processed_sizes[b][0]), int(processed_sizes[b][1])
                gain = min(proc_h / H0, proc_w / W0)
                padw = round((proc_w - W0 * gain) / 2 - 0.1)
                padh = round((proc_h - H0 * gain) / 2 - 0.1)
                m = m[max(padh, 0) : proc_h - max(padh, 0), max(padw, 0) : proc_w - max(padw, 0)]
            m = cv2.resize(m, (W0, H0), interpolation=cv2.INTER_NEAREST)
            res = {"sem_seg": torch.from_numpy(m)}
            if probs is not None:
                tf = sem_seg_probs_transform(
                    (H0, W0), processed_sizes[b], probs.shape[2:], self.keep_ratio
                )
                res["sem_seg_probs"] = torch.from_numpy(probs[b])
                res["sem_seg_probs_tf"] = torch.from_numpy(tf)
            results.append(res)
        return results

    def _postprocess(
        self,
        outputs: List[NDArray],
        processed_sizes: List[Tuple[int, int]],
        original_sizes: List[Tuple[int, int]],
        conf_thresh: float | List[float] | None = None,
        num_top_queries=300,
        use_focal_loss=True,
    ) -> List[Dict[str, NDArray]]:
        """
        returns List with BS length. Each element is a dict {"labels", "boxes", "scores"}
        """
        if self.sem_seg:
            return self._postprocess_sem_seg(outputs, processed_sizes, original_sizes)
        logits, boxes = torch.from_numpy(outputs[0]), torch.from_numpy(outputs[1])
        named = dict(zip(self.output_names, outputs))
        has_masks = self.has_masks
        pred_masks = torch.from_numpy(named["masks"]) if has_masks else None  # [B,Q,Hm,Wm]
        B, Q = logits.shape[:2]

        boxes = self.process_boxes(
            boxes, processed_sizes, original_sizes, self.keep_ratio
        )  # B x TopQ x 4
        kpts = None
        if self.has_kpts:  # [B, Q, K, 4]
            kpts = torch.from_numpy(named["kpts"])
            kpts = self.process_kpts(kpts, processed_sizes, original_sizes, self.keep_ratio)

        # scores/labels and preliminary topK over all Q*C
        if use_focal_loss:
            scores_all = torch.sigmoid(logits)  # [B,Q,C]
            flat = scores_all.flatten(1)  # [B, Q*C]
            # pre-topk to avoid scanning all queries later
            K = min(num_top_queries, flat.shape[1])
            topk_scores, topk_idx = torch.topk(flat, K, dim=-1)  # [B,K]
            topk_labels = topk_idx - (topk_idx // self.n_outputs) * self.n_outputs  # [B,K]
            topk_qidx = topk_idx // self.n_outputs  # [B,K]
        else:
            probs = torch.softmax(logits, dim=-1)[:, :, :-1]  # [B,Q,C-1]
            topk_scores, topk_labels = probs.max(dim=-1)  # [B,Q]
            # keep at most K queries per image by score
            K = min(num_top_queries, Q)
            topk_scores, order = torch.topk(topk_scores, K, dim=-1)  # [B,K]
            topk_labels = topk_labels.gather(1, order)  # [B,K]
            topk_qidx = order

        conf_thresh = self.conf_thresh if conf_thresh is None else self._prep_conf(conf_thresh)

        results = []
        for b in range(B):
            sb = topk_scores[b]
            lb = topk_labels[b]
            qb = topk_qidx[b]
            keep = sb >= (conf_thresh if type(conf_thresh) is float else conf_thresh[lb])
            if self.labels_to_use:  # restrict to requested class ids
                lbl_set = torch.as_tensor(self.labels_to_use, device=lb.device, dtype=lb.dtype)
                keep &= torch.isin(lb, lbl_set)

            # index once and reuse: boolean indexing would re-run nonzero (and re-sync) per tensor
            keep_idx = torch.nonzero(keep, as_tuple=True)[0]
            sb = sb[keep_idx]
            lb = lb[keep_idx]
            qb = qb[keep_idx]
            # gather boxes once
            bb = boxes[b].gather(0, qb.unsqueeze(-1).repeat(1, 4))

            if self.apply_nms and bb.numel() > 0:
                nms_keep = nms(bb, sb, self.nms_iou_thresh)
                sb, lb, bb = sb[nms_keep], lb[nms_keep], bb[nms_keep]
                qb = qb[nms_keep]

            out = {"labels": lb, "boxes": bb, "scores": sb}
            if kpts is not None:
                out["kpts"] = kpts[b, qb]

            if has_masks and qb.numel() > 0:
                # gather only kept masks
                mb = pred_masks[b, qb]  # [K', Hm, Wm] logits or probs
                # resize to original size (list of length 1)
                masks_list = self.process_masks(
                    mb.unsqueeze(0),  # [1,K',Hm,Wm]
                    processed_size=processed_sizes[b],  # (Hin, Win)
                    orig_sizes=[original_sizes[b]],  # host-side (H, W)
                    keep_ratio=self.keep_ratio,
                )
                out["masks"] = masks_list[0]  # [K, H, W]
                if self.binarize_masks:
                    # torch bool is one byte holding 0/1, so reinterpreting it as uint8 is
                    # free; .to(uint8) would copy a tensor the size of the source image.
                    out["masks"] = (out["masks"] >= self.mask_threshold).view(torch.uint8)
                else:
                    out["masks"] = out["masks"].float()
                # clean up masks outside of the corresponding bbox
                out["masks"] = cleanup_masks(out["masks"], out["boxes"])

            results.append(out)
        return results

    def __call__(
        self,
        inputs: NDArray[np.uint8],
        bgr: bool = True,
        conf_thresh: float | List[float] | None = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Input image as ndarray (BGR, HWC) or BHWC. Pass ``bgr=False`` for
        3-channel inputs already in RGB order (e.g., ``.npy`` read via
        ``read_image_hwc``); ignored for >3 channels.
        ``conf_thresh`` (one float or a per-class list) overrides the constructor's value for
        this call only; ignored for sem_seg.
        Output:
            List of batch size length. Each element is a dict {"labels", "boxes", "scores"}
            labels: torch.Tensor of shape (N,), dtype int64
            boxes: torch.Tensor of shape (N, 4), dtype float32, abs values
            scores: torch.Tensor of shape (N,), dtype float32
            masks: torch.Tensor of shape (N, H, W), dtype float32. N = number of objects
            kpts (kpts models): torch.Tensor of shape (N, K, 4), dtype float32: x, y (abs),
                presence score, visible score. All K points are returned
            sem_seg models instead return {"sem_seg": uint8 (H, W) dense label map} per image,
            + "sem_seg_probs" float32 (C, h, w): softmax at 1/4 of the model input, letterbox
            pads included, and "sem_seg_probs_tf" float32 (3, 3) mapping original-image
            pixels to probs pixels, when exported with export.sem_seg_probs.
        """
        processed_inputs, processed_sizes, original_sizes = self._prepare_inputs(inputs, bgr=bgr)
        preds = self._predict(processed_inputs)
        return self._postprocess(preds, processed_sizes, original_sizes, conf_thresh)

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
                # Get the largest contour
                contour = max(contours, key=cv2.contourArea)
                contour = contour.reshape(-1, 2)
                if len(contour) >= 3:  # Need at least 3 points for a valid polygon
                    # Normalize coordinates
                    norm_contour = contour.astype(np.float32)
                    norm_contour[:, 0] /= w
                    norm_contour[:, 1] /= h
                    polys.append(norm_contour)
                else:
                    polys.append(np.array([]))
            else:
                polys.append(np.array([]))

        return polys


def sem_seg_probs_transform(orig_hw, proc_hw, probs_hw, keep_ratio: bool) -> NDArray:
    """3x3 affine mapping an original-image pixel (x, y, 1) to sem_seg_probs pixel coords,
    integers at pixel centers (cv2 / align_corners=False convention). Mirrors preprocessing:
    stretch resize, or letterbox (resize to round(W0 * r) x round(H0 * r), pad floor(dw / 2)
    left / top), then the model's 1/4 scale. Compose with camera intrinsics as K' = tf @ K.
    """
    (H0, W0), (H, W), (h, w) = orig_hw, proc_hw, probs_hw
    if keep_ratio:
        r = min(H / H0, W / W0)
        new_w, new_h = int(round(W0 * r)), int(round(H0 * r))
        left, top = (W - new_w) // 2, (H - new_h) // 2
    else:
        new_w, new_h, left, top = W, H, 0, 0
    # orig -> input: (x + 0.5) * new_w / W0 - 0.5 + left; input -> probs: (x + 0.5) * w / W - 0.5
    sx, sy = new_w / W0 * w / W, new_h / H0 * h / H
    tx = (0.5 * new_w / W0 + left) * w / W - 0.5
    ty = (0.5 * new_h / H0 + top) * h / H - 0.5
    return np.array([[sx, 0, tx], [0, sy, ty], [0, 0, 1]], dtype=np.float32)


def letterbox(
    im,
    new_shape=(640, 640),
    color=None,
    auto=True,
    scale_fill=False,
    scaleup=True,
    stride=32,
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    if color is None:
        c = im.shape[2] if im.ndim == 3 else 1
        color = tuple([114] * c)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # initial uniform width, height ratios (may be updated below)
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(np.floor(dh)), int(np.ceil(dh))
    left, right = int(np.floor(dw)), int(np.ceil(dw))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im, ratio, (dw, dh)


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


def scale_boxes_ratio_kept(boxes, img1_shape, img0_shape, ratio_pad=None, padding=True):
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
    scale_x = orig_shape[1] / resized_shape[1]
    scale_y = orig_shape[0] / resized_shape[0]
    boxes[:, 0] *= scale_x
    boxes[:, 2] *= scale_x
    boxes[:, 1] *= scale_y
    boxes[:, 3] *= scale_y
    return boxes


def norm_xywh_to_abs_xyxy(boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Converts boxes: [N, 4] normalized xywh -> [N, 4] absolute xyxy"""
    x_center = boxes[:, 0] * width
    y_center = boxes[:, 1] * height
    box_width = boxes[:, 2] * width
    box_height = boxes[:, 3] * height

    x_min = x_center - (box_width / 2)
    y_min = y_center - (box_height / 2)
    x_max = x_center + (box_width / 2)
    y_max = y_center + (box_height / 2)

    x_min = torch.clamp(x_min, min=0, max=width)
    y_min = torch.clamp(y_min, min=0, max=height)
    x_max = torch.clamp(x_max, min=0, max=width)
    y_max = torch.clamp(y_max, min=0, max=height)
    return torch.stack([x_min, y_min, x_max, y_max], dim=1)


def cleanup_masks(masks: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Zero each mask outside its own bbox, in place.

    The box test separates into a column term (N,1,W) and a row term (N,H,1), so it is
    applied as two broadcast multiplies. Building the full (N,H,W) `inside` indicator
    instead costs several extra tensors that size at the *original* image resolution.
    """
    N, H, W = masks.shape
    device, dtype = masks.device, masks.dtype

    ys = torch.arange(H, device=device)[None, :, None]  # (1, H, 1)
    xs = torch.arange(W, device=device)[None, None, :]  # (1, 1, W)

    x1, y1, x2, y2 = boxes.T  # each (N,)
    in_x = ((xs >= x1[:, None, None]) & (xs < x2[:, None, None])).to(dtype)  # (N, 1, W)
    in_y = ((ys >= y1[:, None, None]) & (ys < y2[:, None, None])).to(dtype)  # (N, H, 1)
    return masks.mul_(in_x).mul_(in_y)


if __name__ == "__main__":
    model = OVModel(
        model_path="model.xml",
        conf_thresh=0.5,
        half=False,
    )

    img = cv2.imread("test_image.jpg")
    res = model(img)

    for i in range(len(res[0]["boxes"])):
        box = res[0]["boxes"][i].cpu().numpy().astype(int)
        score = res[0]["scores"][i].cpu().numpy()
        label = res[0]["labels"][i].cpu().numpy()
        mask = res[0]["masks"][i].cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8) * 255

        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 180, 100), 2)
        cv2.putText(
            img,
            f"Document {score:.2f}",
            (box[0], box[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 180, 100),
            2,
        )

        colored_mask = np.zeros_like(img)
        colored_mask[:, :, 0] = mask
        colored_mask[:, :, 1] = mask * 120 // 255
        colored_mask[:, :, 2] = mask * 60 // 255
        img = cv2.addWeighted(img, 1.0, colored_mask, 0.2, 0)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (255, 180, 100), 2)

    cv2.imwrite("out.png", img)
