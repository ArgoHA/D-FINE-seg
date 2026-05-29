from typing import Dict, List, Tuple

import cv2
import numpy as np
import tensorrt as trt
import torch
import torch.nn.functional as F  # noqa: N812
from numpy.typing import NDArray
from torchvision.ops import nms


class TRT_model:
    def __init__(
        self,
        model_path: str,
        n_outputs: int,
        conf_thresh: float | List[float] = 0.5,
        binarize_masks: bool = True,
        mask_threshold: float = 0.5,
        rect: bool = False,
        keep_ratio: bool = False,
        device: str = None,
        apply_nms: bool = True,
        nms_iou_thresh: float = 0.7,
        labels_to_use: List[int] = None,  # empty -> keep all classes; else keep only these ids
        use_cuda_graph: bool = True,
    ) -> None:
        self.model_path = model_path
        self.n_outputs = n_outputs
        self.rect = rect
        self.keep_ratio = keep_ratio
        self.binarize_masks = binarize_masks
        self.mask_threshold = mask_threshold
        self.np_dtype = np.float32
        self.apply_nms = apply_nms
        self.nms_iou_thresh = nms_iou_thresh
        self.labels_to_use = labels_to_use or []
        self.use_cuda_graph = use_cuda_graph

        assert not rect, "rect=True is not supported by the current TRT_model implementation"

        if not device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._load_model()
        self._read_engine_metadata()
        self._stream = torch.cuda.Stream(device=self.device) if self.device == "cuda" else None
        self._setup_io_buffers()

        # Per-class confidence thresholds
        if isinstance(conf_thresh, float):
            self.conf_threshs = [conf_thresh] * self.n_outputs
        elif isinstance(conf_thresh, list):
            self.conf_threshs = conf_thresh
        self._conf_threshs_t = torch.tensor(
            self.conf_threshs, device=self.device, dtype=torch.float32
        )

        self._graph = None
        self._test_pred()

        if self.device == "cuda" and self.use_cuda_graph:
            self._capture_cuda_graph()

    def _load_model(self):
        self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(self.model_path, "rb") as f, trt.Runtime(self.TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

    def _read_engine_metadata(self):
        """Auto-read channels + input_size and detect mask presence from the engine."""
        inp_name = self.engine.get_tensor_name(0)
        inp_shape = tuple(self.engine.get_tensor_shape(inp_name))
        self.channels = int(inp_shape[1])
        self.input_size = (inp_shape[2], inp_shape[3])  # (H, W)

        n_outputs = 0
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                n_outputs += 1
        self.has_masks = n_outputs > 3

    @staticmethod
    def _torch_dtype_from_trt(trt_dtype):
        if trt_dtype == trt.float32:
            return torch.float32
        elif trt_dtype == trt.float16:
            return torch.float16
        elif trt_dtype == trt.int32:
            return torch.int32
        elif trt_dtype == trt.int64:
            return torch.int64
        elif trt_dtype == trt.int8:
            return torch.int8
        else:
            raise TypeError(f"Unsupported TensorRT data type: {trt_dtype}")

    def _setup_io_buffers(self):
        """Pre-allocate persistent IO tensors and bind them to the execution context.

        Static-shape engines: allocate at the engine's nominal shape and bind once.
        Dynamic-shape engines (e.g. exported with max_batch_size>1): allocate at
        the optimization profile's max so a single buffer accommodates any valid
        batch; the actual run shape is set on the context per call in `_predict`.
        """
        self._outputs: List[torch.Tensor] = []
        self._input_tensor: torch.Tensor | None = None
        self._input_dtype = torch.float32
        self._is_dynamic = False
        self._max_batch = 1
        self._input_name = ""

        # Pass 1: input. Resolve dynamic dims via the optimization profile so we
        # can allocate a max-sized buffer and set the context's input shape
        # (required before output shapes can be queried in pass 2).
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
                continue
            shape = list(self.engine.get_tensor_shape(name))
            dtype = self._torch_dtype_from_trt(self.engine.get_tensor_dtype(name))

            if any(d < 0 for d in shape):
                self._is_dynamic = True
                _, _, max_shape = self.engine.get_tensor_profile_shape(name, 0)
                max_shape = list(max_shape)
                shape = [m if d < 0 else d for d, m in zip(shape, max_shape)]
            self._max_batch = shape[0]
            self._input_name = name

            tensor = torch.empty(shape, dtype=dtype, device=self.device)
            self.context.set_tensor_address(name, tensor.data_ptr())
            self.context.set_input_shape(name, tuple(shape))
            self._input_tensor = tensor
            self._input_dtype = dtype

        # Pass 2: outputs. With the input shape set, the context resolves each
        # output's concrete (max) shape — use it directly instead of the
        # engine's possibly-symbolic shape.
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
                continue
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = self._torch_dtype_from_trt(self.engine.get_tensor_dtype(name))
            tensor = torch.empty(shape, dtype=dtype, device=self.device)
            self.context.set_tensor_address(name, tensor.data_ptr())
            self._outputs.append(tensor)

        # Pinned host + uint8 GPU staging buffer for fast preprocessing.
        H, W = self.input_size
        if self.device == "cuda":
            self._cpu_pinned_hwc = torch.empty(
                (H, W, self.channels), dtype=torch.uint8, pin_memory=True
            )
            self._gpu_hwc = torch.empty(
                (H, W, self.channels), dtype=torch.uint8, device=self.device
            )

    def _capture_cuda_graph(self):
        """Capture the engine forward into a CUDA graph for low-overhead replay."""
        if self._is_dynamic:
            # Graphs lock in a single shape; useless when batch varies per call.
            self._graph = None
            return
        try:
            self._stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._stream):
                # Warmup so cudnn/cublas/etc. pick algos before capture.
                for _ in range(3):
                    self.context.execute_async_v3(self._stream.cuda_stream)
            self._stream.synchronize()
            torch.cuda.current_stream().wait_stream(self._stream)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self._stream):
                self.context.execute_async_v3(self._stream.cuda_stream)
            self._graph = graph
        except Exception:
            # Some kernels may not be capturable; fall back to plain async exec.
            self._graph = None

    def _test_pred(self) -> None:
        random_image = np.random.randint(0, 255, size=(1100, 1000, self.channels), dtype=np.uint8)
        # Route through __call__ so the dedicated stream context is applied,
        # otherwise the engine (on _stream) and post-processing (on default
        # stream) race and trigger device-side asserts asynchronously.
        self(random_image)

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

            # Single resize directly to original size
            m = torch.nn.functional.interpolate(
                m.unsqueeze(0), size=(H0, W0), mode="bilinear", align_corners=False
            ).squeeze(0)  # [Q, H0, W0]
            out.append(m.clamp_(0, 1))

        if single:
            return [out[0]]
        return out

    def _compute_nearest_size(self, shape, target_size, stride=32) -> Tuple[int, int]:
        """Get nearest size that is divisible by 32"""
        scale = target_size / max(shape)
        new_shape = [int(round(dim * scale)) for dim in shape]
        return [max(stride, int(np.ceil(dim / stride) * stride)) for dim in new_shape]

    def _preprocess_to_pinned(self, img: NDArray, bgr: bool = True) -> None:
        """Resize into the pinned HWC uint8 buffer. 3ch BGR path also does
        BGR->RGB (cv2 source); 3ch RGB (.npy) and >3ch (np.load, RGB+extras)
        skip the swap."""
        H, W = self.input_size
        if not self.keep_ratio:
            resized = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        elif self.rect:
            target_height, target_width = self._compute_nearest_size(img.shape[:2], max(H, W))
            resized = letterbox(img, (target_height, target_width), stride=32, auto=False)[0]
        else:
            resized = letterbox(img, (H, W), stride=32, auto=False)[0]

        if self.channels == 3 and bgr:
            cv2.cvtColor(resized, cv2.COLOR_BGR2RGB, dst=self._cpu_pinned_hwc.numpy())
        else:
            np.copyto(self._cpu_pinned_hwc.numpy(), resized)

    def _preprocess(self, img: NDArray, stride: int = 32, bgr: bool = True) -> NDArray:
        """CPU-only preprocess used by the batched fall-back path."""
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
            img = img[..., ::-1].transpose(2, 0, 1)
        else:
            img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=self.np_dtype)
        img /= 255.0
        return img

    def _prepare_inputs(self, inputs, bgr: bool = True):
        # Single image fast path: avoid np->torch->pin->H2D->fp32 round-trip.
        if isinstance(inputs, np.ndarray) and inputs.ndim == 3 and self.device == "cuda":
            H, W = self.input_size
            self._preprocess_to_pinned(inputs, bgr=bgr)
            self._gpu_hwc.copy_(self._cpu_pinned_hwc, non_blocking=True)
            # HWC uint8 view -> CHW; copy_ does the cast to fp32 into the engine input.
            chw_view = self._gpu_hwc.permute(2, 0, 1)
            self._input_tensor[0].copy_(chw_view, non_blocking=True)
            self._input_tensor.div_(255.0)
            return self._input_tensor, [(H, W)], [(inputs.shape[0], inputs.shape[1])]

        # Generic / batched / CPU path: keep original behavior.
        original_sizes: List[Tuple[int, int]] = []
        processed_sizes: List[Tuple[int, int]] = []

        if isinstance(inputs, np.ndarray) and inputs.ndim == 3:  # single image, CPU
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
        else:
            raise TypeError(f"Unsupported input type: {type(inputs)}")

        tensor = torch.from_numpy(processed_inputs)
        if self.device == "cuda":
            tensor = tensor.pin_memory().to(self.device, non_blocking=True)
        else:
            tensor = tensor.to(self.device)
        return tensor, processed_sizes, original_sizes

    def _predict(self, img: torch.Tensor, actual_batch: int | None = None) -> List[torch.Tensor]:
        # `actual_batch` is the real run-time batch (== len(processed_sizes)).
        # Falling back to img.shape[0] is wrong for the single-image fast path
        # on dynamic engines, where img is the max-sized persistent buffer.
        if actual_batch is None:
            actual_batch = img.shape[0]

        # Fast path: if `img` is the persistent input buffer we already wrote in
        # _prepare_inputs, just (graph-)replay or async-execute.
        if img is self._input_tensor:
            if self._is_dynamic:
                run_shape = (actual_batch,) + tuple(self._input_tensor.shape[1:])
                self.context.set_input_shape(self._input_name, run_shape)
            if self._graph is not None:
                self._graph.replay()
            else:
                stream_handle = self._stream.cuda_stream if self.device == "cuda" else 0
                self.context.execute_async_v3(stream_handle)
            return self._slice_outputs(actual_batch)

        # Generic path: the caller passed a freshly built tensor (e.g. batched
        # numpy fall-back). Copy into the persistent input and run.
        img = img.contiguous()
        rebound = img.shape != tuple(self._input_tensor.shape)
        if rebound:
            self.context.set_input_shape(self._input_name, tuple(img.shape))
            self.context.set_tensor_address(self._input_name, img.data_ptr())
            input_buf = img
        else:
            if self._is_dynamic:
                # Buffer matches img shape, but the context may still hold a
                # smaller batch from a prior call — re-assert it.
                self.context.set_input_shape(self._input_name, tuple(img.shape))
            self._input_tensor.copy_(img, non_blocking=True)
            input_buf = self._input_tensor

        if self.device == "cuda":
            self.context.execute_async_v3(self._stream.cuda_stream)
        else:
            self.context.execute_v2([input_buf.data_ptr()] + [o.data_ptr() for o in self._outputs])

        # Restore the persistent-input binding so subsequent fast-path calls work.
        if rebound:
            self.context.set_tensor_address(self._input_name, self._input_tensor.data_ptr())
        return self._slice_outputs(actual_batch)

    def _slice_outputs(self, batch: int) -> List[torch.Tensor]:
        # Outputs are sized at max batch; trim views to the actual run batch so
        # downstream postprocessing doesn't iterate empty rows. Cheap (no copy).
        return [o[:batch] for o in self._outputs]

    def _postprocess(
        self,
        outputs: List[torch.Tensor],
        processed_sizes: List[Tuple[int, int]],
        original_sizes: List[Tuple[int, int]],
    ) -> List[Dict[str, NDArray]]:
        """
        returns List with BS length. Each element is a dict {"labels", "boxes", "scores"}
        """
        labels = outputs[0]  # [B, K]
        boxes = outputs[1]  # [B, K, 4], absolute xyxy in input_size space
        scores = outputs[2]  # [B, K]
        pred_masks = outputs[3] if self.has_masks else None  # [B, K, Hm, Wm]
        B = labels.shape[0]

        boxes = self.rescale_boxes(boxes, processed_sizes, original_sizes, self.keep_ratio)

        results = []
        for b in range(B):
            sb, lb, bb = scores[b], labels[b], boxes[b]
            # Apply per-class confidence thresholds (cached tensor avoids per-call alloc)
            conf_t = self._conf_threshs_t.to(sb.device, non_blocking=True)
            conf_keep = sb >= conf_t[lb]
            if self.labels_to_use:  # restrict to requested class ids
                lbl_set = torch.as_tensor(self.labels_to_use, device=lb.device, dtype=lb.dtype)
                conf_keep &= torch.isin(lb, lbl_set)
            keep_indices = torch.where(conf_keep)[0]
            sb, lb, bb = sb[conf_keep], lb[conf_keep], bb[conf_keep]

            if self.apply_nms and bb.numel() > 0:
                nms_keep = nms(bb, sb, self.nms_iou_thresh)
                sb, lb, bb = sb[nms_keep], lb[nms_keep], bb[nms_keep]
                keep_indices = keep_indices[nms_keep]

            out = {"labels": lb, "boxes": bb, "scores": sb}

            if pred_masks is not None and lb.numel() > 0:
                mb = pred_masks[b][keep_indices]  # [K, Hm, Wm] — already gathered for top-K
                # resize to original size (list of length 1)
                orig_sizes_tensor = torch.tensor([original_sizes[b]], device=mb.device)
                masks_list = self.process_masks(
                    mb.unsqueeze(0),
                    processed_size=processed_sizes[b],  # (Hin, Win)
                    orig_sizes=orig_sizes_tensor,  # [1,2]
                    keep_ratio=self.keep_ratio,
                )
                out["masks"] = masks_list[0]  # [K, H, W]
                if self.binarize_masks:
                    out["masks"] = (out["masks"] >= self.mask_threshold).to(torch.uint8)
                # clean up masks outside of the corresponding bbox
                out["masks"] = cleanup_masks(out["masks"], out["boxes"])

            results.append(out)
        return results

    def __call__(
        self, inputs: NDArray[np.uint8], bgr: bool = True
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Input image as ndarray (BGR, HWC) or BHWC. Pass ``bgr=False`` for
        3-channel inputs already in RGB order (e.g., ``.npy`` read via
        ``read_image_hwc``); ignored for >3 channels.
        Output:
            List of batch size length. Each element is a dict {"labels", "boxes", "scores"}
            labels: torch.Tensor of shape (N,), dtype int64
            boxes: torch.Tensor of shape (N, 4), dtype float32, abs values
            scores: torch.Tensor of shape (N,), dtype float32
            masks: torch.Tensor of shape (N, H, W), dtype float32. N = number of objects
        """
        # Run all GPU work on the model's dedicated stream so TRT can avoid the
        # extra default-stream synchronisations triggered by enqueueV3, then have
        # the default stream wait so subsequent .cpu() copies stay ordered.
        with torch.cuda.stream(self._stream):
            processed_inputs, processed_sizes, original_sizes = self._prepare_inputs(
                inputs, bgr=bgr
            )
            preds = self._predict(processed_inputs, actual_batch=len(processed_sizes))
            results = self._postprocess(preds, processed_sizes, original_sizes)
        torch.cuda.default_stream(self.device).wait_stream(self._stream)
        return results

    def gpu_run(
        self,
        rgb_chw: "torch.Tensor | List[torch.Tensor]",
        original_sizes: List[Tuple[int, int]] | None = None,
        input_ready: torch.cuda.Event | None = None,
    ) -> Tuple[List[Dict[str, torch.Tensor]], torch.cuda.Event]:
        """GPU-resident entry point. Same output contract as ``__call__`` but the
        caller hands over already-on-device RGB CHW uint8 tensors (no cv2,
        no BGR swap, no pinned-host H2D round-trip).

        Resize + cast + /255 run on the model's private stream directly into
        the persistent engine input buffer; ``_predict`` then takes its fast
        path (input buffer reused -> CUDA-graph replay on static engines, plain
        async exec otherwise) and ``_postprocess`` runs on the same stream.

        Args:
            rgb_chw: one of —
                - ``[3, H, W]`` uint8 RGB CUDA tensor (single image)
                - ``[B, 3, H, W]`` uint8 RGB CUDA tensor (equal-size batch)
                - list of ``[3, H_i, W_i]`` tensors (heterogeneous batch — each
                  image is resized independently before being stacked)
            original_sizes: per-element ``(H, W)`` used for postprocess (boxes
                rescaled into this space, masks resized to it). Defaults to
                each input's own resolution; pass smaller values to get
                coarser masks without paying for a full-res upsample.
            input_ready: optional CUDA event the caller recorded on its stream
                after writing the inputs. ``gpu_run`` inserts a non-blocking
                wait on the model stream so it can't read stale data.

        Returns:
            ``(results, done_event)``. ``results`` is the usual list of B
            ``{labels, boxes, scores, optional masks}`` dicts. ``done_event``
            is recorded on ``self._stream`` after postprocess — wait on it
            from your own stream (``my_stream.wait_event(done_event)``)
            before reading the result tensors. No CPU sync inside.

        Constraints: CUDA-only; ``keep_ratio=False`` engines only (no letterbox
        path); batch must fit the engine's max profile.
        """
        assert self.device == "cuda", "gpu_run requires a CUDA-resident model"
        assert not self.keep_ratio, "gpu_run only supports keep_ratio=False engines (no letterbox)"

        # Normalize to a list of [3, H_i, W_i] tensors so the resize step can
        # handle a heterogeneous-size batch (different clips at different
        # output resolutions converge to the same engine input here).
        if isinstance(rgb_chw, torch.Tensor):
            if rgb_chw.dim() == 3:
                chw_list = [rgb_chw]
            elif rgb_chw.dim() == 4:
                chw_list = [rgb_chw[i] for i in range(rgb_chw.shape[0])]
            else:
                raise ValueError(f"rgb_chw tensor must be 3-D or 4-D, got {rgb_chw.dim()}-D")
        else:
            chw_list = list(rgb_chw)

        B = len(chw_list)
        assert 1 <= B <= self._max_batch, f"batch {B} not in [1, {self._max_batch}] (engine max)"
        Hin, Win = self.input_size

        if original_sizes is None:
            original_sizes = [tuple(t.shape[-2:]) for t in chw_list]
        processed_sizes = [(Hin, Win)] * B

        with torch.cuda.stream(self._stream):
            if input_ready is not None:
                self._stream.wait_event(input_ready)
            # Per-input resize on the model stream (F.interpolate needs float).
            # Heterogeneous sizes => one interpolate call each; if all inputs
            # already match (Hin, Win) we still pay the cast — fine in practice
            # since the input cast happens regardless.
            resized = [
                F.interpolate(
                    t.unsqueeze(0).float(),
                    size=(Hin, Win),
                    mode="bilinear",
                    align_corners=False,
                )[0]
                for t in chw_list
            ]
            batch = torch.stack(resized, dim=0).div_(255.0)
            # Drop into the persistent input buffer to take _predict's fast
            # path (`img is self._input_tensor`). For B < _max_batch the tail
            # rows are left stale and _slice_outputs trims results to [:B];
            # dynamic engines additionally re-set the input shape per call.
            self._input_tensor[:B].copy_(batch, non_blocking=True)
            preds = self._predict(self._input_tensor, actual_batch=B)
            results = self._postprocess(preds, processed_sizes, original_sizes)
            done = torch.cuda.Event()
            done.record(self._stream)
        return results, done

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


def cleanup_masks(masks: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    # clean up masks outside of the corresponding bbox
    N, H, W = masks.shape
    device = masks.device
    dtype = masks.dtype

    ys = torch.arange(H, device=device)[None, :, None]  # (1, H, 1)
    xs = torch.arange(W, device=device)[None, None, :]  # (1, 1, W)

    x1, y1, x2, y2 = boxes.T  # each (N,)
    inside = (
        (xs >= x1[:, None, None])
        & (xs < x2[:, None, None])
        & (ys >= y1[:, None, None])
        & (ys < y2[:, None, None])
    )  # (N, H, W), bool
    masks = masks * inside.to(dtype)
    return masks
