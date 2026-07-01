"""
TensorRT INT8 post-training quantization with calibration.

Mirrors the ov_int8.py workflow:
  1. Loads the ONNX model exported by export.py (the "raw" one without fused postprocessor).
  2. Builds an INT8-calibrated TensorRT engine using the validation dataloader.
  3. Optionally validates F1 accuracy before/after quantization.
  4. Saves a calibration cache so subsequent rebuilds are instant.

Usage:
    make trt_int8          # or:  python -m src.dl.trt_int8
"""

from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
import onnx
import tensorrt as trt
import torch
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from src.dl.dataset import Loader
from src.dl.train import Trainer
from src.dl.utils import get_latest_experiment_name
from src.dl.validator import Validator


# ---------------------------------------------------------------------------
# INT8 Entropy Calibrator
# ---------------------------------------------------------------------------
class Int8EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """
    Feeds batches from a PyTorch DataLoader to TensorRT for INT8 calibration.
    Caches the calibration table so the engine can be rebuilt without re-running
    calibration.
    """

    def __init__(
        self,
        data_loader: torch.utils.data.DataLoader,
        cache_file: Path,
        input_shape: tuple,  # (C, H, W) — no batch dim
    ):
        super().__init__()
        self.data_loader = data_loader
        self.cache_file = Path(cache_file)
        self.batch_iter = iter(data_loader)
        self.input_shape = input_shape

        # Pre-allocate a CUDA buffer large enough for one batch
        self.batch_size = data_loader.batch_size or 1
        nbytes = int(np.prod(input_shape) * self.batch_size * np.dtype(np.float32).itemsize)
        self.device_input = torch.empty(nbytes, dtype=torch.uint8, device="cuda")

    # -- TensorRT calibrator API ------------------------------------------
    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names):  # noqa: ARG002
        try:
            images, _, _ = next(self.batch_iter)  # images: [B, C, H, W] float32
        except StopIteration:
            return None

        images = images.to("cuda", dtype=torch.float32).contiguous()
        # Copy into our pre-allocated buffer
        nbytes = images.nelement() * images.element_size()
        self.device_input[:nbytes].copy_(images.view(-1).view(torch.uint8))
        return [int(images.data_ptr())]

    def read_calibration_cache(self):
        if self.cache_file.exists():
            logger.info(f"Reading calibration cache: {self.cache_file}")
            return self.cache_file.read_bytes()
        return None

    def write_calibration_cache(self, cache):
        self.cache_file.write_bytes(cache)
        logger.info(f"Calibration cache written: {self.cache_file}")


# ---------------------------------------------------------------------------
# Engine builder
# ---------------------------------------------------------------------------
def build_int8_engine(
    onnx_path: Path,
    calibrator: trt.IInt8Calibrator,
    max_batch_size: int = 1,
    workspace_gb: int = 4,
) -> bytes:
    """Parse ONNX and build a TensorRT engine with INT8 + FP16 fallback."""
    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error(parser.get_error(i))
            raise RuntimeError("Failed to parse the ONNX file")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)

    # Enable INT8 with FP16 fallback for layers that don't quantize well
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    config.int8_calibrator = calibrator

    # Force normalization / activation layers to FP16  to prevent TRT from
    # fusing them with preceding Conv2D into an INT8 kernel that doesn't exist
    # (e.g. Conv2D + Sigmoid + SiLU has no INT8 implementation).
    _FP16_LAYER_TYPES = {
        trt.LayerType.NORMALIZATION,
        trt.LayerType.ACTIVATION,
        trt.LayerType.SOFTMAX,
        trt.LayerType.REDUCE,
    }
    _FP16_NAME_HINTS = {"silu", "sigmoid", "swish", "norm", "softmax", "layernorm", "groupnorm"}
    fp16_count = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        name_lower = layer.name.lower()
        if layer.type in _FP16_LAYER_TYPES or any(h in name_lower for h in _FP16_NAME_HINTS):
            layer.precision = trt.float16
            for j in range(layer.num_outputs):
                layer.set_output_type(j, trt.float16)
            fp16_count += 1
    logger.info(f"Marked {fp16_count}/{network.num_layers} layers as FP16 (rest stay INT8)")

    # Dynamic batch profile (if needed)
    if max_batch_size > 1:
        profile = builder.create_optimization_profile()
        inp_name = network.get_input(0).name

        onnx_model = onnx.load(str(onnx_path))
        input_proto = None
        for inp in onnx_model.graph.input:
            if inp.name == inp_name:
                input_proto = inp.type.tensor_type.shape
                break
        if input_proto is None:
            raise ValueError(f"Cannot find input '{inp_name}' in ONNX graph")

        static_dims = []
        for i, dim in enumerate(input_proto.dim[1:], start=1):
            if dim.dim_value:
                static_dims.append(int(dim.dim_value))
            else:
                raise ValueError(f"Dynamic dim at index {i} (beyond batch) not supported")

        profile.set_shape(
            inp_name, (1, *static_dims), (1, *static_dims), (max_batch_size, *static_dims)
        )
        config.add_optimization_profile(profile)

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError(
            "Failed to build TensorRT INT8 engine. Check logs above for details.\n"
            "Common causes: not enough GPU memory, unsupported ops, or bad calibration data."
        )
    return engine_bytes


# ---------------------------------------------------------------------------
# Validation helper (mirrors ov_int8.py)
# ---------------------------------------------------------------------------
def validate_engine(
    engine_path: Path,
    val_loader: torch.utils.data.DataLoader,
    *,
    num_labels: int,
    keep_ratio: bool,
    conf_thresh: float,
    iou_thresh: float,
    label_to_name: dict,
    enable_mask_head: bool,
    input_size: tuple,  # (H, W)
) -> float:
    """
    Load an engine, run the val set through it, return the F1 score.
    Auto-detects whether the engine has raw outputs (logits, boxes) or
    postprocessed outputs (labels, boxes, scores) and handles both.
    """
    trt_logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(trt_logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # Discover I/O by name
    input_idx = None
    output_name_to_idx: Dict[str, int] = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            input_idx = i
        else:
            output_name_to_idx[name] = i

    # Detect format: postprocessed model has "scores"/"labels", raw has "logits"
    is_postprocessed = "scores" in output_name_to_idx or "labels" in output_name_to_idx
    logger.info(
        f"Engine I/O: input idx={input_idx}, "
        f"output names={list(output_name_to_idx.keys())}, "
        f"postprocessed={is_postprocessed}"
    )

    all_preds: List[Dict[str, torch.Tensor]] = []
    all_gt: List[Dict[str, torch.Tensor]] = []

    for inputs, targets, _ in tqdm(val_loader, desc="Validating TRT INT8", leave=False):
        inputs_gpu = inputs.to("cuda", dtype=torch.float32).contiguous()
        batch_shape = tuple(inputs_gpu.shape)

        # Allocate output buffers & run
        bindings: list = [None] * engine.num_io_tensors
        outputs_dict: Dict[str, torch.Tensor] = {}

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if i == input_idx:
                context.set_input_shape(name, batch_shape)
                bindings[i] = inputs_gpu.data_ptr()
            else:
                dims = tuple(engine.get_tensor_shape(name))
                out_shape = (batch_shape[0],) + dims[1:]
                dt = engine.get_tensor_dtype(name)
                buf = torch.empty(out_shape, dtype=_trt_to_torch_dtype(dt), device="cuda")
                outputs_dict[name] = buf
                bindings[i] = buf.data_ptr()

        context.execute_v2(bindings)
        orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).float()

        if is_postprocessed:
            # Outputs: labels [B,K] int, boxes [B,K,4] abs xyxy (input-size), scores [B,K]
            labels_t = outputs_dict["labels"]
            boxes_t = outputs_dict["boxes"]
            scores_t = outputs_dict["scores"]
            B = labels_t.shape[0]

            for b in range(B):
                sb, lb, bb = scores_t[b], labels_t[b], boxes_t[b]
                keep = sb >= conf_thresh
                sb, lb, bb = sb[keep], lb[keep], bb[keep].clone()

                # Rescale boxes: input-size abs xyxy -> original-size abs xyxy
                oh, ow = orig_sizes[b]
                ih, iw = float(input_size[0]), float(input_size[1])
                if keep_ratio:
                    gain = min(ih / oh, iw / ow)
                    pad_w = round((iw - ow * gain) / 2 - 0.1)
                    pad_h = round((ih - oh * gain) / 2 - 0.1)
                    bb[:, [0, 2]] -= pad_w
                    bb[:, [1, 3]] -= pad_h
                    bb[:, :4] /= gain
                else:
                    bb[:, [0, 2]] *= ow / iw
                    bb[:, [1, 3]] *= oh / ih
                bb[:, [0, 2]].clamp_(0, ow)
                bb[:, [1, 3]].clamp_(0, oh)

                all_preds.append(
                    {
                        "labels": lb.cpu(),
                        "boxes": bb.cpu(),
                        "scores": sb.cpu(),
                    }
                )
        else:
            # Raw outputs: logits [B,Q,C], boxes [B,Q,4] normalised cxcywh
            model_out = {
                "pred_logits": outputs_dict["logits"],
                "pred_boxes": outputs_dict["boxes"],
            }
            if enable_mask_head and "masks" in outputs_dict:
                model_out["pred_masks"] = outputs_dict["masks"]

            preds = Trainer.preds_postprocess(
                inputs_gpu,
                model_out,
                orig_sizes,
                num_labels=num_labels,
                keep_ratio=keep_ratio,
                conf_thresh=conf_thresh,
            )
            all_preds.extend(preds)

        gt = Trainer.gt_postprocess(inputs_gpu, targets, orig_sizes, keep_ratio=keep_ratio)
        all_gt.extend(gt)

    validator = Validator(
        all_gt,
        all_preds,
        label_to_name=label_to_name,
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
    )
    metrics = validator.compute_metrics(extended=False)
    f1 = metrics["f1"]
    logger.info(f"TRT INT8 validation F1: {f1:.4f}")
    return f1


def _trt_to_torch_dtype(trt_dtype):
    _map = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.int64: torch.int64,
        trt.int8: torch.int8,
    }
    if trt_dtype not in _map:
        raise TypeError(f"Unsupported TensorRT dtype: {trt_dtype}")
    return _map[trt_dtype]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig):
    """
    Build a TensorRT INT8 engine from the ONNX model using the val set for calibration.
    Expects the ONNX file at <save_dir>/model.onnx (the raw one exported by export.py).
    """
    cfg.exp = get_latest_experiment_name(cfg.exp, cfg.train.path_to_save)
    save_dir = Path(cfg.train.path_to_save)

    onnx_path = save_dir / "model.onnx"
    assert onnx_path.exists(), f"ONNX model not found: {onnx_path}"

    enable_mask_head = cfg.task == "segment"
    logger.info(f"ONNX model: {onnx_path}")
    logger.info(f"Task: {cfg.task}, mask head: {enable_mask_head}")

    # ----- Data ----------------------------------------------------------
    base_loader = Loader(
        root_path=Path(cfg.train.data_path),
        img_size=tuple(cfg.train.img_size),
        batch_size=1,  # calibration is typically batch-1
        num_workers=cfg.train.num_workers,
        cfg=cfg,
        debug_img_processing=False,
    )
    _, val_loader, _ = base_loader.build_dataloaders()
    logger.info(f"Val images (calibration set): {len(val_loader.dataset)}")

    # ----- Build engine --------------------------------------------------
    input_shape = (3, *cfg.train.img_size)  # (C, H, W)
    cache_file = save_dir / "trt_int8_calib.cache"
    workspace_gb = getattr(cfg.export, "trt_int8_workspace_gb", 4)

    calibrator = Int8EntropyCalibrator(val_loader, cache_file, input_shape)

    logger.info("Building TensorRT INT8 engine (this may take several minutes) ...")
    engine_bytes = build_int8_engine(
        onnx_path,
        calibrator,
        max_batch_size=cfg.export.max_batch_size,
        workspace_gb=workspace_gb,
    )

    engine_path = save_dir / "model_int8.engine"
    engine_path.write_bytes(engine_bytes)
    logger.info(f"INT8 engine saved: {engine_path}")

    # ----- Validate (optional, controlled by config) ---------------------
    do_validate = getattr(cfg.export, "trt_int8_validate", True)
    if do_validate:
        label_to_name = cfg.train.label_to_name
        # Need a fresh dataloader because calibration exhausted the iterator
        _, fresh_val_loader, _ = base_loader.build_dataloaders()

        f1 = validate_engine(
            engine_path,
            fresh_val_loader,
            num_labels=len(label_to_name),
            keep_ratio=cfg.train.keep_ratio,
            conf_thresh=cfg.train.conf_thresh,
            iou_thresh=cfg.train.iou_thresh,
            label_to_name=label_to_name,
            enable_mask_head=enable_mask_head,
            input_size=tuple(cfg.train.img_size),
        )
        logger.info(f"Final INT8 F1: {f1:.4f}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
