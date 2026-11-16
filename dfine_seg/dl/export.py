import copy
import platform
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import DictConfig
from tabulate import tabulate
from torch import nn

from dfine_seg.api.ckpt import describe, load_and_describe
from dfine_seg.api.loader import read_image
from dfine_seg.config.resolve import CONFIG_NAME, config_dir
from dfine_seg.dl.utils import get_latest_experiment_name, num_kpts
from dfine_seg.model.configs import base_cfg
from dfine_seg.model.dfine import build_model
from dfine_seg.model.utils import ensure_pretrained, unwrap_checkpoint

# litert is paused (litert-torch caps torch<2.13), so it is built only when asked for by name.
DEFAULT_FORMATS = ("onnx", "tensorrt", "openvino", "coreml")
FORMATS = (*DEFAULT_FORMATS, "litert")

# decoder regenerates these from img_size; dropped so re-export at another resolution works
_SIZE_BUFFERS = ("decoder.anchors", "decoder.valid_mask")
_MASK_HEAD = ("decoder.mask_decoder.", "decoder.mask_head.")


class DFINEPostProcessor(nn.Module):
    """Fused detection postprocessor baked into the exported graph.

    Performs: sigmoid -> topK -> cxcywh -> xyxy (in input-size pixels).
    Outputs: labels [B,K], boxes [B,K,4], scores [B,K].
    Masks (if present) are passed through with sigmoid applied. Keypoints (if present) come out
    as [B,K,num_kpts,4]: x, y in input-size pixels, presence and visible scores.
    """

    def __init__(self, num_classes: int, num_top_queries: int = 300, use_focal_loss: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.num_top_queries = num_top_queries
        self.use_focal_loss = use_focal_loss

    @staticmethod
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

    def forward(self, outputs: dict, input_h: int, input_w: int):
        logits = outputs["pred_logits"]  # [B, Q, C]
        boxes = outputs["pred_boxes"]  # [B, Q, 4]  normalised cxcywh
        pred_masks = outputs.get("pred_masks", None)  # [B, Q, Hm, Wm] or None
        pred_kpts = outputs.get("pred_kpts", None)  # [B, Q, num_kpts, 4] or None

        # box conversion: normalised cxcywh -> absolute xyxy in input-size space
        abs_boxes = self.norm_xywh_to_abs_xyxy(boxes.flatten(0, 1), input_h, input_w).view(
            boxes.shape[0], boxes.shape[1], 4
        )

        # score extraction & topK
        if self.use_focal_loss:
            scores_all = torch.sigmoid(logits)  # [B, Q, C]
            flat = scores_all.flatten(1)  # [B, Q*C]
            K = min(self.num_top_queries, flat.shape[1])
            topk_scores, topk_idx = torch.topk(flat, K, dim=-1)  # [B, K]
            topk_labels = topk_idx % self.num_classes  # [B, K]
            topk_qidx = topk_idx // self.num_classes  # [B, K]
        else:
            probs = F.softmax(logits, dim=-1)[:, :, :-1]  # [B, Q, C-1]
            topk_scores, topk_labels = probs.max(dim=-1)  # [B, Q]
            K = min(self.num_top_queries, topk_scores.shape[1])
            topk_scores, order = torch.topk(topk_scores, K, dim=-1)
            topk_labels = topk_labels.gather(1, order)
            topk_qidx = order

        # gather boxes for top-K queries using advanced indexing (CoreML-friendly)
        batch_idx = (
            torch.arange(abs_boxes.shape[0], device=abs_boxes.device)
            .unsqueeze(1)
            .expand_as(topk_qidx)
        )
        topk_boxes = abs_boxes[batch_idx, topk_qidx]  # [B, K, 4]

        result = (topk_labels, topk_boxes, topk_scores)

        if pred_masks is not None:
            # Gather masks for top-K queries: [B, Q, Hm, Wm] -> [B, K, Hm, Wm]
            topk_masks = pred_masks[batch_idx, topk_qidx]
            result = result + (topk_masks,)

        if pred_kpts is not None:
            scale = torch.tensor([input_w, input_h, 1.0, 1.0], device=pred_kpts.device)
            result = result + (pred_kpts[batch_idx, topk_qidx] * scale,)

        return result


class ExportWrapper(nn.Module):
    """Wraps backbone model + postprocessor for ONNX/TRT export."""

    def __init__(self, model: nn.Module, postprocessor: DFINEPostProcessor):
        super().__init__()
        self.model = model
        self.postprocessor = postprocessor

    def forward(self, x):
        outputs = self.model(x)
        return self.postprocessor(outputs, x.shape[2], x.shape[3])


class SemSegExportWrapper(nn.Module):
    """sem_seg export graph: fused argmax -> int32 [B, H, W] label map ("sem_seg").

    int32 keeps TRT happy (no int64 output) and H*W int32 beats C*H*W fp16 logits
    on output bandwidth. One fused graph serves every backend.
    probs=True adds a second output, the softmax of the 1/4 logits [B, C, H/4, W/4]
    ("sem_seg_probs"), for consumers that need per-class confidence.
    """

    def __init__(self, model: nn.Module, probs: bool = False):
        super().__init__()
        self.model = model
        self.probs = probs

    def forward(self, x):
        self.model.decoder.return_quarter_logits = self.probs
        out = self.model(x)
        labels = out["sem_seg_logits"].argmax(1).to(torch.int32)  # [B, H, W]
        if self.probs:
            return labels, out["sem_seg_logits_q"].softmax(1)
        return labels

    def deploy(self):
        self.model.deploy()
        return self


def _task_ok(ckpt_task: str, task: str) -> bool:
    # Segment weights are a superset of detect (same det heads plus an unused mask head), so
    # exporting detect from them is fine; any other task needs heads the weights don't have.
    return task == ckpt_task or (ckpt_task, task) == ("segment", "detect")


def _check_config(
    ckpt: str,
    task: str,
    n_config: int,
    img_size,
    n_kpts: int = 0,
    in_channels: int = 3,
    pretrained: bool = True,
) -> None:
    """Refuse to export weights that don't match the config.

    Export builds what the weights are, so a mismatch here would ship an artifact that
    disagrees with the config bench and infer read (class names, task, input channels).
    Straight from `dfine init` (2 template classes) this used to export a broken model with a
    green parity check.
    """
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    sd, meta = unwrap_checkpoint(state)
    info = describe(sd, meta)
    name = Path(ckpt).name
    if pretrained:
        fix = (
            "export.from_pretrained exports the released weights as-is. Point "
            "train.pretrained_model_path at a matching release (class lists: "
            "dfine_seg.api.coco_names.COCO_NAMES for the COCO detect / segment releases, "
            "{0: person} for the keypoint releases, the Cityscapes sem-seg releases carry "
            "their own), or train first and export with export.from_pretrained=False."
        )
    else:
        fix = f"Export with the config {name} was trained with (the run's frozen config.yaml)."
    if not _task_ok(info["task"], task):
        raise ValueError(f"{name} is a {info['task']} checkpoint but task is {task}. {fix}")
    if info["num_classes"] != n_config:
        raise ValueError(
            f"{name} has {info['num_classes']} classes but train.label_to_name has {n_config}, "
            f"so bench and infer would name the wrong classes. {fix}"
        )
    if info["num_kpts"] != n_kpts:
        raise ValueError(
            f"{name} has {info['num_kpts']} keypoints but train.kpts.shape asks for {n_kpts}. {fix}"
        )
    if info["in_channels"] != in_channels:
        raise ValueError(
            f"{name} takes {info['in_channels']}-channel input but train.in_channels is "
            f"{in_channels}. {fix}"
        )
    meta_img = meta.get("img_size")
    if meta_img and tuple(meta_img) != tuple(img_size):
        logger.warning(
            f"{name} was trained at {meta_img[0]}x{meta_img[1]} but exports at "
            f"{img_size[0]}x{img_size[1]} - the graph adapts, expect some accuracy drop."
        )


def prepare_model(spec: Dict, state_dict: Dict[str, torch.Tensor], device, dynamic: bool = False):
    """Build the architecture `spec` describes (`api.ckpt.describe` keys, with the export's
    task and img_size) and load `state_dict` into it strictly."""
    # dynamic H/W graphs need the img_size=None (per-input) anchor / pos-embed path: a fixed
    # eval_spatial_size bakes its cached constants into the graph and locks it to that size
    model = build_model(
        spec["model_name"],
        spec["num_classes"],
        enable_mask_head=spec["task"] == "segment",
        device=device,
        img_size=None if dynamic else spec["img_size"],
        in_channels=spec["in_channels"],
        task=spec["task"],
        num_kpts=spec["num_kpts"],
        num_kpt_layers=spec["num_kpt_layers"],
    )
    state = {k: v for k, v in state_dict.items() if k not in _SIZE_BUFFERS}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if spec["task"] == "detect":  # detect from segment weights: the mask head stays behind
        unexpected = [k for k in unexpected if not k.startswith(_MASK_HEAD)]
    missing = sorted(set(missing) - set(_SIZE_BUFFERS))
    if missing or unexpected:
        raise ValueError(
            f"checkpoint does not match its {spec['model_name']} {spec['task']} architecture: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    model.eval()
    # Fold each RepVGG 3x3 + 1x1 pair into one conv (same math). Exporters fold BN on their own
    # but keep the two branches: 3 kernels per block instead of 1 in TensorRT.
    for m in model.encoder.modules():
        if hasattr(m, "convert_to_deploy"):
            m.convert_to_deploy()
    return model


def prepare_export_models(
    spec: Dict,
    state_dict: Dict[str, torch.Tensor],
    device,
    sem_seg_probs: bool = False,
    dynamic: bool = False,
) -> tuple[nn.Module, nn.Module]:
    """(raw model, fused export wrapper), both in eval mode."""
    raw_model = prepare_model(spec, state_dict, device, dynamic=dynamic)
    if spec["task"] == "sem_seg":
        model = SemSegExportWrapper(raw_model, probs=sem_seg_probs)
    else:
        postprocessor = DFINEPostProcessor(
            num_classes=spec["num_classes"],
            num_top_queries=base_cfg["DFINETransformer"]["num_queries"],
            use_focal_loss=base_cfg["matcher"]["use_focal_loss"],
        )
        model = ExportWrapper(raw_model, postprocessor)
    return raw_model.eval(), model.eval()


def _int8_path(out: Path, suffix: str) -> Path:
    """`out/model.pt` -> `out/model_int8<suffix>`: where the INT8 variants live."""
    return out.with_name(f"{out.stem}_int8{suffix}")


def export_to_onnx(
    model: nn.Module,
    model_path: Path,
    x_test: torch.Tensor,
    max_batch_size: int,
    half: bool,
    dynamic_input: bool,
    input_name: str,
    output_names: list[str],
) -> Path:
    import onnx
    from onnxconverter_common import float16
    from onnxscript.function_libs.torch_lib.ops import core as onnx_core

    # torch 2.13 runs type promotion after decomposition, which re-introduces aten.mul.Scalar --
    # an overload onnxscript never registers (it relies on decomposition). Map it to Mul.
    custom_translation_table = {torch.ops.aten.mul.Scalar: onnx_core.aten_mul}

    dynamic_axes = {}
    if max_batch_size > 1:
        for name in [input_name] + output_names:
            dynamic_axes[name] = {0: "batch_size"}
    if dynamic_input:
        if input_name not in dynamic_axes:
            dynamic_axes[input_name] = {}
        dynamic_axes[input_name].update({2: "height", 3: "width"})

    output_path = model_path.with_suffix(".onnx")
    torch.onnx.export(
        model,
        x_test,
        opset_version=19,
        input_names=[input_name],
        output_names=output_names,
        dynamic_axes=dynamic_axes if dynamic_axes else None,
        dynamo=True,
        custom_translation_table=custom_translation_table,
    ).save(output_path)

    onnx_model = onnx.load(output_path)
    if dynamic_input:
        # Embed the preprocessing target: a dynamic-H/W graph has no static size to read,
        # and ONNXModel resolves it from here (explicit args still win).
        kv = onnx_model.metadata_props.add()
        kv.key, kv.value = "img_size", f"{x_test.shape[2]}x{x_test.shape[3]}"
    if half:
        onnx_model = float16.convert_float_to_float16(onnx_model, keep_io_types=True)

    try:
        import onnxsim  # Linux-only dependency: elsewhere the graph is saved unsimplified

        onnx_model, check = onnxsim.simplify(onnx_model)
        assert check
        logger.info("ONNX simplified and exported")
    except Exception as e:
        logger.info(f"Simplification failed: {e}")
    finally:
        onnx.save(onnx_model, output_path)
        return output_path


def export_to_openvino(
    onnx_path: Path, xml_path: Path, x_test, dynamic_input: bool, max_batch_size: int
) -> Path:
    import openvino as ov

    channels = int(x_test.shape[1])
    if not dynamic_input and max_batch_size <= 1:
        inp = None
    elif max_batch_size > 1 and dynamic_input:
        inp = [-1, channels, -1, -1]
    elif max_batch_size > 1:
        inp = [-1, *x_test.shape[1:]]
    elif dynamic_input:
        inp = [1, channels, -1, -1]

    model = ov.convert_model(input_model=str(onnx_path), input=inp, example_input=x_test)

    if dynamic_input:
        # ONNX metadata_props don't survive the conversion; carry the preprocessing target
        # in IR rt_info so OVModel can resolve it (explicit args still win).
        model.set_rt_info(f"{x_test.shape[2]}x{x_test.shape[3]}", ["model_info", "img_size"])

    ov.serialize(model, str(xml_path), str(xml_path.with_suffix(".bin")))
    logger.info("OpenVINO model exported")
    return xml_path


class _ExportGroupNorm(nn.Module):
    """GroupNorm rewrite that survives CoreML conversion with a dynamic batch.

    Stock coremltools bakes the batch dim into a constant reshape inside its
    group_norm op, which fails once batch is symbolic (RangeDim). Folding
    batch*groups into a single -1 dim and computing var as E[x^2]-E[x]^2 keeps
    every reshape dynamic-safe and uses only ops CoreML implements.
    """

    def __init__(self, gn: nn.GroupNorm):
        super().__init__()
        self.g, self.eps, self.c = gn.num_groups, gn.eps, gn.num_channels
        self.weight, self.bias = gn.weight, gn.bias

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        xg = x.reshape(-1, (self.c // self.g) * h * w)  # [B*g, C/g*H*W]
        mean = xg.mean(1, keepdim=True)
        var = (xg * xg).mean(1, keepdim=True) - mean * mean
        xg = (xg - mean) / torch.sqrt(var + self.eps)
        x = xg.reshape(-1, self.c, h, w)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


def _swap_group_norm(module: nn.Module) -> None:
    """Recursively replace nn.GroupNorm with the CoreML-safe variant, in place."""
    for name, child in module.named_children():
        if isinstance(child, nn.GroupNorm):
            setattr(module, name, _ExportGroupNorm(child))
        else:
            _swap_group_norm(child)


def export_to_coreml(
    model: nn.Module,
    model_path: Path,
    x_test: torch.Tensor,
    half: bool,
    max_batch_size: int,
) -> Path:
    """Convert PyTorch model to CoreML (.mlpackage) for iOS / macOS.

    Mirrors the TensorRT path: converts the fused model (with postprocessor).
    Uses torch.jit.trace + coremltools PyTorch converter.
    """

    import coremltools as ct
    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig,
        OptimizationConfig,
        linear_quantize_weights,
    )

    compute_precision = ct.precision.FLOAT16 if half else ct.precision.FLOAT32

    # Dynamic batch + the mask head's GroupNorm crash coremltools' converter;
    # swap in a dynamic-safe GroupNorm on a copy so other exports stay untouched.
    if max_batch_size > 1:
        model = copy.deepcopy(model)
        _swap_group_norm(model)

    traced = torch.jit.trace(model.cpu(), x_test.cpu(), strict=False)

    input_shape = list(x_test.shape)
    if max_batch_size > 1:
        ct_inputs = [
            ct.TensorType(
                name="input",
                shape=ct.Shape(
                    shape=[ct.RangeDim(lower_bound=1, upper_bound=max_batch_size), *input_shape[1:]]
                ),
            )
        ]
    else:
        ct_inputs = [ct.TensorType(name="input", shape=input_shape)]

    mlmodel = ct.convert(
        traced,
        inputs=ct_inputs,
        convert_to="mlprogram",
        compute_precision=compute_precision,
        minimum_deployment_target=ct.target.iOS16,
    )

    output_path = model_path.with_suffix(".mlpackage")
    mlmodel.save(str(output_path))
    logger.info("CoreML model exported")

    # INT8 post-training weight quantization
    op_config = OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
    quant_config = OptimizationConfig(global_config=op_config)
    mlmodel_int8 = linear_quantize_weights(mlmodel, config=quant_config)
    mlmodel_int8.save(str(_int8_path(model_path, ".mlpackage")))
    logger.info("CoreML INT8 model exported")
    return output_path


def export_to_litert(
    model: nn.Module,
    model_path: Path,
    x_test: torch.Tensor,
) -> Path:
    """Convert PyTorch model to LiteRT (.tflite) for on-device inference.

    Uses litert_torch for direct PyTorch -> TFLite conversion.
    Exports FP32 and INT8 (weight-only quantization via ai_edge_quantizer) variants.
    Converts raw model dict output to tensor tuple via a local adapter.

    Imports are deferred to avoid tensorflow/flatbuffers native library
    conflicts that cause segfaults during ONNX/TRT export.
    """
    import litert_torch
    from ai_edge_quantizer import quantizer as aeq
    from ai_edge_quantizer.qtyping import QuantGranularity, TFLOperationName

    class _LiteRTRawAdapter(nn.Module):
        def __init__(self, inner: nn.Module):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            outputs = self.inner(x)
            if not isinstance(outputs, dict):  # sem_seg fused wrapper already returns a tensor
                return outputs
            result = (outputs["pred_logits"], outputs["pred_boxes"])
            pred_masks = outputs.get("pred_masks", None)
            if pred_masks is not None:
                result = result + (pred_masks,)
            return result

    sample = x_test[:1].cpu()
    model = model.cpu().eval()
    if hasattr(model, "deploy"):
        model.deploy()
    litert_model = _LiteRTRawAdapter(model)

    edge_model = litert_torch.convert(litert_model, (sample,))
    output_path = model_path.with_suffix(".tflite")
    edge_model.export(str(output_path))
    logger.info("LiteRT model exported")

    # INT8 weight-only quantization on the exported FP32 tflite
    qt = aeq.Quantizer(str(output_path))
    qt.add_weight_only_config(
        regex=".*",
        operation_name=TFLOperationName.ALL_SUPPORTED,
        num_bits=8,
        granularity=QuantGranularity.CHANNELWISE,
    )
    result = qt.quantize()
    with open(_int8_path(model_path, ".tflite"), "wb") as f:
        f.write(result.quantized_model)
    logger.info("LiteRT INT8 model exported")
    return output_path


def export_to_tensorrt(onnx_file_path: Path, half: bool, max_batch_size: int) -> Path:
    import onnx
    import tensorrt as trt

    tr_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(tr_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, tr_logger)

    with open(onnx_file_path, "rb") as model:
        if not parser.parse(model.read()):
            errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"TensorRT could not parse {onnx_file_path}:\n{errors}")

    config = builder.create_builder_config()
    # Increase workspace memory to help with larger batch sizes
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB
    if half:
        config.set_flag(trt.BuilderFlag.FP16)

    if max_batch_size > 1:
        profile = builder.create_optimization_profile()
        input_tensor = network.get_input(0)
        input_name = input_tensor.name

        # Load ONNX model to get the actual input shape information
        onnx_model = onnx.load(str(onnx_file_path))

        # Find the input by name to ensure we get the correct one
        input_shape_proto = None
        for inp in onnx_model.graph.input:
            if inp.name == input_name:
                input_shape_proto = inp.type.tensor_type.shape
                break

        if input_shape_proto is None:
            raise ValueError(
                f"Could not find input '{input_name}' in ONNX model. "
                f"Available inputs: {[inp.name for inp in onnx_model.graph.input]}"
            )

        # Extract static dimensions from ONNX model
        # The first dimension (batch) should be dynamic, others should be static
        static_dims = []
        for i, dim in enumerate(input_shape_proto.dim[1:], start=1):  # Skip batch dimension
            if dim.dim_value:
                # Static dimension
                static_dims.append(int(dim.dim_value))
            elif dim.dim_param:
                # Dynamic dimension (not allowed for non-batch dims in this case)
                raise ValueError(
                    f"Cannot create TensorRT optimization profile: input shape has dynamic "
                    f"dimension at index {i} (beyond batch). Only batch dimension can be dynamic."
                )
            else:
                raise ValueError(
                    f"Cannot create TensorRT optimization profile: input shape dimension at "
                    f"index {i} is undefined."
                )

        # Set the minimum and optimal batch size to 1, and allow the maximum batch size as provided.
        min_shape = (1, *static_dims)
        opt_shape = (1, *static_dims)
        max_shape = (max_batch_size, *static_dims)

        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError(
            "Failed to build TensorRT engine. This can happen due to:\n"
            "1. Insufficient GPU memory\n"
            "2. Unsupported operations in the ONNX model\n"
            "3. Issues with dynamic batch size configuration\n"
            "Check the TensorRT logs above for more details."
        )

    engine_path = onnx_file_path.with_suffix(".engine")
    with open(engine_path, "wb") as f:
        f.write(engine)
    logger.info("TensorRT model exported")
    return engine_path


def _to_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def _cosine(ref, out) -> float:
    """Cosine similarity between two tensors, flattened."""
    a = _to_np(ref).reshape(-1).astype(np.float64)
    b = _to_np(out).reshape(-1).astype(np.float64)
    if a.shape != b.shape:
        return float("nan")
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(a @ b / denom)


@dataclass(frozen=True)
class _Parity:
    """What the parity checks need to know about one export."""

    out: Path  # <output_dir>/<name>.pt: each format swaps in its own suffix
    formats: Tuple[str, ...]
    task: str
    num_classes: int
    img_size: Tuple[int, int]
    half: bool
    dynamic_input: bool
    image: Optional[Path]


def _parity_input(p: _Parity, raw_model, x_test) -> tuple:
    """p.image (fallback: x_test) as (torch [1,C,H,W], numpy) pair."""
    # CoreML/LiteRT export move the shared model params to CPU in place, so run torch on its device
    dev = next(raw_model.parameters()).device
    if p.image is not None:
        import cv2

        img = read_image(p.image)  # BGR for jpg/png, RGB(+extras) for .npy
        if p.image.suffix.lower() != ".npy":
            img = img[:, :, ::-1]
        if img.shape[2] != x_test.shape[1]:
            raise ValueError(
                f"parity image {p.image.name} has {img.shape[2]} channels, the model takes "
                f"{x_test.shape[1]}"
            )
        h, w = p.img_size
        img = cv2.resize(np.ascontiguousarray(img), (w, h))
        x1 = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1), np.float32) / 255.0)[
            None
        ]
        x1 = x1.to(dev)
    else:
        x1 = x_test[:1].to(dev)
    return x1, x1.detach().cpu().numpy().astype(np.float32)


def _build_parity_backend(key: str, path: Path, p: _Parity):
    """num_classes is only for LiteRT: it exports the raw head, so it decodes labels with the
    class count. Every other backend reads what it needs off its own graph. Thresholds and
    keep_ratio stay at defaults: _predict runs on the prepared input and returns raw outputs."""
    common = dict(model_path=str(path), apply_nms=False)
    if key == "onnx":
        from dfine_seg.infer.onnx_model import ONNXModel

        return ONNXModel(rect=False, **common)
    if key == "openvino":
        from dfine_seg.infer.ov_model import OVModel

        return OVModel(
            rect=p.dynamic_input,
            half=p.half and platform.system() != "Darwin",
            max_batch_size=1,
            **common,
        )
    if key == "litert":
        from dfine_seg.infer.litert_model import LiteRTModel

        return LiteRTModel(n_outputs=p.num_classes, rect=False, **common)
    if key == "coreml":
        from dfine_seg.infer.coreml_model import CoreMLModel

        return CoreMLModel(rect=False, **common)
    from dfine_seg.infer.trt_model import TRTModel

    return TRTModel(rect=False, **common)


def _run_parity_backends(p: _Parity, x1, x_np, add, extract) -> None:
    """Shared per-backend scaffolding: build each exported wrapper, run the parity
    input, and feed extract(key, raw_outputs) into add(tag, is_int8, value)."""
    plat = (
        [
            ("coreml", "CoreML", ".mlpackage", False),
            ("coreml", "CoreML INT8", ".mlpackage", True),
        ]
        if platform.system() == "Darwin"
        else [
            ("tensorrt", "TensorRT", ".engine", False),
            ("tensorrt", "TensorRT INT8", ".engine", True),
        ]
    )
    variants = [
        ("onnx", "ONNX", ".onnx", False),
        ("openvino", "OpenVINO", ".xml", False),
        ("openvino", "OpenVINO INT8", ".xml", True),
        ("litert", "LiteRT", ".tflite", False),
        ("litert", "LiteRT INT8", ".tflite", True),
    ] + plat
    for key, tag, suffix, is8 in variants:
        path = _int8_path(p.out, suffix) if is8 else p.out.with_suffix(suffix)
        if not (key in p.formats and path.exists()):
            continue
        try:
            m = _build_parity_backend(key, path, p)
            if key == "tensorrt":
                # Mirror __call__: _predict enqueues on m._stream, but the metric's
                # .cpu() read runs on the default stream - without this the async
                # execute races the output read.
                with torch.cuda.stream(m._stream):
                    x_trt = x1.to(device=m._input_tensor.device, dtype=m._input_tensor.dtype)
                    outs = m._predict(x_trt.contiguous(), actual_batch=1)
                torch.cuda.default_stream(m.device).wait_stream(m._stream)
            else:
                outs = m._predict(x_np)
            add(tag, is8, extract(key, outs))
        except Exception as e:
            logger.warning(f"Parity skipped for {tag}: {e}")


def _parity_report(rows, low, metric_names: list[str], models_path: Path) -> None:
    if not rows:
        logger.info("Parity: no exported backends available to compare against torch")
        return
    headers = ["format", *metric_names]
    pd.DataFrame(rows, columns=headers).to_csv(models_path / "parity.csv", index=False)
    print("\n" + tabulate(rows, headers=headers, tablefmt="pretty"))
    if low:
        logger.warning("Parity below threshold (vs torch): " + ", ".join(low))
    else:
        logger.info("Parity OK: exported backends match torch within threshold")


def run_parity(p: _Parity, raw_model, model, x_test) -> None:
    """Compare each exported backend's detection scores against torch on one input.

    Scores (sorted top-K confidences) are the reorder-stable, end-to-end signal: they
    flow through the whole network, so any silent export drift moves them. Per-query
    boxes/logits/masks are dominated by background queries that never clear conf
    filtering, so comparing them is just noise - bench covers surviving-box geometry.
    """
    x1, x_np = _parity_input(p, raw_model, x_test)
    with torch.no_grad():
        ref = model(x1)[2]  # [1, K] sorted top-K confidences from the fused torch postproc

    n_queries = base_cfg["DFINETransformer"]["num_queries"]
    rows, low = [], []

    def add(name, is_int8, scores):
        cos = _cosine(ref, scores)
        rows.append([name, round(cos, 6)])
        if not cos >= (0.90 if is_int8 else 0.99):
            low.append(f"{name} cos={cos:.4f}")

    def extract(key, outs):
        if key in ("openvino", "litert"):
            # raw backends emit per-query logits; pick by shape (TFLite reorders outputs)
            logits = next(np.asarray(o) for o in outs if np.ndim(o) == 3 and np.shape(o)[-1] != 4)
            flat = torch.sigmoid(torch.from_numpy(logits)).flatten(1)
            return torch.topk(flat, min(n_queries, flat.shape[1]), dim=-1).values
        return outs[2]  # fused: scores at index 2

    _run_parity_backends(p, x1, x_np, add, extract)
    _parity_report(rows, low, ["scores_cosine"], p.out.parent)


def _sem_seg_outputs(outs) -> list[np.ndarray]:
    """Image 0 of a sem_seg graph's outputs: [label map [H, W], probs [C, h, w] if exported].
    Sorted by rank, not by position: TFLite may reorder outputs."""
    outs = [outs] if isinstance(outs, torch.Tensor) else outs
    return sorted((_to_np(o)[0] for o in outs), key=np.ndim)


def run_parity_sem_seg(p: _Parity, raw_model, model, x_test) -> None:
    """Per-pixel argmax agreement of each exported backend vs torch on one input, plus the
    probs cosine when the graph has the sem_seg_probs output.

    Every sem_seg export is the same fused graph, so the comparison is end-to-end and
    reorder-free by construction; disagreement is confined to class-boundary pixels where
    near-tied logits flip under precision.
    """
    x1, x_np = _parity_input(p, raw_model, x_test)
    with torch.no_grad():
        ref = _sem_seg_outputs(model(x1))

    rows, low = [], []

    def add(name, is_int8, outs):
        thresh = 0.90 if is_int8 else 0.99
        agree = float((outs[0].reshape(ref[0].shape) == ref[0]).mean())
        row = [name, round(agree, 6)]
        if not agree >= thresh:
            low.append(f"{name} agreement={agree:.4f}")
        if len(ref) > 1:
            cos = _cosine(ref[1], outs[1]) if len(outs) > 1 else float("nan")
            row.append(round(cos, 6))
            if not cos >= thresh:
                low.append(f"{name} probs_cosine={cos:.4f}")
        rows.append(row)

    metrics = ["pixel_agreement"] + (["probs_cosine"] if len(ref) > 1 else [])
    _run_parity_backends(p, x1, x_np, add, lambda _, o: _sem_seg_outputs(o))
    _parity_report(rows, low, metrics, p.out.parent)


def run_dynamic_size_parity(p: _Parity, raw_model, model, x_test) -> None:
    """Regression guard for dynamic_input: ONNX must track torch at a second size.

    Standard parity runs at the training resolution, where a graph with baked size
    constants still produces correct output. Rescaling the parity input by 3/4
    (stride-aligned) fails on any baked positional embedding, anchor, or box scale.
    `model` must be the img_size=None export model, so torch itself follows the input size.
    """
    if not (p.dynamic_input and "onnx" in p.formats):
        return
    import onnxruntime as ort
    from torchvision.ops import box_iou

    onnx_path = p.out.with_suffix(".onnx")
    if not onnx_path.exists():
        return
    h, w = p.img_size
    h2, w2 = max(320, round(h * 0.75 / 32) * 32), max(320, round(w * 0.75 / 32) * 32)
    if (h2, w2) == (h, w):
        return

    x1, _ = _parity_input(p, raw_model, x_test)
    x_small = F.interpolate(x1, size=(h2, w2), mode="bilinear", align_corners=False)
    with torch.no_grad():
        ref = model(x_small)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    try:
        outs = sess.run(None, {"input": x_small.detach().cpu().numpy().astype(np.float32)})
    except Exception as e:
        raise RuntimeError(
            f"{onnx_path.name} has dynamic H/W axes but fails at {h2}x{w2}: a size is still "
            f"baked into the graph. {e}"
        ) from e

    if p.task == "sem_seg":
        ref, outs = _sem_seg_outputs(ref), _sem_seg_outputs(outs)
        metrics = {"pixel_agreement": float((outs[0] == ref[0]).mean())}
        if len(ref) > 1:
            metrics["probs_cosine"] = _cosine(ref[1], outs[1])
    else:
        ref = [t.cpu() for t in ref]  # (labels, boxes, scores[, masks])
        ref_scores = torch.sort(ref[2].flatten()).values.numpy()
        metrics = {"scores_cosine": _cosine(ref_scores, np.sort(outs[2].flatten()))}
        # Scores alone miss a baked box scale. Match each confident torch box to its best
        # same-label ONNX box over all queries, which is robust to near-tied rank swaps.
        keep = ref[2][0] >= 0.5  # the confident torch boxes
        if keep.any():
            iou = box_iou(ref[1][0][keep], torch.from_numpy(outs[1][0]))
            same = ref[0][0][keep][:, None] == torch.from_numpy(outs[0][0])[None]
            metrics["box_iou"] = float((iou * same).max(1).values.mean())

    msg = f"Dynamic size parity at {h2}x{w2}: " + ", ".join(
        f"{k}={v:.4f}" for k, v in metrics.items()
    )
    if all(v >= 0.95 if k == "box_iou" else v >= 0.99 for k, v in metrics.items()):
        logger.info(msg)
    else:
        logger.warning(msg + " (below threshold: 0.99, box_iou 0.95)")


def _resolve(ckpt: Path, task: Optional[str], img_size) -> Tuple[Dict[str, torch.Tensor], Dict]:
    """-> (state_dict, spec): the checkpoint's own facts with the export's task and img_size.
    img_size: explicit argument, else checkpoint meta, else sidecar config.yaml, else 640."""
    state_dict, spec = load_and_describe(ckpt)
    if task is not None:
        if not _task_ok(spec["task"], task):
            raise ValueError(f"{ckpt.name} is a {spec['task']} checkpoint, not {task}")
        spec["task"] = task
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    spec["img_size"] = tuple(int(v) for v in img_size or spec["img_size"] or (640, 640))
    return state_dict, spec


def export(
    ckpt: str | Path,
    formats: str | Iterable[str] | None = ("onnx",),
    *,
    half: bool = True,
    max_batch_size: int = 1,
    img_size: int | Tuple[int, int] | None = None,
    task: Optional[str] = None,
    dynamic_input: bool = False,
    sem_seg_probs: bool = False,
    device: Optional[str] = None,
    output_dir: str | Path | None = None,
    name: Optional[str] = None,
    parity: bool = False,
    parity_image: str | Path | None = None,
) -> Dict[str, Path]:
    """Export a `.pt` checkpoint, no config needed; -> {format: artifact path}.

    ```python
    from dfine_seg import export

    paths = export("weights/coco_s.pt", ["tensorrt"])  # {"tensorrt": weights/coco_s.engine}
    ```

    The architecture is read from the weights. Input size (h, w): `img_size` (an int means
    square), else the checkpoint's metadata, else a `config.yaml` next to it, else 640x640.
    `task` defaults to the checkpoint's own; the one alternative is "detect" from segment
    weights. A missing released checkpoint name is downloaded, as in `load_model`.

    formats: any of onnx, tensorrt, openvino, coreml, litert (paused); None = all but litert.
    half: fp16 TensorRT engine.
    max_batch_size: > 1 gives the graphs a dynamic batch axis up to this size.
    dynamic_input: dynamic H/W for ONNX and OpenVINO; TensorRT and CoreML stay static.
    sem_seg_probs: sem_seg only, adds the 1/4-scale softmax output.
    output_dir, name: artifacts are `<output_dir>/<name>.<ext>`; defaults are the checkpoint's
        directory and stem, so `weights/a.pt` -> `weights/a.onnx`, `weights/a.engine`.
        TensorRT leaves its source ONNX there too.
    parity: compare each artifact with torch on `parity_image` (jpg/png, or an RGB(+extras)
        .npy; default: random input), print the table and write `parity.csv` next to the
        artifacts. Low agreement only warns; a dynamic-H/W ONNX graph that fails at a second
        size raises.

    Raises if a requested format is not produced, e.g. TensorRT builds no engine.
    """
    if formats is None:
        formats = DEFAULT_FORMATS
    formats = (formats,) if isinstance(formats, str) else tuple(formats)
    if unknown := [f for f in formats if f not in FORMATS]:
        raise ValueError(f"unknown export format(s) {unknown}; choose from {FORMATS}")
    ckpt = Path(ensure_pretrained(ckpt))
    state_dict, spec = _resolve(ckpt, task, img_size)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # ".pt" is a placeholder suffix: every artifact swaps in its own (dots in `name` survive)
    out = Path(ckpt.parent if output_dir is None else output_dir) / f"{name or ckpt.stem}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)

    input_name = "input"
    sem_seg = spec["task"] == "sem_seg"
    enable_mask_head = spec["task"] == "segment"
    output_names = ["sem_seg"] if sem_seg else ["labels", "boxes", "scores"]
    if sem_seg and sem_seg_probs:
        output_names.append("sem_seg_probs")
    if enable_mask_head:
        output_names.append("masks")
    if spec["task"] == "kpts":
        output_names.append("kpts")

    # TensorRT, CoreML and LiteRT are always static, fixed size for them.
    # dynamic_input adds an img_size=None model used only for the ONNX and OpenVINO
    raw_model, model = prepare_export_models(spec, state_dict, device, sem_seg_probs)
    dyn_raw, dyn_model = raw_model, model
    if dynamic_input:
        dyn_raw, dyn_model = prepare_export_models(
            spec, state_dict, device, sem_seg_probs, dynamic=True
        )

    x_test = torch.randn(max_batch_size, spec["in_channels"], *spec["img_size"]).to(device)
    _ = model(x_test)

    paths: Dict[str, Path] = {}
    # Openvino currently doesn't supprort some operations in the detection postprocessor,
    # so it gets the raw head; sem_seg's fused argmax graph is OV-supported as-is.
    if "openvino" in formats:
        raw_output_names = ["logits", "boxes"]
        if enable_mask_head:
            raw_output_names.append("masks")
        if spec["task"] == "kpts":
            raw_output_names.append("kpts")
        # The raw-head ONNX is only OpenVINO's input. Saved as <name>.onnx it would pass for
        # the fused graph (and ONNXModel would read its two outputs as sem_seg).
        with tempfile.TemporaryDirectory() as tmp:
            raw_onnx_path = export_to_onnx(
                dyn_model if sem_seg else dyn_raw,
                Path(tmp) / out.name,  # same file name: OpenVINO records it in the IR
                x_test,
                max_batch_size,
                half=False,
                dynamic_input=dynamic_input,
                input_name=input_name,
                output_names=output_names if sem_seg else raw_output_names,
            )
            paths["openvino"] = export_to_openvino(
                raw_onnx_path, out.with_suffix(".xml"), x_test, dynamic_input, max_batch_size=1
            )

    # TensorRT is always static, so it is built from a static model.onnx. With dynamic_input,
    # model.onnx is then re-exported as the dynamic graph (same path, replaces the static one).
    if "tensorrt" in formats or ("onnx" in formats and not dynamic_input):
        paths["onnx"] = export_to_onnx(
            model,
            out,
            x_test,
            max_batch_size,
            half=False,
            dynamic_input=False,
            input_name=input_name,
            output_names=output_names,
        )
        if "tensorrt" in formats:
            paths["tensorrt"] = export_to_tensorrt(paths["onnx"], half, max_batch_size)
    if "onnx" in formats and dynamic_input:
        paths["onnx"] = export_to_onnx(
            dyn_model,
            out,
            x_test,
            max_batch_size,
            half=False,
            dynamic_input=True,
            input_name=input_name,
            output_names=output_names,
        )

    if "coreml" in formats:
        paths["coreml"] = export_to_coreml(
            model, out, x_test, half=False, max_batch_size=max_batch_size
        )

    if "litert" in formats:
        # sem_seg ships the fused wrapper (adapter passes non-dict outputs through)
        paths["litert"] = export_to_litert(model if sem_seg else raw_model, out, x_test)

    logger.info(f"Exports saved to: {out.parent}")

    if parity:
        p = _Parity(
            out=out,
            formats=formats,
            task=spec["task"],
            num_classes=spec["num_classes"],
            img_size=spec["img_size"],
            half=half,
            dynamic_input=dynamic_input,
            image=Path(parity_image) if parity_image is not None else None,
        )
        (run_parity_sem_seg if sem_seg else run_parity)(p, raw_model, model, x_test)
        run_dynamic_size_parity(p, dyn_raw, dyn_model, x_test)
    return {f: paths[f] for f in formats}


def run_from_config(cfg: DictConfig) -> Dict[str, Path]:
    """`dfine export`: maps the config onto export(), writing into train.path_to_save."""
    pretrained = cfg.export.from_pretrained
    if pretrained:
        # A released checkpoint, no trained run: export into today's fresh output dir.
        ckpt = ensure_pretrained(cfg.train.pretrained_model_path)
    else:
        cfg.exp = get_latest_experiment_name(cfg.exp, cfg.train.path_to_save)
        ckpt = Path(cfg.train.path_to_save) / "model.pt"
        if not ckpt.exists():
            raise FileNotFoundError(
                f"{ckpt} not found. Train first, or set export.from_pretrained=True "
                "to export pretrained weights directly."
            )
    _check_config(
        ckpt,
        cfg.task,
        len(cfg.train.label_to_name),
        cfg.train.img_size,
        n_kpts=num_kpts(cfg),
        in_channels=cfg.train.in_channels,
        pretrained=pretrained,
    )

    # null = all shipped backends; a list (e.g. [tensorrt]) restricts what is built (research loop).
    formats = cfg.export.get("formats", None)
    # a real training image anchors parity on true predictions (randn has none)
    img_dir = Path(cfg.train.data_path) / "images"
    images = sorted(img_dir.glob("*.jpg")) or sorted(img_dir.glob("*.png"))
    return export(
        ckpt,
        None if formats is None else list(formats),
        half=cfg.export.half,
        max_batch_size=cfg.export.max_batch_size,
        img_size=cfg.train.img_size,
        task=cfg.task,
        dynamic_input=cfg.export.dynamic_input,
        sem_seg_probs=cfg.export.get("sem_seg_probs", False),
        device=cfg.train.device,
        output_dir=cfg.train.path_to_save,
        name="model",
        parity=cfg.export.get("parity", True),
        parity_image=images[0] if images else None,
    )


def main():
    # Decorated on call, not at import: `from dfine_seg import export` must not run Hydra's
    # config discovery (a set-but-wrong $DFINE_SEG_CONFIG_DIR raises there).
    hydra.main(version_base=None, config_path=config_dir(), config_name=CONFIG_NAME)(
        run_from_config
    )()


if __name__ == "__main__":
    main()
