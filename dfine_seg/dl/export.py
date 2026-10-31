import copy
import platform
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import DictConfig
from tabulate import tabulate
from torch import nn

from dfine_seg.api.ckpt import describe
from dfine_seg.config.resolve import CONFIG_NAME, config_dir
from dfine_seg.dl.utils import get_latest_experiment_name, num_kpts
from dfine_seg.model.configs import base_cfg
from dfine_seg.model.dfine import build_model
from dfine_seg.model.utils import (
    ensure_pretrained,
    load_tuning_state,
    unwrap_checkpoint,
)


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


def _check_pretrained(ckpt: str, task: str, n_config: int, img_size, n_kpts: int = 0) -> None:
    """Refuse to export pretrained weights that would produce a broken artifact.

    `load_tuning_state` loads non-strictly, so a class-count mismatch drops every score head
    and a task mismatch leaves the other task's heads at random init - and parity then
    compares that corrupt model against graphs exported from it, so the export "passes".
    Straight from `dfine init` (2 template classes) this produced a green check on a broken
    model.
    """
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    sd, meta = unwrap_checkpoint(state)
    info = describe(sd, meta)
    # Segment weights are a superset of detect (same det heads plus an unused fuser), so
    # exporting detect from them is fine; every other mismatch exports random heads.
    if info["task"] != task and not (info["task"] == "segment" and task == "detect"):
        raise ValueError(
            f"{Path(ckpt).name} is a {info['task']} checkpoint but task is {task}. "
            "export.from_pretrained exports the released weights as-is, so the other task's "
            "heads would be random init and parity would still pass. Point "
            "train.pretrained_model_path at a matching release, or train first and export "
            "with export.from_pretrained=False."
        )
    if info["num_classes"] != n_config:
        raise ValueError(
            f"{Path(ckpt).name} has {info['num_classes']} classes but train.label_to_name has "
            f"{n_config}. export.from_pretrained cannot reconcile that: the mismatched heads "
            "would be dropped and exported at random init. Either list the checkpoint's "
            "classes (dfine_seg.api.coco_names.COCO_NAMES for the COCO releases; the "
            "Cityscapes sem-seg releases carry their own label_to_name), or train on your own "
            "classes first and export with export.from_pretrained=False."
        )
    if info["num_kpts"] != n_kpts:
        raise ValueError(
            f"{Path(ckpt).name} has {info['num_kpts']} keypoints but train.kpts.shape "
            f"asks for {n_kpts}; the keypoint head would be exported at random init."
        )
    meta_img = meta.get("img_size")
    if meta_img and tuple(meta_img) != tuple(img_size):
        logger.warning(
            f"{Path(ckpt).name} was trained at {meta_img[0]}x{meta_img[1]} but exports at "
            f"{img_size[0]}x{img_size[1]} - the graph adapts, expect some accuracy drop."
        )


def prepare_model(cfg, device, dynamic: bool = False):
    # dynamic H/W graphs need the img_size=None (per-input) anchor / pos-embed path: a fixed
    # eval_spatial_size bakes its cached constants into the graph and locks it to that size
    model = build_model(
        cfg.model_name,
        len(cfg.train.label_to_name),
        enable_mask_head=cfg.task == "segment",
        device=device,
        img_size=None if dynamic else cfg.train.img_size,
        in_channels=cfg.train.in_channels,
        task=cfg.task,
        num_kpts=num_kpts(cfg),
    )
    if cfg.export.from_pretrained:
        # Export a released checkpoint directly (no trained model.pt).
        ckpt = ensure_pretrained(cfg.train.pretrained_model_path)
        _check_pretrained(
            ckpt,
            cfg.task,
            len(cfg.train.label_to_name),
            cfg.train.img_size,
            n_kpts=num_kpts(cfg),
        )
        load_tuning_state(model, ckpt)
    else:
        ckpt = Path(cfg.train.path_to_save) / "model.pt"
        if not ckpt.exists():
            raise FileNotFoundError(
                f"{ckpt} not found. Train first, or set export.from_pretrained=True "
                "to export pretrained weights directly."
            )
        state = unwrap_checkpoint(torch.load(ckpt, weights_only=True))[0]
        # decoder regenerates these from img_size; drop so re-export at another resolution works
        for k in ("decoder.anchors", "decoder.valid_mask"):
            state.pop(k, None)
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not unexpected and set(missing) <= {"decoder.anchors", "decoder.valid_mask"}, (
            f"unexpected checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    model.eval()
    return model


def prepare_export_models(cfg, device, dynamic: bool = False) -> tuple[nn.Module, nn.Module]:
    """(raw model, fused export wrapper), both in eval mode."""
    raw_model = prepare_model(cfg, device, dynamic=dynamic)
    if cfg.task == "sem_seg":
        model = SemSegExportWrapper(raw_model, probs=cfg.export.get("sem_seg_probs", False))
    else:
        postprocessor = DFINEPostProcessor(
            num_classes=len(cfg.train.label_to_name),
            num_top_queries=base_cfg["DFINETransformer"]["num_queries"],
            use_focal_loss=base_cfg["matcher"]["use_focal_loss"],
        )
        model = ExportWrapper(raw_model, postprocessor)
    return raw_model.eval(), model.eval()


def export_to_onnx(
    model: nn.Module,
    model_path: Path,
    x_test: torch.Tensor,
    max_batch_size: int,
    half: bool,
    dynamic_input: bool,
    input_name: str,
    output_names: list[str],
) -> None:
    import onnx
    import onnxsim
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
        onnx_model, check = onnxsim.simplify(onnx_model)
        assert check
        logger.info("ONNX simplified and exported")
    except Exception as e:
        logger.info(f"Simplification failed: {e}")
    finally:
        onnx.save(onnx_model, output_path)
        return output_path


def export_to_openvino(onnx_path: Path, x_test, dynamic_input: bool, max_batch_size: int) -> None:
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

    ov.serialize(model, str(onnx_path.with_suffix(".xml")), str(onnx_path.with_suffix(".bin")))
    logger.info("OpenVINO model exported")


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
) -> None:
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
    int8_path = model_path.with_name("model_int8").with_suffix(".mlpackage")
    mlmodel_int8.save(str(int8_path))
    logger.info("CoreML INT8 model exported")


def export_to_litert(
    model: nn.Module,
    model_path: Path,
    x_test: torch.Tensor,
) -> None:
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
    int8_path = model_path.with_name("model_int8").with_suffix(".tflite")
    with open(int8_path, "wb") as f:
        f.write(result.quantized_model)
    logger.info("LiteRT INT8 model exported")


def export_to_tensorrt(
    onnx_file_path: Path,
    half: bool,
    max_batch_size: int,
    opt_bs: int = 1,
    fp32_grid_sample: bool = False,
) -> None:
    import onnx
    import tensorrt as trt

    opt_bs = min(opt_bs, max_batch_size)

    tr_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(tr_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, tr_logger)

    with open(onnx_file_path, "rb") as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return

    config = builder.create_builder_config()
    # Increase workspace memory to help with larger batch sizes
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB
    if half:
        config.set_flag(trt.BuilderFlag.FP16)
        if fp32_grid_sample:
            # The keypoint decoder's deformable attention is mis-fused in fp16 (TRT 10.13):
            # kpt AP 0.19 vs 0.29 in torch; fp32 GridSample matches torch at the same latency.
            # Pins the box decoder's GridSamples too: keypoint-decoder-only gave the same AP
            # and latency, and the simplified graph carries no module names to select by.
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            for i in range(network.num_layers):
                layer = network.get_layer(i)
                if layer.type == trt.LayerType.GRID_SAMPLE:
                    layer.precision = trt.float32
                    layer.set_output_type(0, trt.float32)

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

    with open(onnx_file_path.with_suffix(".engine"), "wb") as f:
        f.write(engine)
    logger.info("TensorRT model exported")


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


def _parity_input(cfg, raw_model, x_test) -> tuple:
    """One real training image (fallback: x_test) as (torch [1,C,H,W], numpy) pair."""
    # CoreML/LiteRT export move the shared model params to CPU in place, so run torch on its device
    dev = next(raw_model.parameters()).device
    # a real training image anchors parity on true predictions (randn has none)
    img_dir = Path(cfg.train.data_path) / "images"
    imgs = sorted(img_dir.glob("*.jpg")) or sorted(img_dir.glob("*.png"))
    if imgs:
        import cv2

        h, w = cfg.train.img_size
        img = cv2.resize(cv2.imread(str(imgs[0])), (w, h))[:, :, ::-1]  # BGR->RGB
        x1 = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1), np.float32) / 255.0)[
            None
        ]
        x1 = x1.to(dev)
    else:
        x1 = x_test[:1].to(dev)
    return x1, x1.detach().cpu().numpy().astype(np.float32)


def _build_parity_backend(key: str, path: Path, cfg, n_out: int):
    """n_out is only for LiteRT: it exports the raw head, so it decodes labels with the
    class count. Every other backend reads what it needs off its own graph."""
    conf, kr = cfg.train.conf_thresh, cfg.train.keep_ratio
    common = dict(model_path=str(path), conf_thresh=conf, keep_ratio=kr, apply_nms=False)
    if key == "onnx":
        from dfine_seg.infer.onnx_model import ONNXModel

        return ONNXModel(rect=False, **common)
    if key == "openvino":
        from dfine_seg.infer.ov_model import OVModel

        return OVModel(
            rect=cfg.export.dynamic_input,
            half=cfg.export.half and platform.system() != "Darwin",
            max_batch_size=1,
            **common,
        )
    if key == "litert":
        from dfine_seg.infer.litert_model import LiteRTModel

        return LiteRTModel(n_outputs=n_out, rect=False, **common)
    if key == "coreml":
        from dfine_seg.infer.coreml_model import CoreMLModel

        return CoreMLModel(rect=False, **common)
    from dfine_seg.infer.trt_model import TRTModel

    return TRTModel(rect=False, **common)


def _run_parity_backends(cfg, x1, x_np, want, models_path: Path, add, extract) -> None:
    """Shared per-backend scaffolding: build each exported wrapper, run the parity
    input, and feed extract(key, raw_outputs) into add(tag, is_int8, value)."""
    n_out = len(cfg.train.label_to_name)
    plat = (
        [
            ("coreml", "CoreML", "model.mlpackage", False),
            ("coreml", "CoreML INT8", "model_int8.mlpackage", True),
        ]
        if platform.system() == "Darwin"
        else [
            ("tensorrt", "TensorRT", "model.engine", False),
            ("tensorrt", "TensorRT INT8", "model_int8.engine", True),
        ]
    )
    variants = [
        ("onnx", "ONNX", "model.onnx", False),
        ("openvino", "OpenVINO", "model.xml", False),
        ("openvino", "OpenVINO INT8", "model_int8.xml", True),
        ("litert", "LiteRT", "model.tflite", False),
        ("litert", "LiteRT INT8", "model_int8.tflite", True),
    ] + plat
    for key, tag, fn, is8 in variants:
        if not (want(key) and (models_path / fn).exists()):
            continue
        try:
            m = _build_parity_backend(key, models_path / fn, cfg, n_out)
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


def run_parity(cfg, raw_model, model, x_test, want, models_path: Path) -> None:
    """Compare each exported backend's detection scores against torch on one input.

    Scores (sorted top-K confidences) are the reorder-stable, end-to-end signal: they
    flow through the whole network, so any silent export drift moves them. Per-query
    boxes/logits/masks are dominated by background queries that never clear conf
    filtering, so comparing them is just noise - bench covers surviving-box geometry.
    """
    x1, x_np = _parity_input(cfg, raw_model, x_test)
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

    _run_parity_backends(cfg, x1, x_np, want, models_path, add, extract)
    _parity_report(rows, low, ["scores_cosine"], models_path)


def _sem_seg_outputs(outs) -> list[np.ndarray]:
    """Image 0 of a sem_seg graph's outputs: [label map [H, W], probs [C, h, w] if exported].
    Sorted by rank, not by position: TFLite may reorder outputs."""
    outs = [outs] if isinstance(outs, torch.Tensor) else outs
    return sorted((_to_np(o)[0] for o in outs), key=np.ndim)


def run_parity_sem_seg(cfg, raw_model, model, x_test, want, models_path: Path) -> None:
    """Per-pixel argmax agreement of each exported backend vs torch on one input, plus the
    probs cosine when the graph has the sem_seg_probs output.

    Every sem_seg export is the same fused graph, so the comparison is end-to-end and
    reorder-free by construction; disagreement is confined to class-boundary pixels where
    near-tied logits flip under precision.
    """
    x1, x_np = _parity_input(cfg, raw_model, x_test)
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
    _run_parity_backends(cfg, x1, x_np, want, models_path, add, lambda _, o: _sem_seg_outputs(o))
    _parity_report(rows, low, metrics, models_path)


def run_dynamic_size_parity(cfg, raw_model, model, x_test, want, models_path: Path) -> None:
    """Regression guard for export.dynamic_input: ONNX must track torch at a second size.

    Standard parity runs at the training resolution, where a graph with baked size
    constants still produces correct output. Rescaling the parity input by 3/4
    (stride-aligned) fails on any baked positional embedding, anchor, or box scale.
    `model` must be the img_size=None export model, so torch itself follows the input size.
    """
    if not (cfg.export.dynamic_input and want("onnx")):
        return
    import onnxruntime as ort
    from torchvision.ops import box_iou

    onnx_path = models_path / "model.onnx"
    if not onnx_path.exists():
        return
    h, w = cfg.train.img_size
    h2, w2 = max(320, round(h * 0.75 / 32) * 32), max(320, round(w * 0.75 / 32) * 32)
    if (h2, w2) == (h, w):
        return

    x1, _ = _parity_input(cfg, raw_model, x_test)
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

    if cfg.task == "sem_seg":
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
        keep = ref[2][0] >= cfg.train.conf_thresh
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


@hydra.main(version_base=None, config_path=config_dir(), config_name=CONFIG_NAME)
def main(cfg: DictConfig):
    input_name = "input"
    sem_seg = cfg.task == "sem_seg"
    enable_mask_head = cfg.task == "segment"
    output_names = ["sem_seg"] if sem_seg else ["labels", "boxes", "scores"]
    if sem_seg and cfg.export.get("sem_seg_probs", False):
        output_names.append("sem_seg_probs")
    if enable_mask_head:
        output_names.append("masks")
    if cfg.task == "kpts":
        output_names.append("kpts")

    device = cfg.train.device
    # from_pretrained needs no trained run - it writes into today's fresh output dir.
    if not cfg.export.from_pretrained:
        cfg.exp = get_latest_experiment_name(cfg.exp, cfg.train.path_to_save)

    # from_pretrained exports before any training run, so the output dir may not exist yet
    Path(cfg.train.path_to_save).mkdir(parents=True, exist_ok=True)
    model_path = Path(cfg.train.path_to_save) / "model.pt"

    # TensorRT, CoreML and LiteRT are always static, fixed size for them.
    # dynamic_input adds an img_size=None model used only for the ONNX and OpenVINO
    raw_model, model = prepare_export_models(cfg, device)
    dyn_raw, dyn_model = raw_model, model
    if cfg.export.dynamic_input:
        dyn_raw, dyn_model = prepare_export_models(cfg, device, dynamic=True)

    x_test = torch.randn(cfg.export.max_batch_size, cfg.train.in_channels, *cfg.train.img_size).to(
        device
    )
    _ = model(x_test)

    # null = all shipped backends; a list (e.g. [tensorrt]) restricts what is built (research loop).
    # litert is paused (litert-torch caps torch<2.13), so it is built only when asked for by name.
    default_formats = ["onnx", "tensorrt", "openvino", "coreml"]
    formats = cfg.export.get("formats", None)
    want = (lambda f: f in default_formats) if formats is None else (lambda f: f in formats)

    # Openvino currently doesn't supprort some operations in the detection postprocessor,
    # so it gets the raw head; sem_seg's fused argmax graph is OV-supported as-is.
    if want("openvino"):
        raw_output_names = ["logits", "boxes"]
        if enable_mask_head:
            raw_output_names.append("masks")
        if cfg.task == "kpts":
            raw_output_names.append("kpts")
        raw_onnx_path = export_to_onnx(
            dyn_model if sem_seg else dyn_raw,
            model_path,
            x_test,
            cfg.export.max_batch_size,
            half=False,
            dynamic_input=cfg.export.dynamic_input,
            input_name=input_name,
            output_names=output_names if sem_seg else raw_output_names,
        )
        export_to_openvino(raw_onnx_path, x_test, cfg.export.dynamic_input, max_batch_size=1)

    # TensorRT is always static, so it is built from a static model.onnx. With dynamic_input,
    # model.onnx is then re-exported as the dynamic graph (same path, replaces the static one).
    if want("tensorrt") or (want("onnx") and not cfg.export.dynamic_input):
        full_onnx_path = export_to_onnx(
            model,
            model_path,
            x_test,
            cfg.export.max_batch_size,
            half=False,
            dynamic_input=False,
            input_name=input_name,
            output_names=output_names,
        )
        if want("tensorrt"):
            export_to_tensorrt(
                full_onnx_path,
                cfg.export.half,
                cfg.export.max_batch_size,
                opt_bs=cfg.export.get("opt_batch_size", 1) or 1,
                fp32_grid_sample=cfg.task == "kpts",
            )
    if want("onnx") and cfg.export.dynamic_input:
        export_to_onnx(
            dyn_model,
            model_path,
            x_test,
            cfg.export.max_batch_size,
            half=False,
            dynamic_input=True,
            input_name=input_name,
            output_names=output_names,
        )

    if want("coreml"):
        export_to_coreml(
            model, model_path, x_test, half=False, max_batch_size=cfg.export.max_batch_size
        )

    if want("litert"):
        # sem_seg ships the fused wrapper (adapter passes non-dict outputs through)
        export_to_litert(model if sem_seg else raw_model, model_path, x_test)

    logger.info(f"Exports saved to: {model_path.parent}")

    if cfg.export.get("parity", True):
        parity_fn = run_parity_sem_seg if sem_seg else run_parity
        parity_fn(cfg, raw_model, model, x_test, want, model_path.parent)
        run_dynamic_size_parity(cfg, dyn_raw, dyn_model, x_test, want, model_path.parent)


if __name__ == "__main__":
    main()
