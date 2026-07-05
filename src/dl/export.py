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

from src.d_fine.configs import base_cfg
from src.d_fine.dfine import build_model
from src.d_fine.utils import ensure_pretrained, load_tuning_state
from src.dl.utils import get_latest_experiment_name


class DFINEPostProcessor(nn.Module):
    """Fused detection postprocessor baked into the exported graph.

    Performs: sigmoid -> topK -> cxcywh -> xyxy (in input-size pixels).
    Outputs: labels [B,K], boxes [B,K,4], scores [B,K].
    Masks (if present) are passed through with sigmoid applied.
    """

    def __init__(self, num_classes: int, num_top_queries: int = 300, use_focal_loss: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.num_top_queries = num_top_queries
        self.use_focal_loss = use_focal_loss

    @staticmethod
    def norm_xywh_to_abs_xyxy(
        boxes: torch.Tensor, height: int, width: int, to_round=True
    ) -> torch.Tensor:
        """Converts boxes: [N, 4] normalized xywh -> [N, 4] absolute xyxy"""
        x_center = boxes[:, 0] * width
        y_center = boxes[:, 1] * height
        box_width = boxes[:, 2] * width
        box_height = boxes[:, 3] * height

        x_min = x_center - (box_width / 2)
        y_min = y_center - (box_height / 2)
        x_max = x_center + (box_width / 2)
        y_max = y_center + (box_height / 2)

        if to_round:
            x_min = torch.clamp(torch.floor(x_min), min=0, max=width - 1)
            y_min = torch.clamp(torch.floor(y_min), min=0, max=height - 1)
            x_max = torch.clamp(torch.ceil(x_max), min=0, max=width - 1)
            y_max = torch.clamp(torch.ceil(y_max), min=0, max=height - 1)
        else:
            x_min = torch.clamp(x_min, min=0, max=width)
            y_min = torch.clamp(y_min, min=0, max=height)
            x_max = torch.clamp(x_max, min=0, max=width)
            y_max = torch.clamp(y_max, min=0, max=height)
        return torch.stack([x_min, y_min, x_max, y_max], dim=1)

    def forward(self, outputs: dict, input_h: int, input_w: int):
        logits = outputs["pred_logits"]  # [B, Q, C]
        boxes = outputs["pred_boxes"]  # [B, Q, 4]  normalised cxcywh
        pred_masks = outputs.get("pred_masks", None)  # [B, Q, Hm, Wm] or None

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

        return result


class ExportWrapper(nn.Module):
    """Wraps backbone model + postprocessor for ONNX/TRT export."""

    def __init__(self, model: nn.Module, postprocessor: DFINEPostProcessor, input_size):
        super().__init__()
        self.model = model
        self.postprocessor = postprocessor
        self.input_h = input_size[0]
        self.input_w = input_size[1]

    def forward(self, x):
        outputs = self.model(x)
        return self.postprocessor(outputs, self.input_h, self.input_w)


class SemSegExportWrapper(nn.Module):
    """sem_seg export graph: fused argmax -> int32 [B, H, W] label map ("sem_seg").

    int32 keeps TRT happy (no int64 output) and H*W int32 beats C*H*W fp16 logits
    on output bandwidth. One fused graph serves every backend.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits = self.model(x)["sem_seg_logits"]  # [B, C, H, W]
        return logits.argmax(1).to(torch.int32)

    def deploy(self):
        self.model.deploy()
        return self


def prepare_model(cfg, device):
    model = build_model(
        cfg.model_name,
        len(cfg.train.label_to_name),
        enable_mask_head=cfg.task == "segment",
        device=device,
        img_size=cfg.train.img_size,
        in_channels=cfg.train.in_channels,
        task=cfg.task,
    )
    if cfg.export.from_pretrained:
        # Export the COCO/obj2coco pretrained weights directly (no trained model.pt).
        load_tuning_state(model, ensure_pretrained(cfg.train.pretrained_model_path))
    else:
        ckpt = Path(cfg.train.path_to_save) / "model.pt"
        if not ckpt.exists():
            raise FileNotFoundError(
                f"{ckpt} not found. Train first, or set export.from_pretrained=True "
                "to export pretrained weights directly."
            )
        state = torch.load(ckpt, weights_only=True)
        # decoder regenerates these from img_size; drop so re-export at another resolution works
        for k in ("decoder.anchors", "decoder.valid_mask"):
            state.pop(k, None)
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not unexpected and set(missing) <= {"decoder.anchors", "decoder.valid_mask"}, (
            f"unexpected checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    model.eval()
    return model


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
    ).save(output_path)

    onnx_model = onnx.load(output_path)
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
    onnx_file_path: Path, half: bool, max_batch_size: int, opt_bs: int = 1
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


def run_parity(cfg, raw_model, model, x_test, want, models_path: Path) -> None:
    """Compare each exported backend's detection scores against torch on one input.

    Scores (sorted top-K confidences) are the reorder-stable, end-to-end signal: they
    flow through the whole network, so any silent export drift moves them. Per-query
    boxes/logits/masks are dominated by background queries that never clear conf
    filtering, so comparing them is just noise — bench covers surviving-box geometry.
    """

    x1, x_np = _parity_input(cfg, raw_model, x_test)
    with torch.no_grad():
        ref = model(x1)[2]  # [1, K] sorted top-K confidences from the fused torch postproc

    n_out = len(cfg.train.label_to_name)
    n_queries = base_cfg["DFINETransformer"]["num_queries"]
    conf, kr, dyn = cfg.train.conf_thresh, cfg.train.keep_ratio, cfg.export.dynamic_input
    rows, low = [], []

    def add(name, is_int8, scores):
        cos = _cosine(ref, scores)
        rows.append([name, round(cos, 6)])
        if not cos >= (0.90 if is_int8 else 0.99):
            low.append(f"{name} cos={cos:.4f}")

    def raw_scores(outs):
        # raw backends emit per-query logits; pick by shape (TFLite reorders outputs)
        logits = next(np.asarray(o) for o in outs if np.ndim(o) == 3 and np.shape(o)[-1] != 4)
        flat = torch.sigmoid(torch.from_numpy(logits)).flatten(1)
        return torch.topk(flat, min(n_queries, flat.shape[1]), dim=-1).values

    if want("onnx") and (models_path / "model.onnx").exists():
        try:
            from src.infer.onnx_model import ONNX_model

            m = ONNX_model(
                model_path=str(models_path / "model.onnx"),
                n_outputs=n_out,
                conf_thresh=conf,
                rect=False,
                keep_ratio=kr,
                apply_nms=False,
            )
            add("ONNX", False, m._predict(x_np)[2])  # fused: scores at index 2
        except Exception as e:
            logger.warning(f"Parity skipped for ONNX: {e}")

    ov_variants = [("OpenVINO", "model.xml", False), ("OpenVINO INT8", "model_int8.xml", True)]
    for tag, fn, is8 in ov_variants:
        if want("openvino") and (models_path / fn).exists():
            try:
                from src.infer.ov_model import OV_model

                m = OV_model(
                    model_path=str(models_path / fn),
                    conf_thresh=conf,
                    rect=dyn,
                    half=cfg.export.half and platform.system() != "Darwin",
                    keep_ratio=kr,
                    max_batch_size=1,
                    apply_nms=False,
                )
                add(tag, is8, raw_scores(m._predict(x_np)))
            except Exception as e:
                logger.warning(f"Parity skipped for {tag}: {e}")

    litert_variants = [
        ("LiteRT", "model.tflite", False),
        ("LiteRT INT8", "model_int8.tflite", True),
    ]
    for tag, fn, is8 in litert_variants:
        if want("litert") and (models_path / fn).exists():
            try:
                from src.infer.litert_model import LiteRT_model

                m = LiteRT_model(
                    model_path=str(models_path / fn),
                    n_outputs=n_out,
                    conf_thresh=conf,
                    rect=False,
                    keep_ratio=kr,
                    apply_nms=False,
                )
                add(tag, is8, raw_scores(m._predict(x_np)))
            except Exception as e:
                logger.warning(f"Parity skipped for {tag}: {e}")

    if platform.system() == "Darwin":
        coreml_variants = [
            ("CoreML", "model.mlpackage", False),
            ("CoreML INT8", "model_int8.mlpackage", True),
        ]
        for tag, fn, is8 in coreml_variants:
            if want("coreml") and (models_path / fn).exists():
                try:
                    from src.infer.coreml_model import CoreML_model

                    m = CoreML_model(
                        model_path=str(models_path / fn),
                        n_outputs=n_out,
                        conf_thresh=conf,
                        rect=False,
                        keep_ratio=kr,
                        apply_nms=False,
                    )
                    add(tag, is8, m._predict(x_np)[2])  # fused: scores at index 2
                except Exception as e:
                    logger.warning(f"Parity skipped for {tag}: {e}")
    else:
        trt_variants = [
            ("TensorRT", "model.engine", False),
            ("TensorRT INT8", "model_int8.engine", True),
        ]
        for tag, fn, is8 in trt_variants:
            if want("tensorrt") and (models_path / fn).exists():
                try:
                    from src.infer.trt_model import TRT_model

                    m = TRT_model(
                        model_path=str(models_path / fn),
                        n_outputs=n_out,
                        conf_thresh=conf,
                        rect=False,
                        keep_ratio=kr,
                        apply_nms=False,
                    )
                    # Mirror __call__: _predict enqueues on m._stream, but
                    # _cosine's .cpu() read runs on the default stream — without
                    # this wrapper the async execute races the output read.
                    with torch.cuda.stream(m._stream):
                        x_trt = x1.to(device=m._input_tensor.device, dtype=m._input_tensor.dtype)
                        scores = m._predict(x_trt.contiguous(), actual_batch=1)[2]
                    torch.cuda.default_stream(m.device).wait_stream(m._stream)
                    add(tag, is8, scores)  # scores idx 2
                except Exception as e:
                    logger.warning(f"Parity skipped for {tag}: {e}")

    if not rows:
        logger.info("Parity: no exported backends available to compare against torch")
        return
    headers = ["format", "scores_cosine"]
    pd.DataFrame(rows, columns=headers).to_csv(models_path / "parity.csv", index=False)
    print("\n" + tabulate(rows, headers=headers, tablefmt="pretty"))
    if low:
        logger.warning("Parity below threshold (vs torch): " + ", ".join(low))
    else:
        logger.info("Parity OK: exported backends match torch within threshold")


def run_parity_sem_seg(cfg, raw_model, model, x_test, want, models_path: Path) -> None:
    """Per-pixel argmax agreement of each exported backend vs torch on one input.

    Every sem_seg export is the same fused graph (single int label-map output), so
    the comparison is end-to-end and reorder-free by construction; disagreement is
    confined to class-boundary pixels where near-tied logits flip under precision.
    """
    x1, x_np = _parity_input(cfg, raw_model, x_test)
    with torch.no_grad():
        ref = model(x1)[0].cpu().numpy()  # [H, W] int32 label map

    n_out = len(cfg.train.label_to_name)
    conf, kr = cfg.train.conf_thresh, cfg.train.keep_ratio
    rows, low = [], []

    def add(name, is_int8, label_map):
        m = _to_np(label_map).reshape(ref.shape)
        agree = float((m == ref).mean())
        rows.append([name, round(agree, 6)])
        if not agree >= (0.90 if is_int8 else 0.99):
            low.append(f"{name} agreement={agree:.4f}")

    if want("onnx") and (models_path / "model.onnx").exists():
        try:
            from src.infer.onnx_model import ONNX_model

            m = ONNX_model(
                model_path=str(models_path / "model.onnx"),
                n_outputs=n_out,
                conf_thresh=conf,
                rect=False,
                keep_ratio=kr,
            )
            add("ONNX", False, m._predict(x_np)[0][0])
        except Exception as e:
            logger.warning(f"Parity skipped for ONNX: {e}")

    for tag, fn, is8 in [("OpenVINO", "model.xml", False), ("OpenVINO INT8", "model_int8.xml", True)]:
        if want("openvino") and (models_path / fn).exists():
            try:
                from src.infer.ov_model import OV_model

                m = OV_model(
                    model_path=str(models_path / fn),
                    conf_thresh=conf,
                    rect=cfg.export.dynamic_input,
                    half=cfg.export.half and platform.system() != "Darwin",
                    keep_ratio=kr,
                    max_batch_size=1,
                )
                add(tag, is8, m._predict(x_np)[0][0])
            except Exception as e:
                logger.warning(f"Parity skipped for {tag}: {e}")

    for tag, fn, is8 in [("LiteRT", "model.tflite", False), ("LiteRT INT8", "model_int8.tflite", True)]:
        if want("litert") and (models_path / fn).exists():
            try:
                from src.infer.litert_model import LiteRT_model

                m = LiteRT_model(
                    model_path=str(models_path / fn),
                    n_outputs=n_out,
                    conf_thresh=conf,
                    rect=False,
                    keep_ratio=kr,
                )
                add(tag, is8, m._predict(x_np)[0][0])
            except Exception as e:
                logger.warning(f"Parity skipped for {tag}: {e}")

    if platform.system() == "Darwin":
        for tag, fn, is8 in [("CoreML", "model.mlpackage", False), ("CoreML INT8", "model_int8.mlpackage", True)]:
            if want("coreml") and (models_path / fn).exists():
                try:
                    from src.infer.coreml_model import CoreML_model

                    m = CoreML_model(
                        model_path=str(models_path / fn),
                        n_outputs=n_out,
                        conf_thresh=conf,
                        rect=False,
                        keep_ratio=kr,
                    )
                    add(tag, is8, m._predict(x_np)[0][0])
                except Exception as e:
                    logger.warning(f"Parity skipped for {tag}: {e}")
    else:
        for tag, fn, is8 in [("TensorRT", "model.engine", False), ("TensorRT INT8", "model_int8.engine", True)]:
            if want("tensorrt") and (models_path / fn).exists():
                try:
                    from src.infer.trt_model import TRT_model

                    m = TRT_model(
                        model_path=str(models_path / fn),
                        n_outputs=n_out,
                        conf_thresh=conf,
                        rect=False,
                        keep_ratio=kr,
                    )
                    # same stream discipline as the detection parity (see run_parity)
                    with torch.cuda.stream(m._stream):
                        x_trt = x1.to(device=m._input_tensor.device, dtype=m._input_tensor.dtype)
                        label_map = m._predict(x_trt.contiguous(), actual_batch=1)[0][0]
                    torch.cuda.default_stream(m.device).wait_stream(m._stream)
                    add(tag, is8, label_map)
                except Exception as e:
                    logger.warning(f"Parity skipped for {tag}: {e}")

    if not rows:
        logger.info("Parity: no exported backends available to compare against torch")
        return
    headers = ["format", "pixel_agreement"]
    pd.DataFrame(rows, columns=headers).to_csv(models_path / "parity.csv", index=False)
    print("\n" + tabulate(rows, headers=headers, tablefmt="pretty"))
    if low:
        logger.warning("Parity below threshold (vs torch): " + ", ".join(low))
    else:
        logger.info("Parity OK: exported backends match torch within threshold")


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig):
    input_name = "input"
    sem_seg = cfg.task == "sem_seg"
    enable_mask_head = cfg.task == "segment"
    output_names = ["sem_seg"] if sem_seg else ["labels", "boxes", "scores"]
    if enable_mask_head:
        output_names.append("masks")

    device = cfg.train.device
    cfg.exp = get_latest_experiment_name(cfg.exp, cfg.train.path_to_save)

    model_path = Path(cfg.train.path_to_save) / "model.pt"

    raw_model = prepare_model(cfg, device)

    if sem_seg:
        model = SemSegExportWrapper(raw_model)
    else:
        # Wrap model with fused postprocessor
        postprocessor = DFINEPostProcessor(
            num_classes=len(cfg.train.label_to_name),
            num_top_queries=base_cfg["DFINETransformer"]["num_queries"],
            use_focal_loss=base_cfg["matcher"]["use_focal_loss"],
        )
        model = ExportWrapper(raw_model, postprocessor, input_size=cfg.train.img_size)
    model.eval()
    raw_model.eval()

    x_test = torch.randn(cfg.export.max_batch_size, cfg.train.in_channels, *cfg.train.img_size).to(
        device
    )
    _ = model(x_test)

    # null = all backends; a list (e.g. [tensorrt]) restricts what is built (research loop)
    formats = cfg.export.get("formats", None)
    want = (lambda f: True) if formats is None else (lambda f: f in formats)

    # Openvino currently doesn't supprort some operations in the detection postprocessor,
    # so it gets the raw head; sem_seg's fused argmax graph is OV-supported as-is.
    if want("openvino"):
        raw_output_names = ["logits", "boxes"]
        if enable_mask_head:
            raw_output_names.append("masks")
        raw_onnx_path = export_to_onnx(
            model if sem_seg else raw_model,
            model_path,
            x_test,
            cfg.export.max_batch_size,
            half=False,
            dynamic_input=False,
            input_name=input_name,
            output_names=output_names if sem_seg else raw_output_names,
        )
        export_to_openvino(raw_onnx_path, x_test, cfg.export.dynamic_input, max_batch_size=1)

    if want("onnx") or want("tensorrt"):
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


if __name__ == "__main__":
    main()
