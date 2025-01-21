from pathlib import Path

import hydra
import onnx
import onnxsim
import openvino as ov
import tensorrt as trt
import torch
from loguru import logger
from omegaconf import DictConfig
from torch import nn

from src.d_fine.dfine import build_model

INPUT_NAME = "input"
OUTPUT_NAME = "output"


def prepare_model(cfg, model_path, device):
    model = build_model(cfg.model_name, len(cfg.train.label_to_name), device)
    model.load_state_dict(torch.load(Path(cfg.train.path_to_save) / "model.pt", weights_only=True))
    model.eval()
    return model


def add_suffix(output_path, dynamic_input: bool, half: bool):
    if dynamic_input:
        output_path = f"{output_path}_dynamic"
    if half:
        output_path = f"{output_path}_half"
    return Path(output_path).with_suffix(".onnx")


def export_to_onnx(
    model: nn.Module,
    model_path: Path,
    x_test: torch.Tensor,
    max_batch_size: int,
    half: bool,
    dynamic_input: bool,
) -> None:
    if half:
        model = model.half()
        x_test = x_test.half()

    dynamic_axes = {}
    if max_batch_size > 1:
        dynamic_axes = {INPUT_NAME: {0: "batch_size"}, OUTPUT_NAME: {0: "batch_size"}}
    if dynamic_input:
        if INPUT_NAME not in dynamic_axes:
            dynamic_axes[INPUT_NAME] = {}
        dynamic_axes[INPUT_NAME].update({2: "height", 3: "width"})

    output_path = add_suffix(
        model_path.parent / model_path.stem, dynamic_input=dynamic_input, half=half
    )
    torch.onnx.export(
        model,
        x_test,
        output_path,
        opset_version=17,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes=dynamic_axes,
    )

    onnx_model = onnx.load(output_path)

    try:
        onnx_model, check = onnxsim.simplify(onnx_model)
        assert check
        logger.info("ONNX simplified and exported")
    except Exception as e:
        logger.info(f"Simplification failed: {e}")
    finally:
        onnx.save(onnx_model, output_path)
        return output_path


def export_to_openvino(onnx_path: Path, x_test, dynamic_input: bool, half: bool) -> None:
    model = ov.convert_model(
        input_model=str(onnx_path),
        input=None if dynamic_input else [x_test.shape],
        example_input=x_test,
    )
    ov.serialize(model, str(onnx_path.with_suffix(".xml")), str(onnx_path.with_suffix(".bin")))
    logger.info("OpenVINO model exported")


def export_to_tensorrt(
    onnx_file_path: Path,
    half: bool,
    max_batch_size: int,
    dynamic_input: bool,
    max_shape: tuple,
) -> None:
    opt_shape = max_shape
    min_shape = max_shape[0], max_shape[1], max_shape[2] // 2, max_shape[3] // 2

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
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    if half:
        config.set_flag(trt.BuilderFlag.FP16)

    # Create an optimization profile for batching and dynamic shapes
    if False:
        profile = builder.create_optimization_profile()
        input_name = network.get_input(0).name  # Assumes single input
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
    else:
        if max_batch_size > 1:
            profile = builder.create_optimization_profile()
            input_name = network.get_input(0).name  # Assumes single input
            input_shape = network.get_input(0).shape
            static_shape = (max_batch_size, *input_shape[1:])
            profile.set_shape(input_name, static_shape, static_shape, static_shape)
            config.add_optimization_profile(profile)

    engine = builder.build_serialized_network(network, config)
    with open(onnx_file_path.with_suffix(".engine"), "wb") as f:
        f.write(engine)
    logger.info("TensorRT model exported")


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig):
    device = cfg.train.device
    model_path = Path(cfg.train.path_to_save) / "model.pt"

    model = prepare_model(cfg, model_path, device)
    x_test = torch.randn(cfg.export.max_batch_size, 3, *cfg.train.img_size).to(device)
    _ = model(x_test)

    onnx_path = export_to_onnx(
        model,
        model_path,
        x_test,
        cfg.export.max_batch_size,
        cfg.export.half,
        cfg.export.dynamic_input,
    )
    export_to_openvino(onnx_path, x_test, cfg.export.dynamic_input, cfg.export.half)

    static_onnx_path = export_to_onnx(
        model,
        model_path,
        x_test,
        cfg.export.max_batch_size,
        cfg.export.half,
        False,
    )
    export_to_tensorrt(
        static_onnx_path,
        cfg.export.half,
        cfg.export.max_batch_size,
        cfg.export.dynamic_input,
        x_test.shape,
    )

    logger.info(f"Exports saved to: {model_path.parent}")


if __name__ == "__main__":
    main()
