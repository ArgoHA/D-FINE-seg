"""`TorchModel(path)` must recover its input size, not silently fall back to 640x640."""

import shutil
from pathlib import Path

import pytest
import yaml

from dfine_seg.infer.torch_model import TorchModel

CKPT = Path(__file__).resolve().parents[2] / "pretrained" / "dfine_n_coco.pt"
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    """A checkpoint beside the config training freezes next to it."""
    if not CKPT.is_file():
        pytest.skip(f"{CKPT.name} not downloaded")
    d = tmp_path_factory.mktemp("run")
    shutil.copy(CKPT, d / "model.pt")
    (d / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "train": {
                    "img_size": [544, 960],
                    "keep_ratio": True,
                    "label_to_name": {i: f"c{i}" for i in range(80)},
                }
            }
        )
    )
    return d


def test_input_size_and_keep_ratio_come_from_the_sidecar(run_dir):
    m = TorchModel(str(run_dir / "model.pt"), device="cpu")
    assert m.input_size == (544, 960)
    assert m.keep_ratio is True
    assert m.names[0] == "c0"


def test_explicit_arguments_still_win(run_dir):
    m = TorchModel(
        str(run_dir / "model.pt"), input_height=320, input_width=320, keep_ratio=False, device="cpu"
    )
    assert m.input_size == (320, 320)
    assert m.keep_ratio is False


def test_falls_back_to_640_without_a_sidecar(tmp_path):
    if not CKPT.is_file():
        pytest.skip(f"{CKPT.name} not downloaded")
    shutil.copy(CKPT, tmp_path / "model.pt")
    m = TorchModel(str(tmp_path / "model.pt"), device="cpu")
    assert m.input_size == (640, 640)
    assert m.keep_ratio is False
