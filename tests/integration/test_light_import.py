"""`import dfine_seg` must stay torch-only.

The pip default install has no export backends, so anything the public API imports at
module scope has to work without them. Runs in a subprocess because pytest itself has
already imported half of these.
"""

import subprocess
import sys
import textwrap

import pytest

# Modules the light path must never pull in. hydra/wandb/albumentations are installed
# (they are core deps) but belong to training, not to `load_model(...)(img)`.
FORBIDDEN = [
    "onnx",
    "onnxruntime",
    "openvino",
    "tensorrt",
    "coremltools",
    "nncf",
    "transformers",
    "gradio",
    "hydra",
    "wandb",
    "albumentations",
    "matplotlib",
    "pandas",
    "sklearn",
    "torchmetrics",
    "faster_coco_eval",
]


def _leaked(snippet: str) -> list[str]:
    code = textwrap.dedent(snippet) + textwrap.dedent(f"""
        import sys
        print(",".join(m for m in {FORBIDDEN!r} if m in sys.modules))
    """)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return [m for m in proc.stdout.strip().split(",") if m]


def test_import_package_is_light():
    assert _leaked("import dfine_seg") == []


def test_import_api_symbols_is_light():
    assert _leaked("from dfine_seg import load_model, read_image, Visualizer") == []


def test_torch_wrapper_import_is_light():
    assert _leaked("from dfine_seg.infer.torch_model import TorchModel") == []


@pytest.mark.slow
def test_predict_is_light(tmp_path):
    """A full build+predict cycle must not drag a backend in either."""
    leaked = _leaked("""
        import numpy as np
        from dfine_seg import load_model
        m = load_model("s")
        m(np.zeros((64, 64, 3), dtype=np.uint8))
    """)
    assert leaked == []
