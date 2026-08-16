"""`load_model()` — resolve weights and hand back the matching inference wrapper.

Deliberately not a wrapper around the wrappers: `load_model` returns the very same
`TorchModel` / `TRTModel` / … object you would construct yourself, so its call
signature and output contract are the wrappers' own. Keyword arguments pass straight
through. Torch-only at import time; a backend is imported only when its file is loaded.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from numpy.typing import NDArray

from dfine_seg.data.coco_names import COCO_NAMES
from dfine_seg.model.utils import _FILENAME_RE, ensure_pretrained

SIZES = ("n", "s", "m", "l", "x")
TASKS = ("detect", "segment", "sem_seg")

# file suffix -> (module, class). Only the one that matches is imported.
_BACKENDS = {
    ".pt": ("dfine_seg.infer.torch_model", "TorchModel"),
    ".engine": ("dfine_seg.infer.trt_model", "TRTModel"),
    ".onnx": ("dfine_seg.infer.onnx_model", "ONNXModel"),
    ".xml": ("dfine_seg.infer.ov_model", "OVModel"),
    ".mlpackage": ("dfine_seg.infer.coreml_model", "CoreMLModel"),
    ".tflite": ("dfine_seg.infer.litert_model", "LiteRTModel"),
}

_EXTRA_FOR = {".onnx": "export", ".xml": "export", ".mlpackage": "export", ".engine": "trt"}


def pretrained_path(
    size: str,
    task: str = "detect",
    dataset: str = "coco",
    weights_dir: Union[str, Path] = "pretrained",
) -> str:
    """Path to a released checkpoint, downloading it from Hugging Face on first use."""
    if size not in SIZES:
        raise ValueError(f"size must be one of {SIZES}, got {size!r}")
    if task == "sem_seg":
        raise ValueError(
            "no pretrained sem_seg weights exist — train one, then load its path: "
            'load_model("output/models/<exp>/model.pt")'
        )
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    if task == "segment":
        filename = f"dfine_seg_{size}_coco.pt"
    elif dataset in ("coco", "obj2coco"):
        filename = f"dfine_{size}_{dataset}.pt"
    else:
        raise ValueError(f"dataset must be coco|obj2coco, got {dataset!r}")
    return ensure_pretrained(Path(weights_dir) / filename)


def load_model(
    model: Union[str, Path] = "s",
    task: Optional[str] = None,
    *,
    dataset: str = "coco",
    weights_dir: Union[str, Path] = "pretrained",
    names: Optional[Dict[int, str]] = None,
    **kwargs: Any,
):
    """Load a model by size string (`"s"`) or by artifact path.

    ```python
    model = load_model("s")                      # COCO detection weights, auto-downloaded
    model = load_model("s", task="segment")
    model = load_model("runs/exp/model.pt")      # TorchModel
    model = load_model("runs/exp/model.engine")  # TRTModel
    out = model(image)                     # the wrapper's own contract, tensors on device
    ```

    Extra keyword arguments go straight to the backend wrapper, e.g.
    `load_model("s", conf_thresh=0.3, input_height=960, input_width=960)`.
    """
    spec = str(model)
    if spec in SIZES:
        path = Path(pretrained_path(spec, task or "detect", dataset, weights_dir))
        default_names = dict(COCO_NAMES)
    else:
        path = Path(spec)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Pass a model path, or one of {SIZES} for pretrained "
                "COCO weights."
            )
        # A released checkpoint loaded by path is still a COCO model — don't make
        # load_model("pretrained/dfine_n_coco.pt") behave differently from load_model("n").
        default_names = dict(COCO_NAMES) if _FILENAME_RE.match(path.name) else None
        if task is not None:
            kwargs.setdefault("task", task)

    suffix = path.suffix.lower()
    if suffix not in _BACKENDS:
        raise ValueError(
            f"unsupported model file {path.name!r}; expected one of {sorted(_BACKENDS)}"
        )

    module_name, class_name = _BACKENDS[suffix]
    try:
        module = __import__(module_name, fromlist=[class_name])
    except ImportError as e:
        extra = _EXTRA_FOR.get(suffix)
        hint = f" Install it with `pip install 'dfine-seg[{extra}]'`." if extra else ""
        raise ImportError(f"backend for {suffix} is not available: {e}.{hint}") from e

    wrapper = getattr(module, class_name)(str(path), **kwargs)
    # Names never live in the weights. Wrapper-supplied (from a sidecar config) wins over
    # the bundled COCO map; an explicit names= wins over both.
    wrapper.names = names or getattr(wrapper, "names", None) or default_names
    return wrapper


def read_image(source: Union[str, Path, Any]) -> NDArray[np.uint8]:
    """Read an image for a wrapper's `__call__`, which takes HWC uint8 arrays.

    Returns BGR for `.jpg`/`.png` (pass to the wrapper as-is) and RGB for `.npy` and PIL
    images (pass `bgr=False`). Mirrors `dl/dataset.read_image_hwc` but stays import-light,
    so the inference path never pulls the training stack.
    """
    if hasattr(source, "convert"):  # PIL.Image -> RGB
        return np.asarray(source.convert("RGB"))
    p = Path(source)
    if p.suffix.lower() == ".npy":  # RGB, or RGB+extras for multi-channel stacks
        return np.load(p)
    import cv2

    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read image: {p}")
    return img
