"""`load_model()` - resolve weights and hand back the matching inference wrapper.

Deliberately not a wrapper around the wrappers: `load_model` returns the very same
`TorchModel` / `TRTModel` / … object you would construct yourself, so its call
signature and output contract are the wrappers' own. Keyword arguments pass straight
through. Torch-only at import time; a backend is imported only when its file is loaded.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from numpy.typing import NDArray

from dfine_seg.api.coco_names import COCO_NAMES
from dfine_seg.model.utils import _COCO_FILENAME_RE, ensure_pretrained, pretrained_from_hub

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
    dataset: Optional[str] = None,
    weights_dir: Union[str, Path, None] = None,
) -> str:
    """Path to a released checkpoint, downloading it from Hugging Face on first use.

    `weights_dir=None` (the default) reuses a clone's `pretrained/` when it already holds
    the file, and otherwise the shared HF cache - so the API doesn't leave a `pretrained/`
    copy in every directory it is called from. Pass a directory to force one.

    Tasks resolve to: detect -> `dfine_<size>_<coco|obj2coco>.pt`,
    segment -> `dfine_seg_<size>_coco.pt`, sem_seg -> `dfine_sem_<size>_cityscapes.pt`.
    `dataset=None` picks each task's only released family; detect is the one task with
    two (coco | obj2coco), so that is the only place dataset= actually selects anything.
    """
    if size not in SIZES:
        raise ValueError(f"size must be one of {SIZES}, got {size!r}")
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    if task == "segment":
        if dataset not in (None, "coco"):
            raise ValueError(
                f"pretrained segment weights exist only for dataset='coco', got {dataset!r}"
            )
        filename = f"dfine_seg_{size}_coco.pt"
    elif task == "sem_seg":
        if dataset not in (None, "cityscapes"):
            raise ValueError(
                f"pretrained sem_seg weights exist only for dataset='cityscapes', got {dataset!r}"
            )
        filename = f"dfine_sem_{size}_cityscapes.pt"
    else:  # detect - the only task with two released families
        if dataset not in (None, "coco", "obj2coco"):
            raise ValueError(
                f"dataset must be coco|obj2coco for task={task}, got {dataset!r}; "
                'use task="sem_seg" for the Cityscapes weights'
            )
        filename = f"dfine_{size}_{dataset or 'coco'}.pt"
    if weights_dir is not None:
        return ensure_pretrained(Path(weights_dir) / filename)
    local = Path("pretrained") / filename
    return str(local) if local.is_file() else pretrained_from_hub(filename)


def load_model(
    model: Union[str, Path] = "s",
    task: Optional[str] = None,
    *,
    dataset: Optional[str] = None,
    weights_dir: Union[str, Path, None] = None,
    names: Optional[Dict[int, str]] = None,
    **kwargs: Any,
):
    """Load a model by size string (`"s"`) or by artifact path.

    ```python
    model = load_model("s")                      # COCO detection weights, auto-downloaded
    model = load_model("s", task="segment")
    model = load_model("x", task="sem_seg")      # Cityscapes semantic segmentation
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
    else:
        path = Path(spec)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Pass a model path, or one of {SIZES} for pretrained weights."
            )
        # Only .pt builds its architecture from `task`; graph artifacts have it baked in
        # and their wrappers take no task= at all.
        if task is not None and path.suffix.lower() == ".pt":
            kwargs.setdefault("task", task)

    # A released COCO checkpoint is a COCO model even without meta - don't make
    # load_model("pretrained/dfine_n_coco.pt") behave differently from load_model("n").
    # The Cityscapes sem-seg releases carry their own label_to_name, which wins below.
    default_names = dict(COCO_NAMES) if _COCO_FILENAME_RE.match(path.name) else None

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
    # Wrapper-supplied names (checkpoint meta, or a sidecar config for meta-less files) beat
    # the bundled COCO default; an explicit names= beats everything.
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
