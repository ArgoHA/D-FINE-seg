"""Recover `model_name`, `task`, `num_classes` and class names from a checkpoint.

Checkpoints are bare `state_dict()`s (dl/train.py:643,657), so architecture is inferred
from key structure. `(backbone key count, encoder hidden dim)` is unique per size and
identical across tasks - verified on all 14 released checkpoints.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import yaml
from loguru import logger

# (n backbone keys, encoder hidden dim) -> model size
_FINGERPRINT: Dict[Tuple[int, int], str] = {
    (312, 128): "n",
    (312, 256): "s",
    (442, 256): "m",
    (400, 256): "l",
    (650, 384): "x",
}

_ENC_PROJ = "encoder.input_proj.0"
_DET_HEAD = "decoder.enc_score_head.weight"
_SEM_HEAD = "decoder.classifier.weight"
_MASK_PREFIX = "decoder.mask_decoder."


def _size_from(sd: Dict[str, torch.Tensor]) -> str:
    n_bb = sum(1 for k in sd if k.startswith("backbone."))
    proj = next((k for k in sd if k.startswith(_ENC_PROJ)), None)
    if proj is None:
        raise ValueError(f"not a D-FINE-seg checkpoint: no '{_ENC_PROJ}*' key")
    fp = (n_bb, sd[proj].shape[0])
    if fp not in _FINGERPRINT:
        raise ValueError(
            f"unrecognized architecture {fp}; pass model_name= explicitly "
            f"(known: {sorted(_FINGERPRINT.values())})"
        )
    return _FINGERPRINT[fp]


def sibling_config(ckpt: Path) -> Dict[str, Any]:
    """`config.yaml` that training freezes next to the checkpoint (dl/train.py)."""
    p = ckpt.parent / "config.yaml"
    if not p.is_file():
        return {}
    try:
        return yaml.safe_load(p.read_text()) or {}
    except Exception as e:  # a malformed sidecar must not block loading the weights
        logger.warning(f"ignoring unreadable {p}: {e}")
        return {}


def describe(sd: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """-> architecture facts recoverable from an in-memory state_dict.

    `names`, `img_size` and `keep_ratio` are preprocessing, not architecture: nothing in
    the weights carries them, so they stay None here and `load_and_describe` fills them
    from the sidecar config.
    """
    if _DET_HEAD in sd:  # a mask decoder over the detection head = instance segmentation
        task = "segment" if any(k.startswith(_MASK_PREFIX) for k in sd) else "detect"
        num_classes = sd[_DET_HEAD].shape[0]
    elif _SEM_HEAD in sd:
        task, num_classes = "sem_seg", sd[_SEM_HEAD].shape[0]
    else:
        raise ValueError("not a D-FINE-seg checkpoint: no detection or sem_seg head found")

    return {
        "model_name": _size_from(sd),
        "task": task,
        "num_classes": num_classes,
        "names": None,
        "in_channels": _in_channels(sd),
        "img_size": None,
        "keep_ratio": None,
    }


def load_and_describe(path: str | Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """-> (state_dict, info). Reads the file once; callers reuse the state_dict."""
    p = Path(path)
    sd = torch.load(p, map_location="cpu", weights_only=True)
    info = describe(sd)
    cfg = sibling_config(p).get("train", {})
    info["names"] = _coerce_names(cfg.get("label_to_name"))
    # A model trained at 1024x2048 still runs at the 640x640 default, silently and worse --
    # so preprocessing comes from the frozen config too, not just the class names.
    size = cfg.get("img_size")
    info["img_size"] = (int(size[0]), int(size[1])) if size else None
    info["keep_ratio"] = cfg.get("keep_ratio")
    return sd, info


def inspect(path: str | Path) -> Dict[str, Any]:
    """-> {model_name, task, num_classes, names, in_channels, img_size, keep_ratio}."""
    return load_and_describe(path)[1]


def _in_channels(sd: Dict[str, torch.Tensor]) -> int:
    stem = next((k for k in sd if k.startswith("backbone.stem.stem1.conv.weight")), None)
    return int(sd[stem].shape[1]) if stem else 3


def _coerce_names(raw: Any) -> Optional[Dict[int, str]]:
    if not raw:
        return None
    if isinstance(raw, dict):
        return {int(k): str(v) for k, v in raw.items()}
    return {i: str(v) for i, v in enumerate(raw)}
