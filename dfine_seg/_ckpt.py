"""Recover `model_name`, `task`, `num_classes` and class names from a checkpoint.

Checkpoints are bare `state_dict()`s (dl/train.py:643,657), so architecture is inferred
from key structure. `(backbone key count, encoder hidden dim)` is unique per size and
identical across tasks — verified on all 14 released checkpoints.
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


def _task_and_classes(sd: Dict[str, torch.Tensor]) -> Tuple[str, int]:
    if _DET_HEAD in sd:
        task = "segment" if any(k.startswith(_MASK_PREFIX) for k in sd) else "detect"
        return task, sd[_DET_HEAD].shape[0]
    if _SEM_HEAD in sd:
        return "sem_seg", sd[_SEM_HEAD].shape[0]
    raise ValueError("not a D-FINE-seg checkpoint: no detection or sem_seg head found")


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
    """-> {model_name, task, num_classes, names, in_channels} from an in-memory state_dict."""
    task, num_classes = _task_and_classes(sd)
    return {
        "model_name": _size_from(sd),
        "task": task,
        "num_classes": num_classes,
        "names": None,  # not in the weights; load_and_describe reads the sidecar config
        "in_channels": _in_channels(sd),
    }


def load_and_describe(path: str | Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """-> (state_dict, info). Reads the file once; callers reuse the state_dict."""
    p = Path(path)
    sd = torch.load(p, map_location="cpu", weights_only=True)
    info = describe(sd)
    info["names"] = _coerce_names(sibling_config(p).get("train", {}).get("label_to_name"))
    return sd, info


def inspect(path: str | Path) -> Dict[str, Any]:
    """-> {model_name, task, num_classes, names, in_channels}. Never raises on names."""
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
