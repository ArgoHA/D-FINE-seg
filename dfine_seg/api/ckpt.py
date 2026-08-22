"""Recover `model_name`, `task`, `num_classes` and class names from a checkpoint.

Training writes a `meta` block (`model/utils.save_checkpoint`), but the released
checkpoints and anything trained before it are bare `state_dict()`s, so architecture is
always inferred from key structure: `(backbone key count, encoder hidden dim)` is unique
per size and identical across tasks - verified on all 14 released checkpoints. `meta`
carries what weights cannot: class names and the preprocessing the model was trained with.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import yaml
from loguru import logger

from dfine_seg.model.utils import unwrap_checkpoint

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


def _size_from(sd: Dict[str, torch.Tensor], meta: Dict[str, Any]) -> str:
    n_bb = sum(1 for k in sd if k.startswith("backbone."))
    proj = next((k for k in sd if k.startswith(_ENC_PROJ)), None)
    if proj is None:
        raise ValueError(f"not a D-FINE-seg checkpoint: no '{_ENC_PROJ}*' key")
    fp = (n_bb, sd[proj].shape[0])
    if fp in _FINGERPRINT:  # derived from the weights themselves, so it cannot disagree
        return _FINGERPRINT[fp]
    if meta.get("model_name"):  # architecture newer than the table - trust the writer
        return str(meta["model_name"])
    raise ValueError(
        f"unrecognized architecture {fp}; pass model_name= explicitly "
        f"(known: {sorted(_FINGERPRINT.values())})"
    )


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


def describe(sd: Dict[str, torch.Tensor], meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """-> architecture facts recoverable from an in-memory state_dict.

    `meta` is consulted only for `model_name`, and only when the fingerprint table doesn't
    know the architecture. `names`, `img_size` and `keep_ratio` are preprocessing, not
    architecture: nothing in the weights carries them, so they stay None here and
    `load_and_describe` fills them from the checkpoint's meta or the sidecar config.
    """
    meta = meta or {}
    if _DET_HEAD in sd:  # a mask decoder over the detection head = instance segmentation
        task = "segment" if any(k.startswith(_MASK_PREFIX) for k in sd) else "detect"
        num_classes = sd[_DET_HEAD].shape[0]
    elif _SEM_HEAD in sd:
        task, num_classes = "sem_seg", sd[_SEM_HEAD].shape[0]
    else:
        raise ValueError("not a D-FINE-seg checkpoint: no detection or sem_seg head found")

    return {
        "model_name": _size_from(sd, meta),
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
    sd, meta = unwrap_checkpoint(torch.load(p, map_location="cpu", weights_only=True))
    info = describe(sd, meta)
    # A model trained at 1024x2048 still runs at the 640x640 default, silently and worse --
    # so preprocessing is recovered too, not just the class names. The checkpoint's own meta
    # travels with the file; the frozen config is the fallback for one left in its run dir.
    cfg = sibling_config(p).get("train", {})
    info["names"] = _coerce_names(meta.get("label_to_name") or cfg.get("label_to_name"))
    size = meta.get("img_size") or cfg.get("img_size")
    info["img_size"] = (int(size[0]), int(size[1])) if size else None
    info["keep_ratio"] = meta.get("keep_ratio", cfg.get("keep_ratio"))
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
