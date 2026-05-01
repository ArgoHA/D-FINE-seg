import re
from pathlib import Path
from typing import Dict

import torch
from loguru import logger

from .dist_utils import is_main_process

HF_REPO_ID = "ArgoSA/D-FINE-seg"
_FILENAME_RE = re.compile(r"^dfine_(?:seg_(n|s|m|l|x)_coco|(n|s|m|l|x)_(coco|obj2coco))\.pt$")

obj365_ids = [
    0,
    46,
    5,
    58,
    114,
    55,
    116,
    65,
    21,
    40,
    176,
    127,
    249,
    24,
    56,
    139,
    92,
    78,
    99,
    96,
    144,
    295,
    178,
    180,
    38,
    39,
    13,
    43,
    120,
    219,
    148,
    173,
    165,
    154,
    137,
    113,
    145,
    146,
    204,
    8,
    35,
    10,
    88,
    84,
    93,
    26,
    112,
    82,
    265,
    104,
    141,
    152,
    234,
    143,
    150,
    97,
    2,
    50,
    25,
    75,
    98,
    153,
    37,
    73,
    115,
    132,
    106,
    61,
    163,
    134,
    277,
    81,
    133,
    18,
    94,
    30,
    169,
    70,
    328,
    226,
]


def map_class_weights(cur_tensor, pretrain_tensor):
    """Map class weights from pretrain model to current model based on class IDs."""
    if pretrain_tensor.size() == cur_tensor.size():
        return pretrain_tensor

    adjusted_tensor = cur_tensor.clone()
    adjusted_tensor.requires_grad = False

    if pretrain_tensor.size() > cur_tensor.size():
        for coco_id, obj_id in enumerate(obj365_ids):
            adjusted_tensor[coco_id] = pretrain_tensor[obj_id + 1]
    else:
        for coco_id, obj_id in enumerate(obj365_ids):
            adjusted_tensor[obj_id + 1] = pretrain_tensor[coco_id]

    return adjusted_tensor


def adjust_head_parameters(cur_state_dict, pretrain_state_dict):
    """Adjust head parameters between datasets."""
    # List of parameters to adjust
    if (
        pretrain_state_dict["decoder.denoising_class_embed.weight"].size()
        != cur_state_dict["decoder.denoising_class_embed.weight"].size()
    ):
        del pretrain_state_dict["decoder.denoising_class_embed.weight"]

    head_param_names = ["decoder.enc_score_head.weight", "decoder.enc_score_head.bias"]
    for i in range(8):
        head_param_names.append(f"decoder.dec_score_head.{i}.weight")
        head_param_names.append(f"decoder.dec_score_head.{i}.bias")

    adjusted_params = []

    for param_name in head_param_names:
        if param_name in cur_state_dict and param_name in pretrain_state_dict:
            cur_tensor = cur_state_dict[param_name]
            pretrain_tensor = pretrain_state_dict[param_name]
            adjusted_tensor = map_class_weights(cur_tensor, pretrain_tensor)
            if adjusted_tensor is not None:
                pretrain_state_dict[param_name] = adjusted_tensor
                adjusted_params.append(param_name)
            else:
                print(f"Cannot adjust parameter '{param_name}' due to size mismatch.")

    return pretrain_state_dict


def matched_state(state: Dict[str, torch.Tensor], params: Dict[str, torch.Tensor]):
    missed_list = []
    unmatched_list = []
    matched_state = {}
    for k, v in state.items():
        if k in params:
            if v.shape == params[k].shape:
                matched_state[k] = params[k]
            else:
                unmatched_list.append(k)
        else:
            missed_list.append(k)

    return matched_state, {"missed": missed_list, "unmatched": unmatched_list}


def extract_pretrained_state_dict(state: Dict[str, torch.Tensor]):
    """Extract the raw model state dict from legacy or current checkpoint formats."""
    if "ema" in state:
        return state["ema"]["module"]
    if "model" in state:
        return state["model"]
    return state


def load_tuning_state(model, path: str):
    """Load model for tuning and adjust mismatched head parameters"""
    if path.startswith("http"):
        state = torch.hub.load_state_dict_from_url(path, map_location="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)

    pretrain_state_dict = extract_pretrained_state_dict(state)

    # Adjust head parameters between datasets
    try:
        adjusted_state_dict = adjust_head_parameters(model.state_dict(), pretrain_state_dict)
        stat, infos = matched_state(model.state_dict(), adjusted_state_dict)
    except Exception:
        stat, infos = matched_state(model.state_dict(), pretrain_state_dict)

    model.load_state_dict(stat, strict=False)
    if is_main_process():
        logger.info(f"Pretrained weigts from {path}, {infos}")
    return model


def ensure_pretrained(path: str | Path) -> str:
    """Resolve a pretrained checkpoint path, fetching from HF if missing.

    If the file already exists, returns the path unchanged. If it's missing and
    the filename matches the standard `dfine_<size>_<dataset>.pt` pattern, it's
    downloaded from `HF_REPO_ID` into the same directory. For non-standard
    filenames (e.g. user-supplied fine-tuning checkpoints), returns the path
    as-is so the caller raises FileNotFoundError as before.
    """
    p = Path(path)
    if p.exists():
        return str(p)

    if not _FILENAME_RE.match(p.name):
        return str(p)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required to auto-download pretrained weights. "
            "Install with `pip install huggingface_hub` or place the file at "
            f"{p} manually."
        ) from e

    p.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Pretrained weights not found at {p}; downloading from {HF_REPO_ID}")
    downloaded = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=p.name,
        local_dir=str(p.parent),
    )
    return downloaded
