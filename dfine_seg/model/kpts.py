"""Keypoint helpers shared by the criterion, matcher and validator."""

from typing import Optional, Sequence

import torch

# COCO person keypoint OKS sigmas (pycocotools)
COCO_SIGMAS = [
    0.026,
    0.025,
    0.025,
    0.035,
    0.035,
    0.079,
    0.079,
    0.072,
    0.072,
    0.062,
    0.062,
    0.107,
    0.107,
    0.087,
    0.087,
    0.089,
    0.089,
]
DEFAULT_SIGMA = 0.07  # ~COCO mean, used for every point when K != 17
# OKS scale from the box when no COCO area exists: a person fills ~half of its box
BOX_AREA_FACTOR = 0.53


def resolve_sigmas(num_kpts: int, sigmas: Optional[Sequence[float]] = None) -> list:
    if sigmas is not None:
        if len(sigmas) != num_kpts:
            raise ValueError(f"train.kpts.sigmas has {len(sigmas)} values, expected {num_kpts}")
        return [float(s) for s in sigmas]
    if num_kpts == len(COCO_SIGMAS):
        return list(COCO_SIGMAS)
    return [DEFAULT_SIGMA] * num_kpts


def kpt_oks(
    pred: torch.Tensor,
    gt: torch.Tensor,
    visible: torch.Tensor,
    area: torch.Tensor,
    sigmas: torch.Tensor,
) -> torch.Tensor:
    """COCO OKS of broadcastable points -> [...].

    pred / gt [..., K, 2], visible [..., K] bool (v > 0), area [...] in the coords' squared
    units, sigmas [K]. Instances without labeled points get 0.
    Pairwise [P, G]: kpt_oks(pred[:, None], gt[None], visible[None], area[None], sigmas).
    """
    d2 = (pred - gt).pow(2).sum(-1)
    e = d2 / ((2 * sigmas).pow(2) * (area[..., None] + 1e-9) * 2)
    vis = visible.to(e.dtype)
    return (torch.exp(-e) * vis).sum(-1) / vis.sum(-1).clamp(min=1)
