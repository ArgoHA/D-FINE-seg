"""Shared fixtures.

Determinism: `seeded` is autouse so every test starts from the same RNG state.
Heavy work (pretrained weights, model build) lives in session-scoped fixtures so
the expensive setup happens at most once per pytest run.
"""

from __future__ import annotations

import shutil
from importlib import util as importlib_util
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
import torch

from src.dl.utils import set_seeds

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(autouse=True)
def seeded():
    """Deterministic RNGs for every test (CPU-deterministic; cudnn untouched)."""
    set_seeds(42, cudnn_fixed=False)


@pytest.fixture
def tiny_image() -> np.ndarray:
    """128x128 RGB image with two painted rectangles (uint8 HWC)."""
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    img[20:60, 30:80] = (200, 50, 50)   # red box
    img[70:110, 40:100] = (50, 200, 50) # green box
    return img


@pytest.fixture
def synthetic_preds_gt() -> Dict[str, List[Dict[str, torch.Tensor]]]:
    """Two-image batch in Validator's expected format (xyxy abs pixels).

    The first image has perfect preds; the second has a near-perfect pred plus
    one extra false positive. Useful for sanity-checking the metrics surface.
    """
    gt = [
        {
            "labels": torch.tensor([0, 1], dtype=torch.long),
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0],
                                   [60.0, 60.0, 100.0, 100.0]]),
        },
        {
            "labels": torch.tensor([0], dtype=torch.long),
            "boxes": torch.tensor([[5.0, 5.0, 40.0, 40.0]]),
        },
    ]
    preds = [
        {
            "labels": torch.tensor([0, 1], dtype=torch.long),
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0],
                                   [60.0, 60.0, 100.0, 100.0]]),
            "scores": torch.tensor([0.99, 0.95]),
        },
        {
            "labels": torch.tensor([0, 0], dtype=torch.long),
            "boxes": torch.tensor([[5.0, 5.0, 40.0, 40.0],
                                   [200.0, 200.0, 250.0, 250.0]]),
            "scores": torch.tensor([0.9, 0.8]),
        },
    ]
    return {"gt": gt, "preds": preds}


@pytest.fixture
def cuda_available() -> bool:
    return torch.cuda.is_available()


@pytest.fixture
def trt_available() -> bool:
    return importlib_util.find_spec("tensorrt") is not None


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


# ---- Slow / heavy fixtures ---------------------------------------------------

@pytest.fixture(scope="session")
def coco_pretrained_path(tmp_path_factory) -> Path:
    """Ensure dfine_s_coco.pt is on disk (HF auto-download). Skip if offline."""
    from src.d_fine.utils import ensure_pretrained

    target = REPO_ROOT / "pretrained" / "dfine_s_coco.pt"
    try:
        resolved = ensure_pretrained(target)
    except Exception as e:
        pytest.skip(f"pretrained weights unavailable: {e}")
    if not Path(resolved).exists():
        pytest.skip(f"pretrained weights missing at {resolved}")
    return Path(resolved)


@pytest.fixture(scope="session")
def coco_pretrained_s_cpu(coco_pretrained_path):
    """CPU `dfine_s` model with COCO weights, eval mode. Session-scoped."""
    from src.d_fine.dfine import build_model

    model = build_model(
        model_name="s",
        num_classes=80,
        enable_mask_head=False,
        device="cpu",
        img_size=[640, 640],
        pretrained_model_path=str(coco_pretrained_path),
    )
    model.eval()
    return model
