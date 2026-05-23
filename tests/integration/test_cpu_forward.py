"""CPU forward-pass smoke tests + loose latency ceiling.

These don't need pretrained weights — the model is instantiated with random
init. The point is to catch shape regressions and gross slowdowns (O(N^2)
regressions, accidentally re-enabling something heavy on the hot path).
"""

import time

import pytest
import torch

from src.d_fine.dfine import build_model


@pytest.fixture(scope="module")
def model_n_detect_cpu():
    model = build_model(
        model_name="n",
        num_classes=80,
        enable_mask_head=False,
        device="cpu",
        img_size=[640, 640],
    )
    model.eval()
    return model


@pytest.fixture(scope="module")
def model_n_segment_cpu():
    model = build_model(
        model_name="n",
        num_classes=80,
        enable_mask_head=True,
        device="cpu",
        img_size=[640, 640],
    )
    model.eval()
    return model


def test_forward_shapes_cpu_detect(model_n_detect_cpu):
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        out = model_n_detect_cpu(x)
    assert "pred_logits" in out and "pred_boxes" in out
    bs, q, c = out["pred_logits"].shape
    assert bs == 1 and c == 80
    assert out["pred_boxes"].shape == (bs, q, 4)
    assert torch.isfinite(out["pred_logits"]).all()
    assert torch.isfinite(out["pred_boxes"]).all()


def test_forward_shapes_cpu_segment(model_n_segment_cpu):
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        out = model_n_segment_cpu(x)
    assert "pred_masks" in out
    masks = out["pred_masks"]
    assert masks.dim() == 4 and masks.shape[0] == 1
    # Masks are sigmoid-bound in [0, 1] post-decoder.
    assert masks.min().item() >= 0.0 - 1e-4
    assert masks.max().item() <= 1.0 + 1e-4


def test_cpu_forward_latency_smoke(model_n_detect_cpu):
    """Loose ceiling: catches O(N^2) regressions, not microbenchmarks.

    The `n` model on CPU with batch=1 / 640x640 should land well under 10s/iter
    on any sane machine. The hard latency budgets live in `bench.py`.
    """
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        _ = model_n_detect_cpu(x)  # warmup (alloc, kernel select)
        t0 = time.perf_counter()
        _ = model_n_detect_cpu(x)
        elapsed = time.perf_counter() - t0
    assert elapsed < 10.0, f"CPU forward took {elapsed:.2f}s, expected < 10s"


def test_forward_shapes_cpu_4channel():
    """N-channel input: stem rewires for in_channels=4 and forwards cleanly."""
    model = build_model(
        model_name="n",
        num_classes=80,
        enable_mask_head=False,
        device="cpu",
        img_size=[640, 640],
        in_channels=4,
    ).eval()
    assert model.backbone.stem.stem1.conv.weight.shape[1] == 4
    x = torch.randn(1, 4, 640, 640)
    with torch.no_grad():
        out = model(x)
    bs, q, c = out["pred_logits"].shape
    assert bs == 1 and c == 80
    assert out["pred_boxes"].shape == (bs, q, 4)
    assert torch.isfinite(out["pred_logits"]).all()
    assert torch.isfinite(out["pred_boxes"]).all()


@pytest.mark.gpu
def test_forward_cuda_detect(cuda_available):
    if not cuda_available:
        pytest.skip("CUDA not available")
    model = build_model(
        model_name="n",
        num_classes=80,
        enable_mask_head=False,
        device="cuda",
        img_size=[640, 640],
    ).eval()
    x = torch.randn(1, 3, 640, 640, device="cuda")
    with torch.no_grad():
        out = model(x)
    assert out["pred_logits"].is_cuda
    assert torch.isfinite(out["pred_logits"]).all()
