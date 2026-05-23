"""Pin coordinate / box conversions in `src/dl/utils.py`.

These are the workhorses behind every detection postprocess and the YOLO label
writer — any silent change here corrupts both eval and exported labels.
"""

import numpy as np
import pytest
import torch

from src.dl.utils import (
    abs_xyxy_to_norm_xywh,
    clip_boxes,
    norm_xywh_to_abs_xyxy,
    process_boxes,
    scale_boxes,
    scale_boxes_ratio_kept,
)


def test_norm_xywh_to_abs_xyxy_centered_box_no_round():
    # A centered box covering 50% of a 100x200 image (h, w).
    boxes = np.array([[0.5, 0.5, 0.5, 0.5]])
    out = norm_xywh_to_abs_xyxy(boxes, height=100, width=200, to_round=False)
    np.testing.assert_allclose(out, [[50.0, 25.0, 150.0, 75.0]])


def test_norm_xywh_to_abs_xyxy_rounds_and_clamps():
    boxes = np.array([[0.5, 0.5, 1.5, 1.5]])  # blown-out box should clamp
    out = norm_xywh_to_abs_xyxy(boxes, height=100, width=200, to_round=True)
    # to_round=True clamps to width-1 / height-1 (REPO_AUDIT B5 — exclusive vs inclusive)
    assert out[0, 0] == 0
    assert out[0, 1] == 0
    assert out[0, 2] == 199
    assert out[0, 3] == 99


def test_abs_xyxy_to_norm_xywh_inverts_no_round_path():
    H, W = 480, 640
    rng = np.random.default_rng(0)
    # build random normalized xywh, fully inside the image
    xy = rng.uniform(0.2, 0.8, size=(8, 2))
    wh = rng.uniform(0.05, 0.3, size=(8, 2))
    boxes_norm = np.concatenate([xy, wh], axis=1)
    abs_xyxy = norm_xywh_to_abs_xyxy(boxes_norm, height=H, width=W, to_round=False)
    boxes_back = abs_xyxy_to_norm_xywh(abs_xyxy, height=H, width=W)
    np.testing.assert_allclose(boxes_back, boxes_norm, atol=1e-6)


def test_clip_boxes_numpy_and_torch_agree():
    shape = (100, 200)  # h, w
    raw = np.array([[-5.0, -10.0, 250.0, 150.0]])
    expected = np.array([[0.0, 0.0, 200.0, 100.0]])

    np_boxes = raw.copy()
    clip_boxes(np_boxes, shape)
    np.testing.assert_array_equal(np_boxes, expected)

    t_boxes = torch.tensor(raw)
    clip_boxes(t_boxes, shape)
    np.testing.assert_array_equal(t_boxes.numpy(), expected)


def test_scale_boxes_uniform():
    # resized 320x320 -> orig 640x480; scale x=2, y=1.5
    boxes = np.array([[10.0, 20.0, 100.0, 120.0]])
    out = scale_boxes(boxes.copy(), orig_shape=(480, 640), resized_shape=(320, 320))
    np.testing.assert_allclose(out, [[20.0, 30.0, 200.0, 180.0]])


def test_scale_boxes_ratio_kept_undoes_letterbox():
    # original 100x200, letterboxed into 640x640 with gain = min(640/100, 640/200) = 3.2
    # padded H -> 320, padding 160 top/bot
    boxes = np.array([[160.0, 320.0, 320.0, 480.0]])
    out = scale_boxes_ratio_kept(
        boxes.copy(), img0_shape=(100, 200), img1_shape=(640, 640), padding=True
    )
    # gain 3.2, x padding 0, y padding 160 -> ((160-0)/3.2, (320-160)/3.2, ...) = (50, 50, 100, 100)
    np.testing.assert_allclose(out, [[50.0, 50.0, 100.0, 100.0]], atol=1.0)


def test_process_boxes_returns_torch_tensor_with_correct_shape():
    boxes = torch.tensor([[[0.5, 0.5, 0.5, 0.5]]])  # [B, Q, 4]
    orig_sizes = torch.tensor([[100, 200]])  # h, w
    out = process_boxes(
        boxes,
        processed_size=(100, 200),
        orig_sizes=orig_sizes,
        keep_ratio=False,
        device="cpu",
    )
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 1, 4)
