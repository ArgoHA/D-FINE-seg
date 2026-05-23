"""Pin letterbox + scale-back path used by inference and bench.

`letterbox` (preprocess) and `scale_boxes_ratio_kept` (postprocess) are inverses
on the keep_ratio path. If either drifts, predictions land in the wrong place.
"""

import numpy as np

from src.dl.utils import scale_boxes_ratio_kept
from src.infer.torch_model import letterbox


def test_letterbox_preserves_aspect_ratio_no_auto():
    # Non-square input padded to a square net input.
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    out, ratio, pad = letterbox(img, new_shape=(640, 640), auto=False, scaleup=True)
    assert out.shape == (640, 640, 3)
    # gain = 640/200 = 3.2 in both x and y (aspect-preserving).
    assert abs(ratio[0] - 3.2) < 1e-6
    assert abs(ratio[1] - 3.2) < 1e-6
    # Padding only in the vertical direction (image is wider than tall after scaling).
    assert pad[0] == 0.0
    assert pad[1] == 160.0


def test_letterbox_no_upscale_when_scaleup_false():
    # Larger input than net shape: scaleup=False keeps r <= 1.
    img = np.zeros((1280, 1280, 3), dtype=np.uint8)
    _, ratio, _ = letterbox(img, new_shape=(640, 640), auto=False, scaleup=False)
    assert ratio[0] <= 1.0 + 1e-6


def test_scale_boxes_ratio_kept_round_trips_letterbox():
    # Original 100x200; letterbox to 640x640 -> ratio 3.2, pad (0, 160).
    # Pick a box in original-image coords and project it forward + back.
    orig_box = np.array([[10.0, 20.0, 80.0, 60.0]])
    gain = 3.2
    pad_x, pad_y = 0.0, 160.0
    fwd = orig_box * gain
    fwd[:, [0, 2]] += pad_x
    fwd[:, [1, 3]] += pad_y

    back = scale_boxes_ratio_kept(
        fwd.copy(), img0_shape=(100, 200), img1_shape=(640, 640), padding=True
    )
    np.testing.assert_allclose(back, orig_box, atol=0.5)
