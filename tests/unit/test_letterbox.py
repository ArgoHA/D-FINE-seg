"""Pin letterbox + scale-back path used by inference and bench.

`letterbox` (preprocess) and `scale_boxes_ratio_kept` (postprocess) are inverses
on the keep_ratio path. If either drifts, predictions land in the wrong place.
"""

import albumentations as A
import numpy as np

from src.dl.utils import LetterboxRect, scale_boxes_ratio_kept
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


def test_letterbox_rect_dense_mask_nearest_and_ignore_pad():
    # 100x200 -> 640x640: vertical pad of 160 top/bottom (gain 3.2).
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    mask = np.full((100, 200), 3, dtype=np.uint8)
    mask[:, :100] = 7  # two distinct class ids, no id between them
    t = A.Compose(
        [LetterboxRect(640, 640, dense_mask=True, mask_fill=255)],
        mask_interpolation=None,  # LetterboxRect overrides apply_to_mask; Compose setting ignored
    )
    out = t(image=img, mask=mask)["mask"]
    assert out.shape == (640, 640)
    assert set(np.unique(out).tolist()) <= {3, 7, 255}  # NEAREST: no interpolated ids
    assert (out[:160] == 255).all() and (out[480:] == 255).all()  # pad = ignore_index
    assert (out[160:480] != 255).all()  # interior fully labeled


def test_letterbox_rect_binary_mask_unchanged_pads_zero():
    # Default (binary) mode must keep old behavior: pad with 0, values stay {0, 1}.
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    mask = np.ones((100, 200), dtype=np.uint8)
    out = A.Compose([LetterboxRect(640, 640)])(image=img, mask=mask)["mask"]
    assert set(np.unique(out).tolist()) <= {0, 1}
    assert (out[:160] == 0).all() and (out[480:] == 0).all()
