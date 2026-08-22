"""Pin polygon / mask conversion helpers.

These sit on the segmentation hot path (label rasterization, GT post-processing
in `train.gt_postprocess`) and the ETL conversion utilities. Bad numerics here
shift mask boundaries — and silently degrade mask mAP.
"""

import numpy as np

from dfine_seg.dl.utils import (
    clip_polygon_to_rect,
    norm_poly_to_abs,
    poly_abs_to_mask,
    resample_segments,
    segment2box,
)
from dfine_seg.etl.polys2bbox import polygon_to_bbox


def test_norm_poly_to_abs_basic():
    poly_norm = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    out = norm_poly_to_abs(poly_norm, H=100, W=200)
    np.testing.assert_array_equal(out, [[0, 0], [200, 0], [200, 100], [0, 100]])


def test_norm_poly_to_abs_empty():
    out = norm_poly_to_abs(np.array([]), H=10, W=10)
    assert out.shape == (0, 2)


def test_poly_abs_to_mask_full_image_filled():
    poly = np.array([[0, 0], [49, 0], [49, 49], [0, 49]], dtype=np.float32)
    mask = poly_abs_to_mask(poly, h=50, w=50)
    assert mask.shape == (50, 50)
    # Allow a small boundary tolerance: at minimum the interior should be filled.
    assert mask[1:48, 1:48].all()


def _square(x, y, side):
    return np.array([[x, y], [x + side, y], [x + side, y + side], [x, y + side]], dtype=np.float32)


def test_poly_abs_to_mask_fills_every_part():
    """One instance split into islands: all of them must land in its mask."""
    parts = [_square(2, 2, 10), _square(30, 30, 10), _square(2, 30, 10)]
    mask = poly_abs_to_mask(parts, h=50, w=50)
    for p in parts:
        x, y = p[0]
        assert mask[int(y) + 5, int(x) + 5] == 1


def test_poly_abs_to_mask_parts_union_not_even_odd():
    """Nested parts must OR together (pycocotools semantics), not punch a hole.

    A single multi-contour cv2.fillPoly call would apply the even-odd rule here.
    """
    mask = poly_abs_to_mask([_square(0, 0, 40), _square(10, 10, 20)], h=50, w=50)
    assert mask[20, 20] == 1


def test_poly_abs_to_mask_accepts_bare_array_and_empty_list():
    bare = poly_abs_to_mask(_square(0, 0, 20), h=30, w=30)
    assert bare[5, 5] == 1
    assert poly_abs_to_mask([], h=30, w=30).sum() == 0


def test_clip_polygon_to_rect_keeps_inside():
    # A square fully inside the clipping rect.
    poly = np.array([[10.0, 10.0], [40.0, 10.0], [40.0, 40.0], [10.0, 40.0]])
    out = clip_polygon_to_rect(poly, width=50.0, height=50.0)
    assert out.shape[0] == 4
    np.testing.assert_allclose(sorted(out[:, 0].tolist()), [10, 10, 40, 40])


def test_clip_polygon_to_rect_clips_overhang():
    # Polygon partially outside the right edge.
    poly = np.array([[10.0, 10.0], [60.0, 10.0], [60.0, 40.0], [10.0, 40.0]])
    out = clip_polygon_to_rect(poly, width=50.0, height=50.0)
    assert out.shape[0] >= 3
    assert out[:, 0].max() <= 50.0 + 1e-5  # clipped to right edge


def test_clip_polygon_to_rect_fully_outside_returns_empty():
    poly = np.array([[100.0, 100.0], [110.0, 100.0], [110.0, 110.0], [100.0, 110.0]])
    out = clip_polygon_to_rect(poly, width=50.0, height=50.0)
    assert out.shape == (0, 2)


def test_segment2box_bounds_polygon():
    poly = np.array([[10.0, 15.0], [80.0, 15.0], [80.0, 90.0], [10.0, 90.0]])
    box = segment2box(poly, width=640, height=640)
    np.testing.assert_array_equal(box, [10.0, 15.0, 80.0, 90.0])


def test_resample_segments_to_fixed_length():
    seg = [np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])]
    out = resample_segments(seg, n=128)
    assert out[0].shape == (128, 2)


def test_polygon_to_bbox_matches_min_max():
    # Triangle with vertices (0.1, 0.2), (0.7, 0.3), (0.4, 0.9).
    poly = [0.1, 0.2, 0.7, 0.3, 0.4, 0.9]
    xc, yc, w, h = polygon_to_bbox(poly)
    # bounding box is x: [0.1, 0.7], y: [0.2, 0.9]
    assert abs(xc - 0.4) < 1e-9
    assert abs(yc - 0.55) < 1e-9
    assert abs(w - 0.6) < 1e-9
    assert abs(h - 0.7) < 1e-9
