"""Pin ETL conversion helpers used to prepare datasets.

Bad numerics here corrupt labels at the source — every downstream metric drops
without any error message.
"""

import cv2
import numpy as np

from src.etl.png_mask_to_yolo import find_contours, mask_to_yolo_lines, to_yolo_poly


def _binary_square(h=64, w=64, x0=10, y0=10, x1=40, y1=50):
    img = np.zeros((h, w), dtype=np.uint8)
    img[y0:y1, x0:x1] = 255
    return img


def test_find_contours_returns_one_for_single_blob():
    img = _binary_square()
    contours = find_contours(img)
    assert len(contours) == 1


def test_find_contours_returns_two_for_two_blobs():
    img = _binary_square()
    img[5, 5] = 255  # tiny separate pixel
    cv2.rectangle(img, (50, 0), (60, 10), 255, -1)  # second blob in top-right
    contours = find_contours(img)
    assert len(contours) >= 2


def test_to_yolo_poly_returns_normalized_coords():
    img = _binary_square()
    contours = find_contours(img)
    poly = to_yolo_poly(contours[0], w=64, h=64, epsilon_ratio=0.005, n_points_max=None)
    assert len(poly) >= 3
    for x, y in poly:
        assert 0.0 <= x <= 1.0
        assert 0.0 <= y <= 1.0


def test_to_yolo_poly_respects_n_points_max():
    # Complex contour: draw a noisy boundary so Douglas-Peucker keeps many points.
    img = np.zeros((128, 128), dtype=np.uint8)
    pts = np.array([[20, 20], [80, 25], [110, 60], [85, 100], [40, 110], [15, 70]], dtype=np.int32)
    cv2.fillPoly(img, [pts], 255)
    contour = find_contours(img)[0]
    poly = to_yolo_poly(contour, w=128, h=128, epsilon_ratio=0.0001, n_points_max=8)
    assert len(poly) <= 8


def test_mask_to_yolo_lines_emits_one_line_per_blob():
    img = _binary_square()
    lines = mask_to_yolo_lines(
        img, class_id=0, thresh_invert=False, min_area_px=1, epsilon_ratio=0.005, n_points_max=None
    )
    assert len(lines) == 1
    parts = lines[0].split()
    assert parts[0] == "0"  # class id
    # Remaining tokens come in (x, y) pairs.
    assert (len(parts) - 1) % 2 == 0
    coords = [float(p) for p in parts[1:]]
    assert all(0.0 <= c <= 1.0 for c in coords)


def test_mask_to_yolo_lines_filters_small_blobs():
    img = np.zeros((64, 64), dtype=np.uint8)
    img[10:12, 10:12] = 255  # tiny 2x2 region, area = 4
    lines = mask_to_yolo_lines(
        img, class_id=0, thresh_invert=False, min_area_px=100,
        epsilon_ratio=0.005, n_points_max=None,
    )
    assert lines == []
