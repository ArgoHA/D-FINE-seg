"""Channel-order contract for CustomDataset._read_image.

After loading any supported format, the first 3 channels must be in RGB order
(matching the pretrained backbone) with any extras (e.g. thermal) preserved.
"""

import cv2
import numpy as np
import pytest

from src.dl.dataset import CustomDataset


def _build(in_channels):
    """Bypass __init__ to keep the test focused on _read_image."""
    ds = CustomDataset.__new__(CustomDataset)
    ds.in_channels = in_channels
    return ds


def _write_bgr_jpeg(path, r, g, b):
    bgr = np.dstack(
        [np.full((4, 4), b, np.uint8), np.full((4, 4), g, np.uint8), np.full((4, 4), r, np.uint8)]
    )
    cv2.imwrite(str(path), bgr)


def _write_bgrt_tiff(path, r, g, b, t):
    bgrt = np.dstack(
        [
            np.full((4, 4), b, np.uint8),
            np.full((4, 4), g, np.uint8),
            np.full((4, 4), r, np.uint8),
            np.full((4, 4), t, np.uint8),
        ]
    )
    cv2.imwrite(str(path), bgrt, [cv2.IMWRITE_TIFF_COMPRESSION, 5])


def test_3ch_jpeg_returns_rgb(tmp_path):
    p = tmp_path / "img.jpg"
    _write_bgr_jpeg(p, r=200, g=50, b=80)
    ds = _build(in_channels=3)

    img = ds._read_image(p)
    assert img.shape == (4, 4, 3)
    px = img[0, 0].tolist()
    # JPEG is lossy; allow small tolerance but channel ordering must be RGB.
    assert abs(px[0] - 200) < 4 and abs(px[1] - 50) < 4 and abs(px[2] - 80) < 4


def test_4ch_tiff_returns_rgbt(tmp_path):
    p = tmp_path / "img.tiff"
    _write_bgrt_tiff(p, r=200, g=50, b=80, t=42)
    ds = _build(in_channels=4)

    img = ds._read_image(p)
    assert img.shape == (4, 4, 4)
    assert img[0, 0].tolist() == [200, 50, 80, 42]


def test_3ch_run_on_4ch_tiff_drops_thermal(tmp_path):
    """RGB-only ablation on multi-channel TIFFs: drop trailing channels, keep RGB."""
    p = tmp_path / "img.tiff"
    _write_bgrt_tiff(p, r=200, g=50, b=80, t=42)
    ds = _build(in_channels=3)

    img = ds._read_image(p)
    assert img.shape == (4, 4, 3)
    assert img[0, 0].tolist() == [200, 50, 80]


def test_3ch_tiff_returns_rgb(tmp_path):
    """A 3-channel TIFF written via cv2 (BGR convention) must come back RGB."""
    p = tmp_path / "img.tiff"
    bgr = np.dstack(
        [
            np.full((4, 4), 80, np.uint8),
            np.full((4, 4), 50, np.uint8),
            np.full((4, 4), 200, np.uint8),
        ]
    )
    cv2.imwrite(str(p), bgr, [cv2.IMWRITE_TIFF_COMPRESSION, 5])
    ds = _build(in_channels=3)

    img = ds._read_image(p)
    assert img.shape == (4, 4, 3)
    assert img[0, 0].tolist() == [200, 50, 80]


def test_returns_none_on_missing(tmp_path):
    ds = _build(in_channels=3)
    assert ds._read_image(tmp_path / "nope.jpg") is None


def test_raises_on_wrong_channel_count(tmp_path):
    p = tmp_path / "img.jpg"
    _write_bgr_jpeg(p, r=10, g=10, b=10)
    ds = _build(in_channels=4)
    with pytest.raises(ValueError):
        ds._read_image(p)
