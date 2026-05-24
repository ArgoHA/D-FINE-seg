"""Channel-order contract for CustomDataset._read_image.

3-channel: cv2 reads PNG/JPEG as BGR; reader swaps to RGB.
>3 channel: stored as .npy in RGB(+extras) order; reader returns it as-is.
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


def _save_rgbt_npy(path, r, g, b, t):
    arr = np.dstack(
        [
            np.full((4, 4), r, np.uint8),
            np.full((4, 4), g, np.uint8),
            np.full((4, 4), b, np.uint8),
            np.full((4, 4), t, np.uint8),
        ]
    )
    np.save(str(path), arr)


def test_3ch_jpeg_returns_rgb(tmp_path):
    p = tmp_path / "img.jpg"
    _write_bgr_jpeg(p, r=200, g=50, b=80)
    ds = _build(in_channels=3)

    img = ds._read_image(p)
    assert img.shape == (4, 4, 3)
    px = img[0, 0].tolist()
    # JPEG is lossy; allow small tolerance but channel ordering must be RGB.
    assert abs(px[0] - 200) < 4 and abs(px[1] - 50) < 4 and abs(px[2] - 80) < 4


def test_4ch_npy_returns_rgbt(tmp_path):
    p = tmp_path / "img.npy"
    _save_rgbt_npy(p, r=200, g=50, b=80, t=42)
    ds = _build(in_channels=4)

    img = ds._read_image(p)
    assert img.shape == (4, 4, 4)
    assert img[0, 0].tolist() == [200, 50, 80, 42]


def test_grayscale_replicated_to_3ch(tmp_path):
    """Default IMREAD_COLOR replicates grayscale to BGR; reader returns 3-ch RGB."""
    p = tmp_path / "gray.png"
    gray = np.full((4, 4), 137, dtype=np.uint8)
    cv2.imwrite(str(p), gray)
    ds = _build(in_channels=3)

    img = ds._read_image(p)
    assert img.shape == (4, 4, 3)
    assert img[0, 0].tolist() == [137, 137, 137]


def test_returns_none_on_missing(tmp_path):
    ds = _build(in_channels=3)
    assert ds._read_image(tmp_path / "nope.jpg") is None


def test_returns_none_on_missing_npy(tmp_path):
    ds = _build(in_channels=4)
    assert ds._read_image(tmp_path / "nope.npy") is None


def test_raises_on_channel_mismatch(tmp_path):
    p = tmp_path / "img.npy"
    _save_rgbt_npy(p, r=10, g=10, b=10, t=10)
    for n in (3, 5):
        ds = _build(in_channels=n)
        with pytest.raises(ValueError):
            ds._read_image(p)
