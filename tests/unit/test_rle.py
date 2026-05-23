"""Pin RLE encode/decode used to keep eval-time mask memory in check.

Validator stores predictions RLE-encoded between epochs (see REPO_AUDIT H1/H2);
a bad codec here would silently corrupt mask mAP.
"""

import numpy as np
import torch

from src.dl.utils import (
    decode_sample_rle_to_masks,
    encode_sample_masks_to_rle,
    masks_to_rle,
    rle_to_masks,
)


def _random_masks(n, h, w, seed=0):
    rng = np.random.default_rng(seed)
    arr = (rng.random((n, h, w)) > 0.7).astype(np.uint8)
    return torch.from_numpy(arr)


def test_masks_to_rle_to_masks_round_trip():
    masks = _random_masks(4, 32, 48)
    rles = masks_to_rle(masks)
    assert len(rles) == 4
    decoded = rle_to_masks(rles)
    assert decoded.shape == masks.shape
    assert torch.equal(decoded, masks)


def test_masks_to_rle_empty_returns_empty_list():
    assert masks_to_rle(torch.empty(0, 10, 10, dtype=torch.uint8)) == []


def test_rle_to_masks_empty_returns_zero_tensor():
    out = rle_to_masks([])
    assert out.numel() == 0


def test_rle_counts_are_strings_after_encode():
    # Validator's deepcopy / JSON-compat path requires str counts, not bytes.
    rles = masks_to_rle(_random_masks(2, 16, 16))
    for rle in rles:
        assert isinstance(rle["counts"], str)


def test_sample_encode_decode_round_trip():
    masks = _random_masks(3, 24, 32, seed=1)
    sample = {"boxes": torch.zeros((3, 4)), "masks": masks}
    encoded = encode_sample_masks_to_rle(dict(sample))
    assert "masks" not in encoded
    assert "masks_rle" in encoded and encoded["masks_size"] == (24, 32)
    decoded = decode_sample_rle_to_masks(encoded)
    assert torch.equal(decoded["masks"], masks)


def test_sample_encode_with_empty_masks():
    sample = {"masks": torch.empty(0, 10, 10, dtype=torch.uint8)}
    encoded = encode_sample_masks_to_rle(dict(sample))
    assert encoded["masks_rle"] == []
    decoded = decode_sample_rle_to_masks(encoded)
    assert decoded["masks"].shape[0] == 0
