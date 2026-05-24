"""Stem-weight inflation: 3-channel pretrained -> N-channel target."""

import pytest
import torch

from src.d_fine.utils import inflate_stem_weight, maybe_inflate_stem


def test_inflate_shape_and_values():
    torch.manual_seed(0)
    pretrained = torch.randn(16, 3, 3, 3)
    out = inflate_stem_weight(pretrained, target_in_ch=5)
    assert out.shape == (16, 5, 3, 3)
    # First 3 channels copied verbatim.
    assert torch.equal(out[:, :3], pretrained)
    # Remaining channels are the mean of the 3 RGB filters.
    mean_w = pretrained.mean(dim=1, keepdim=True)
    assert torch.allclose(out[:, 3:4], mean_w)
    assert torch.allclose(out[:, 4:5], mean_w)


def test_inflate_rejects_non_3_pretrained():
    with pytest.raises(ValueError):
        inflate_stem_weight(torch.randn(8, 4, 3, 3), target_in_ch=5)


def test_inflate_rejects_no_op_target():
    with pytest.raises(ValueError):
        inflate_stem_weight(torch.randn(8, 3, 3, 3), target_in_ch=3)


def test_maybe_inflate_swaps_in_place():
    pretrained = torch.randn(8, 3, 3, 3)
    target = torch.zeros(8, 4, 3, 3)
    model_state = {"backbone.stem.stem1.conv.weight": target}
    pretrain_state = {"backbone.stem.stem1.conv.weight": pretrained.clone()}
    maybe_inflate_stem(model_state, pretrain_state)
    after = pretrain_state["backbone.stem.stem1.conv.weight"]
    assert after.shape == (8, 4, 3, 3)
    assert torch.equal(after[:, :3], pretrained)


def test_maybe_inflate_skips_when_shapes_match():
    pretrained = torch.randn(8, 3, 3, 3)
    model_state = {"backbone.stem.stem1.conv.weight": torch.zeros(8, 3, 3, 3)}
    pretrain_state = {"backbone.stem.stem1.conv.weight": pretrained.clone()}
    maybe_inflate_stem(model_state, pretrain_state)
    assert torch.equal(pretrain_state["backbone.stem.stem1.conv.weight"], pretrained)
