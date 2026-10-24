"""Class-head remapping when loading pretrained weights with a different class count."""

import pytest
import torch

from dfine_seg.model.utils import (
    adjust_head_parameters,
    map_class_weights,
    matched_state,
    obj365_ids,
)

HIDDEN = 8


def _head_state(num_classes):
    state = {
        "decoder.denoising_class_embed.weight": torch.randn(num_classes + 1, HIDDEN),
        "decoder.enc_score_head.weight": torch.randn(num_classes, HIDDEN),
        "decoder.enc_score_head.bias": torch.randn(num_classes),
    }
    for i in range(8):
        state[f"decoder.dec_score_head.{i}.weight"] = torch.randn(num_classes, HIDDEN)
        state[f"decoder.dec_score_head.{i}.bias"] = torch.randn(num_classes)
    return state


@pytest.mark.parametrize("num_classes", [1, 10, 120])
def test_coco_to_custom_drops_heads(num_classes):
    # Regression: 0.6.0/0.6.1 raised IndexError here, breaking fine-tuning from COCO weights.
    cur, pretrain = _head_state(num_classes), _head_state(80)
    stat, infos = matched_state(cur, adjust_head_parameters(cur, pretrain))
    assert not any("score_head" in k or "denoising_class_embed" in k for k in stat)
    assert "decoder.enc_score_head.weight" in infos["unmatched"]


def test_obj365_to_coco_remaps_rows():
    pretrain, cur = torch.randn(366, HIDDEN), torch.randn(80, HIDDEN)
    out = map_class_weights(cur, pretrain)
    for coco_id, obj_id in enumerate(obj365_ids):
        assert torch.equal(out[coco_id], pretrain[obj_id + 1])


def test_coco_to_obj365_remaps_rows():
    pretrain, cur = torch.randn(80, HIDDEN), torch.randn(366, HIDDEN)
    out = map_class_weights(cur, pretrain)
    for coco_id, obj_id in enumerate(obj365_ids):
        assert torch.equal(out[obj_id + 1], pretrain[coco_id])


def test_same_size_returns_pretrained():
    pretrain = torch.randn(80, HIDDEN)
    assert map_class_weights(torch.randn(80, HIDDEN), pretrain) is pretrain
