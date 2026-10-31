"""HGNetv2 stem freezing: freeze_at is honored for RGB, an inflated stem always trains."""

from dfine_seg.model.arch.hgnetv2 import HGNetv2
from dfine_seg.model.configs import sizes_cfg


def _stem_trainable(in_channels, freeze_at):
    m = HGNetv2("B0", freeze_at=freeze_at, pretrained=False, in_channels=in_channels)
    flags = {p.requires_grad for p in m.stem.parameters()}
    assert len(flags) == 1
    return flags.pop()


def test_freeze_at_freezes_rgb_stem():
    assert not _stem_trainable(3, freeze_at=0)
    assert _stem_trainable(3, freeze_at=-1)


def test_inflated_stem_stays_trainable():
    assert _stem_trainable(4, freeze_at=0)


def test_no_size_freezes_backbone_by_default():
    for size, cfg in sizes_cfg.items():
        assert cfg["HGNetv2"]["freeze_at"] == -1, size
