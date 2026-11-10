import torch

from dfine_seg.model.arch.hybrid_encoder import HybridEncoder


def test_pos_embed_is_row_major_and_square_unchanged():
    pe = HybridEncoder.build_2d_sincos_position_embedding(w=28, h=14, embed_dim=8)[0]
    assert pe.shape == (14 * 28, 8)
    y, x = 3, 20  # token y * w + x: first half encodes the row, second half the column
    assert torch.allclose(pe[y * 28 + x, [0, 4]], torch.tensor([y, x]).float().sin())

    # square stays identical to upstream's (w, h) meshgrid, so pretrained weights keep meaning
    gw, gh = torch.meshgrid(torch.arange(5.0), torch.arange(5.0), indexing="ij")
    omega = 1.0 / (10000.0 ** (torch.arange(2.0) / 2))
    ow, oh = gw.flatten()[:, None] @ omega[None], gh.flatten()[:, None] @ omega[None]
    upstream = torch.cat([ow.sin(), ow.cos(), oh.sin(), oh.cos()], 1)
    assert torch.equal(HybridEncoder.build_2d_sincos_position_embedding(5, 5, 8)[0], upstream)


def test_repvgg_fusion_keeps_outputs():
    """Export folds every RepVGG 3x3 + 1x1 pair (and conv + BN) into one conv: same function."""
    from dfine_seg.model.configs import models

    torch.manual_seed(0)
    enc = HybridEncoder(**models["s"]["HybridEncoder"]).eval()
    for m in enc.modules():  # fresh BN stats are the identity; make the fold non-trivial
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.normal_(0, 0.1)
            m.running_var.uniform_(0.5, 1.5)
    feats = [torch.randn(1, c, 64 // s, 64 // s) for c, s in zip(enc.in_channels, enc.feat_strides)]
    with torch.no_grad():
        want = enc(feats)
        fused = [m for m in enc.modules() if hasattr(m, "convert_to_deploy")]
        for m in fused:
            m.convert_to_deploy()
        got = enc(feats)
    assert fused and not any(hasattr(m, "conv1") for m in fused)
    for a, b in zip(want, got):
        assert torch.allclose(a, b, atol=1e-4)
