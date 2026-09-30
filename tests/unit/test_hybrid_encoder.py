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
