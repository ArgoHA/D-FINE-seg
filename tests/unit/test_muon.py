"""Unit tests for the Muon optimizer and its enc/dec param-selection rule.

Pure: no model is built and no weights are loaded — `_is_muon_param` is exercised
with synthetic (name, tensor) pairs and `MuonWithAuxAdam` with hand-made param
groups. The model-level partitioning (build_optimizer over a real DFINE) lives in
tests/integration/test_muon_param_groups.py.
"""

import torch

from src.d_fine.dfine import MUON_TOKENS, _is_muon_param
from src.d_fine.muon import MuonWithAuxAdam, _zeropower_via_newtonschulz5


def _p(ndim):
    return torch.zeros(*([4] * ndim))


# --- _is_muon_param: allowlist = 2D enc/dec attn/MLP matrix, never the mask head ---


def test_is_muon_param_accepts_encdec_attn_mlp_2d():
    for name in (
        "decoder.decoder.layers.0.self_attn.out_proj.weight",
        "encoder.encoder.0.layers.0.linear1.weight",
        "decoder.decoder.layers.0.cross_attn.attention_weights.weight",
        "decoder.decoder.layers.0.gateway.gate.weight",
    ):
        assert _is_muon_param(name, _p(2)), name


def test_is_muon_param_rejects_mask_head_even_with_token():
    # "mask" guard wins even when a Muon token is present in the name.
    assert not _is_muon_param("decoder.mask_decoder.self_attn.weight", _p(2))


def test_is_muon_param_rejects_non_2d():
    assert not _is_muon_param("decoder.decoder.layers.0.self_attn.in_proj_bias", _p(1))
    assert not _is_muon_param("decoder.mask_decoder.lateral.0.weight", _p(4))


def test_is_muon_param_rejects_backbone_and_tokenless():
    assert not _is_muon_param("backbone.stem.self_attn.weight", _p(2))  # not enc/dec
    assert not _is_muon_param("decoder.norm.weight", _p(2))  # no token
    assert not _is_muon_param("decoder.query_pos_head.layers.0.weight", _p(2))  # det head, no token


def test_muon_tokens_are_the_documented_set():
    assert MUON_TOKENS == ("self_attn", "cross_attn", "linear1", "linear2", "gateway.gate")


# --- MuonWithAuxAdam: per-group routing + the Newton-Schulz core ---


def test_newtonschulz_approximately_orthogonalizes():
    torch.manual_seed(0)
    X = _zeropower_via_newtonschulz5(torch.randn(64, 48))
    sv = torch.linalg.svdvals(X.float())
    # Quintic NS pulls every singular value toward 1 (approximate, not exact).
    assert sv.min() > 0.5 and sv.max() < 1.5
    assert torch.isfinite(X).all()


def test_muon_subclasses_optimizer():
    p = torch.nn.Parameter(torch.zeros(4, 4))
    opt = MuonWithAuxAdam([{"params": [p], "use_muon": True, "lr": 0.1}])
    assert isinstance(opt, torch.optim.Optimizer)


def test_step_routes_muon_vs_adamw_differently():
    # Identical param + identical grad: the Muon group's orthogonalized update must
    # differ from the AdamW group's, proving per-group routing is wired by the flag.
    torch.manual_seed(0)
    g = torch.randn(8, 8)
    p_m = torch.nn.Parameter(torch.zeros(8, 8))
    p_a = torch.nn.Parameter(torch.zeros(8, 8))
    p_m.grad, p_a.grad = g.clone(), g.clone()
    opt = MuonWithAuxAdam(
        [
            {"params": [p_m], "use_muon": True, "lr": 0.1, "weight_decay": 0.0},
            {
                "params": [p_a],
                "use_muon": False,
                "lr": 0.1,
                "betas": (0.9, 0.999),
                "weight_decay": 0.0,
            },
        ]
    )
    opt.step()
    assert torch.isfinite(p_m).all() and torch.isfinite(p_a).all()
    assert not torch.allclose(p_m, torch.zeros(8, 8))  # muon moved
    assert not torch.allclose(p_a, torch.zeros(8, 8))  # adamw moved
    assert not torch.allclose(p_m, p_a, atol=1e-4)  # different update


def test_muon_weight_decay_decoupled_from_grad():
    # Decoupled (AdamW-style) decay: shrinks by 1 - lr*wd even when the grad is zero.
    p = torch.nn.Parameter(torch.ones(4, 4))
    p.grad = torch.zeros(4, 4)
    MuonWithAuxAdam([{"params": [p], "use_muon": True, "lr": 0.1, "weight_decay": 0.5}]).step()
    assert torch.allclose(p.detach(), torch.full((4, 4), 0.95), atol=1e-6)


def test_step_skips_params_without_grad():
    # p.grad is None -> param untouched (no decay, no update applied).
    p = torch.nn.Parameter(torch.ones(4, 4))
    MuonWithAuxAdam([{"params": [p], "use_muon": True, "lr": 0.1, "weight_decay": 0.5}]).step()
    assert torch.allclose(p.detach(), torch.ones(4, 4))
