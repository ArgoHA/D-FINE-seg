"""build_optimizer's Muon partitioning over a real (nano, random-init) DFINE.

Guards the allowlist end-to-end: a future arch rename that moved the mask head or
det head under a Muon token would either leak it into the Muon group or empty the
group — both caught here. No weights loaded (random init), CPU only.
"""

import pytest
from torch import optim

from dfine_seg.model.dfine import _is_muon_param, build_model, build_optimizer
from dfine_seg.model.muon import MuonWithAuxAdam


@pytest.fixture(scope="module")
def segment_model_n():
    return build_model("n", num_classes=4, enable_mask_head=True, device="cpu", img_size=[640, 640])


def _opt_kwargs(**extra):
    return dict(
        lr=1e-4, backbone_lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-4, base_lr=1e-4, **extra
    )


def test_use_muon_false_is_plain_adamw(segment_model_n):
    opt = build_optimizer(segment_model_n, **_opt_kwargs(use_muon=False))
    assert isinstance(opt, optim.AdamW)
    assert len(opt.param_groups) == 4
    assert not any(g.get("use_muon") for g in opt.param_groups)


def test_use_muon_true_appends_muon_group(segment_model_n):
    opt = build_optimizer(segment_model_n, **_opt_kwargs(use_muon=True, muon_lr=1e-3))
    assert isinstance(opt, MuonWithAuxAdam)
    assert len(opt.param_groups) == 5
    muon_group = opt.param_groups[-1]  # appended LAST (scheduler's max_lr indexes it)
    assert muon_group["use_muon"] is True
    assert muon_group["lr"] == 1e-3
    assert len(muon_group["params"]) > 0  # non-empty: tokens still match this arch
    assert all(g["use_muon"] is False for g in opt.param_groups[:-1])


def test_muon_group_is_exactly_the_allowlist(segment_model_n):
    # The optimizer's Muon group must equal {p : _is_muon_param(name, p)} — no drift
    # between the build_optimizer loop and the helper it is meant to mirror.
    opt = build_optimizer(segment_model_n, **_opt_kwargs(use_muon=True))
    muon_ids = {id(p) for p in opt.param_groups[-1]["params"]}
    expected = {id(p) for n, p in segment_model_n.named_parameters() if _is_muon_param(n, p)}
    assert muon_ids == expected


def test_mask_and_det_heads_never_in_muon_group(segment_model_n):
    opt = build_optimizer(segment_model_n, **_opt_kwargs(use_muon=True))
    muon_ids = {id(p) for p in opt.param_groups[-1]["params"]}
    name_by_id = {id(p): n for n, p in segment_model_n.named_parameters()}
    for pid in muon_ids:
        name = name_by_id[pid]
        assert "mask" not in name  # mask head excluded
        assert "backbone" not in name  # backbone stays AdamW
        assert "encoder" in name or "decoder" in name  # only enc/dec matrices
    # Known 2D enc/dec params that must stay on AdamW (lack a token, or are mask head).
    excluded = (
        "mask_decoder",
        "mask_head",
        "lqe_layers",
        "denoising_class_embed",
        "query_pos_head",
    )
    for name, p in segment_model_n.named_parameters():
        if any(tok in name for tok in excluded):
            assert id(p) not in muon_ids, name


def test_every_param_in_exactly_one_group(segment_model_n):
    # Partition: the Muon split must neither drop nor double-count any parameter.
    opt = build_optimizer(segment_model_n, **_opt_kwargs(use_muon=True))
    seen = [id(p) for g in opt.param_groups for p in g["params"]]
    all_ids = [id(p) for _, p in segment_model_n.named_parameters()]
    assert sorted(seen) == sorted(all_ids)  # complete + disjoint
    assert len(seen) == len(set(seen))  # no duplicates
