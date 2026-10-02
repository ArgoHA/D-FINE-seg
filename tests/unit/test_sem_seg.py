"""Unit tests for the sem_seg task: decoder shapes, criterion ignore_index handling,
SemSegValidator math, and NEAREST/ignore-fill behavior of the aug pipeline."""

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from dfine_seg.model.arch.dfine_decoder import SemSegDecoder
from dfine_seg.model.sem_seg_criterion import SemSegCriterion
from dfine_seg.dl.validator import SemSegValidator

N_CLASSES = 7
WEIGHTS = {"loss_ce": 1, "loss_dice": 1, "loss_aux": 0.4}


# ── decoder ──────────────────────────────────────────────────────────────


def test_decoder_shapes_and_aux():
    dec = SemSegDecoder(num_classes=N_CLASSES, feat_channels=[256, 256, 256])
    feats = [torch.randn(2, 256, 8, 8), torch.randn(2, 256, 4, 4), torch.randn(2, 256, 2, 2)]

    dec.train()
    out = dec(feats)
    assert out["sem_seg_logits"].shape == (2, N_CLASSES, 64, 64)  # 1/8 base -> 1/4 -> x4
    assert out["sem_seg_logits_aux"].shape == out["sem_seg_logits"].shape

    dec.eval()
    out = dec(feats)
    assert "sem_seg_logits_aux" not in out  # aux is train-only (dropped at export)


def test_decoder_nano_low_level():
    # nano: encoder feats at 1/16+1/32, backbone 1/8 passed as low_level_feat
    dec = SemSegDecoder(
        num_classes=N_CLASSES, feat_channels=[128, 128], mask_dim=128, mask_low_level_ch=64
    )
    feats = [torch.randn(1, 128, 4, 4), torch.randn(1, 128, 2, 2)]
    low = torch.randn(1, 64, 8, 8)
    out = dec.eval()(feats, low_level_feat=low)
    assert out["sem_seg_logits"].shape == (1, N_CLASSES, 64, 64)


def test_str4_feat_zero_init():
    # S/M/L/X sem_seg: backbone 1/4 feat added after MaskDecoder's x2 upsample
    from dfine_seg.api.ckpt import describe
    from dfine_seg.model.dfine import build_model

    torch.manual_seed(0)
    str4 = build_model("s", N_CLASSES, False, "cpu", img_size=[448, 896], task="sem_seg").eval()
    base = build_model(
        "s", N_CLASSES, False, "cpu", img_size=[448, 896], task="sem_seg", str4_feat=False
    ).eval()
    assert str4.decoder.str4_proj.in_channels == 64 and base.decoder.str4_proj is None
    missing, unexpected = str4.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and missing == ["decoder.str4_proj.weight"]

    x = torch.randn(1, 3, 448, 896)
    with torch.no_grad():
        feats = str4.backbone(x)
        assert feats[0].shape == (1, 64, 112, 224) and len(feats) == 4  # 1/4 + encoder's 3
        out = str4(x)["sem_seg_logits"]
        assert out.shape == (1, N_CLASSES, 448, 896)
        assert torch.equal(out, base(x)["sem_seg_logits"])  # zero weight -> no-op at init

    # old (no-str4) and new checkpoints are told apart by weights alone
    assert describe(str4.state_dict())["str4_feat"]
    assert not describe(base.state_dict())["str4_feat"]


def test_mask_decoder_skip_none_unchanged():
    from dfine_seg.model.arch.dfine_decoder import MaskDecoder

    dec = MaskDecoder(in_chs=[256, 256, 256]).eval()
    feats = [torch.randn(1, 256, 8, 16), torch.randn(1, 256, 4, 8), torch.randn(1, 256, 2, 4)]
    with torch.no_grad():
        assert torch.equal(dec(feats), dec(feats, str4_feat=None))
        assert dec(feats, str4_feat=torch.zeros(1, 256, 16, 32)).shape == (1, 256, 16, 32)


# ── criterion ────────────────────────────────────────────────────────────


def _targets(mask):
    return [{"sem_mask": mask}]


def test_criterion_finite_and_weighted():
    crit = SemSegCriterion(WEIGHTS, num_classes=N_CLASSES)
    logits = torch.randn(1, N_CLASSES, 16, 16)
    aux = torch.randn(1, N_CLASSES, 16, 16)
    target = torch.randint(0, N_CLASSES, (16, 16))
    losses = crit({"sem_seg_logits": logits, "sem_seg_logits_aux": aux}, _targets(target))
    assert set(losses) == {"loss_ce", "loss_dice", "loss_aux"}
    assert all(torch.isfinite(v) for v in losses.values())


def test_criterion_ignore_index():
    """Loss with ignored top half == loss computed on the bottom half alone."""
    crit = SemSegCriterion(WEIGHTS, num_classes=N_CLASSES, ignore_index=255)
    logits = torch.randn(1, N_CLASSES, 16, 16)
    target = torch.randint(0, N_CLASSES, (16, 16))
    target[:8] = 255

    masked = crit({"sem_seg_logits": logits}, _targets(target))
    cropped = crit({"sem_seg_logits": logits[..., 8:, :]}, _targets(target[8:]))
    for k in masked:
        assert torch.allclose(masked[k], cropped[k], atol=1e-6)


def test_criterion_all_ignore_is_zero():
    crit = SemSegCriterion(WEIGHTS, num_classes=N_CLASSES, ignore_index=255)
    logits = torch.randn(2, N_CLASSES, 8, 8, requires_grad=True)
    aux = torch.randn(2, N_CLASSES, 8, 8, requires_grad=True)
    target = torch.full((8, 8), 255, dtype=torch.long)
    losses = crit({"sem_seg_logits": logits, "sem_seg_logits_aux": aux}, [{"sem_mask": target}] * 2)
    total = sum(losses.values())
    assert total.item() == 0.0
    total.backward()  # graph stays intact for DDP/AMP
    assert torch.isfinite(logits.grad).all()


# ── validator ────────────────────────────────────────────────────────────


def test_validator_perfect_and_known_confusion():
    label_to_name = {0: "a", 1: "b", 2: "c"}
    v = SemSegValidator(3, label_to_name)
    gt = torch.tensor([[0, 0], [1, 2]])
    v.update(gt.clone(), gt)
    m = v.compute_metrics(extended=True)
    assert m["mIoU"] == 1.0 and m["pixel_acc"] == 1.0

    # one class-1 pixel predicted as 2: IoU_0=1, IoU_1=0, IoU_2=1/2 -> mIoU=0.5
    v2 = SemSegValidator(3, label_to_name)
    v2.update(torch.tensor([[0, 0], [2, 2]]), gt)
    m2 = v2.compute_metrics(extended=True)
    assert m2["mIoU"] == pytest.approx(0.5)
    assert m2["extended_metrics"]["iou_b"] == 0.0
    assert m2["extended_metrics"]["iou_c"] == 0.5


def test_validator_ignores_255():
    v = SemSegValidator(2, {0: "a", 1: "b"})
    gt = torch.tensor([[0, 255], [255, 255]])
    pred = torch.tensor([[0, 1], [1, 1]])  # wrong only on ignored pixels
    v.update(pred, gt)
    m = v.compute_metrics()
    assert m["mIoU"] == 1.0
    assert v.cm.sum().item() == 1  # only the single valid pixel counted


def test_validator_absent_class_excluded_from_miou():
    v = SemSegValidator(3, {0: "a", 1: "b", 2: "c"})
    gt = torch.tensor([[0, 1]])
    v.update(torch.tensor([[0, 1]]), gt)  # class 2 has no GT pixels
    assert v.compute_metrics()["mIoU"] == 1.0


# ── dataset augs ─────────────────────────────────────────────────────────


def _cfg(tmp_path, rotation_p=0.0, keep_ratio=False, rare_class_sampling=False):
    return OmegaConf.create(
        {
            "task": "sem_seg",
            "train": {
                "seed": 42,
                "use_one_class": False,
                "in_channels": 3,
                "keep_ratio": keep_ratio,
                "debug_img_path": str(tmp_path / "debug"),
                "label_to_name": {i: str(i) for i in range(N_CLASSES)},
                "sem_seg": {
                    "ignore_index": 255,
                    "class_weights": None,
                    "rare_class_sampling": rare_class_sampling,
                },
                "mosaic_augs": {
                    "mosaic_prob": 0.0,
                    "mosaic_scale": [0.5, 1.5],
                    "degrees": 0.0,
                    "translate": 0.2,
                    "shear": 2.0,
                },
                "augs": {
                    "rotation_degree": 45,
                    "rotation_p": rotation_p,
                    "scale_jitter": None,
                    "rotate_90": 0.0,
                    "left_right_flip": 0.0,
                    "up_down_flip": 0.0,
                    "to_gray": 0.0,
                    "blur": 0.0,
                    "gamma": 0.0,
                    "brightness": 0.0,
                    "noise": 0.0,
                    "coarse_dropout": 0.0,
                    "multiscale_prob": 0.0,
                },
            },
        }
    )


def _make_dataset(tmp_path, rotation_p=0.0, keep_ratio=False, mode="train"):
    from dfine_seg.dl.dataset import SemSegDataset

    cfg = _cfg(tmp_path, rotation_p, keep_ratio)
    return SemSegDataset((64, 64), tmp_path, pd.DataFrame(["x.jpg"]), False, mode, cfg)


def test_augs_preserve_class_ids(tmp_path):
    """Resize must be NEAREST for masks: no interpolated (invented) class ids."""
    ds = _make_dataset(tmp_path)
    img = np.random.randint(0, 255, (100, 120, 3), dtype=np.uint8)
    mask = np.zeros((100, 120), dtype=np.uint8)
    mask[40:, :] = 6  # hard 0/6 boundary; LINEAR would produce 1..5
    out = ds.transform(image=img, mask=mask)["mask"]
    assert set(np.unique(out.numpy())) <= {0, 6}


def test_rotate_fills_mask_with_ignore(tmp_path):
    ds = _make_dataset(tmp_path, rotation_p=1.0)
    img = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
    mask = np.full((80, 80), 3, dtype=np.uint8)
    ids = set()
    for _ in range(8):
        ids |= set(np.unique(ds.transform(image=img, mask=mask)["mask"].numpy()))
    assert ids <= {3, 255} and 255 in ids  # rotate corners -> ignore, never class 0


def test_keep_ratio_letterbox_pads_mask_with_ignore(tmp_path):
    """keep_ratio=True val letterbox: dense mask padded with ignore_index (not class 0),
    class ids preserved (NEAREST). Pad rows never supervise."""
    ds = _make_dataset(tmp_path, keep_ratio=True, mode="val")
    img = np.random.randint(0, 255, (40, 120, 3), dtype=np.uint8)  # 3:1 -> vertical pad to 64x64
    mask = np.full((40, 120), 4, dtype=np.uint8)
    mask[:, :60] = 2
    out = ds.transform(image=img, mask=mask)["mask"].numpy()
    assert out.shape == (64, 64)
    assert set(np.unique(out).tolist()) <= {2, 4, 255}  # NEAREST + ignore pad, no invented ids
    assert (out[0] == 255).all() and (out[-1] == 255).all()  # top/bottom rows are pad
    assert (out == 2).any() and (out == 4).any()  # both classes survive


def _write_sample(tmp_path, mask, stem="x"):
    (tmp_path / "images").mkdir(exist_ok=True)
    (tmp_path / "labels").mkdir(exist_ok=True)
    img = np.random.randint(0, 255, (*mask.shape, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "images" / f"{stem}.jpg"), img)
    cv2.imwrite(str(tmp_path / "labels" / f"{stem}.png"), mask)


def test_load_mosaic_target_size_and_ignore_fill(tmp_path):
    """Mosaic emits a target-size pair; mask ids stay {source ids, ignore} — NEAREST
    tiling/warp (no interpolated ids) with ignore_index border fill (never class 0)."""
    mask = np.zeros((48, 80), dtype=np.uint8)
    mask[24:] = 6  # hard 0/6 boundary; interpolation would produce 1..5
    _write_sample(tmp_path, mask)

    ds = _make_dataset(tmp_path)
    ids = set()
    for _ in range(8):
        img4, mask4 = ds._load_mosaic(0)
        assert img4.shape == (64, 64, 3) and mask4.shape == (64, 64)
        ids |= set(np.unique(mask4).tolist())
    assert ids <= {0, 6, 255} and ids & {0, 6}


def test_rcs_weights_favor_rare_class_and_stay_finite():
    from dfine_seg.dl.dataset import rcs_weights

    # (images, classes) pixel counts: class 0 dominant, 1 common, 2 rare (one image), 3 never
    pixels = np.array(
        [[900, 100, 0, 0]] * 4 + [[1000, 0, 0, 0]] * 4 + [[880, 100, 20, 0], [0, 0, 0, 0]]
    )
    w, p = rcs_weights(pixels, temperature=0.01)
    assert np.isfinite(w).all() and np.isfinite(p).all()  # raw exp((1-f)/0.01) would overflow
    assert p[3] == 0 and np.isclose(p.sum(), 1) and p[2] > p[1] > p[0]
    assert w[8] > 10 * w[0] and w[-1] == 0  # rare-class image dominates; empty image never drawn


def test_rare_class_sampling_flag(tmp_path):
    from torch.utils.data import RandomSampler, WeightedRandomSampler

    from dfine_seg.dl.dataset import Loader

    for i, cls in enumerate([0, 0, 0, 5]):
        _write_sample(tmp_path, np.full((16, 16), cls, dtype=np.uint8), stem=f"s{i}")
    pd.DataFrame([f"s{i}.jpg" for i in range(4)]).to_csv(
        tmp_path / "train.csv", header=False, index=False
    )
    pd.DataFrame(["s0.jpg"]).to_csv(tmp_path / "val.csv", header=False, index=False)

    def train_loader(flag):
        loader = Loader(tmp_path, (64, 64), 2, 0, _cfg(tmp_path, rare_class_sampling=flag))
        return loader._build_dataloader_impl(loader._make_dataset("train"), shuffle=True)

    off = train_loader(False)
    assert isinstance(off.sampler, RandomSampler)  # today's plain shuffle
    on = train_loader(True)
    assert isinstance(on.sampler, WeightedRandomSampler)
    assert (tmp_path / "class_presence.json").exists()
    assert np.mean(list(on.sampler)) > 2.5  # rare-class image s3 drawn nearly always
    assert list(train_loader(True).sampler) == list(train_loader(True).sampler)  # seeded


def test_sem_seg_collate_filters_none():
    from dfine_seg.dl.dataset import sem_seg_collate_fn

    item = (torch.zeros(3, 8, 8), torch.zeros(8, 8).long(), "a.jpg", torch.tensor([8, 8]))
    images, targets, paths = sem_seg_collate_fn([item, None])
    assert images.shape == (1, 3, 8, 8)
    assert targets[0]["sem_mask"].shape == (8, 8) and paths == ["a.jpg"]
    assert sem_seg_collate_fn([None, None]) == (None, None, None)


def test_out_of_range_class_ids_fail_loudly(tmp_path):
    """Both GT entry points reject ids >= num_classes (that aren't ignore_index)."""
    v = SemSegValidator(2, {0: "a", 1: "b"})
    with pytest.raises(ValueError, match="class id"):
        v.update(torch.zeros(1, 2).long(), torch.tensor([[0, 26]]))

    mask = np.full((32, 32), 200, dtype=np.uint8)  # e.g. raw labelIds instead of trainIds
    _write_sample(tmp_path, mask)
    with pytest.raises(ValueError, match="class id 200"):
        _make_dataset(tmp_path)._load_image_mask(0)


def test_torch_model_process_sem_seg():
    """Wrapper postprocess: argmax -> NEAREST to original size, uint8, per-image dicts."""
    from dfine_seg.infer.torch_model import TorchModel

    logits = torch.full((1, 4, 8, 8), -5.0)
    logits[:, 2, :4] = 5.0  # top half -> class 2, bottom half -> class 0
    logits[:, 0, 4:] = 5.0
    out = TorchModel.process_sem_seg(logits, [(8, 8)], [(32, 16)], keep_ratio=False)
    m = out[0]["sem_seg"]
    assert m.shape == (32, 16) and m.dtype == torch.uint8
    assert set(torch.unique(m).tolist()) == {0, 2}
    assert (m[:16] == 2).all() and (m[16:] == 0).all()  # NEAREST keeps the hard boundary

    # labels_to_use: ids not requested -> 255 (void); keep only class 2, class 0 -> 255
    out = TorchModel.process_sem_seg(
        logits, [(8, 8)], [(32, 16)], keep_ratio=False, labels_to_use=[2]
    )
    m = out[0]["sem_seg"]
    assert (m[:16] == 2).all() and (m[16:] == 255).all()  # class 0 pixels become void, not 0


# ── non-square input ─────────────────────────────────────────────────────


def test_sem_seg_forward_non_square():
    from dfine_seg.dl.export import SemSegExportWrapper
    from dfine_seg.model.dfine import build_model

    model = build_model("n", N_CLASSES, False, "cpu", img_size=[448, 896], task="sem_seg").eval()
    x = torch.randn(1, 3, 448, 896)
    with torch.no_grad():
        logits = model(x)["sem_seg_logits"]
        label_map = SemSegExportWrapper(model)(x)
    assert logits.shape == (1, N_CLASSES, 448, 896)
    assert label_map.shape == (1, 448, 896)
