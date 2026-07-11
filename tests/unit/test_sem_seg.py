"""Unit tests for the sem_seg task: decoder shapes, criterion ignore_index handling,
SemSegValidator math, and NEAREST/ignore-fill behavior of the aug pipeline."""

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from src.d_fine.arch.dfine_decoder import SemSegDecoder
from src.d_fine.sem_seg_criterion import SemSegCriterion
from src.dl.validator import SemSegValidator

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


def _make_dataset(tmp_path, rotation_p=0.0):
    from src.dl.dataset import SemSegDataset

    cfg = OmegaConf.create(
        {
            "task": "sem_seg",
            "train": {
                "in_channels": 3,
                "keep_ratio": False,
                "debug_img_path": str(tmp_path / "debug"),
                "label_to_name": {i: str(i) for i in range(N_CLASSES)},
                "sem_seg": {"ignore_index": 255, "class_weights": None},
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
                },
            },
        }
    )
    return SemSegDataset((64, 64), tmp_path, pd.DataFrame(["x.jpg"]), False, "train", cfg)


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


def _write_sample(tmp_path, mask, stem="x"):
    (tmp_path / "images").mkdir(exist_ok=True)
    (tmp_path / "masks").mkdir(exist_ok=True)
    img = np.random.randint(0, 255, (*mask.shape, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "images" / f"{stem}.jpg"), img)
    cv2.imwrite(str(tmp_path / "masks" / f"{stem}.png"), mask)


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


def test_sem_seg_collate_filters_none():
    from src.dl.dataset import sem_seg_collate_fn

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
    from src.infer.torch_model import Torch_model

    logits = torch.full((1, 4, 8, 8), -5.0)
    logits[:, 2, :4] = 5.0  # top half -> class 2, bottom half -> class 0
    logits[:, 0, 4:] = 5.0
    out = Torch_model.process_sem_seg(logits, [(8, 8)], [(32, 16)], keep_ratio=False)
    m = out[0]["sem_seg"]
    assert m.shape == (32, 16) and m.dtype == torch.uint8
    assert set(torch.unique(m).tolist()) == {0, 2}
    assert (m[:16] == 2).all() and (m[16:] == 0).all()  # NEAREST keeps the hard boundary
