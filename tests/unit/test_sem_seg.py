"""Unit tests for the sem_seg task: decoder shapes, criterion ignore_index handling,
SemSegValidator math, and NEAREST/ignore-fill behavior of the aug pipeline."""

import os

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
                "rare_class_sampling": rare_class_sampling,
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


def test_rfs_weights_floor_and_order():
    from dfine_seg.dl.dataset import rfs_weights

    # (images, classes) instance counts: class 0 everywhere, 1 in half, 2 in one image
    counts = np.array([[3, 0, 0], [2, 1, 0], [1, 1, 1], [4, 0, 0], [0, 0, 0]])
    w = rfs_weights(counts)
    assert w[4] == 1 and w.min() >= 1  # background-only image keeps weight 1, none drop out
    assert w[2] == pytest.approx(np.sqrt(5)) and w[1] == pytest.approx(np.sqrt(5 / 2))


@pytest.fixture
def sampling_data(tmp_path):
    for i, cls in enumerate([0, 0, 0, 5]):
        _write_sample(tmp_path, np.full((16, 16), cls, dtype=np.uint8), stem=f"s{i}")
    pd.DataFrame([f"s{i}.jpg" for i in range(4)]).to_csv(
        tmp_path / "train.csv", header=False, index=False
    )
    pd.DataFrame(["s0.jpg"]).to_csv(tmp_path / "val.csv", header=False, index=False)
    return tmp_path


def test_rare_class_sampling_flag(sampling_data):
    from torch.utils.data import RandomSampler, WeightedRandomSampler

    from dfine_seg.dl.dataset import Loader

    tmp_path = sampling_data

    def train_loader(flag):
        cfg = _cfg(tmp_path, rare_class_sampling=flag)
        if flag is None:
            del cfg.train.rare_class_sampling
        loader = Loader(tmp_path, (64, 64), 2, 0, cfg)
        return loader._build_dataloader_impl(loader._make_dataset("train"), shuffle=True)

    off = train_loader(False)
    assert isinstance(off.sampler, RandomSampler)  # today's plain shuffle
    assert isinstance(train_loader(None).sampler, RandomSampler)  # configs predating the flag
    on = train_loader(True)
    assert isinstance(on.sampler, WeightedRandomSampler)
    assert (tmp_path / "class_presence.json").exists()
    w = on.sampler.weights  # class 5 is in 1 of 4 images: r = sqrt(4) vs sqrt(4/3) for class 0
    assert w[3] == pytest.approx(2.0) and w[0] == pytest.approx((4 / 3) ** 0.5)
    assert list(train_loader(True).sampler) == list(train_loader(True).sampler)  # seeded


def test_sampling_cache_tracks_config_and_label_changes(sampling_data):
    from dfine_seg.dl.dataset import Loader

    cfg = _cfg(sampling_data, rare_class_sampling=True)

    def weights():
        return Loader(sampling_data, (64, 64), 2, 0, cfg).build_dataloaders()[0].sampler.weights

    cfg.train.sem_seg.ignore_index = 0
    assert weights().tolist() == [1.0, 1.0, 1.0, 2.0]
    cfg.train.sem_seg.ignore_index = 255
    assert weights().tolist() == pytest.approx([(4 / 3) ** 0.5] * 3 + [2.0])

    cfg.train.label_to_name = {0: "merged"}
    with pytest.raises(ValueError, match="s3.png: class id 5.*label_to_name"):
        weights()
    cv2.imwrite(str(sampling_data / "labels/s3.png"), np.zeros((16, 16), dtype=np.uint8))
    assert weights().tolist() == [1.0] * 4


def test_sampling_cache_recovers_from_corruption_and_interrupted_write(tmp_path, monkeypatch):
    import json

    from dfine_seg.dl.dataset import load_class_pixels

    _write_sample(tmp_path, np.array([[0, 1, 255]], dtype=np.uint8))
    split = pd.DataFrame(["x.jpg"])
    cache = tmp_path / "class_presence.json"
    cache.write_text("{")
    assert load_class_pixels(tmp_path, split, 255) == {"x.png": {"0": 1, "1": 1}}
    previous = cache.read_text()
    mask = tmp_path / "labels/x.png"
    cv2.imwrite(str(mask), np.array([[1, 1, 255]], dtype=np.uint8))
    # a same-size rewrite can land on the same coarse mtime tick, leaving the cache key equal
    st = mask.stat()
    os.utime(mask, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))

    def interrupt_write(value, stream):
        stream.write("{")
        stream.flush()
        raise OSError("Interrupted cache write")

    with monkeypatch.context() as patch:
        patch.setattr(json, "dump", interrupt_write)
        with pytest.raises(OSError, match="Interrupted cache write"):
            load_class_pixels(tmp_path, split, 255)
    assert cache.read_text() == previous
    assert load_class_pixels(tmp_path, split, 255) == {"x.png": {"1": 2}}


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


# ── fine-tuning from a released checkpoint ───────────────────────────────


def test_finetune_from_sem_seg_ckpt_with_fewer_classes(tmp_path):
    """Cityscapes-style init with a different class count: backbone, encoder, mask fuser
    and seg neck load byte-identical; only the class-count heads re-initialize. This is
    the fine-tune recipe promised in the README/CHANGELOG."""
    from dfine_seg.model.dfine import build_model
    from dfine_seg.model.utils import save_checkpoint

    src = build_model("n", 19, False, "cpu", img_size=[448, 896], task="sem_seg").eval()
    ckpt = tmp_path / "dfine_sem_n_cityscapes.pt"
    save_checkpoint(ckpt, src.state_dict(), {"task": "sem_seg", "model_name": "n"})

    torch.manual_seed(123)  # different init than src, so equality proves a weight was loaded
    model = build_model(
        "n",
        11,
        False,
        "cpu",
        img_size=[448, 896],
        task="sem_seg",
        pretrained_model_path=str(ckpt),
    ).eval()

    pre, new = src.state_dict(), model.state_dict()
    reinit, loaded = [], []
    for k, v in new.items():
        if k not in pre:
            continue
        if v.shape != pre[k].shape:
            reinit.append(k)
        elif torch.equal(v, pre[k]):
            loaded.append(k)

    assert reinit == [
        "decoder.classifier.weight",
        "decoder.classifier.bias",
        "decoder.aux_head.2.weight",
        "decoder.aux_head.2.bias",
    ]
    shape_matched = [k for k, v in new.items() if k in pre and v.shape == pre[k].shape]
    assert set(shape_matched) == set(loaded)  # nothing shape-matching left at random init
    assert "backbone.stem.stem1.conv.weight" in loaded
    assert any(k.startswith("decoder.mask_decoder") for k in loaded)  # the pretrained fuser
    assert any(k.startswith("decoder.neck") for k in loaded)  # the seg neck survives too
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 448, 896))["sem_seg_logits"]  # the built eval_spatial_size
    assert out.shape == (1, 11, 448, 896)


# ── export.sem_seg_probs ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "orig, proc, keep_ratio",
    [
        ((1024, 1365), (512, 1024), True),  # pad 171 px: not a multiple of 4
        ((321, 643), (640, 640), True),  # odd sizes, vertical pad
        ((100, 90), (640, 640), True),  # upscale
        ((1080, 1920), (640, 640), False),  # stretch
    ],
)
def test_probs_transform_matches_letterbox(orig, proc, keep_ratio):
    """Image edges map exactly onto the content edges the real preprocessing produces."""
    from dfine_seg.infer.onnx_model import letterbox, sem_seg_probs_transform

    (H0, W0), (H, W) = orig, proc
    if keep_ratio:
        _, _, (dw, dh) = letterbox(np.zeros((H0, W0, 3), np.uint8), proc, auto=False)
        left, top = int(np.floor(dw)), int(np.floor(dh))
        new_w, new_h = W - 2 * dw, H - 2 * dh  # dw/dh are the half pads
    else:
        left, top, new_w, new_h = 0, 0, W, H
    tf = sem_seg_probs_transform(orig, proc, (H // 4, W // 4), keep_ratio).astype(np.float64)
    # original pixel edges -0.5 / W0 - 0.5 -> input content edges -> probs coords (x + 0.5) / 4 - 0.5
    lo = tf @ [-0.5, -0.5, 1]
    hi = tf @ [W0 - 0.5, H0 - 0.5, 1]
    assert lo[:2] == pytest.approx([left / 4 - 0.5, top / 4 - 0.5], abs=1e-4)
    assert hi[:2] == pytest.approx([(left + new_w) / 4 - 0.5, (top + new_h) / 4 - 0.5], abs=1e-4)


def test_probs_transform_matches_head_upsample():
    """The 1/4 geometry is the head's x4 bilinear (align_corners=False) upsample: upsampling a
    ramp of 1/4-pixel coordinates must give back tf @ x at every interior input pixel."""
    from dfine_seg.infer.onnx_model import sem_seg_probs_transform

    H, W = 64, 128
    tf = sem_seg_probs_transform((H, W), (H, W), (H // 4, W // 4), keep_ratio=False)
    ramp = torch.arange(W // 4, dtype=torch.float64).expand(1, 1, H // 4, W // 4)
    up = torch.nn.functional.interpolate(ramp, scale_factor=4.0, mode="bilinear")[0, 0, 0]
    want = torch.arange(W, dtype=torch.float64) * float(tf[0, 0]) + float(tf[0, 2])
    interior = (want >= 0) & (want <= W // 4 - 1)  # border pixels clamp in the upsample
    assert torch.equal(up[interior], want[interior])


def test_probs_transform_copies_identical():
    """infer/ modules are self-contained copies (AGENTS.md); this pins them to each other. Parsed
    from source, not imported: tensorrt / coremltools are optional and absent in CI."""
    import ast
    from pathlib import Path

    import dfine_seg.infer

    copies = [
        ast.dump(node)
        for path in Path(dfine_seg.infer.__file__).parent.glob("*_model.py")
        for node in ast.parse(path.read_text()).body
        if getattr(node, "name", None) == "sem_seg_probs_transform"
    ]
    assert len(copies) == 7 and len(set(copies)) == 1


def test_postprocess_with_probs_output():
    """A graph with the probs output adds the uncropped probs + transform; the label map is the
    same as without it (labels_to_use filters only the label map)."""
    from dfine_seg.infer.litert_model import LiteRTModel
    from dfine_seg.infer.onnx_model import ONNXModel
    from dfine_seg.infer.ov_model import OVModel

    labels = np.zeros((1, 64, 64), dtype=np.int32)
    labels[:, 32:] = 2
    probs = np.random.rand(1, 3, 16, 16).astype(np.float32)
    sizes = ([(64, 64)], [(32, 64)])  # 32x64 image letterboxed into 64x64
    for cls in (ONNXModel, OVModel, LiteRTModel):
        m = cls.__new__(cls)
        m.keep_ratio, m.labels_to_use = True, [2]
        base = m._postprocess_sem_seg([labels], *sizes)[0]
        outs = [labels, probs][:: -1 if cls is LiteRTModel else 1]  # TFLite may reorder
        out = m._postprocess_sem_seg(outs, *sizes)[0]
        assert set(base) == {"sem_seg"}
        assert set(out) == {"sem_seg", "sem_seg_probs", "sem_seg_probs_tf"}
        assert torch.equal(out["sem_seg"], base["sem_seg"])
        assert torch.equal(out["sem_seg_probs"], torch.from_numpy(probs[0]))  # pads kept
        # the image's top edge is input row 16 (letterbox pad) -> probs edge 16 / 4 - 0.5
        assert out["sem_seg_probs_tf"].numpy() @ [0, -0.5, 1] == pytest.approx([-0.375, 3.5, 1])


def test_torch_model_probs(tmp_path):
    """.pt path: sem_seg_probs=True adds the 1/4 softmax + transform, the label map is unchanged."""
    from dfine_seg.infer.torch_model import TorchModel
    from dfine_seg.model.dfine import build_model
    from dfine_seg.model.utils import save_checkpoint

    src = build_model("n", N_CLASSES, False, "cpu", img_size=[128, 256], task="sem_seg").eval()
    ckpt = tmp_path / "model.pt"
    save_checkpoint(ckpt, src.state_dict(), {"task": "sem_seg", "img_size": [128, 256]})

    img = np.random.randint(0, 255, (100, 300, 3), dtype=np.uint8)
    base = TorchModel(str(ckpt), device="cpu")(img)[0]
    out = TorchModel(str(ckpt), device="cpu", sem_seg_probs=True)(img)[0]
    assert set(base) == {"sem_seg"}
    assert set(out) == {"sem_seg", "sem_seg_probs", "sem_seg_probs_tf"}
    assert torch.equal(out["sem_seg"], base["sem_seg"])
    assert out["sem_seg_probs"].shape == (N_CLASSES, 32, 64)
    assert torch.allclose(out["sem_seg_probs"].sum(0), torch.ones(32, 64), atol=1e-5)
    assert out["sem_seg_probs_tf"].shape == (3, 3)
