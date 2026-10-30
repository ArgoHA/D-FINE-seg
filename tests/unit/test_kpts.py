"""task=kpts: label parsing, keypoint augmentation, loss masking, matching, metrics, export.

The geometry tests use one synthetic object whose two keypoints sit at known offsets inside
its box, so every transform can be checked against where the points must end up.
"""

import random

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from dfine_seg.api.ckpt import describe
from dfine_seg.dl.dataset import (
    CustomDataset,
    kpt_config,
    kpts_to_canvas,
    load_coco_split,
    parse_yolo_pose_file,
)
from dfine_seg.dl.export import DFINEPostProcessor
from dfine_seg.dl.utils import KptCoarseDropout, process_boxes, process_kpts
from dfine_seg.dl.validator import Validator
from dfine_seg.model.arch import dfine_decoder
from dfine_seg.model.dfine import build_loss, build_model
from dfine_seg.model.dfine_criterion import DFINECriterion
from dfine_seg.model.kpts import COCO_SIGMAS, DEFAULT_SIGMA, kpt_oks, resolve_sigmas
from dfine_seg.model.matcher import HungarianMatcher

H, W = 200, 300
BOX = (100, 50, 200, 150)  # abs xyxy
# left point v=2 (visible), right point v=1 (occluded), third slot unlabeled
KPTS = [(110, 60, 2), (190, 60, 1), (0, 0, 0)]


def _cfg(tmp_path, lr=0.0, ud=0.0, mosaic=0.0, keep_ratio=False, shape=(3, 3), flip_idx=(1, 0, 2)):
    return OmegaConf.create(
        {
            "task": "kpts",
            "train": {
                "in_channels": 3,
                "keep_ratio": keep_ratio,
                "use_one_class": False,
                "debug_img_path": str(tmp_path / "debug"),
                "label_to_name": {0: "obj"},
                "kpts": {"shape": list(shape), "flip_idx": list(flip_idx) if flip_idx else None},
                "mosaic_augs": {
                    "mosaic_prob": mosaic,
                    "mosaic_scale": [1.0, 1.0],
                    "degrees": 0.0,
                    "translate": 0.0,
                    "shear": 0.0,
                },
                "augs": {
                    "rotation_degree": 0,
                    "rotation_p": 0,
                    "multiscale_prob": 0,
                    "rotate_90": 0,
                    "left_right_flip": lr,
                    "up_down_flip": ud,
                    "to_gray": 0,
                    "blur": 0,
                    "gamma": 0,
                    "brightness": 0,
                    "noise": 0,
                    "coarse_dropout": 0,
                },
            },
        }
    )


@pytest.fixture
def pose_root(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    cv2.imwrite(str(tmp_path / "images" / "a.jpg"), np.zeros((H, W, 3), np.uint8))
    x1, y1, x2, y2 = BOX
    box = f"{(x1 + x2) / 2 / W} {(y1 + y2) / 2 / H} {(x2 - x1) / W} {(y2 - y1) / H}"
    pts = " ".join(f"{x / W} {y / H} {v}" for x, y, v in KPTS)
    (tmp_path / "labels" / "a.txt").write_text(f"0 {box} {pts}\n")
    return tmp_path


def _item(root, cfg, size=(64, 64)):
    ds = CustomDataset(size, root, pd.DataFrame(["a.jpg"]), False, "train", cfg)
    out = ds[0]
    return out[2], out[7]  # boxes (cxcywh norm), keypoints (x, y norm, v)


# ── parsing ──────────────────────────────────────────────────────────────


def test_parse_pose_with_flags(pose_root):
    boxes, kpts = parse_yolo_pose_file(pose_root / "labels" / "a.txt", (3, 3))
    assert boxes.shape == (1, 5) and kpts.shape == (1, 3, 3)
    np.testing.assert_allclose(kpts[0, :, 2], [2, 1, 0])
    np.testing.assert_allclose(kpts[0, 0, :2], [110 / W, 60 / H], rtol=1e-5)


def test_parse_pose_without_flags(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("0 0.5 0.5 0.2 0.2 0.4 0.4 0 0\n")
    _, kpts = parse_yolo_pose_file(p, (2, 2))
    np.testing.assert_allclose(kpts[0], [[0.4, 0.4, 2], [0, 0, 0]])  # (0, 0) = unlabeled


@pytest.mark.parametrize(
    "line", ["0 0.5 0.5 0.2 0.2 0.4 0.4 2", "0 0.5 0.5 0.2 0.2 0.4 0.4 2 0.1 0.1 2 0.3"]
)
def test_parse_pose_wrong_length_raises(tmp_path, line):
    p = tmp_path / "a.txt"
    p.write_text(line + "\n")
    with pytest.raises(ValueError, match="expects 11 values"):
        parse_yolo_pose_file(p, (2, 3))


def test_parse_pose_bad_flag_raises(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("0 0.5 0.5 0.2 0.2 0.4 0.4 3\n")
    with pytest.raises(ValueError, match="flags"):
        parse_yolo_pose_file(p, (1, 3))


def test_coco_keypoints_and_area(tmp_path):
    import json

    coco = {
        "images": [{"id": 1, "file_name": "a.jpg"}],
        "categories": [{"id": 1, "name": "person"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 42.0,
             "keypoints": [1, 2, 2, 0, 0, 0]},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 20]},  # no keypoints
        ],
    }  # fmt: skip
    (tmp_path / "train.json").write_text(json.dumps(coco))
    entry = load_coco_split(tmp_path / "train.json", num_kpts=2)[0][0]
    np.testing.assert_allclose(entry["kpts_abs"][0], [[1, 2, 2], [0, 0, 0]])
    assert not entry["kpts_abs"][1].any()  # missing keypoints -> unlabeled slots
    np.testing.assert_allclose(entry["areas"], [42.0, 10 * 20 * 0.53])

    with pytest.raises(ValueError, match="expects 3"):
        load_coco_split(tmp_path / "train.json", num_kpts=3)


# ── config ───────────────────────────────────────────────────────────────


def test_flip_idx_required_with_reflections(tmp_path):
    with pytest.raises(ValueError, match="flip_idx is required"):
        kpt_config(_cfg(tmp_path, lr=0.5, flip_idx=None))
    with pytest.raises(ValueError, match="flip_idx is required"):
        kpt_config(_cfg(tmp_path, ud=0.5, flip_idx=None))
    assert kpt_config(_cfg(tmp_path, flip_idx=None))[1] is None  # no reflections: not needed
    assert kpt_config(_cfg(tmp_path, lr=0.5, shape=(1, 3), flip_idx=None))[1] == [0]  # K=1
    with pytest.raises(ValueError, match="permutation"):
        kpt_config(_cfg(tmp_path, flip_idx=(0, 0, 2)))


def test_sigmas():
    assert resolve_sigmas(17) == COCO_SIGMAS
    assert resolve_sigmas(2) == [DEFAULT_SIGMA] * 2
    assert resolve_sigmas(2, [0.1, 0.2]) == [0.1, 0.2]
    with pytest.raises(ValueError):
        resolve_sigmas(2, [0.1])


# ── augmentation geometry ────────────────────────────────────────────────


def _rel(boxes, kpts):
    """Keypoint offsets relative to their box, in box units."""
    return (kpts[..., :2] - boxes[:, None, :2]) / boxes[:, None, 2:]


def test_plain_resize(pose_root):
    boxes, kpts = _item(pose_root, _cfg(pose_root))
    torch.testing.assert_close(_rel(boxes, kpts)[0, :2], torch.tensor([[-0.4, -0.4], [0.4, -0.4]]))
    assert kpts[0, :, 2].tolist() == [2, 1, 0]
    assert kpts[0, 2, :2].tolist() == [0, 0]


def test_hflip_swaps_slots_and_flags(pose_root):
    boxes, kpts = _item(pose_root, _cfg(pose_root, lr=1.0))
    # mirrored left point lands at +0.4 and moves into the right slot together with its flag
    torch.testing.assert_close(_rel(boxes, kpts)[0, :2], torch.tensor([[-0.4, -0.4], [0.4, -0.4]]))
    assert kpts[0, :, 2].tolist() == [1, 2, 0]


def test_vflip_swaps_slots(pose_root):
    boxes, kpts = _item(pose_root, _cfg(pose_root, ud=1.0))
    # x stays, y mirrors; the slots still swap (a reflection turns left into right)
    torch.testing.assert_close(_rel(boxes, kpts)[0, :2], torch.tensor([[0.4, 0.4], [-0.4, 0.4]]))
    assert kpts[0, :, 2].tolist() == [1, 2, 0]


def test_letterbox_keeps_points_on_object(pose_root):
    boxes, kpts = _item(pose_root, _cfg(pose_root, keep_ratio=True))
    torch.testing.assert_close(_rel(boxes, kpts)[0, :2], torch.tensor([[-0.4, -0.4], [0.4, -0.4]]))


def test_mosaic_carries_keypoints(pose_root):
    random.seed(0)
    cfg = _cfg(pose_root, mosaic=1.0)
    ds = CustomDataset((128, 128), pose_root, pd.DataFrame(["a.jpg"]), False, "train", cfg)
    seen = 0
    for _ in range(10):
        out = ds[0]
        boxes, kpts = out[2], out[7]
        assert kpts.shape == (len(boxes), 3, 3)
        # untruncated copies: each tile is the image resized to the canvas, box = 1/3 x 1/2
        whole = ((boxes[:, 2:] - torch.tensor([1 / 3, 1 / 2])).abs() < 1e-3).all(1)
        rel = _rel(boxes[whole], kpts[whole])[:, :2]
        expected = torch.tensor([[-0.4, -0.4], [0.4, -0.4]]).expand_as(rel)
        torch.testing.assert_close(rel, expected, atol=1e-3, rtol=0)
        seen += int(whole.sum())
    assert seen


def test_points_outside_canvas_become_unlabeled():
    kpts = np.array([[[5, 5, 2], [-1, 5, 2], [5, 11, 1]]], np.float32)
    out = kpts_to_canvas(kpts, h=10, w=10)
    np.testing.assert_allclose(out[0, :, 2], [2, 0, 0])
    np.testing.assert_allclose(out[0, 1:, :2], 0)
    np.testing.assert_allclose(out[0, 0, :2], [0.5, 0.5])


# ── loss / matching ──────────────────────────────────────────────────────


def _criterion(visibility=True):
    return DFINECriterion(
        matcher=None,
        weight_dict={},
        losses=["kpts"],
        kpt_sigmas=[0.07] * 3,
        kpt_visibility=visibility,
    )


def _loss_inputs(perfect_labeled=True):
    tgt = torch.tensor([[[0.4, 0.4, 2], [0.6, 0.4, 1], [0.0, 0.0, 0]]])
    pred = torch.zeros(1, 2, 3, 4)
    pred[0, 0, :, :2] = tgt[0, :, :2] if perfect_labeled else 0.9
    pred[0, 0, 2, :2] = torch.tensor([0.95, 0.05])  # unlabeled slot: far off, must not count
    targets = [{"boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]]), "kpts": tgt}]
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    return {"pred_kpts": pred}, targets, indices


def test_coordinate_losses_ignore_unlabeled_points():
    losses = _criterion().loss_kpts(*_loss_inputs(), num_boxes=1)
    assert losses["loss_kpt_l1"].item() == pytest.approx(0, abs=1e-6)
    assert losses["loss_kpt_oks"].item() == pytest.approx(0, abs=1e-6)
    off = _criterion().loss_kpts(*_loss_inputs(perfect_labeled=False), num_boxes=1)
    assert off["loss_kpt_l1"] > 0 and off["loss_kpt_oks"] > 0


def test_presence_and_visible_targets():
    outputs, targets, indices = _loss_inputs()
    logits = torch.tensor([20.0, 20.0, -20.0])  # presence matches v > 0 for every slot
    outputs["pred_kpts"][0, 0, :, 2] = logits
    # visible: v=2 -> 1, v=1 -> 0; the unlabeled slot is wrong on purpose and must not count
    outputs["pred_kpts"][0, 0, :, 3] = torch.tensor([20.0, -20.0, 20.0])
    losses = _criterion().loss_kpts(outputs, targets, indices, num_boxes=1)
    assert losses["loss_kpt_presence"].item() == pytest.approx(0, abs=1e-6)
    assert losses["loss_kpt_visible"].item() == pytest.approx(0, abs=1e-6)


def test_visible_loss_skipped_without_flags():
    losses = _criterion(visibility=False).loss_kpts(*_loss_inputs(), num_boxes=1)
    assert "loss_kpt_visible" not in losses and "loss_kpt_presence" in losses


def test_matcher_keypoint_cost_separates_identical_boxes():
    """Two people with the same box: only the keypoint cost can tell the queries apart."""
    box = torch.tensor([0.5, 0.5, 0.4, 0.6])
    kp_a = torch.tensor([[0.40, 0.30, 2], [0.60, 0.30, 2]])
    kp_b = torch.tensor([[0.45, 0.70, 2], [0.55, 0.75, 2]])
    targets = [
        {"labels": torch.tensor([0, 0]), "boxes": box.repeat(2, 1), "kpts": torch.stack([kp_a, kp_b])}
    ]  # fmt: skip
    weights = {"cost_class": 2, "cost_bbox": 5, "cost_giou": 2, "cost_kpt_l1": 5, "cost_kpt_oks": 2}
    matcher = HungarianMatcher(weights, use_focal_loss=True, kpt_sigmas=[0.07, 0.07])
    for order in ([kp_b, kp_a], [kp_a, kp_b]):
        pred = torch.zeros(1, 2, 2, 4)
        pred[0, :, :, :2] = torch.stack(order)[..., :2]
        outputs = {
            "pred_logits": torch.full((1, 2, 1), 5.0),
            "pred_boxes": box.repeat(1, 2, 1),
            "pred_kpts": pred,
        }
        q, t = matcher(outputs, targets)["indices"][0]
        expected = [1, 0] if order[0] is kp_b else [0, 1]
        assert t[q.argsort()].tolist() == expected

    # without keypoint costs (pre/encoder outputs) the same call still works on boxes alone
    outputs.pop("pred_kpts")
    assert len(matcher(outputs, targets)["indices"][0][0]) == 2


def test_model_and_loss_end_to_end():
    model = build_model("n", 2, False, "cpu", img_size=[320, 320], num_kpts=3).train()
    loss_fn = build_loss(
        "n", 2, 0.0, False, task="kpts", kpts_meta={"shape": [3, 2], "sigmas": [0.07] * 3}
    )
    kp = torch.tensor([[[0.4, 0.4, 2], [0.6, 0.4, 2], [0, 0, 0]]])
    targets = [
        {
            "labels": torch.tensor([1]),
            "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]]),
            "kpts": kp,
        }
    ]
    out = model(torch.rand(1, 3, 320, 320), targets)
    # training: keypoint decoder points + per-axis log scales
    assert out["pred_kpts"].shape == (1, 300, 3, 6)
    assert all("pred_kpts" in a for a in out["aux_outputs"] + out["dn_outputs"])
    assert "pred_kpts" not in out["pre_outputs"]
    losses = loss_fn(out, targets)
    assert "loss_kpt_l1_aux_0" in losses and "loss_kpt_oks_dn_0" in losses
    # per-query head points, keypoint decoder layers, keypoint denoising, quality
    assert "loss_kpt_l1_kpt_0" in losses and "loss_kpt_l1_kpt_dn_0" in losses
    assert "loss_kpt_quality" in losses
    assert not any("visible" in k for k in losses)
    sum(losses.values()).backward()
    assert model.decoder.kpt_head[0].layers[-1].weight.grad.abs().sum() > 0

    model.eval()
    with torch.no_grad():
        out = model(torch.rand(1, 3, 320, 320))
    kpts = out["pred_kpts"]
    assert kpts.shape == (1, 300, 3, 4) and ((kpts[..., 2:] >= 0) & (kpts[..., 2:] <= 1)).all()
    # zero-initialized head: every point starts at its box center
    torch.testing.assert_close(kpts[..., 0, :2], out["pred_boxes"][..., :2])


def test_describe_infers_kpts():
    sd = build_model("s", 2, False, "cpu", num_kpts=5).state_dict()
    info = describe(sd)
    assert info["task"] == "kpts" and info["num_kpts"] == 5 and info["num_classes"] == 2
    assert info["num_kpt_layers"] == 2
    # per-query head only (no keypoint decoder weights) -> 0 layers
    head_only = build_model("s", 2, False, "cpu", num_kpts=5, num_kpt_layers=0)
    assert describe(head_only.state_dict())["num_kpt_layers"] == 0
    det = build_model("s", 2, False, "cpu").state_dict()
    assert describe(det)["task"] == "detect" and describe(det)["num_kpts"] == 0


# ── keypoint decoder ─────────────────────────────────────────────────────


def test_kpt_decoder_chunked_cross_attention_is_exact(monkeypatch):
    """The TensorRT GridSample workaround splits tokens into chunks; results must not change."""
    torch.manual_seed(0)
    model = build_model("n", 1, False, "cpu", img_size=[320, 320], num_kpts=17).eval()
    for p in model.decoder.kpt_decoder.reg_head.parameters():  # zero-init = attention-blind
        p.data.normal_(0, 0.1)
    x = torch.rand(1, 3, 320, 320)
    with torch.no_grad():
        ref = model(x)["pred_kpts"]
        monkeypatch.setattr(dfine_decoder, "TRT_GRID_SAMPLE_MAX_ROWS", 7)
        chunked = model(x)["pred_kpts"]
        monkeypatch.setattr(model.decoder.kpt_decoder, "ref_scale", 0.5)
        moved = model(x)["pred_kpts"]
    torch.testing.assert_close(chunked, ref)
    assert not torch.allclose(moved, ref)  # the points do depend on what is sampled


def test_rescore_only_touches_refined_queries():
    model = build_model("n", 1, False, "cpu", num_kpts=1).decoder
    model.kpt_rescore = 0.3
    logits = torch.tensor([[[2.0], [-1.0]]])
    quality = torch.tensor([[[0.0, 1.0], [5.0, 0.0]]])  # query 0 refined, quality 0.5
    out = model._rescore(logits, quality)
    want = torch.sigmoid(torch.tensor(2.0)) ** 0.7 * 0.5**0.3
    torch.testing.assert_close(torch.sigmoid(out[0, 0, 0]), want, atol=1e-4, rtol=0)
    assert out[0, 1, 0] == logits[0, 1, 0]


def test_kpt_denoising_groups_map_slots_to_gts(monkeypatch):
    """Keypoint denoising slot g * M + j supervises GT j; noise-free points give zero L1."""
    model = build_model("n", 1, False, "cpu", img_size=[320, 320], num_kpts=2).train()
    model.decoder.kpt_dn_groups, model.decoder.kpt_dn_noise = 3, 1e-9
    # identity keypoint decoder: refined points = the (GT) starting points
    for p in model.decoder.kpt_decoder.reg_head.parameters():
        p.data.zero_()
    loss_fn = build_loss(
        "n", 1, 0.0, False, task="kpts", kpts_meta={"shape": [2, 2], "sigmas": [0.07] * 2}
    )
    kp = torch.tensor([[[0.4, 0.4, 2], [0.6, 0.4, 2]], [[0.2, 0.7, 2], [0.3, 0.8, 2]]])
    targets = [
        {
            "labels": torch.tensor([0, 0]),
            "boxes": torch.tensor([[0.5, 0.4, 0.3, 0.3]] * 2),
            "kpts": kp,
        },
        {
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[0.25, 0.75, 0.2, 0.2]]),
            "kpts": kp[1:],
        },
    ]
    out = model(torch.rand(2, 3, 320, 320), targets)
    assert out["kpt_dn_outputs"][0]["pred_kpts"].shape[1] == 3 * 2  # groups x max GTs
    losses = loss_fn(out, targets)
    assert losses["loss_kpt_l1_kpt_dn_0"] < 1e-4


# ── metrics / postprocess / export ───────────────────────────────────────


def _pose_gt():
    return [
        {
            "labels": torch.tensor([0, 0]),
            "boxes": torch.tensor([[10.0, 10.0, 60.0, 110.0], [100.0, 10.0, 150.0, 110.0]]),
            # second person has no labeled points: ignored by the keypoint metrics (COCO rule)
            "kpts": torch.tensor(
                [[[20.0, 20.0, 2], [50.0, 20.0, 1]], [[0.0, 0.0, 0], [0.0, 0.0, 0]]]
            ),
            "areas": None,
        }
    ]


def _pose_preds(shift=0.0):
    kp = torch.tensor([[[20.0 + shift, 20.0, 0.9, 0.9], [50.0, 20.0, 0.9, 0.1]]] * 2)
    pred = {
        "labels": torch.tensor([0, 0]),
        "boxes": torch.tensor([[10.0, 10.0, 60.0, 110.0], [100.0, 10.0, 150.0, 110.0]]),
        "scores": torch.tensor([0.9, 0.8]),
        "kpts": kp,
    }
    return [pred | {f"all_{k}": v for k, v in pred.items()}]


def _validator(preds):
    return Validator(
        _pose_gt(),
        preds,
        label_to_name={0: "person"},
        kpts_meta={"shape": [2, 3], "sigmas": [0.07] * 2, "conf_thresh": 0.5},
    ).compute_metrics(extended=True)


def test_kpt_metrics_perfect():
    m = _validator(_pose_preds())
    ext = m["extended_metrics"]
    # the pred on the unlabeled person is neither TP nor FP
    assert m["f1_kpt"] == 1 and ext["precision_kpt"] == 1 and ext["oks_kpt"] == pytest.approx(1)
    assert m["kpt_px_err"] == 0 and ext["pck_kpt"] == 1
    assert ext["mAP_50_kpt"] == pytest.approx(1) and m["mAP_50_95_kpt"] == pytest.approx(1)
    assert m["visible_acc"] == 1  # 0.9 -> visible (v=2), 0.1 -> occluded (v=1)
    assert m["presence_f1"] == 1 and ext["presence_recall"] == 1


def test_kpt_metrics_penalize_error():
    m = _validator(_pose_preds(shift=6.0))
    assert m["kpt_px_err"] == pytest.approx(3.0)  # one of two points off by 6 px
    assert m["mAP_50_95_kpt"] < 1
    assert m["extended_metrics"]["oks_kpt"] < 1


def test_kpt_oks_pairwise_shape():
    pred, gt = torch.rand(4, 3, 2), torch.rand(2, 3, 2)
    vis = torch.ones(2, 3, dtype=torch.bool)
    oks = kpt_oks(pred[:, None], gt[None], vis[None], torch.ones(1, 2), torch.full((3,), 0.07))
    assert oks.shape == (4, 2)


@pytest.mark.parametrize("keep_ratio", [False, True])
def test_keypoints_rescale_like_boxes(keep_ratio):
    """A point on a box corner must land on the rescaled box corner."""
    boxes = torch.tensor([[[0.5, 0.4, 0.2, 0.3]]])  # norm cxcywh
    corner = torch.tensor([[[[0.4, 0.25, 1.0, 1.0]]]])  # its top-left corner
    orig = torch.tensor([[480.0, 640.0]])
    b = process_boxes(boxes, (640, 640), orig, keep_ratio, "cpu")
    k = process_kpts(corner, (640, 640), orig, keep_ratio)
    torch.testing.assert_close(k[0, 0, 0, :2], b[0, 0, :2].float())


def test_postprocessor_emits_keypoints_in_input_pixels():
    pp = DFINEPostProcessor(num_classes=2, num_top_queries=3)
    kp = torch.rand(1, 5, 4, 4)
    outputs = {
        "pred_logits": torch.randn(1, 5, 2),
        "pred_boxes": torch.rand(1, 5, 4),
        "pred_kpts": kp,
    }
    labels, boxes, scores, kpts = pp(outputs, 320, 640)
    qidx = torch.topk(outputs["pred_logits"].sigmoid().flatten(1), 3).indices // 2
    torch.testing.assert_close(kpts[0], kp[0, qidx[0]] * torch.tensor([640, 320, 1, 1]))


def test_etl_keypoints_round_trip(tmp_path):
    """COCO json -> YOLO-pose lines -> COCO json keeps every point and flag."""
    import json

    from dfine_seg.etl.coco2yolo import convert_coco_json
    from dfine_seg.etl.yolo2coco import yolo_to_coco

    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir()
    kps = [10, 20, 2, 0, 0, 0, 30, 40, 1]
    coco = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 50}],
        "categories": [{"id": 1, "name": "person"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [5, 10, 40, 30], "keypoints": kps}],
    }  # fmt: skip
    (ann_dir / "train.json").write_text(json.dumps(coco))
    convert_coco_json(str(ann_dir), use_kpts=True)
    label = tmp_path / "yolo_labels" / "train" / "a.txt"
    boxes, kpts = parse_yolo_pose_file(label, (3, 3))
    np.testing.assert_allclose(kpts[0], [[0.1, 0.4, 2], [0, 0, 0], [0.3, 0.8, 1]], atol=1e-6)

    (tmp_path / "images").mkdir()
    cv2.imwrite(str(tmp_path / "images" / "a.jpg"), np.zeros((50, 100, 3), np.uint8))
    out = tmp_path / "back.json"
    yolo_to_coco(label.parent, tmp_path / "images", out, None, ["person"], kpt_shape=(3, 3))
    ann = json.loads(out.read_text())["annotations"][0]
    np.testing.assert_allclose(ann["keypoints"], kps, atol=1e-3)
    assert ann["num_keypoints"] == 2
    assert ann["area"] == pytest.approx(40 * 30 * 0.53)
    with pytest.raises(ValueError, match="expected 5 values"):  # pose lines are not boxes
        yolo_to_coco(label.parent, tmp_path / "images", out, None, ["person"])


@pytest.mark.parametrize("task", ["kpts", "segment"])
def test_frozen_training_keeps_aux_supervision(task):
    """freeze_except_* + eval() on the returned modules must leave the decoder in train mode."""
    from dfine_seg.model.dfine import freeze_except_kpt, freeze_except_mask

    model = build_model("n", 1, task == "segment", "cpu", num_kpts=3 * (task == "kpts"))
    model.train()
    for m in (freeze_except_kpt if task == "kpts" else freeze_except_mask)(model):
        m.eval()
    assert all(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules() if not m.training)
    targets = [
        {
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]]),
            "kpts": torch.tensor([[[0.4, 0.4, 2], [0.6, 0.4, 2], [0, 0, 0]]]),
            "masks": torch.ones(1, 320, 320, dtype=torch.uint8),
        }
    ]
    out = model(torch.rand(1, 3, 320, 320), targets)
    assert len(out["aux_outputs"]) == model.decoder.num_layers - 1


def test_freeze_except_kpt_trains_every_keypoint_module():
    """The keypoint decoder (incl. the rescoring quality head) is randomly initialized from
    detection weights, so it must train with the per-query head."""
    from dfine_seg.model.dfine import freeze_except_kpt

    model = build_model("n", 1, False, "cpu", num_kpts=2)
    freeze_except_kpt(model)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert any(n.startswith("decoder.kpt_decoder.quality_head") for n in trainable)
    assert all(n.startswith(("decoder.kpt_head", "decoder.kpt_decoder")) for n in trainable)


def test_frozen_batchnorms_survive_syncbn_conversion():
    from dfine_seg.model.dfine import freeze_except_kpt, frozen_batchnorms

    model = build_model("n", 1, False, "cpu", num_kpts=2)
    assert freeze_except_kpt(model)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).train()
    frozen = frozen_batchnorms(model)
    assert frozen and all(isinstance(m, torch.nn.SyncBatchNorm) for m in frozen)
    for m in frozen:
        m.eval()
    assert not any(m.training for m in model.modules() if isinstance(m, torch.nn.SyncBatchNorm))


def test_coco2yolo_pads_anns_without_keypoints(tmp_path):
    import json

    from dfine_seg.etl.coco2yolo import convert_coco_json

    (tmp_path / "ann").mkdir()
    coco = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 50}],
        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "car"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [5, 10, 40, 30], "keypoints": [10, 20, 2]},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [50, 10, 40, 30]},
        ],
    }  # fmt: skip
    (tmp_path / "ann" / "train.json").write_text(json.dumps(coco))
    convert_coco_json(str(tmp_path / "ann"), use_kpts=True)
    _, kpts = parse_yolo_pose_file(tmp_path / "yolo_labels" / "train" / "a.txt", (1, 3))
    assert kpts[:, 0, 2].tolist() == [2, 0]


def test_visualizer_zero_threshold_is_respected():
    from dfine_seg.viz import Visualizer

    v = Visualizer(2, kpt_conf_thresh=0.0, kpt_visible_thresh=0.0)
    assert v.kpt_conf_thresh == 0.0 and v.kpt_visible_thresh == 0.0


def test_coarse_dropout_marks_hidden_points_occluded():
    # one 100 x 100 px hole (int ranges are pixels) on a 100 x 100 image: covers [0, 100)
    hole = dict(num_holes_range=(1, 1), hole_height_range=(100, 100), hole_width_range=(100, 100))
    t = A.Compose(
        [KptCoarseDropout(**hole, p=1)],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )
    kpts = np.array([[10, 10, 2], [60, 60, 1], [30, 70, 0], [100, 50, 2]], np.float32)
    out = np.asarray(t(image=np.zeros((100, 100, 3), np.uint8), keypoints=kpts)["keypoints"])
    # hidden visible point -> occluded, coordinates kept; other flags and the edge point stay
    np.testing.assert_array_equal(out[:, :2], kpts[:, :2])
    np.testing.assert_array_equal(out[:, 2], [1, 1, 0, 2])
