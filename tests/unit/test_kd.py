import torch

from dfine_seg.model.arch.utils import weighting_function
from dfine_seg.model.dfine import build_loss
from dfine_seg.model.kd import channel_wise_kd, pair_through_gt, rebin_corners

R = 32
UP, RS = torch.tensor([0.5]), torch.tensor([4.0])
KD = {"cls_weight": 1.0, "loc_weight": 1.5, "mask_weight": 1.0}


def _left_edge(prob, ref):
    """Expected absolute x1 of a (M, R+1) left-edge distribution measured from ref (M, 4)."""
    half = 0.5 * RS + weighting_function(R, UP, RS)
    return ref[:, 0] - (prob * half).sum(-1) * ref[:, 2] / RS


def test_rebin_same_frame_is_identity():
    torch.manual_seed(0)
    corners = torch.randn(5, 4 * (R + 1))
    ref = torch.tensor([[0.5, 0.5, 0.2, 0.3]]).repeat(5, 1)
    target = rebin_corners(corners, ref, UP, RS, ref, UP, RS, R)
    expected = corners.reshape(-1, R + 1).softmax(-1)
    assert torch.allclose(target, expected, atol=1e-4)


def test_rebin_keeps_expected_edge_position_across_frames():
    k = torch.arange(R + 1, dtype=torch.float32)
    logits = -((k - R / 2) ** 2) / 4  # mass on central bins, away from the clamped ends
    corners = logits.repeat(1, 4)
    t_ref = torch.tensor([[0.50, 0.50, 0.20, 0.30]])
    s_ref = torch.tensor([[0.51, 0.49, 0.22, 0.28]])
    target = rebin_corners(corners, t_ref, UP, RS, s_ref, UP, RS, R)
    assert torch.allclose(target.sum(-1), torch.ones(4), atol=1e-5)
    t_x1 = _left_edge(logits.softmax(-1)[None], t_ref)
    s_x1 = _left_edge(target[:1], s_ref)
    assert torch.allclose(t_x1, s_x1, atol=1e-4)


def test_pair_through_gt():
    indices = [(torch.tensor([7, 3]), torch.tensor([1, 0])), (torch.tensor([]), torch.tensor([]))]
    t_indices = [
        (torch.tensor([10, 20]), torch.tensor([0, 1])),
        (torch.tensor([]), torch.tensor([])),
    ]
    b, s_q, t_q, gt = pair_through_gt(indices, t_indices)
    assert b.tolist() == [0, 0]
    assert s_q.tolist() == [7, 3]
    assert t_q.tolist() == [20, 10]  # teacher queries matched to GT 1 and 0
    assert gt.tolist() == [1, 0]


def test_channel_wise_kd_zero_for_identical_logits():
    x = torch.randn(2, 3, 8, 16)
    assert channel_wise_kd(x, x).abs() < 1e-6
    assert channel_wise_kd(x, torch.randn_like(x)) > 0


def _outputs(b=2, q=10, c=3):
    torch.manual_seed(1)
    return {
        "pred_logits": torch.randn(b, q, c),
        "pred_boxes": torch.rand(b, q, 4) * 0.3 + 0.3,
        "pred_corners": torch.randn(b, q, 4 * (R + 1)),
        "ref_points": torch.rand(b, q, 4) * 0.3 + 0.3,
        "up": UP,
        "reg_scale": RS,
    }


def test_loss_kd_loc_zero_for_identical_teacher_and_grads_reach_student_only():
    crit = build_loss("s", 3, 0.0, False, task="detect", kd=KD)
    student = _outputs()
    student["pred_corners"].requires_grad_(True)
    teacher = {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in student.items()}
    targets = [
        {
            "labels": torch.tensor([0, 2]),
            "boxes": torch.tensor([[0.4, 0.4, 0.1, 0.1], [0.5, 0.5, 0.2, 0.2]]),
        },
        {"labels": torch.tensor([1]), "boxes": torch.tensor([[0.45, 0.45, 0.1, 0.2]])},
    ]
    indices = crit.matcher(student, targets)["indices"]
    losses = crit.loss_kd(student | {"kd_teacher": teacher}, targets, indices)
    assert set(losses) == {"loss_kd_cls", "loss_kd_loc"}
    assert losses["loss_kd_loc"].abs() < 1e-4

    # matching reads only logits and boxes, so other corners keep the same pairs
    teacher["pred_corners"] = torch.randn_like(teacher["pred_corners"])
    losses = crit.loss_kd(student | {"kd_teacher": teacher}, targets, indices)
    assert losses["loss_kd_loc"] > 0
    losses["loss_kd_loc"].backward()
    assert student["pred_corners"].grad.abs().sum() > 0
    assert teacher["pred_corners"].grad is None


def test_loss_kd_mask_term_for_segment():
    crit = build_loss("s", 3, 0.0, True, task="segment", kd=KD)
    student = _outputs() | {"pred_masks": torch.randn(2, 10, 16, 32, requires_grad=True)}
    teacher = {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in student.items()}
    teacher["pred_mask_logits"] = torch.randn(2, 10, 32, 64)  # teacher at 2x the resolution
    targets = [
        {"labels": torch.tensor([0]), "boxes": torch.tensor([[0.4, 0.4, 0.2, 0.2]])},
        {"labels": torch.tensor([1]), "boxes": torch.tensor([[0.5, 0.5, 0.3, 0.2]])},
    ]
    indices = crit.matcher({k: student[k] for k in ("pred_logits", "pred_boxes")}, targets)
    losses = crit.loss_kd(student | {"kd_teacher": teacher}, targets, indices["indices"])
    assert torch.isfinite(losses["loss_kd_mask"]) and losses["loss_kd_mask"] > 0
    losses["loss_kd_mask"].backward()
    assert student["pred_masks"].grad.abs().sum() > 0


def test_criterion_without_teacher_has_no_kd_terms():
    crit = build_loss("s", 3, 0.0, False, task="detect")
    assert crit.kd_weights == {}
