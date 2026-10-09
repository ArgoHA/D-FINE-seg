"""Knowledge distillation from a frozen teacher of the same task and classes (train-only).

detect / segment: teacher and student queries are paired through the GT object each one is
Hungarian-matched to; class scores, edge distributions and (segment) masks are distilled on
those pairs. sem_seg: channel-wise distillation of the 1/4-resolution logits.
"""

import torch

from .arch.utils import translate_gt, weighting_function


def channel_wise_kd(student, teacher, tau=4.0):
    """CWD (Shu et al., ICCV 2021; mmrazor ChannelWiseDivergence): per class channel, softmax
    over spatial positions, KL(teacher || student) summed over positions, * tau^2 / (N * C)."""
    n, c = student.shape[:2]
    s = student.float().flatten(2) / tau
    t = teacher.float().flatten(2) / tau
    kl = t.softmax(-1) * (t.log_softmax(-1) - s.log_softmax(-1))
    return kl.sum() * tau**2 / (n * c)


def rebin_corners(t_corners, t_ref, t_up, t_reg_scale, s_ref, s_up, s_reg_scale, reg_max):
    """Teacher edge distributions moved into the student's frame -> (M * 4, reg_max + 1) probs.

    D-FINE predicts each edge as a distribution over bins W(n) measured from the model's own
    reference box, so the same bin means a different image position in teacher and student.
    Each teacher bin is decoded to its absolute edge coordinate (distance2bbox), re-encoded as
    a distance from the student's reference box (bbox2distance) and its probability is split
    between the two nearest student bins, the same two-bin split FGL uses for the GT edge.
    """
    m, n = t_corners.shape[0], reg_max + 1
    prob = t_corners.float().reshape(m, 4, n).softmax(-1)
    rs_t, rs_s = t_reg_scale.abs().float(), s_reg_scale.abs().float()
    half = 0.5 * rs_t + weighting_function(reg_max, t_up, t_reg_scale).float()  # (n,)

    cx, cy, w, h = (v[:, None] for v in t_ref.float().unbind(-1))
    x1, x2 = cx - half * w / rs_t, cx + half * w / rs_t  # (m, n)
    y1, y2 = cy - half * h / rs_t, cy + half * h / rs_t

    sx, sy, sw, sh = (v[:, None] for v in s_ref.float().unbind(-1))
    sw, sh = sw / rs_s + 1e-16, sh / rs_s + 1e-16
    dist = torch.stack([(sx - x1) / sw, (sy - y1) / sh, (x2 - sx) / sw, (y2 - sy) / sh], 1)
    idx, w_right, w_left = translate_gt(dist - 0.5 * rs_s, reg_max, rs_s, s_up)

    left = idx.floor().long()  # out-of-range values land on the end bin with weight 1
    rows = torch.arange(m * 4, device=prob.device).repeat_interleave(n)
    p = prob.reshape(-1)
    target = torch.zeros(m * 4, n, device=prob.device)
    target.index_put_((rows, left), p * w_left, accumulate=True)
    target.index_put_((rows, left + 1), p * w_right, accumulate=True)
    return target


def pair_through_gt(indices, t_indices):
    """Student (image, query) pairs and the teacher query matched to the same GT.

    -> (batch_idx, student_query, teacher_query, gt_idx), one entry per student match whose GT
    the teacher also matched (all of them when the teacher has more queries than objects).
    """
    b_all, s_all, t_all, g_all = [], [], [], []
    for b, ((s_q, s_gt), (t_q, t_gt)) in enumerate(zip(indices, t_indices)):
        gts = torch.cat([s_gt, t_gt]).long().cpu()
        t_of_gt = torch.full((int(gts.max()) + 1 if gts.numel() else 0,), -1, dtype=torch.long)
        t_of_gt[t_gt.long().cpu()] = t_q.long().cpu()
        q = t_of_gt[s_gt.long().cpu()]
        keep = q >= 0
        b_all.append(torch.full((int(keep.sum()),), b, dtype=torch.long))
        s_all.append(s_q.long().cpu()[keep])
        t_all.append(q[keep])
        g_all.append(s_gt.long().cpu()[keep])
    return tuple(torch.cat(x) for x in (b_all, s_all, t_all, g_all))
