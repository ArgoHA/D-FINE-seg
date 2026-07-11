"""Semantic segmentation loss: CE + multi-class soft Dice + auxiliary CE (deep supervision)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemSegCriterion(nn.Module):
    def __init__(
        self,
        weight_dict,
        num_classes,
        ignore_index=255,
        class_weights=None,
        label_smoothing=0.0,
    ):
        super().__init__()
        self.weight_dict = weight_dict
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.class_weights = (
            torch.tensor(list(class_weights), dtype=torch.float32) if class_weights else None
        )

    def _dice(self, logits, target, valid):
        # multi-class soft dice; ignored pixels are masked out of both prob and one-hot
        prob = logits.softmax(1)
        one_hot = F.one_hot(torch.where(valid, target, 0), self.num_classes)
        one_hot = one_hot.permute(0, 3, 1, 2).to(prob.dtype)
        v = valid.unsqueeze(1).to(prob.dtype)
        prob, one_hot = prob * v, one_hot * v
        inter = (prob * one_hot).sum((0, 2, 3))
        denom = prob.sum((0, 2, 3)) + one_hot.sum((0, 2, 3))
        dice = (2 * inter + 1.0) / (denom + 1.0)  # absent classes -> dice 1 -> zero loss
        return 1.0 - dice.mean()

    def forward(self, outputs, targets):
        logits = outputs["sem_seg_logits"].float()
        target = torch.stack([t["sem_mask"] for t in targets])  # (B, H, W) long
        valid = target != self.ignore_index

        if not valid.any():  # all-ignore batch (e.g. fully padded) contributes zero loss
            zero = logits.sum() * 0.0
            losses = {"loss_ce": zero, "loss_dice": zero}
            if "sem_seg_logits_aux" in outputs:
                losses["loss_aux"] = outputs["sem_seg_logits_aux"].float().sum() * 0.0
        else:
            weight = (
                self.class_weights.to(logits.device) if self.class_weights is not None else None
            )
            losses = {
                "loss_ce": F.cross_entropy(
                    logits,
                    target,
                    weight=weight,
                    ignore_index=self.ignore_index,
                    label_smoothing=self.label_smoothing,
                ),
                "loss_dice": self._dice(logits, target, valid),
            }
            if "sem_seg_logits_aux" in outputs:
                losses["loss_aux"] = F.cross_entropy(
                    outputs["sem_seg_logits_aux"].float(),
                    target,
                    ignore_index=self.ignore_index,
                )
        return {k: v * self.weight_dict[k] for k, v in losses.items()}
