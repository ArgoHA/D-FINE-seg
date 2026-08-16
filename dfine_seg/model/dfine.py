from copy import deepcopy
from pathlib import Path

import torch.nn as nn
import torch.optim as optim

from dfine_seg.model.dfine_criterion import DFINECriterion
from dfine_seg.model.sem_seg_criterion import SemSegCriterion
from dfine_seg.model.utils import ensure_pretrained

from .arch.dfine_decoder import DFINETransformer, SemSegDecoder
from .arch.hgnetv2 import HGNetv2
from .arch.hybrid_encoder import HybridEncoder
from .configs import models
from .matcher import HungarianMatcher
from .muon import MuonWithAuxAdam
from .utils import load_tuning_state

__all__ = ["DFINE"]


class DFINE(nn.Module):
    __inject__ = [
        "backbone",
        "encoder",
        "decoder",
    ]

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder

    def forward(self, x, targets=None):
        feats = self.backbone(x)

        # When backbone returns more features than encoder expects (e.g. nano + seg),
        # the extra leading feature is the low-level 1/8 map for MaskDecoder.
        low_level_feat = None
        if len(feats) > len(self.encoder.in_channels):
            low_level_feat = feats[0]
            feats = feats[1:]

        x = self.encoder(feats)
        x = self.decoder(x, targets, low_level_feat=low_level_feat)
        return x

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self


def build_model(
    model_name,
    num_classes,
    enable_mask_head,
    device,
    img_size=None,
    in_channels: int = 3,
    pretrained_model_path=None,
    pretrained_backbone=False,
    task=None,  # "sem_seg" swaps DFINETransformer for SemSegDecoder; else by enable_mask_head
):
    if int(in_channels) not in (3, 4):
        raise ValueError(
            f"train.in_channels must be 3 (RGB) or 4 (RGB+one extra modality); got {in_channels}. "
            "Stacks with >4 channels are not supported (cv2 Scalar / Albumentations cap)."
        )
    model_cfg = deepcopy(models[model_name])
    # research: ImageNet stage1 backbone init (constant across experiments), random neck/head
    model_cfg["HGNetv2"]["pretrained"] = pretrained_backbone

    model_cfg["HybridEncoder"]["eval_spatial_size"] = img_size
    model_cfg["DFINETransformer"]["eval_spatial_size"] = img_size
    model_cfg["DFINETransformer"]["enable_mask_head"] = enable_mask_head

    # For models without 1/8 stride (nano), when a mask-producing head is enabled,
    # pass the backbone 1/8 feature as a low-level input for MaskDecoder
    sem_seg = task == "sem_seg"
    enc_strides = model_cfg["HybridEncoder"]["feat_strides"]
    if (enable_mask_head or sem_seg) and 8 not in enc_strides:
        return_idx = model_cfg["HGNetv2"]["return_idx"]
        if 1 not in return_idx:  # stage index 1 = stride 8
            model_cfg["HGNetv2"]["return_idx"] = [1] + return_idx
        backbone_name = model_cfg["HGNetv2"]["name"]
        stage2_ch = HGNetv2.arch_configs[backbone_name]["stage_config"]["stage2"][2]
        model_cfg["DFINETransformer"]["mask_low_level_ch"] = stage2_ch

    backbone = HGNetv2(in_channels=in_channels, **model_cfg["HGNetv2"])
    encoder = HybridEncoder(**model_cfg["HybridEncoder"])
    if sem_seg:
        dec_cfg = model_cfg["DFINETransformer"]
        decoder = SemSegDecoder(
            num_classes=num_classes,
            feat_channels=dec_cfg["feat_channels"],
            mask_dim=dec_cfg["mask_dim"],
            mask_low_level_ch=dec_cfg.get("mask_low_level_ch"),
        )
    else:
        decoder = DFINETransformer(num_classes=num_classes, **model_cfg["DFINETransformer"])

    model = DFINE(backbone, encoder, decoder)

    if pretrained_model_path:
        resolved = ensure_pretrained(pretrained_model_path)
        if not Path(resolved).exists():
            raise FileNotFoundError(f"{pretrained_model_path} does not exist")
        model = load_tuning_state(model, resolved)
    return model.to(device)


def build_loss(
    model_name,
    num_classes,
    label_smoothing,
    enable_mask_head,
    task=None,
    ignore_index=255,
    class_weights=None,
):
    # deepcopy so appending "masks" mutates a private copy, not the shared
    # global `models` config — build_loss may be called more than once per run.
    model_cfg = deepcopy(models[model_name])
    if task == "sem_seg":
        return SemSegCriterion(
            model_cfg["SemSegCriterion"]["weight_dict"],
            num_classes=num_classes,
            ignore_index=ignore_index,
            class_weights=class_weights,
            label_smoothing=label_smoothing,
        )
    if enable_mask_head and "masks" not in model_cfg["DFINECriterion"]["losses"]:
        model_cfg["DFINECriterion"]["losses"].append("masks")
    matcher = HungarianMatcher(**model_cfg["matcher"])
    loss_fn = DFINECriterion(
        matcher,
        num_classes=num_classes,
        label_smoothing=label_smoothing,
        **model_cfg["DFINECriterion"],
    )
    return loss_fn


# Encoder/decoder transformer attention/MLP weight matrices — Muon's target when
# train.use_muon is on. Allowlist (not denylist) so det/mask heads, embeddings, LQE
# and norms/biases can never leak in: those names lack these tokens.
MUON_TOKENS = ("self_attn", "cross_attn", "linear1", "linear2", "gateway.gate")


def _is_muon_param(name, param):
    return (
        param.ndim == 2
        and ("encoder" in name or "decoder" in name)
        and "mask" not in name
        and any(t in name for t in MUON_TOKENS)
    )


# The only randomly-initialized modules when a detection checkpoint is loaded for
# task=segment — everything else comes pretrained (see train.freeze_except_mask).
MASK_PREFIXES = ("decoder.mask_decoder", "decoder.mask_head")


def freeze_except_mask(model):
    """Freeze every parameter outside the mask head. Returns the fully-frozen submodules.

    Call before any DDP wrap — named_parameters() gains a "module." prefix after it.
    requires_grad=False does not stop BatchNorm from updating its running stats, so the
    caller must re-apply .eval() to the returned modules after every model.train() or
    the "frozen" features still drift epoch to epoch.
    """
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith(MASK_PREFIXES))
    return [
        m
        for m in model.modules()
        if list(m.parameters(recurse=True))
        and not any(p.requires_grad for p in m.parameters(recurse=True))
    ]


def build_optimizer(
    model,
    lr,
    backbone_lr,
    betas,
    weight_decay,
    base_lr,
    use_muon=False,
    muon_lr=None,
    aux_optimizer="adamw",
    respect_backbone_lr=False,
    adan_betas=(0.98, 0.92, 0.99),
):
    backbone_exclude_norm = []
    backbone_norm = []
    encdec_norm_bias = []
    rest = []
    muon = []

    for name, param in model.named_parameters():
        # Group 1 and 2: "backbone" in name
        if "backbone" in name:
            if "norm" in name or "bn" in name:
                # Group 2: backbone + norm/bn
                backbone_norm.append(param)
            else:
                # Group 1: backbone but not norm/bn
                backbone_exclude_norm.append(param)

        # Group 3: "encoder" or "decoder" plus "norm"/"bn"/"bias"
        elif ("encoder" in name or "decoder" in name) and (
            "norm" in name or "bn" in name or "bias" in name
        ):
            encdec_norm_bias.append(param)

        # Muon group: enc/dec attention+MLP matrices (only when enabled)
        elif use_muon and _is_muon_param(name, param):
            muon.append(param)

        else:
            rest.append(param)

    group1 = {"params": backbone_exclude_norm, "lr": backbone_lr, "initial_lr": backbone_lr}
    group2 = {
        "params": backbone_norm,
        "lr": backbone_lr,
        "weight_decay": 0.0,
        "initial_lr": backbone_lr,
    }
    group3 = {"params": encdec_norm_bias, "weight_decay": 0.0, "lr": base_lr, "initial_lr": base_lr}
    group4 = {"params": rest, "lr": base_lr, "initial_lr": base_lr}

    param_groups = [group1, group2, group3, group4]

    if not use_muon:
        return optim.AdamW(param_groups, lr=lr, betas=betas, weight_decay=weight_decay)

    if aux_optimizer == "adan":
        assert len(adan_betas) == 3, (
            "adan requires exactly 3 betas (b1,b2,b3); see train.adan_betas"
        )

    # Muon path: same AdamW groups (use_muon=False, explicit betas/wd since the custom
    # optimizer has no top-level defaults) + a Muon group appended LAST so the scheduler's
    # per-group max_lr list can target it by index.
    # respect_backbone_lr: keep backbone on AdamW at its own low LR (l/x, backbone_lr << base_lr).
    for i, g in enumerate(param_groups):  # groups 0,1 = backbone; 2 = encdec norm/bias; 3 = rest
        g["use_muon"] = False
        g.setdefault("weight_decay", weight_decay)
        is_backbone = i < 2
        if aux_optimizer == "adan" and not (respect_backbone_lr and is_backbone):
            # Adan on the aux groups: betas from cfg (train.adan_betas), eps intrinsic; per-group
            # weight_decay kept identical to the AdamW recipe so only the update rule (+LR) changes.
            g["aux_optimizer"] = "adan"
            g["betas"] = tuple(adan_betas)
            g["eps"] = 1e-8
        else:
            g["betas"] = betas
    muon_lr = muon_lr if muon_lr is not None else base_lr * 10
    param_groups.append(
        {
            "params": muon,
            "use_muon": True,
            "lr": muon_lr,
            "initial_lr": muon_lr,
            "weight_decay": weight_decay,
        }
    )
    return MuonWithAuxAdam(param_groups)
