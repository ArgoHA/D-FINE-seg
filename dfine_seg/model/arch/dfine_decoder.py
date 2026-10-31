"""
D-FINE-seg: Object Detection and Instance Segmentation Framework with Multi-backend Deployment
Copyright (c) 2026 The D-FINE-seg Authors. All Rights Reserved.
---------------------------------------------------------------------------------
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import copy
import functools
import math
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .utils import (
    bias_init_with_prob,
    deformable_attention_core_func_v2,
    distance2bbox,
    get_activation,
    get_contrastive_denoising_training_group,
    inverse_sigmoid,
    weighting_function,
)

__all__ = ["DFINETransformer"]


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        act: str = "relu",
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.act = get_activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MSDeformableAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: Union[int, List[int]] = 4,
        method: str = "default",
        offset_scale: float = 0.5,
    ) -> None:
        """Multi-Scale Deformable Attention"""
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.offset_scale = offset_scale

        if isinstance(num_points, list):
            assert len(num_points) == num_levels, ""
            num_points_list = num_points
        else:
            num_points_list = [num_points for _ in range(num_levels)]

        self.num_points_list = num_points_list

        num_points_scale = [1 / n for n in num_points_list for _ in range(n)]
        self.register_buffer(
            "num_points_scale", torch.tensor(num_points_scale, dtype=torch.float32)
        )

        self.total_points = num_heads * sum(num_points_list)
        self.method = method

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, (
            "embed_dim must be divisible by num_heads"
        )

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)

        self.ms_deformable_attn_core = functools.partial(
            deformable_attention_core_func_v2, method=self.method
        )

        self._reset_parameters()

        if method == "discrete":
            for p in self.sampling_offsets.parameters():
                p.requires_grad = False

    def _reset_parameters(self) -> None:
        # sampling_offsets
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 2).tile([1, sum(self.num_points_list), 1])
        scaling = torch.concat([torch.arange(1, n + 1) for n in self.num_points_list]).reshape(
            1, -1, 1
        )
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        value_spatial_shapes: List[List[int]],
    ) -> torch.Tensor:
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]

        sampling_offsets: torch.Tensor = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.reshape(
            bs, Len_q, self.num_heads, sum(self.num_points_list), 2
        )

        attention_weights = self.attention_weights(query).reshape(
            bs, Len_q, self.num_heads, sum(self.num_points_list)
        )
        attention_weights = F.softmax(attention_weights, dim=-1)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = (
                reference_points.reshape(bs, Len_q, 1, self.num_levels, 1, 2)
                + sampling_offsets / offset_normalizer
            )
        elif reference_points.shape[-1] == 4:
            # reference_points [8, 480, None, 1,  4]
            # sampling_offsets [8, 480, 8,    12, 2]
            num_points_scale = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
            offset = (
                sampling_offsets
                * num_points_scale
                * reference_points[:, :, None, :, 2:]
                * self.offset_scale
            )
            sampling_locations = reference_points[:, :, None, :, :2] + offset
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".format(
                    reference_points.shape[-1]
                )
            )

        output = self.ms_deformable_attn_core(
            value, value_spatial_shapes, sampling_locations, attention_weights, self.num_points_list
        )

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        activation: str = "relu",
        n_levels: int = 4,
        n_points: int = 4,
        cross_attn_method: str = "default",
        layer_scale: Optional[float] = None,
    ) -> None:
        super(TransformerDecoderLayer, self).__init__()
        if layer_scale is not None:
            dim_feedforward = round(layer_scale * dim_feedforward)
            d_model = round(layer_scale * d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformableAttention(
            d_model, n_head, n_levels, n_points, method=cross_attn_method
        )
        self.dropout2 = nn.Dropout(dropout)

        # gate
        self.gateway = Gate(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor: torch.Tensor, pos: Optional[torch.Tensor]) -> torch.Tensor:
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(
        self,
        target: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        spatial_shapes: List[List[int]],
        attn_mask: Optional[torch.Tensor] = None,
        query_pos_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # self attention
        q = k = self.with_pos_embed(target, query_pos_embed)

        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target = target + self.dropout1(target2)
        target = self.norm1(target)

        # cross attention
        target2 = self.cross_attn(
            self.with_pos_embed(target, query_pos_embed), reference_points, value, spatial_shapes
        )

        target = self.gateway(target, self.dropout2(target2))

        # ffn
        target2 = self.forward_ffn(target)
        target = target + self.dropout4(target2)
        target = self.norm3(target.clamp(min=-65504, max=65504))

        return target


TRT_GRID_SAMPLE_MAX_ROWS = 1024


class KeypointDecoderLayer(TransformerDecoderLayer):
    """Decoder layer over keypoint tokens [B, N, K, C]: self-attention among the K points of
    one instance, deformable cross-attention around each point's own reference."""

    def forward(
        self,
        target: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        spatial_shapes: List[List[int]],
        query_pos_embed: torch.Tensor,
    ) -> torch.Tensor:
        b, n, k, c = target.shape
        q = (target + query_pos_embed).reshape(b * n, k, c)
        target2, _ = self.self_attn(q, q, value=target.reshape(b * n, k, c))
        target = self.norm1(target + target2.reshape(b, n, k, c))

        # TensorRT 10.13 mis-executes chained GridSamples on a shared value tensor (layer 2's
        # grid depends on layer 1, see trt_bugs/TENSORRT_ISSUES.md); splitting the tokens into
        # >= 2 chunks avoids it. Exported graphs only: eager / compiled runs keep one call.
        # The count must divide n * k: uneven torch splits export as even ONNX Splits. Each
        # query samples independently, so the result is unchanged.
        q = (target + query_pos_embed).reshape(b, n * k, c)
        ref = reference_points.reshape(b, n * k, 1, 4)
        num_chunks = 1
        if torch.onnx.is_in_onnx_export():
            min_chunks = max(2, -(-n * k // TRT_GRID_SAMPLE_MAX_ROWS))
            divisors = (i for i in range(min_chunks, n * k + 1) if n * k % i == 0)
            num_chunks = next(divisors, 1)
        chunks = zip(q.chunk(num_chunks, 1), ref.chunk(num_chunks, 1))
        target2 = torch.cat(
            [self.cross_attn(q_i, r_i, value, spatial_shapes) for q_i, r_i in chunks], 1
        )
        target = self.gateway(target, target2.reshape(b, n, k, c))
        target = target + self.forward_ffn(target)
        return self.norm3(target.clamp(min=-65504, max=65504))


class KeypointDecoder(nn.Module):
    """Refines the keypoints of the top-scoring instances with one token per point.

    The per-query head regresses all K points from one instance vector whose attention samples
    around the box, so nothing looks at the image where a point is. Here each point is a token
    that samples the encoder memory around its current location, sees the instance's other
    points and predicts a box-relative residual; presence/visible logits refine the same way.
    """

    def __init__(
        self,
        num_kpts: int,
        hidden_dim: int,
        nhead: int,
        dim_feedforward: int,
        num_levels: int,
        num_points: Union[int, List[int]],
        num_layers: int,
        ref_scale: float = 0.25,
    ) -> None:
        super().__init__()
        self.ref_scale = ref_scale  # sampling extent around a point, relative to its box
        layer = KeypointDecoderLayer(
            hidden_dim, nhead, dim_feedforward, 0.0, "relu", num_levels, num_points
        )
        self.layers = nn.ModuleList(copy.deepcopy(layer) for _ in range(num_layers))
        self.kpt_embed = nn.Embedding(num_kpts, hidden_dim)
        self.instance_proj = nn.Linear(hidden_dim, hidden_dim)
        self.pos_head = MLP(4, 2 * hidden_dim, hidden_dim, 2)
        # per point: dx, dy, log Laplace scale x / y (box-relative, train-only uncertainty)
        self.reg_head = nn.ModuleList(MLP(hidden_dim, hidden_dim, 4, 3) for _ in range(num_layers))
        self.score_head = nn.ModuleList(nn.Linear(hidden_dim, 2) for _ in range(num_layers))
        # every layer starts as identity: the per-query head's points and scores
        for head in list(self.reg_head) + list(self.score_head):
            last = head.layers[-1] if isinstance(head, MLP) else head
            init.constant_(last.weight, 0)
            init.constant_(last.bias, 0)
        init.xavier_uniform_(self.pos_head.layers[0].weight)
        init.xavier_uniform_(self.pos_head.layers[1].weight)
        # instance keypoint quality: logit of the predicted OKS, for rescoring
        self.quality_head = MLP(hidden_dim, hidden_dim, 1, 2)

    def forward(
        self,
        h: torch.Tensor,
        boxes: torch.Tensor,
        kpts: torch.Tensor,
        value: Tuple[torch.Tensor, ...],
        spatial_shapes: List[List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """h [B, N, C], boxes [B, N, 4] cxcywh, kpts [B, N, K, 4] initial (x, y, presence,
        visible logits) -> ([L, B, N, K, C], quality logit [B, N]): every layer + log scales
        (C=6) in training, the last layer (C=4) in eval."""
        wh = boxes[..., None, 2:].detach()
        xy, logits = kpts[..., :2].detach(), kpts[..., 2:].detach()
        target = self.instance_proj(h)[:, :, None] + self.kpt_embed.weight
        out = []
        for i, layer in enumerate(self.layers):
            ref = torch.cat([xy, (wh * self.ref_scale).expand_as(xy)], -1)
            pos = self.pos_head(ref).clamp(min=-10, max=10)
            target = layer(target, ref, value, spatial_shapes, pos)
            reg = self.reg_head[i](target)
            xy = xy + reg[..., :2] * wh
            logits = logits + self.score_head[i](target)
            if self.training:
                out.append(torch.cat([xy, logits, reg[..., 2:]], -1))
            elif i == len(self.layers) - 1:
                out.append(torch.cat([xy, logits], -1))
            xy, logits = xy.detach(), logits.detach()
        return torch.stack(out), self.quality_head(target.mean(2)).squeeze(-1)


class Gate(nn.Module):
    def __init__(self, d_model: int) -> None:
        super(Gate, self).__init__()
        self.gate = nn.Linear(2 * d_model, 2 * d_model)
        bias = bias_init_with_prob(0.5)
        init.constant_(self.gate.bias, bias)
        init.constant_(self.gate.weight, 0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        gate_input = torch.cat([x1, x2], dim=-1)
        gates = torch.sigmoid(self.gate(gate_input))
        gate1, gate2 = gates.chunk(2, dim=-1)
        return self.norm(gate1 * x1 + gate2 * x2)


class Integral(nn.Module):
    """
    A static layer that calculates integral results from a distribution.

    This layer computes the target location using the formula: `sum{Pr(n) * W(n)}`,
    where Pr(n) is the softmax probability vector representing the discrete
    distribution, and W(n) is the non-uniform Weighting Function.

    Args:
        reg_max (int): Max number of the discrete bins. Default is 32.
                       It can be adjusted based on the dataset or task requirements.
    """

    def __init__(self, reg_max: int = 32) -> None:
        super(Integral, self).__init__()
        self.reg_max = reg_max

    def forward(self, x: torch.Tensor, project: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = (x @ project.to(x.device).unsqueeze(-1)).reshape(-1, 4)
        return x.reshape(list(shape[:-1]) + [-1])


class LQE(nn.Module):
    def __init__(self, k: int, hidden_dim: int, num_layers: int, reg_max: int) -> None:
        super(LQE, self).__init__()
        self.k = k
        self.reg_max = reg_max
        self.reg_conf = MLP(4 * (k + 1), hidden_dim, 1, num_layers)
        init.constant_(self.reg_conf.layers[-1].bias, 0)
        init.constant_(self.reg_conf.layers[-1].weight, 0)
        self.deploy = False

    def convert_to_deploy(self) -> None:
        self.deploy = True

    def _topk_no_sort(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """TFLite-compatible top-k using iterative argmax (avoids vhlo.sort_v1)."""
        top_vals = []
        for _ in range(k):
            idx = x.argmax(dim=-1, keepdim=True)
            val = x.gather(-1, idx)
            top_vals.append(val)
            x = x.scatter(-1, idx, torch.zeros_like(val))
        return torch.cat(top_vals, dim=-1)

    def forward(self, scores: torch.Tensor, pred_corners: torch.Tensor) -> torch.Tensor:
        B, L, _ = pred_corners.size()
        prob = F.softmax(pred_corners.reshape(B, L, 4, self.reg_max + 1), dim=-1)
        if self.deploy:
            prob_topk = self._topk_no_sort(prob, self.k)
        else:
            prob_topk, _ = prob.topk(self.k, dim=-1)
        stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)
        quality_score = self.reg_conf(stat.reshape(B, L, -1))
        return scores + quality_score


class MaskDecoder(nn.Module):
    """
    Fuses multi-scale PAN features from HybridEncoder to produce high-resolution mask features.

    This module takes the PAN (bottom-up) outputs from HybridEncoder, which contain
    multi-scale features after full bidirectional (FPN + PAN) fusion, enriched with
    both local and global context.

    Input:
        feats: List of PAN features [F_s8, F_s16, F_s32] from HybridEncoder's outs.
               Shape: [(B, C, H/8, W/8), (B, C, H/16, W/16), (B, C, H/32, W/32)]

    Output:
        Fused mask features at 1/4 resolution. Shape: (B, out_ch, H/4, W/4)
    """

    def __init__(self, in_chs: List[int], out_ch: int = 256) -> None:
        super().__init__()
        n_groups = 32
        # 1x1 proj for each backbone level
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_ch, 1, bias=False) for c in in_chs])
        self.bn = nn.ModuleList([nn.GroupNorm(n_groups, out_ch) for _ in in_chs])

        # 3x3 conv after fusion to smooth aliasing and add spatial context
        self.fusion_conv = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.fusion_norm = nn.GroupNorm(n_groups, out_ch)

        # Upsample 1/8 -> 1/4 using bilinear + conv (avoids checkerboard artifacts)
        self.up_conv = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(n_groups, out_ch)
        self.act = nn.ReLU(inplace=True)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        init.kaiming_normal_(self.up_conv.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        # feats: PAN features [F_s8, F_s16, F_s32] from HybridEncoder's outs
        # Take the biggest resolution feature as the base
        f0 = self.bn[0](self.lateral[0](feats[0]))  # (B, out_ch, H/8, W/8)
        x = f0

        # Fuse upsampled coarser levels into finest resolution
        for i in range(1, len(feats)):
            t = self.bn[i](self.lateral[i](feats[i]))
            x = x + F.interpolate(t, size=f0.shape[-2:], mode="bilinear", align_corners=False)

        # Smooth fused features
        x = self.act(self.fusion_norm(self.fusion_conv(x)))

        # Upsample: 1/8 -> 1/4 using bilinear interpolation + conv (smoother than ConvTranspose)
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.act(self.bn1(self.up_conv(x)))
        return x  # (B, out_ch, H/4, W/4)


def conv_gn_act(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.GroupNorm(32, out_ch),
        nn.ReLU(inplace=True),
    )


class SemSegDecoder(nn.Module):
    """Dense per-pixel head for task=sem_seg: MaskDecoder fuser -> seg neck -> classifier.

    Plugs into the DFINE "decoder" slot, bypassing the whole query/matcher path.
    Attr name "mask_decoder" keeps keys aligned with dfine_seg_<size>_coco.pt so the
    pretrained fuser weights transfer (neck/classifier/aux train from scratch).
    Logits are produced at 1/4 scale and bilinearly upsampled x4 to input resolution.
    """

    def __init__(
        self,
        num_classes: int,
        feat_channels: List[int],
        mask_dim: int = 256,
        mask_low_level_ch: Optional[int] = None,
        neck_dim: int = 128,
        dropout: float = 0.1,
        aux: bool = True,
    ) -> None:
        super().__init__()
        in_chs = list(feat_channels)
        if mask_low_level_ch is not None:  # nano: prepend backbone 1/8 feat
            in_chs = [mask_low_level_ch] + in_chs
        self.mask_decoder = MaskDecoder(in_chs=in_chs, out_ch=mask_dim)
        self.neck = nn.Sequential(conv_gn_act(mask_dim, neck_dim), conv_gn_act(neck_dim, neck_dim))
        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(neck_dim, num_classes, 1)
        # train-only deep supervision on the finest PAN feature (stride 8; 16 for nano)
        self.aux_head = (
            nn.Sequential(
                conv_gn_act(feat_channels[0], neck_dim),
                nn.Dropout2d(dropout),
                nn.Conv2d(neck_dim, num_classes, 1),
            )
            if aux
            else None
        )
        self.return_quarter_logits = False  # train-only: KD reads the 1/4 logits

    def forward(
        self,
        feats: List[torch.Tensor],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
        low_level_feat: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        mask_feats = list(feats) if low_level_feat is None else [low_level_feat] + list(feats)
        x = self.mask_decoder(mask_feats)  # (B, mask_dim, H/4, W/4)
        logits_q = self.classifier(self.dropout(self.neck(x)))  # (B, C, H/4, W/4)
        logits = F.interpolate(logits_q, scale_factor=4.0, mode="bilinear", align_corners=False)
        out = {"sem_seg_logits": logits}  # (B, C, H, W)
        if self.return_quarter_logits:
            out["sem_seg_logits_q"] = logits_q
        if self.training and self.aux_head is not None:
            aux = self.aux_head(feats[0])
            out["sem_seg_logits_aux"] = F.interpolate(
                aux, size=logits.shape[-2:], mode="bilinear", align_corners=False
            )
        return out


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder implementing Fine-grained Distribution Refinement (FDR).

    This decoder refines object detection predictions through iterative updates across multiple layers,
    utilizing attention mechanisms, location quality estimators, and distribution refinement techniques
    to improve bounding box accuracy and robustness.
    """

    def __init__(
        self,
        hidden_dim: int,
        decoder_layer: TransformerDecoderLayer,
        decoder_layer_wide: TransformerDecoderLayer,
        num_layers: int,
        num_head: int,
        reg_max: int,
        reg_scale: torch.Tensor,
        up: torch.Tensor,
        eval_idx: int = -1,
        layer_scale: int = 2,
    ) -> None:
        super(TransformerDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_scale = layer_scale
        self.num_head = num_head
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.up, self.reg_scale, self.reg_max = up, reg_scale, reg_max
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(self.eval_idx + 1)]
            + [copy.deepcopy(decoder_layer_wide) for _ in range(num_layers - self.eval_idx - 1)]
        )
        self.lqe_layers = nn.ModuleList(
            [copy.deepcopy(LQE(4, 64, 2, reg_max)) for _ in range(num_layers)]
        )

    def value_op(
        self,
        memory: torch.Tensor,
        value_proj: Optional[nn.Module],
        value_scale: Optional[int],
        memory_mask: Optional[torch.Tensor],
        memory_spatial_shapes: List[List[int]],
    ) -> Tuple[torch.Tensor, ...]:
        """
        Preprocess values for MSDeformableAttention.
        """
        value = value_proj(memory) if value_proj is not None else memory
        value = F.interpolate(memory, size=value_scale) if value_scale is not None else value
        if memory_mask is not None:
            value = value * memory_mask.to(value.dtype).unsqueeze(-1)
        value = value.reshape(value.shape[0], value.shape[1], self.num_head, -1)
        split_shape = [h * w for h, w in memory_spatial_shapes]
        return value.permute(0, 2, 3, 1).split(split_shape, dim=-1)

    def convert_to_deploy(self) -> None:
        self.project = weighting_function(self.reg_max, self.up, self.reg_scale, deploy=True)
        self.layers = self.layers[: self.eval_idx + 1]
        self.lqe_layers = nn.ModuleList(
            [nn.Identity()] * (self.eval_idx) + [self.lqe_layers[self.eval_idx]]
        )

    def forward(
        self,
        target: torch.Tensor,
        ref_points_unact: torch.Tensor,
        memory: torch.Tensor,
        spatial_shapes: List[List[int]],
        bbox_head: nn.ModuleList,
        score_head: nn.ModuleList,
        query_pos_head: nn.Module,
        pre_bbox_head: nn.Module,
        integral: Integral,
        up: torch.Tensor,
        reg_scale: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        return_queries: bool = False,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        output = target
        output_detach = pred_corners_undetach = 0
        value = self.value_op(memory, None, None, memory_mask, spatial_shapes)

        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_pred_corners = []
        dec_out_refs = []
        if not hasattr(self, "project"):
            project = weighting_function(self.reg_max, up, reg_scale)
        else:
            project = self.project

        ref_points_detach = F.sigmoid(ref_points_unact)
        dec_out_queries = (
            [] if return_queries else None
        )  # Collect per-layer query features for segmentation head

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)

            # TODO Adjust scale if needed for detachable wider layers
            if i >= self.eval_idx + 1 and self.layer_scale > 1:
                query_pos_embed = F.interpolate(query_pos_embed, scale_factor=self.layer_scale)
                value = self.value_op(
                    memory, None, query_pos_embed.shape[-1], memory_mask, spatial_shapes
                )
                output = F.interpolate(output, size=query_pos_embed.shape[-1])
                output_detach = output.detach()

            output = layer(
                output, ref_points_input, value, spatial_shapes, attn_mask, query_pos_embed
            )  # decoder query embeddings
            if return_queries:
                dec_out_queries.append(output)

            if i == 0:
                # Initial bounding box predictions with inverse sigmoid refinement
                pre_bboxes = F.sigmoid(pre_bbox_head(output) + inverse_sigmoid(ref_points_detach))
                pre_scores = score_head[0](output)
                ref_points_initial = pre_bboxes.detach()

            # Refine bounding box corners using FDR, integrating previous layer's corrections
            pred_corners = bbox_head[i](output + output_detach) + pred_corners_undetach
            inter_ref_bbox = distance2bbox(
                ref_points_initial,
                integral(pred_corners, project),
                reg_scale,
                deploy=hasattr(self, "project"),
            )

            if self.training or i == self.eval_idx:
                scores = score_head[i](output)
                # Lqe does not affect the performance here.
                scores = self.lqe_layers[i](scores, pred_corners)
                dec_out_logits.append(scores)
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_pred_corners.append(pred_corners)
                dec_out_refs.append(ref_points_initial)

                if not self.training:
                    break

            pred_corners_undetach = pred_corners
            ref_points_detach = inter_ref_bbox.detach()
            output_detach = output.detach()

        if return_queries:
            hs = torch.stack(dec_out_queries)
        else:
            hs = None

        return (
            torch.stack(dec_out_bboxes),
            torch.stack(dec_out_logits),
            torch.stack(dec_out_pred_corners),
            torch.stack(dec_out_refs),
            pre_bboxes,
            pre_scores,
            hs,
        )


class DFINETransformer(nn.Module):
    __share__ = ["num_classes", "eval_spatial_size"]

    def __init__(
        self,
        num_classes: int = 80,
        hidden_dim: int = 256,
        num_queries: int = 300,
        feat_channels: List[int] = [512, 1024, 2048],
        feat_strides: List[int] = [8, 16, 32],
        num_levels: int = 3,
        num_points: int = 4,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        activation: str = "relu",
        num_denoising: int = 100,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        learn_query_content: bool = False,
        eval_spatial_size: Optional[List[int]] = None,
        eval_idx: int = -1,
        eps: float = 1e-2,
        aux_loss: bool = True,
        cross_attn_method: str = "default",
        query_select_method: str = "default",
        reg_max: int = 32,
        reg_scale: float = 4.0,
        layer_scale: int = 1,
        enable_mask_head: bool = False,
        mask_dim: int = 256,
        mask_low_level_ch: Optional[int] = None,
        num_kpts: int = 0,  # K points per object; 0 = no keypoint head
        num_kpt_layers: int = 0,  # keypoint decoder depth; 0 = per-query heads only
        num_kpt_queries: int = 50,  # top-scoring queries the keypoint decoder refines
        kpt_dn_noise: float = 0.0,  # keypoint denoising noise, x GT box size; 0 = off
        kpt_rescore: float = 0.0,  # refined queries score cls^(1 - a) * quality^a
        kpt_ref_scale: float = 0.25,  # sampling extent around each point, x box size
        kpt_dn_groups: int = 1,  # box-denoising groups that also denoise keypoints
    ) -> None:
        super().__init__()
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)

        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        scaled_dim = round(layer_scale * hidden_dim)
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_layers = num_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        self.reg_max = reg_max
        self.mask_dim = mask_dim
        self.enable_mask_head = enable_mask_head
        self.num_kpts = num_kpts

        assert query_select_method in ("default", "one2many", "agnostic"), ""
        assert cross_attn_method in ("default", "discrete"), ""
        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method

        # backbone feature projection
        self._build_input_proj_layer(feat_channels)

        # Transformer module
        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)
        self.kd_outputs = False  # train-only: eval forward also returns what KD distills
        decoder_layer = TransformerDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            num_levels,
            num_points,
            cross_attn_method=cross_attn_method,
        )
        decoder_layer_wide = TransformerDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            num_levels,
            num_points,
            cross_attn_method=cross_attn_method,
            layer_scale=layer_scale,
        )
        self.decoder = TransformerDecoder(
            hidden_dim,
            decoder_layer,
            decoder_layer_wide,
            num_layers,
            nhead,
            reg_max,
            self.reg_scale,
            self.up,
            eval_idx,
            layer_scale,
        )
        # denoising
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(
                num_classes + 1, hidden_dim, padding_idx=num_classes
            )
            init.normal_(self.denoising_class_embed.weight[:-1])

        # segmentation head
        if self.enable_mask_head:
            # When a low-level backbone feature is available (e.g. nano + seg),
            # prepend its channel dim so MaskDecoder builds an extra lateral conv.
            mask_in_chs = list(feat_channels)
            if mask_low_level_ch is not None:
                mask_in_chs = [mask_low_level_ch] + mask_in_chs
            self.mask_decoder = MaskDecoder(in_chs=mask_in_chs, out_ch=self.mask_dim)
            self.mask_head = MLP(self.hidden_dim, self.hidden_dim, self.mask_dim, num_layers=3)

        # decoder embedding
        self.learn_query_content = learn_query_content
        if learn_query_content:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, 2)

        self.enc_output = nn.Sequential(
            OrderedDict(
                [
                    ("proj", nn.Linear(hidden_dim, hidden_dim)),
                    (
                        "norm",
                        nn.LayerNorm(
                            hidden_dim,
                        ),
                    ),
                ]
            )
        )

        if query_select_method == "agnostic":
            self.enc_score_head = nn.Linear(hidden_dim, 1)
        else:
            self.enc_score_head = nn.Linear(hidden_dim, num_classes)

        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3)

        # decoder head
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.dec_score_head = nn.ModuleList(
            [nn.Linear(hidden_dim, num_classes) for _ in range(self.eval_idx + 1)]
            + [nn.Linear(scaled_dim, num_classes) for _ in range(num_layers - self.eval_idx - 1)]
        )
        self.pre_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3)
        self.dec_bbox_head = nn.ModuleList(
            [
                MLP(hidden_dim, hidden_dim, 4 * (self.reg_max + 1), 3)
                for _ in range(self.eval_idx + 1)
            ]
            + [
                MLP(scaled_dim, scaled_dim, 4 * (self.reg_max + 1), 3)
                for _ in range(num_layers - self.eval_idx - 1)
            ]
        )
        self.integral = Integral(self.reg_max)

        # keypoint head: per layer, per point (dx, dy, presence logit, visible logit)
        if num_kpts:
            self.kpt_head = nn.ModuleList(
                [MLP(hidden_dim, hidden_dim, num_kpts * 4, 3) for _ in range(self.eval_idx + 1)]
                + [
                    MLP(scaled_dim, scaled_dim, num_kpts * 4, 3)
                    for _ in range(num_layers - self.eval_idx - 1)
                ]
            )
        self.num_kpt_queries = num_kpt_queries
        self.kpt_dn_noise = kpt_dn_noise
        self.kpt_dn_groups = kpt_dn_groups
        self.kpt_rescore = kpt_rescore
        if num_kpts and num_kpt_layers:
            self.kpt_decoder = KeypointDecoder(
                num_kpts,
                hidden_dim,
                nhead,
                dim_feedforward,
                num_levels,
                num_points,
                num_kpt_layers,
                kpt_ref_scale,
            )

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer("anchors", anchors)
            self.register_buffer("valid_mask", valid_mask)
        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            self.anchors, self.valid_mask = self._generate_anchors()

        self._reset_parameters(feat_channels)

    def convert_to_deploy(self) -> None:
        self.dec_score_head = nn.ModuleList(
            [nn.Identity()] * (self.eval_idx) + [self.dec_score_head[self.eval_idx]]
        )
        self.dec_bbox_head = nn.ModuleList(
            [
                self.dec_bbox_head[i] if i <= self.eval_idx else nn.Identity()
                for i in range(len(self.dec_bbox_head))
            ]
        )
        # Pre-compute abs so downstream ops (distance2bbox) don't need tfl.abs
        self.reg_scale = nn.Parameter(self.reg_scale.abs(), requires_grad=False)
        self.up = nn.Parameter(self.up.abs(), requires_grad=False)

    def _reset_parameters(self, feat_channels: List[int]) -> None:
        bias = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        init.constant_(self.pre_bbox_head.layers[-1].weight, 0)
        init.constant_(self.pre_bbox_head.layers[-1].bias, 0)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, bias)
            if hasattr(reg_, "layers"):
                init.constant_(reg_.layers[-1].weight, 0)
                init.constant_(reg_.layers[-1].bias, 0)

        if self.num_kpts:  # points start at the box center
            for head in self.kpt_head:
                init.constant_(head.layers[-1].weight, 0)
                init.constant_(head.layers[-1].bias, 0)

        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learn_query_content:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        for m, in_channels in zip(self.input_proj, feat_channels):
            if in_channels != self.hidden_dim:
                init.xavier_uniform_(m[0].weight)

    def _build_input_proj_layer(self, feat_channels: List[int]) -> None:
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(
                        OrderedDict(
                            [
                                ("conv", nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)),
                                (
                                    "norm",
                                    nn.BatchNorm2d(
                                        self.hidden_dim,
                                    ),
                                ),
                            ]
                        )
                    )
                )

        in_channels = feat_channels[-1]

        for _ in range(self.num_levels - len(feat_channels)):
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(
                        OrderedDict(
                            [
                                (
                                    "conv",
                                    nn.Conv2d(
                                        in_channels, self.hidden_dim, 3, 2, padding=1, bias=False
                                    ),
                                ),
                                ("norm", nn.BatchNorm2d(self.hidden_dim)),
                            ]
                        )
                    )
                )
                in_channels = self.hidden_dim

    def _get_encoder_input(self, feats: List[torch.Tensor]) -> Tuple[torch.Tensor, List[List[int]]]:
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])

        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)
        return feat_flatten, spatial_shapes

    def _generate_anchors(
        self,
        spatial_shapes: Optional[List[List[int]]] = None,
        grid_size: float = 0.05,
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if spatial_shapes is None:
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size
            for s in self.feat_strides:
                spatial_shapes.append([int(eval_h / s), int(eval_w / s)])

        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype)
            wh = torch.ones_like(grid_xy) * grid_size * (2.0**lvl)
            lvl_anchors = torch.concat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4)
            anchors.append(lvl_anchors)

        anchors = torch.concat(anchors, dim=1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)

        return anchors, valid_mask

    def _get_decoder_input(
        self,
        memory: torch.Tensor,
        spatial_shapes: List[List[int]],
        denoising_logits: Optional[torch.Tensor] = None,
        denoising_bbox_unact: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        # prepare input for decoder
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors = self.anchors
            valid_mask = self.valid_mask
        if memory.shape[0] > 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)

        # memory = torch.where(valid_mask, memory, 0)
        # TODO fix type error for onnx export
        memory = valid_mask.to(memory.dtype) * memory

        output_memory: torch.Tensor = self.enc_output(memory)
        enc_outputs_logits: torch.Tensor = self.enc_score_head(output_memory)

        enc_topk_bboxes_list, enc_topk_logits_list = [], []
        enc_topk_memory, enc_topk_logits, enc_topk_anchors = self._select_topk(
            output_memory, enc_outputs_logits, anchors, self.num_queries
        )

        enc_topk_bbox_unact: torch.Tensor = self.enc_bbox_head(enc_topk_memory) + enc_topk_anchors

        if self.training:
            enc_topk_bboxes = F.sigmoid(enc_topk_bbox_unact)
            enc_topk_bboxes_list.append(enc_topk_bboxes)
            enc_topk_logits_list.append(enc_topk_logits)

        # if self.num_select_queries != self.num_queries:
        #     raise NotImplementedError('')

        if self.learn_query_content:
            content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1])
        else:
            content = enc_topk_memory.detach()

        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()

        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.concat([denoising_bbox_unact, enc_topk_bbox_unact], dim=1)
            content = torch.concat([denoising_logits, content], dim=1)

        return content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list

    def _select_topk(
        self,
        memory: torch.Tensor,
        outputs_logits: torch.Tensor,
        outputs_anchors_unact: torch.Tensor,
        topk: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        if self.query_select_method == "default":
            _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)

        elif self.query_select_method == "one2many":
            _, topk_ind = torch.topk(outputs_logits.flatten(1), topk, dim=-1)
            topk_ind = topk_ind // self.num_classes

        elif self.query_select_method == "agnostic":
            _, topk_ind = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)

        topk_ind: torch.Tensor

        topk_anchors = outputs_anchors_unact.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_anchors_unact.shape[-1])
        )

        topk_logits = (
            outputs_logits.gather(
                dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[-1])
            )
            if self.training
            else None
        )

        topk_memory = memory.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1])
        )

        return topk_memory, topk_logits, topk_anchors

    def _should_do_masks(self, targets: Optional[List[Dict[str, torch.Tensor]]]) -> bool:
        if not self.enable_mask_head:
            return False
        if targets is None:
            # inference path – allow masks if branch is enabled
            return True
        # train path – only if any sample has non-empty masks
        for t in targets:
            m = t.get("masks", None)
            if m is not None and hasattr(m, "numel") and m.numel() > 0:
                return True
        return False

    def _mask_logits_from_h(
        self, h: torch.Tensor, mask_feat: torch.Tensor
    ) -> torch.Tensor:  # h: [B,Q,C]
        mask_embed = self.mask_head(h)  # [B,Q,Cmask]
        # Scale by 1/sqrt(C) to normalize dot product (like attention scaling)
        # This prevents large logits from the sum of C multiplications
        scale = mask_embed.shape[-1] ** -0.5
        mask_embed = mask_embed * scale
        # einsum: (B,Q,C) x (B,C,H,W) -> (B,Q,H,W)
        return torch.einsum("bqc,bchw->bqhw", mask_embed, mask_feat)

    def _kpts_from_h(self, hs: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """-> [L, B, Q, K, 4]: (x, y) normalized, presence and visible logits.

        A point is its box center + a box-relative offset, unbounded so it may leave the box
        (an occluded foot below a cut box). Every layer adds a correction to the previous
        layer's offset (detached, like the box refinement). Eval decodes only the last layer.
        """
        offsets, out = None, []
        for i, h in enumerate(hs):
            raw = self.kpt_head[i](h).reshape(h.shape[0], h.shape[1], self.num_kpts, 4)
            offsets = raw[..., :2] if offsets is None else raw[..., :2] + offsets.detach()
            if self.training or i == len(hs) - 1:
                box = boxes[i if self.training else -1].detach().unsqueeze(-2)  # [B, Q, 1, 4]
                out.append(torch.cat([box[..., :2] + offsets * box[..., 2:], raw[..., 2:]], -1))
        return torch.stack(out)

    def _refine_kpts(
        self,
        h: torch.Tensor,
        logits: torch.Tensor,
        boxes: torch.Tensor,
        kpts: torch.Tensor,
        value: Tuple[torch.Tensor, ...],
        spatial_shapes: List[List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Keypoint decoder on the top num_kpt_queries queries by score; the rest keep the
        per-query head's points. -> (points [L, B, Q, K, C], quality [B, Q, 2])"""
        n = min(self.num_kpt_queries, logits.shape[1])
        idx = torch.topk(logits.detach().max(-1).values, n, dim=1).indices  # [B, N]

        def take(x: torch.Tensor) -> torch.Tensor:
            index = idx.reshape(*idx.shape, *[1] * (x.dim() - 2)).expand(-1, -1, *x.shape[2:])
            return x.gather(1, index)

        refined, quality = self.kpt_decoder(take(h), take(boxes), take(kpts), value, spatial_shapes)
        # queries outside the top N keep the per-query points and log scale 0
        base = F.pad(kpts, (0, refined.shape[-1] - kpts.shape[-1]))
        index = idx[:, :, None, None].expand(-1, -1, *refined.shape[-2:])
        refined = torch.stack([base.scatter(1, index, r) for r in refined])
        # [B, Q, 2]: quality logit, 1 where the query was refined
        quality = torch.stack([quality, torch.ones_like(quality)], -1)
        quality = quality.new_zeros(*logits.shape[:2], 2).scatter(
            1, idx[..., None].expand(-1, -1, 2), quality
        )
        return refined, quality

    def _rescore(self, logits: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
        """Refined queries score cls^(1 - a) * quality^a, a = kpt_rescore: a detection whose
        points are likely off ranks lower (COCO keypoint AP ranks by score)."""
        a = self.kpt_rescore
        score = F.sigmoid(logits).pow(1 - a) * F.sigmoid(quality[..., :1]).pow(a)
        return torch.where(quality[..., 1:] > 0, inverse_sigmoid(score), logits)

    def _denoise_kpts(
        self,
        h: torch.Tensor,
        boxes: torch.Tensor,
        kpts: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
        num_groups: int,
        value: Tuple[torch.Tensor, ...],
        spatial_shapes: List[List[int]],
    ) -> torch.Tensor:
        """Train-only keypoint denoising: the keypoint decoder starts from GT points + uniform
        noise (kpt_dn_noise x GT box size per axis) and learns to pull them back. Runs on the
        positive queries of the first kpt_dn_groups denoising groups: slot g * M + j = GT j.
        Unlabeled points start from the per-query head. -> [L, B, G * M, K, C], M = max GTs
        per image."""
        m = max(len(t["labels"]) for t in targets)
        g = min(self.kpt_dn_groups, num_groups)
        # denoising group i holds M positive, then M negative queries
        arange = functools.partial(torch.arange, device=h.device)
        slots = (arange(g)[:, None] * 2 * m + arange(m)).flatten()
        gt = kpts.new_zeros(kpts.shape[0], m, *kpts.shape[2:3], 3)
        gt_wh = kpts.new_zeros(kpts.shape[0], m, 1, 2)
        for i, t in enumerate(targets):
            gt[i, : len(t["labels"])] = t["kpts"]
            gt_wh[i, : len(t["labels"]), 0] = t["boxes"][:, 2:]
        gt, gt_wh, kpts = gt.repeat(1, g, 1, 1), gt_wh.repeat(1, g, 1, 1), kpts[:, slots]
        noise = (torch.rand_like(gt[..., :2]) * 2 - 1) * self.kpt_dn_noise * gt_wh
        xy = torch.where(gt[..., 2:] > 0, gt[..., :2] + noise, kpts[..., :2])
        kpts = torch.cat([xy, kpts[..., 2:]], -1)
        return self.kpt_decoder(h[:, slots], boxes[:, slots], kpts, value, spatial_shapes)[0]

    def forward(
        self,
        feats: List[torch.Tensor],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
        low_level_feat: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        enable_mask_head = self._should_do_masks(targets)
        enable_kpt_head = self.num_kpts > 0
        # input projection and embedding
        memory, spatial_shapes = self._get_encoder_input(feats)
        # prepare denoising training
        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = (
                get_contrastive_denoising_training_group(
                    targets,
                    self.num_classes,
                    self.num_queries,
                    self.denoising_class_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=1.0,
                )
            )
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list = (
            self._get_decoder_input(memory, spatial_shapes, denoising_logits, denoising_bbox_unact)
        )

        # decoder
        out_bboxes, out_logits, out_corners, out_refs, pre_bboxes, pre_logits, hs = self.decoder(
            init_ref_contents,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            self.pre_bbox_head,
            self.integral,
            self.up,
            self.reg_scale,
            attn_mask=attn_mask,
            return_queries=enable_mask_head or enable_kpt_head,
        )  # hs: [num_layers, B, Q, C]
        out_kpts = self._kpts_from_h(hs, out_bboxes) if enable_kpt_head else None

        if self.training and dn_meta is not None:
            dn_pre_logits, pre_logits = torch.split(pre_logits, dn_meta["dn_num_split"], dim=1)
            dn_pre_bboxes, pre_bboxes = torch.split(pre_bboxes, dn_meta["dn_num_split"], dim=1)
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta["dn_num_split"], dim=2)

            dn_out_corners, out_corners = torch.split(out_corners, dn_meta["dn_num_split"], dim=2)
            dn_out_refs, out_refs = torch.split(out_refs, dn_meta["dn_num_split"], dim=2)

            # split hs for denoising mask computation
            if hs is not None:
                dn_hs, hs = torch.split(hs, dn_meta["dn_num_split"], dim=2)
            if enable_kpt_head:
                dn_out_kpts, out_kpts = torch.split(out_kpts, dn_meta["dn_num_split"], dim=2)

        # keypoint decoder: final points, plus [per-query head, refine layers but the last]
        # for the loss
        if enable_kpt_head:
            kpts, kpt_refine_aux, kpt_dn = out_kpts[-1], [], []
            if hasattr(self, "kpt_decoder"):
                value = self.decoder.value_op(memory, None, None, None, spatial_shapes)
                refined, kpt_quality = self._refine_kpts(
                    hs[-1], out_logits[-1], out_bboxes[-1], kpts, value, spatial_shapes
                )
                kpt_refine_aux = [kpts] + list(refined[:-1])
                kpts = refined[-1]
                if not self.training and self.kpt_rescore > 0:
                    rescored = self._rescore(out_logits[-1], kpt_quality)
                    out_logits = torch.cat([out_logits[:-1], rescored[None]])
                has_dn = dn_meta is not None and dn_meta["dn_num_group"] > 0
                if self.training and self.kpt_dn_noise > 0 and has_dn:
                    kpt_dn = self._denoise_kpts(
                        dn_hs[-1],
                        dn_out_bboxes[-1],
                        dn_out_kpts[-1],
                        targets,
                        dn_meta["dn_num_group"],
                        value,
                        spatial_shapes,
                    )

        # segmentation
        if enable_mask_head:
            pred_masks = None
            aux_masks = None
            dn_aux_masks = None
            mask_feats = list(feats) if low_level_feat is None else [low_level_feat] + list(feats)
            mask_feat = self.mask_decoder(mask_feats)

            # Compute masks for regular queries
            pred_masks = self._mask_logits_from_h(hs[-1], mask_feat)  # logits
            if self.training:  # aux masks only feed the loss
                aux_masks = [
                    self._mask_logits_from_h(h, mask_feat) for h in hs[:-1]
                ]  # list of [B,Q,Hm,Wm]

            # Denoising masks for every layer, final included: dn_outputs zips them per layer
            if self.training and dn_meta is not None and dn_hs is not None:
                dn_aux_masks = [self._mask_logits_from_h(h, mask_feat) for h in dn_hs]

        if self.training:
            out = {
                "pred_logits": out_logits[-1],
                "pred_boxes": out_bboxes[-1],
                "pred_corners": out_corners[-1],
                "ref_points": out_refs[-1],
                "up": self.up,
                "reg_scale": self.reg_scale,
            }
            if enable_mask_head:
                out["pred_masks"] = pred_masks
            if enable_kpt_head:
                out["pred_kpts"] = kpts
                out["kpt_aux_outputs"] = [{"pred_kpts": k} for k in kpt_refine_aux]
                out["kpt_dn_outputs"] = [{"pred_kpts": k} for k in kpt_dn]
                if hasattr(self, "kpt_decoder"):
                    out["pred_kpt_quality"] = kpt_quality
        else:
            out = {
                "pred_logits": out_logits[-1],
                "pred_boxes": out_bboxes[-1],
            }
            if self.kd_outputs:  # eval mode for KD teacher; masks only as logits (VRAM)
                out |= {"pred_corners": out_corners[-1], "ref_points": out_refs[-1]}
                out |= {"up": self.up, "reg_scale": self.reg_scale}
                if enable_mask_head:
                    out["pred_mask_logits"] = pred_masks
            elif enable_mask_head:
                out["pred_masks"] = torch.sigmoid(pred_masks)
            if enable_kpt_head:
                out["pred_kpts"] = torch.cat([kpts[..., :2], kpts[..., 2:].sigmoid()], -1)

        if self.training and self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss2(
                out_logits[:-1],
                out_bboxes[:-1],
                out_corners[:-1],
                out_refs[:-1],
                out_corners[-1],
                out_logits[-1],
                aux_masks=aux_masks if enable_mask_head else None,
                aux_kpts=out_kpts[:-1] if enable_kpt_head else None,
            )
            out["enc_aux_outputs"] = self._set_aux_loss(enc_topk_logits_list, enc_topk_bboxes_list)
            out["pre_outputs"] = {"pred_logits": pre_logits, "pred_boxes": pre_bboxes}
            out["enc_meta"] = {"class_agnostic": self.query_select_method == "agnostic"}

            if dn_meta is not None:
                # Include denoising masks for mask supervision on denoising queries
                out["dn_outputs"] = self._set_aux_loss2(
                    dn_out_logits,
                    dn_out_bboxes,
                    dn_out_corners,
                    dn_out_refs,
                    dn_out_corners[-1],
                    dn_out_logits[-1],
                    aux_masks=dn_aux_masks if enable_mask_head else None,
                    aux_kpts=dn_out_kpts if enable_kpt_head else None,
                )
                out["dn_pre_outputs"] = {"pred_logits": dn_pre_logits, "pred_boxes": dn_pre_bboxes}
                out["dn_meta"] = dn_meta

        return out

    @torch.jit.unused
    def _set_aux_loss(
        self, outputs_class: List[torch.Tensor], outputs_coord: List[torch.Tensor]
    ) -> List[Dict[str, torch.Tensor]]:
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a, "pred_boxes": b} for a, b in zip(outputs_class, outputs_coord)]

    @torch.jit.unused
    def _set_aux_loss2(
        self,
        outputs_class: torch.Tensor,
        outputs_coord: torch.Tensor,
        outputs_corners: torch.Tensor,
        outputs_ref: torch.Tensor,
        teacher_corners: Optional[torch.Tensor] = None,
        teacher_logits: Optional[torch.Tensor] = None,
        aux_masks: Optional[List[torch.Tensor]] = None,
        aux_kpts: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        results = []
        for i, (a, b, c, d) in enumerate(
            zip(outputs_class, outputs_coord, outputs_corners, outputs_ref)
        ):
            result = {
                "pred_logits": a,
                "pred_boxes": b,
                "pred_corners": c,
                "ref_points": d,
                "teacher_corners": teacher_corners,
                "teacher_logits": teacher_logits,
            }
            if aux_masks is not None:
                result["pred_masks"] = aux_masks[i]
            if aux_kpts is not None:
                result["pred_kpts"] = aux_kpts[i]
            results.append(result)
        for extra in (aux_masks, aux_kpts):
            assert extra is None or len(extra) == len(results)
        return results
