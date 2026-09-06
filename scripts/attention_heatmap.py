"""
Attention heatmaps for D-FINE (issue #14): for each detection, a Grad-CAM-style
overlay showing where its decoder query looked.

Why plain Grad-CAM / "attention after the decoder" gives garbage here:
  - Cross-attention in D-FINE is multi-scale DEFORMABLE: each query samples
    n_heads * n_levels * n_points locations (8*3*4 = 96 for dfine_s) with softmax
    weights. There is no dense attention matrix to plot, and backpropagating a
    class score through 96 scattered samples makes CAM-style maps look like noise.
  - The decoder outputs D-dim query vectors, so anything drawn "after the decoder"
    must be projected back to pixels via the attention's sampling locations.

This script hooks every executed decoder layer's MSDeformableAttention, recomputes
its softmax attention weights + sampling locations (exact same math as the module
forward), and splats the weighted samples onto each encoder level. High response
inside a query's box is the evidence it used; response elsewhere shows what else
it considered. Works with detect and segment .pt checkpoints (torch backend only).

Usage:
    uv run python scripts/attention_heatmap.py <model|n|s|m|l|x> <image|dir> [options]

Examples:
    uv run python scripts/attention_heatmap.py pretrained/dfine_s_coco.pt img.jpg
    uv run python scripts/attention_heatmap.py s assets/ --topk 4 --layer all -o attn_out
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from dfine_seg import load_model

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def make_capture(model):
    """Register hooks on all decoder layers' cross-attention; returns (captured, handles).

    Each entry: {"attn": [B, Q, heads, points], "locs": [B, Q, heads, points, 2] in
    normalized xy, "shapes": [(h, w) per level], "n_points": points per level}.
    """
    captured, handles = [], []

    def hook(module, args, output):
        query, ref_points, _, shapes = args
        bs, Lq = query.shape[:2]
        pts = module.total_points // module.num_heads  # points per head (= sum(n_points_list))
        attn = module.attention_weights(query).reshape(bs, Lq, module.num_heads, pts).softmax(-1)
        offs = module.sampling_offsets(query).reshape(bs, Lq, module.num_heads, pts, 2)
        if ref_points.shape[-1] == 4:  # box references: offsets scale with box wh
            scale = module.num_points_scale.to(query.dtype).unsqueeze(-1)
            locs = ref_points[:, :, None, :, :2] + (
                offs * scale * ref_points[:, :, None, :, 2:] * module.offset_scale
            )
        else:  # point references: offsets scaled by level size
            norm = (
                torch.tensor(shapes, device=query.device)
                .flip([1])
                .reshape(1, 1, 1, module.num_levels, 1, 2)
            )
            locs = ref_points.reshape(bs, Lq, 1, module.num_levels, 1, 2) + offs / norm
        captured.append(
            {
                "attn": attn.detach(),
                "locs": locs.detach(),
                "shapes": [tuple(s) for s in shapes],
                "n_points": list(module.num_points_list),
            }
        )

    for layer in model.decoder.decoder.layers:
        handles.append(layer.cross_attn.register_forward_hook(hook))
    return captured, handles


def bilinear_splat(pts, wts, h, w):
    """Scatter weights `wts` [N] at normalized xy `pts` [N, 2] into an [h, w] grid."""
    acc = torch.zeros(h, w)
    x = (pts[:, 0] * w - 0.5).clamp(0, w - 1)
    y = (pts[:, 1] * h - 0.5).clamp(0, h - 1)
    x0, y0 = x.long(), y.long()
    fx, fy = x - x0, y - y0
    for dx in (0, 1):
        for dy in (0, 1):
            xi = (x0 + dx).clamp(0, w - 1)
            yi = (y0 + dy).clamp(0, h - 1)
            wx = fx if dx else 1 - fx
            wy = fy if dy else 1 - fy
            acc.index_put_((yi, xi), wts * wx * wy, accumulate=True)
    return acc


def layer_heatmap(cap, q, H, W):
    """One query's attention over the encoder levels -> [H, W] in [0, 1]."""
    attn, locs = cap["attn"][0, q].cpu(), cap["locs"][0, q].cpu()
    levels, start = [], 0
    for (h, w), n_pts in zip(cap["shapes"], cap["n_points"]):
        a = attn[:, start : start + n_pts].reshape(-1)
        p = locs[:, start : start + n_pts].reshape(-1, 2)
        start += n_pts
        acc = bilinear_splat(p, a, h, w)
        up = F.interpolate(acc[None, None] / (acc.max() + 1e-9), size=(H, W), mode="bilinear")[0, 0]
        levels.append(up)
    return torch.stack(levels).mean(0)


def render_panel(base, heatmap, box, label):
    """Overlay one heatmap + box on the (processed-size) BGR image."""
    hm = heatmap.numpy()
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-9)
    heat = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    panel = cv2.addWeighted(base, 0.55, heat, 0.45, 0)
    H, W = base.shape[:2]
    cx, cy, bw, bh = box
    x1, y1 = int((cx - bw / 2) * W), int((cy - bh / 2) * H)
    x2, y2 = int((cx + bw / 2) * W), int((cy + bh / 2) * H)
    cv2.rectangle(panel, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.putText(
        panel, label, (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
    )
    return panel


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("model", help="size (n|s|m|l|x) or path to a .pt checkpoint")
    ap.add_argument("images", help="image file or directory")
    ap.add_argument("-o", "--out", default="attention_heatmaps", help="output directory")
    ap.add_argument("--topk", type=int, default=6, help="detections to visualize per image")
    ap.add_argument("--conf", type=float, default=0.25, help="score threshold for the top-k pool")
    ap.add_argument(
        "--layer",
        default="last",
        help="decoder layer to visualize: 0-based index, 'last' (default) or 'all' (mean)",
    )
    ap.add_argument("--size", type=int, default=None, help="square input size (default: ckpt's)")
    ap.add_argument("--device", default=None, help="cuda | cpu | mps (default: auto)")
    args = ap.parse_args()

    kwargs = {}
    if args.size:
        kwargs["input_height"] = kwargs["input_width"] = args.size
    if args.device:
        kwargs["device"] = args.device
    wm = load_model(args.model, **kwargs)
    names = wm.names or {}

    paths = (
        sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in IMG_EXTS)
        if Path(args.images).is_dir()
        else [Path(args.images)]
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        bgr = cv2.imread(str(path))
        if bgr is None:
            print(f"skip (unreadable): {path}")
            continue
        tensor, _, _ = wm._prepare_inputs(bgr)  # same preprocessing as inference
        captured, handles = make_capture(wm.model)
        try:
            with torch.no_grad():
                out = wm.model(tensor)
        finally:
            for h in handles:
                h.remove()

        scores, cls_ids = out["pred_logits"][0].sigmoid().max(-1)
        keep = (scores >= args.conf).nonzero().flatten()
        n = min(args.topk, len(keep))
        if n == 0:
            keep = scores.topk(1).indices.flatten()  # always show something
            n = 1
        queries = keep[scores[keep].topk(n).indices]

        caps = captured
        if args.layer != "last":
            caps = captured if args.layer == "all" else [captured[int(args.layer)]]

        base = tensor[0].mul(255).byte().permute(1, 2, 0).cpu().numpy()
        base = np.ascontiguousarray(base[:, :, ::-1])  # RGB -> BGR, network input size
        H, W = base.shape[:2]
        panels = []
        for rank, q in enumerate(queries.tolist()):
            hm = torch.stack([layer_heatmap(c, q, H, W) for c in caps]).mean(0)
            label = f"{names.get(cls_ids[q].item(), cls_ids[q].item())} {scores[q]:.2f}"
            panel = render_panel(base, hm, out["pred_boxes"][0, q].tolist(), label)
            cv2.imwrite(
                str(
                    out_dir
                    / f"{path.stem}_attn_q{rank}_{label.replace(' ', '_').replace('.', '')}.jpg"
                ),
                panel,
            )
            panels.append(panel)

        cols = 3
        rows = []
        for i in range(0, len(panels), cols):
            row = panels[i : i + cols]
            row += [np.full_like(row[0], 32)] * (cols - len(row))
            rows.append(np.hstack(row))
        cv2.imwrite(str(out_dir / f"{path.stem}_attn_grid.jpg"), np.vstack(rows))
        print(f"{path.name}: {len(panels)} heatmaps -> {out_dir}/")


if __name__ == "__main__":
    main()
