#!/usr/bin/env python
"""Good-looking sem_seg showcase video renderer (built on temporal_smooth.py).

Runs the trained sem_seg model per frame with flow-warped temporal smoothing of
the softmax probabilities (temporal_smooth.py logic — kills small-object flicker),
then renders up to three cinematic styles over the RGB, each an independent toggle
(all on by default):

  * --desat_pop  world greyed out, chosen classes kept in full colour (the base fill)
  * --contours   palette-coloured glowing class boundaries drawn on top
  * --wipe       animated seam sweeping between the seg-view and the raw RGB

They compose: desat_pop builds the base (falls back to a flat palette overlay when
off), contours glow on top, wipe blends that seg-view against the raw frame. Turn
any off to isolate a look.

Run (CPU, from repo root):
  uv run python sem_seg_showcase.py \
      --model_path /abs/model.pt --model_name m --config /abs/frozen/config.yaml \
      --input clip.mp4 --out showcase.mp4 --device cpu
"""
import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.dl.utils import overlay_sem_seg, sem_seg_palette
from src.infer.torch_model import Torch_model

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
# Cityscapes "things" (movers + signs) — the classes worth popping into colour.
DEFAULT_POP = [6, 7, 11, 12, 13, 14, 15, 16, 17, 18]


# --- flow-warped temporal smoothing (temporal_smooth.py logic) ---
def backward_flow(cur_gray, prev_gray):
    """Flow that, for each current pixel, points to its source in the previous frame."""
    return cv2.calcOpticalFlowFarneback(cur_gray, prev_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)


def warp_probs(probs, flow, device):
    """Sample probs[1,C,H,W] at (x+flow_x, y+flow_y) via grid_sample -> warped [1,C,H,W]."""
    h, w = flow.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    gx = 2.0 * (xx + flow[..., 0]) / (w - 1) - 1.0
    gy = 2.0 * (yy + flow[..., 1]) / (h - 1) - 1.0
    grid = torch.from_numpy(np.stack([gx, gy], -1)).float().unsqueeze(0).to(device)
    return F.grid_sample(probs, grid, mode="bilinear", padding_mode="border", align_corners=True)


def input_gray(proc):
    """Grayscale of the exact preprocessed input (matches prob-map layout incl. any padding)."""
    return (proc[0, :3].mean(0) * 255).clamp(0, 255).byte().cpu().numpy()


def desat_pop(frame, label_map, pop_ids, palette, tint):
    """Greyscale the world, keep pop_ids classes in full colour (+ optional palette tint)."""
    gray = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    pop = np.isin(label_map, pop_ids)
    gray[pop] = frame[pop]
    if tint > 0:
        gray[pop] = cv2.addWeighted(frame, 1 - tint, palette[label_map], tint, 0)[pop]
    return gray


def class_boundaries(label_map, ignore_index):
    """Boolean map of pixels sitting on a class edge (4-neighbour), excluding ignore pixels."""
    lm = label_map
    b = np.zeros(lm.shape, bool)
    dx = lm[:, :-1] != lm[:, 1:]
    dy = lm[:-1, :] != lm[1:, :]
    b[:, :-1] |= dx
    b[:, 1:] |= dx
    b[:-1, :] |= dy
    b[1:, :] |= dy
    return b & (lm != ignore_index)


def glowing_contours(base, label_map, palette, thickness, glow, ignore_index):
    """Additive palette-coloured halo + crisp core line along class boundaries."""
    b = class_boundaries(label_map, ignore_index)
    edge = np.zeros_like(base)
    edge[b] = palette[label_map][b]
    if thickness > 1:
        edge = cv2.dilate(edge, np.ones((thickness, thickness), np.uint8))
    halo = cv2.GaussianBlur(edge, (0, 0), sigmaX=thickness * 2.5)
    out = cv2.addWeighted(base, 1.0, halo, glow, 0)
    core = edge.any(2)
    out[core] = np.clip(edge[core].astype(np.int16) + 60, 0, 255).astype(np.uint8)
    return out


def wipe(seg_view, raw, x, seam):
    """Show seg_view left of column x, raw RGB right of it, with a bright seam."""
    out = raw.copy()
    out[:, :x] = seg_view[:, :x]
    x0, x1 = max(x - seam, 0), min(x + seam, raw.shape[1])
    out[:, x0:x1] = np.clip(out[:, x0:x1].astype(np.int16) + 120, 0, 255).astype(np.uint8)
    return out


def render(frame, label_map, palette, args, x_wipe):
    if args.desat_pop:
        seg = desat_pop(frame, label_map, args.pop_ids, palette, args.pop_tint)
    else:
        seg = overlay_sem_seg(frame, label_map, palette, args.overlay_alpha)
    if args.contours:
        seg = glowing_contours(seg, label_map, palette, args.thickness, args.glow, args.ignore_index)
    if args.wipe:
        return wipe(seg, frame, x_wipe, args.seam)
    return seg


def process_video(tm, palette, in_path, out_path, args):
    vid = cv2.VideoCapture(str(in_path))
    if not vid.isOpened():
        print(f"  !! could not open {in_path}, skipping")
        return
    fps = args.fps or vid.get(cv2.CAP_PROP_FPS) or 17.0
    w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    ema, prev_gray, n = None, None, 0
    ok, frame = vid.read()
    while ok:
        proc, psizes, osizes = tm._prepare_inputs(frame, bgr=True)
        probs = torch.softmax(tm._predict(proc)["sem_seg_logits"], dim=1)  # [1,C,H,W]
        cur_gray = input_gray(proc)
        if ema is None or args.alpha == 0:
            ema = probs
        else:
            ema = args.alpha * warp_probs(ema, backward_flow(cur_gray, prev_gray), tm.device)
            ema = ema + (1.0 - args.alpha) * probs
        prev_gray = cur_gray

        lm = tm.process_sem_seg(ema, psizes, osizes, tm.keep_ratio)[0]["sem_seg"].cpu().numpy()
        # cosine-eased ping-pong seam, loop-friendly
        frac = 0.5 - 0.5 * math.cos(2 * math.pi * (n / fps) / args.wipe_period)
        writer.write(render(frame, lm, palette, args, int(frac * w)))

        n += 1
        if args.max_frames and n >= args.max_frames:
            break
        ok, frame = vid.read()

    writer.release()
    vid.release()
    print(f"  wrote {out_path}  ({n} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model_name", default=None, help="override cfg.model_name (n/s/m/l/x)")
    ap.add_argument("--input", required=True, help="video file or folder of videos")
    ap.add_argument("--out", required=True, help="output file (single input) or folder")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--device", default=None, help="cpu / cuda / mps (default: auto)")
    # temporal smoothing
    ap.add_argument("--alpha", type=float, default=0.85, help="temporal weight on history (0=off)")
    # style toggles (all on by default; --no-<x> to disable)
    ap.add_argument("--desat_pop", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--contours", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--wipe", action=argparse.BooleanOptionalAction, default=True)
    # style knobs
    ap.add_argument("--pop_classes", default=None, help="comma ids to keep in colour (default: things)")
    ap.add_argument("--pop_tint", type=float, default=0.0, help="palette tint on popped classes [0..1]")
    ap.add_argument("--overlay_alpha", type=float, default=0.5, help="flat overlay alpha (desat_pop off)")
    ap.add_argument("--thickness", type=int, default=2, help="contour line thickness (px)")
    ap.add_argument("--glow", type=float, default=1.6, help="contour halo intensity")
    ap.add_argument("--wipe_period", type=float, default=5.0, help="seam there-and-back period (s)")
    ap.add_argument("--seam", type=int, default=6, help="wipe seam half-width (px)")
    # io / model
    ap.add_argument("--fps", type=float, default=None, help="output fps (default: source)")
    ap.add_argument("--max_frames", type=int, default=0, help="cap frames per clip (0=all)")
    ap.add_argument("--img_size", type=int, nargs=2, default=None, help="H W override")
    ap.add_argument("--keep_ratio", choices=["auto", "true", "false"], default="auto")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    label_to_name = OmegaConf.to_container(cfg.train.label_to_name, resolve=True)
    img_h, img_w = args.img_size or list(cfg.train.img_size)
    keep_ratio = (
        bool(cfg.train.get("keep_ratio", False)) if args.keep_ratio == "auto"
        else args.keep_ratio == "true"
    )
    args.ignore_index = int(cfg.train.get("sem_seg", {}).get("ignore_index", 255))
    args.pop_ids = (
        [int(i) for i in args.pop_classes.split(",")] if args.pop_classes else DEFAULT_POP
    )

    tm = Torch_model(
        model_name=args.model_name or cfg.model_name,
        model_path=args.model_path,
        n_outputs=len(label_to_name),
        input_width=img_w,
        input_height=img_h,
        conf_thresh=float(cfg.train.conf_thresh),
        keep_ratio=keep_ratio,
        channels=int(cfg.train.in_channels),
        task="sem_seg",
        device=args.device,
    )
    palette = sem_seg_palette(len(label_to_name))
    styles = [s for s, on in [("desat_pop", args.desat_pop), ("contours", args.contours),
                              ("wipe", args.wipe)] if on]
    print(f"alpha={args.alpha} input={img_h}x{img_w} keep_ratio={keep_ratio} "
          f"styles={styles or ['flat']} pop={args.pop_ids}")

    in_path = Path(args.input)
    if in_path.is_dir():
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        for v in sorted(f for f in in_path.iterdir() if f.suffix.lower() in VIDEO_EXTS):
            process_video(tm, palette, v, out_dir / f"{v.stem}_showcase.mp4", args)
    else:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        process_video(tm, palette, in_path, args.out, args)


if __name__ == "__main__":
    main()
