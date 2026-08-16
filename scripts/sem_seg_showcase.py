"""Cinematic sem_seg showcase renderer for self-driving demo videos.

Per frame: runs the trained sem_seg model with flow-warped temporal EMA of the
softmax probs (kills small-object flicker), then renders a flat class-coloured
palette overlay and layers optional cinematic elements over the RGB:

  * --wipe            one-shot reveal seam (plays once at the start, then locks)
  * --det_model_path  a second detect/segment checkpoint whose class boxes +
                      labels are drawn on top — only after the wipe finishes.
                      ByteTrack (this repo) steadies them across frames.
  * --hud             bottom class legend + optional title card

Run sem_seg-only (no boxes): just drop --det_model_path.

  uv run python sem_seg_showcase.py \
      --model_path /abs/sem_seg/model.pt --model_name m \
      --config /abs/sem_seg/config.yaml \
      --input clip.mp4 --out showcase.mp4 --device cpu \
      --title "D-FINE-seg · Cityscapes"

Add detection boxes + tracking:

uv run python sem_seg_showcase.py \
  --model_path .../model.pt \
  --model_name m --config  .../config.yaml \
  --det_model_path .../model.pt \
  --det_config  .../config.yaml \
  --input test/video/path/here.mp4 \
  --out output_path/showcase.mp4 --device cuda \
  --alpha 0.4 --title "D-FINE-seg" --overlay_alpha 0.4
"""

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from dfine_seg.dl.utils import overlay_sem_seg, sem_seg_palette
from dfine_seg.infer.byte_track import ByteTrack, Detection
from dfine_seg.infer.torch_model import Torch_model

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
TITLE_FADE_IN, TITLE_FADE_OUT = 0.4, 0.5


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


# --- wipe (one-shot reveal) ---
def wipe(seg_view, raw, x, seam):
    """Show seg_view left of column x, raw RGB right of it, with a bright seam."""
    out = raw.copy()
    out[:, :x] = seg_view[:, :x]
    x0, x1 = max(x - seam, 0), min(x + seam, raw.shape[1])
    out[:, x0:x1] = np.clip(out[:, x0:x1].astype(np.int16) + 120, 0, 255).astype(np.uint8)
    return out


# --- detection boxes + labels ---
def draw_boxes(img, items, names, palette):
    """items: list of (cls_id, (x1,y1,x2,y2))."""
    for label, box in items:
        x1, y1, x2, y2 = (int(v) for v in box)
        c = palette[int(label)].tolist()
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2, cv2.LINE_AA)
        text = str(names[int(label)])
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y0 = max(0, y1 - th - 6)
        cv2.rectangle(img, (x1 - 1, y0 - 1), (x1 + tw + 6, y0 + th + 4), c, -1)
        cv2.putText(
            img, text, (x1 + 2, y0 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
        )
    return img


# --- HUD: bottom legend + title card ---
def make_legend(w, names, palette, row_h=22, pad=6, slot_min=100):
    """Render the class-colour legend strip once: bar_h x W x 3 BGR."""
    names = dict(names)
    cols = max(1, w // slot_min)
    rows = math.ceil(len(names) / cols)
    bar_h = rows * row_h + pad * 2
    bar = np.full((bar_h, w, 3), 14, np.uint8)
    slot_w = w // cols
    for i, (cid, name) in enumerate(names.items()):
        r, c = divmod(i, cols)
        x = c * slot_w + pad
        y = pad + r * row_h
        cv2.rectangle(bar, (x, y + 2), (x + 12, y + 14), palette[int(cid)].tolist(), -1)
        cv2.putText(
            bar,
            str(name),
            (x + 18, y + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    return bar


def draw_hud(img, legend, title, t, title_dur):
    h, w = img.shape[:2]
    if legend is not None:
        lh = legend.shape[0]
        img[h - lh : h, :] = cv2.addWeighted(img[h - lh : h, :], 0.3, legend, 0.85, 0)
    if title and title_dur > 0 and t < title_dur:
        fi = min(TITLE_FADE_IN, title_dur * 0.3)
        fo = min(TITLE_FADE_OUT, title_dur * 0.3)
        fade = min(1.0, t / max(fi, 1e-6)) * min(1.0, (title_dur - t) / max(fo, 1e-6))
        a = 0.7 * max(0.0, min(1.0, fade))
        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 2)
        pad_x, pad_y = 40, 18
        bw, bh = tw + 2 * pad_x, th + 2 * pad_y
        band = np.full((bh, bw, 3), 0, np.uint8)
        cv2.putText(
            band,
            title,
            (pad_x, pad_y + th),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        x0 = (w - bw) // 2
        yv = int(h * 0.10)  # upper area, not full width
        img[yv : yv + bh, x0 : x0 + bw] = cv2.addWeighted(
            img[yv : yv + bh, x0 : x0 + bw], 1 - a, band, a, 0
        )
    return img


def class_edges(label_map, width):
    """Boolean mask of class-boundary pixels, thickened to `width` px."""
    e = np.zeros(label_map.shape, np.uint8)
    e[:, :-1] |= (label_map[:, 1:] != label_map[:, :-1]).astype(np.uint8)
    e[:-1, :] |= (label_map[1:, :] != label_map[:-1, :]).astype(np.uint8)
    if width > 1:
        e = cv2.dilate(e, np.ones((width, width), np.uint8))
    return e.astype(bool)


def render(frame, label_map, palette, args, x_wipe, w, items, det_names, det_palette):
    seg = overlay_sem_seg(frame, label_map, palette, args.overlay_alpha, args.ignore_index)
    if args.edge_width > 0:
        e = class_edges(label_map, args.edge_width)
        seg[e] = args.edge_color
    if items:
        seg = draw_boxes(seg, items, det_names, det_palette)
    if args.wipe and x_wipe < w:
        return wipe(seg, frame, x_wipe, args.seam)
    return seg


def process_video(
    sem_tm, det_tm, det_names, det_palette, palette, names_all, in_path, out_path, args
):
    vid = cv2.VideoCapture(str(in_path))
    if not vid.isOpened():
        print(f"  !! could not open {in_path}, skipping")
        return
    fps = args.fps or vid.get(cv2.CAP_PROP_FPS) or 17.0
    w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    legend = make_legend(w, names_all, palette) if args.hud else None
    tracker = (
        ByteTrack(detrack_thresh=max(args.box_thresh, 0.4))
        if (det_tm is not None and args.track)
        else None
    )

    ema, prev_gray, n = None, None, 0
    ok, frame = vid.read()
    while ok:
        proc, psizes, osizes = sem_tm._prepare_inputs(frame, bgr=True)
        probs = torch.softmax(sem_tm._predict(proc)["sem_seg_logits"], dim=1)  # [1,C,H,W]
        cur_gray = input_gray(proc)
        if ema is None or args.alpha == 0:
            ema = probs
        else:
            ema = args.alpha * warp_probs(ema, backward_flow(cur_gray, prev_gray), sem_tm.device)
            ema = ema + (1.0 - args.alpha) * probs
        prev_gray = cur_gray

        lm = (
            sem_tm.process_sem_seg(ema, psizes, osizes, sem_tm.keep_ratio)[0]["sem_seg"]
            .cpu()
            .numpy()
        )

        # detection runs first so boxes can be baked into the seg-view and then
        # revealed by the wipe (drawn before wipe composition). Tracker runs from
        # frame 0 so tracks steady fast.
        items = []
        if det_tm is not None:
            res = det_tm(frame, bgr=True)[0]
            boxes = res["boxes"].cpu().numpy()
            labels = res["labels"].cpu().numpy()
            scores = res["scores"].cpu().numpy()
            if tracker:
                dets = [
                    Detection(tuple(b.tolist()), float(s), int(c))
                    for b, c, s in zip(boxes, labels, scores)
                    if s >= args.box_thresh
                ]
                tracked = tracker.update(dets, frame_shape=(h, w))
                items = [(cls, box) for _, cls, box, _ in tracked]
            else:
                items = [
                    (int(c), b.tolist())
                    for b, c, s in zip(boxes, labels, scores)
                    if s >= args.box_thresh
                ]

        # one-shot reveal: eased 0->1 over wipe_period seconds, then locked at 1
        frac = min(1.0, (n / fps) / args.wipe_period)
        frac = 0.5 - 0.5 * math.cos(math.pi * frac)
        out = render(frame, lm, palette, args, int(frac * w), w, items, det_names, det_palette)

        if args.hud:
            out = draw_hud(out, legend, args.title, n / fps, args.title_dur)

        writer.write(out)
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
    ap.add_argument("--alpha", type=float, default=0.4, help="temporal weight on history (0=off)")
    # base overlay
    ap.add_argument("--overlay_alpha", type=float, default=0.5, help="palette overlay alpha")
    ap.add_argument(
        "--edge_width", type=int, default=1, help="class-boundary line thickness in px (0=off)"
    )
    ap.add_argument(
        "--edge_color",
        type=int,
        nargs=3,
        default=(200, 200, 200),
        metavar=("B", "G", "R"),
        help="edge line color BGR (e.g. 255 255 255 for white)",
    )
    # wipe (one-shot reveal)
    ap.add_argument("--wipe", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--wipe_period", type=float, default=2.5, help="one-shot reveal duration (s)")
    ap.add_argument("--seam", type=int, default=6, help="wipe seam half-width (px)")
    # detector (optional) — omit for sem_seg-only
    ap.add_argument("--det_model_path", default=None, help="detect/segment checkpoint for boxes")
    ap.add_argument("--det_config", default=None, help="detector config (default: --config)")
    ap.add_argument("--det_name", default=None, help="override detector model_name")
    ap.add_argument("--box_thresh", type=float, default=0.35, help="detector confidence thresh")
    ap.add_argument(
        "--track",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ByteTrack on the detection boxes (default on when a detector is set)",
    )
    # HUD
    ap.add_argument("--hud", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--title", default=None, help="centered title card text")
    ap.add_argument("--title_dur", type=float, default=1.0, help="title card on-screen seconds")
    # io / model
    ap.add_argument("--fps", type=float, default=None, help="output fps (default: source)")
    ap.add_argument("--max_frames", type=int, default=0, help="cap frames per clip (0=all)")
    ap.add_argument("--img_size", type=int, nargs=2, default=None, help="H W override")
    ap.add_argument("--keep_ratio", choices=["auto", "true", "false"], default="auto")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    names_all = OmegaConf.to_container(cfg.train.label_to_name, resolve=True)
    img_h, img_w = args.img_size or list(cfg.train.img_size)
    keep_ratio = (
        bool(cfg.train.get("keep_ratio", False))
        if args.keep_ratio == "auto"
        else args.keep_ratio == "true"
    )
    args.ignore_index = int(cfg.train.get("sem_seg", {}).get("ignore_index", 255))

    sem_tm = Torch_model(
        model_name=args.model_name or cfg.model_name,
        model_path=args.model_path,
        n_outputs=len(names_all),
        input_width=img_w,
        input_height=img_h,
        conf_thresh=float(cfg.train.conf_thresh),
        keep_ratio=keep_ratio,
        channels=int(cfg.train.in_channels),
        task="sem_seg",
        device=args.device,
    )
    palette = sem_seg_palette(len(names_all))

    # optional detector (detect or segment task) for class boxes + labels
    det_tm, det_names, det_palette = None, None, None
    if args.det_model_path:
        dcfg = OmegaConf.load(args.det_config or args.config)
        det_names = OmegaConf.to_container(dcfg.train.label_to_name, resolve=True)
        dh, dw = args.img_size or list(dcfg.train.img_size)
        dkr = (
            bool(dcfg.train.get("keep_ratio", False))
            if args.keep_ratio == "auto"
            else args.keep_ratio == "true"
        )
        det_tm = Torch_model(
            model_name=args.det_name or dcfg.model_name,
            model_path=args.det_model_path,
            n_outputs=len(det_names),
            input_width=dw,
            input_height=dh,
            conf_thresh=args.box_thresh,
            keep_ratio=dkr,
            channels=int(dcfg.train.in_channels),
            task=dcfg.task,
            device=args.device,
        )
        det_palette = sem_seg_palette(len(det_names))

    styles = [
        s
        for s, on in [
            ("wipe", args.wipe),
            ("edges", args.edge_width > 0),
            ("boxes", det_tm is not None),
            ("track", bool(det_tm is not None and args.track)),
            ("hud", args.hud),
        ]
        if on
    ]
    print(
        f"alpha={args.alpha} input={img_h}x{img_w} keep_ratio={keep_ratio} "
        f"styles={styles or ['flat']}"
    )

    in_path = Path(args.input)
    if in_path.is_dir():
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        for v in sorted(f for f in in_path.iterdir() if f.suffix.lower() in VIDEO_EXTS):
            process_video(
                sem_tm,
                det_tm,
                det_names,
                det_palette,
                palette,
                names_all,
                v,
                out_dir / f"{v.stem}_showcase.mp4",
                args,
            )
    else:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        process_video(
            sem_tm, det_tm, det_names, det_palette, palette, names_all, in_path, args.out, args
        )


if __name__ == "__main__":
    main()
