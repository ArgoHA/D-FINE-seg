"""Build the README hero mosaic: input + detection + instance seg + semantic seg (2x2).

All three prediction panels use S models trained on Cityscapes (paths under ROOT below).
Renders one mosaic per --image. The committed assets/mosaic.jpg was made with:

    uv run python scripts/make_mosaic.py --images munster_000165_000019 --out-dir assets
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.dl.utils import Visualizer, overlay_sem_seg, sem_seg_palette
from src.infer.torch_model import Torch_model

ROOT = Path("/home/argo/Desktop/Projects/cityscapes")
IMG_DIR = ROOT / "data/dataset/images"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

DET_NAMES = {0: "person", 1: "rider", 2: "car", 3: "truck", 4: "bus", 5: "train", 6: "motorcycle", 7: "bicycle"}
SEM_NAMES = {
    0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence", 5: "pole",
    6: "traffic-light", 7: "traffic-sign", 8: "vegetation", 9: "terrain", 10: "sky",
    11: "person", 12: "rider", 13: "car", 14: "truck", 15: "bus", 16: "train",
    17: "motorcycle", 18: "bicycle",
}

PANEL_H = 900
GUTTER = 10
FINAL_W = 2560


def latest(exp_name):
    d = ROOT / "output/models"
    cands = sorted(p for p in d.glob(f"{exp_name}_*") if p.is_dir())
    return cands[-1] / "model.pt"


def draw_dets(img, res, names, colors, with_masks, label_min_px=26):
    """Boxes (+ masks), class names only - no scores, no Hershey font."""
    boxes = res["boxes"].cpu().numpy()
    labels = res["labels"].cpu().numpy().astype(int)

    if with_masks:
        masks = res["masks"].cpu().numpy()
        overlay = img.copy()
        for m, lb in zip(masks, labels):
            overlay[m.astype(bool)] = colors[lb]
        img = cv2.addWeighted(img, 0.55, overlay, 0.45, 0)
        for m, lb in zip(masks, labels):
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, colors[lb], 2, cv2.LINE_AA)

    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil, "RGBA")
    font = ImageFont.truetype(FONT, 17)
    for (x1, y1, x2, y2), lb in zip(boxes, labels):
        c = tuple(int(v) for v in colors[lb][::-1])  # BGR -> RGB
        d.rectangle([x1, y1, x2, y2], outline=c + (235,), width=3)
        if (x2 - x1) < label_min_px:  # skip labels on tiny far-away objects
            continue
        txt = names[lb]
        tw = d.textlength(txt, font=font)
        tx = min(x1, img.shape[1] - tw - 13)  # keep the chip inside the frame
        ty = max(0, y1 - 24)
        d.rounded_rectangle([tx, ty, tx + tw + 12, ty + 23], radius=5, fill=c + (235,))
        d.text((tx + 6, ty + 4), txt, font=font, fill=(255, 255, 255, 255))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def caption(img, title, sub):
    """Translucent chip inside the panel - theme-neutral, no background bar."""
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil, "RGBA")
    f_t = ImageFont.truetype(FONT, 32)
    f_s = ImageFont.truetype(FONT, 23)
    pad, x, y = 18, 20, 20
    w = max(d.textlength(title, font=f_t), d.textlength(sub, font=f_s)) + 2 * pad
    d.rounded_rectangle([x, y, x + w, y + 88], radius=12, fill=(13, 15, 20, 210))
    d.text((x + pad, y + 13), title, font=f_t, fill=(255, 255, 255, 255))
    d.text((x + pad, y + 54), sub, font=f_s, fill=(165, 174, 188, 255))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def fit(img, crop):
    """crop = (x0, y0, y1) as fractions of the frame; None keeps the full frame."""
    if crop:
        x0, y0, y1 = crop
        h, w = img.shape[:2]
        img = img[int(h * y0) : int(h * y1), int(w * x0) :]
    return cv2.resize(img, (int(img.shape[1] * PANEL_H / img.shape[0]), PANEL_H), cv2.INTER_AREA)


def build(img, models, colors, crop):
    det, seg, sem = models
    p1 = draw_dets(img.copy(), det(img, bgr=True)[0], DET_NAMES, colors, with_masks=False)
    p2 = draw_dets(img.copy(), seg(img, bgr=True)[0], DET_NAMES, colors, with_masks=True)
    sem_map = sem(img, bgr=True)[0]["sem_seg"].cpu().numpy()
    p3 = overlay_sem_seg(img.copy(), sem_map, sem_seg_palette(len(SEM_NAMES)), alpha=0.55)

    tiles = [
        (img.copy(), "Input", "Cityscapes 2048x1024"),
        (p1, "Detection", "D-FINE  S"),
        (p2, "Instance Segmentation", "D-FINE-seg  S"),
        (p3, "Semantic Segmentation", "D-FINE-seg  S"),
    ]
    panels = [caption(fit(t, crop), title, sub) for t, title, sub in tiles]

    vgap = np.full((PANEL_H, GUTTER, 3), 255, np.uint8)
    rows = [cv2.hconcat([panels[0], vgap, panels[1]]), cv2.hconcat([panels[2], vgap, panels[3]])]
    hgap = np.full((GUTTER, rows[0].shape[1], 3), 255, np.uint8)
    mosaic = cv2.vconcat([rows[0], hgap, rows[1]])
    return cv2.resize(mosaic, (FINAL_W, int(FINAL_W * mosaic.shape[0] / mosaic.shape[1])), cv2.INTER_AREA)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images", nargs="+", required=True, help="frame stems in IMG_DIR")
    p.add_argument("--out-dir", required=True, help="one <stem>.jpg per image lands here")
    p.add_argument("--crop-x0", type=float, default=0.22, help="cut this fraction off the left")
    p.add_argument("--crop-y0", type=float, default=0.02, help="cut the corrupted top scanline")
    p.add_argument("--crop-y1", type=float, default=0.87, help="cut the ego-hood below this")
    p.add_argument("--no-crop", action="store_true", help="keep the full 2048x1024 frame")
    p.add_argument("--device", default="cpu", help="cpu | cuda")
    args = p.parse_args()

    crop = None if args.no_crop else (args.crop_x0, args.crop_y0, args.crop_y1)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = Visualizer.generate_colors(len(DET_NAMES))
    kw = dict(device=args.device)
    models = (
        Torch_model("s", str(latest("det_s")), n_outputs=8, task="detect", **kw),
        Torch_model("s", str(latest("seg_s")), n_outputs=8, task="segment", **kw),
        Torch_model("s", str(latest("sem_seg_s")), n_outputs=19, task="sem_seg", **kw),
    )

    for name in args.images:
        stem = Path(name).stem
        img = cv2.imread(str(IMG_DIR / f"{stem}.png"))
        if img is None:
            print(f"MISSING {stem}")
            continue
        out = out_dir / f"{stem}.jpg"
        cv2.imwrite(str(out), build(img, models, colors, crop), [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"saved {out}")
