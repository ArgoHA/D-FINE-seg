import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.infer.sam3_model import SAM3_model

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

BOTTLE_CLASS_ID = 0


def box_to_yolo(box, img_w: int, img_h: int):
    """Convert (x1, y1, x2, y2) → YOLO (cx, cy, w, h) normalised."""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return cx, cy, w, h


MASK_COLORS = [
    (0, 200, 255),
    (255, 100, 0),
    (0, 255, 100),
    (200, 0, 255),
    (255, 255, 0),
    (0, 150, 200),
]
MASK_ALPHA = 0.45
BBOX_THICKNESS = 2


def draw_masks(image_rgb: np.ndarray, masks: np.ndarray, boxes: np.ndarray, scores: np.ndarray):
    """Overlay coloured semi-transparent masks and bboxes onto the image."""
    vis = image_rgb.copy()
    img_h, img_w = vis.shape[:2]

    for i, mask in enumerate(masks):
        if mask.ndim == 3:
            mask = mask[0]

        if mask.shape[:2] != (img_h, img_w):
            mask = cv2.resize(
                mask.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        color = MASK_COLORS[i % len(MASK_COLORS)]
        overlay = vis.copy()
        overlay[mask] = color
        cv2.addWeighted(overlay, MASK_ALPHA, vis, 1 - MASK_ALPHA, 0, vis)

        x1, y1, x2, y2 = int(boxes[i][0]), int(boxes[i][1]), int(boxes[i][2]), int(boxes[i][3])
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, BBOX_THICKNESS)
        cv2.putText(
            vis,
            f"{scores[i]:.2f}",
            (x1, max(y1 - 6, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    return vis


def process_image(
    image_path: Path,
    model: SAM3_model,
    output_dir: Path,
    vis_dir: Path | None = None,
):
    txt_path = output_dir / (image_path.stem + ".txt")

    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size
    image_rgb = np.array(image)

    res = model(image_rgb, bgr=False)[0]

    boxes = res["boxes"].numpy()
    scores = res["scores"].numpy()
    masks = res["masks"].numpy().astype(bool) if vis_dir is not None else None

    if len(boxes) == 0:
        return 0

    yolo_lines = []
    for box in boxes:
        cx, cy, w, h = box_to_yolo(box, img_w, img_h)
        yolo_lines.append(f"{BOTTLE_CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    txt_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

    if vis_dir is not None:
        vis = draw_masks(image_rgb, masks, boxes, scores)
        cv2.imwrite(str(vis_dir / (image_path.stem + ".jpg")), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    return len(yolo_lines)


def main():
    image_dir = Path("")  # your folder with images to label
    prompt = "person"
    score_threshold = 0.5  # combined presence × per-object confidence
    to_visualize = True  # if True, save image + masks + bboxes overlay alongside the labels

    output_dir = image_dir.parent / (image_dir.name + "_labels")
    output_dir.mkdir(parents=True, exist_ok=True)

    vis_dir = None
    if to_visualize:
        vis_dir = image_dir.parent / (image_dir.name + "_vis")
        vis_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not image_files:
        sys.exit(f"No images found in {image_dir}")

    print(f"Images : {len(image_files)} found in {image_dir}")
    print(f"Labels : {output_dir}")
    if vis_dir is not None:
        print(f"Vis    : {vis_dir}")
    print(f"Prompt : '{prompt}'")
    print(f"Score threshold: {score_threshold}")

    model = SAM3_model(prompt=prompt, conf_thresh=score_threshold, binarize_masks=False)

    total_detections = 0
    with tqdm(image_files, unit="img") as pbar:
        for image_path in pbar:
            pbar.set_description(image_path.name)
            n = process_image(
                image_path=image_path,
                model=model,
                output_dir=output_dir,
                vis_dir=vis_dir,
            )
            total_detections += n
            pbar.set_postfix(bottles=n, total=total_detections)

    print(
        f"\nDone. {len(image_files)} image(s) → {total_detections} total detections → {output_dir}"
    )


if __name__ == "__main__":
    main()
