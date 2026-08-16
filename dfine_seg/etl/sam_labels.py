#!/usr/bin/env python
"""Pre-annotate a folder of images with a local SAM3 and write COCO or YOLO labels.

Runs the local SAM3 wrapper ([dfine_seg/infer/sam3_model.py](dfine_seg/infer/sam3_model.py), HF
weights, GPU when available) over every image, turns the returned masks into polygons
and writes one of four dataset shapes:

    --format coco --task segment   `coco.json`; `segmentation` carries one polygon per
                                   island of an instance
    --format coco --task detect    `coco.json`; `bbox` only
    --format yolo --task segment   `<stem>.txt` per image, `cls x1 y1 x2 y2 …`
                                   normalized, one line per instance
    --format yolo --task detect    `<stem>.txt` per image, `cls cx cy w h` normalized

`detect` labels the boxes SAM3's box head predicts; `segment` measures its boxes off the
polygons it writes, so every box contains its own segmentation.

Usage:

    python -m dfine_seg.etl.sam_labels ~/data/frames --prompt person --format coco --task segment

Output lands in `~/data/frames_labels` unless `--out` says otherwise; the COCO file is
named `coco.json` so `make split` can consume it directly. Repeat `--prompt` for a
multi-class dataset: COCO category ids and YOLO class indices follow the order the
prompts are given in, and the YOLO output records the names in `labels.txt`.

Needs `transformers`, which the project venv does not carry — run it with an interpreter
that has it. `facebook/sam3` is gated, so without a token that has access the processor's
`chat_template.json` probe 401s; prefix `HF_HUB_OFFLINE=1` (or pass the cached snapshot
directory as `--model`) to load a snapshot that is already on disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from dfine_seg.infer.sam3_model import SAM3Model

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})
COCO_NAME = "coco.json"  # what dfine_seg.etl.split expects as its single-file COCO source

MASK_COLORS = ((0, 200, 255), (255, 100, 0), (0, 255, 100), (200, 0, 255), (255, 255, 0))
MASK_ALPHA = 0.45


@dataclass(frozen=True)
class Options:
    prompts: list[str]
    task: str
    format: str
    min_island_px: int
    polygon_epsilon: float


@dataclass
class Instance:
    category_index: int
    score: float
    bbox: tuple[float, float, float, float]  # x, y, w, h
    area: int
    polygons: list[list[float]] = field(default_factory=list)


@dataclass
class ImageRecord:
    file_name: str
    width: int
    height: int
    instances: list[Instance]


def _iter_images(root: Path, recursive: bool) -> list[Path]:
    paths = root.rglob("*") if recursive else root.glob("*")
    return sorted(p for p in paths if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def mask_to_polygons(
    mask: np.ndarray, min_island_px: int, epsilon_ratio: float
) -> list[list[float]]:
    """Trace a binary mask into one simplified `[x, y, …]` ring per island."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < min_island_px:
            continue
        epsilon = max(1.0, epsilon_ratio * cv2.arcLength(contour, True))
        points = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(points) >= 3:
            polygons.append([float(v) for point in points for v in point])
    return polygons


def merge_islands(polygons: list[list[float]]) -> list[float]:
    """Splice islands into the single ring YOLO's one-line-per-instance format needs.

    Each island is spliced into the largest ring at their closest pair of points and the
    seam is walked back, so the bridges have zero width and the filled ring covers the
    same pixels as the separate islands did.
    """
    rings = [np.asarray(polygon, np.float32).reshape(-1, 2) for polygon in polygons]
    rings.sort(key=cv2.contourArea, reverse=True)
    merged = rings[0]
    for ring in rings[1:]:
        distances = np.linalg.norm(merged[:, None, :] - ring[None, :, :], axis=2)
        into, from_ = np.unravel_index(int(distances.argmin()), distances.shape)
        bridged = np.concatenate([ring[from_:], ring[:from_], ring[from_ : from_ + 1]])
        merged = np.concatenate([merged[: into + 1], bridged, merged[into:]])
    return [float(v) for point in merged for v in point]


def _polygon_bbox_area(
    polygons: list[list[float]],
) -> tuple[tuple[float, float, float, float], int]:
    """Measure the rings we emit, rasterizing only their own bounding box."""
    rings = [np.asarray(polygon, np.int32).reshape(-1, 2) for polygon in polygons]
    corner = np.concatenate(rings).min(axis=0)
    extent = np.concatenate(rings).max(axis=0) - corner
    canvas = np.zeros((extent[1] + 1, extent[0] + 1), np.uint8)
    cv2.fillPoly(canvas, [ring - corner for ring in rings], 1)
    bbox = (float(corner[0]), float(corner[1]), float(extent[0]) + 1.0, float(extent[1]) + 1.0)
    return bbox, int(canvas.sum())


def build_instances(
    res: dict, category_index: int, width: int, height: int, options: Options
) -> list[Instance]:
    """Convert one prompt's SAM3 output into instances in original-image pixels.

    `detect` takes SAM3's own box head, clamped to the image. `segment` measures the rings
    it emits instead, because a box that doesn't contain its polygons is mask area the
    training loader flags as unreachable at inference.
    """
    scores = res["scores"].numpy()
    if options.task == "detect":
        boxes = res["boxes"].numpy().clip((0, 0, 0, 0), (width, height, width, height))
        return [
            Instance(
                category_index=category_index,
                score=float(score),
                bbox=(x1, y1, x2 - x1, y2 - y1),
                area=int(round((x2 - x1) * (y2 - y1))),
            )
            for (x1, y1, x2, y2), score in zip(boxes, scores)
            if x2 > x1 and y2 > y1
        ]

    instances = []
    for mask, score in zip(res["masks"].numpy(), scores):
        mask = np.ascontiguousarray(mask.reshape(mask.shape[-2:]), np.uint8)
        polygons = mask_to_polygons(mask, options.min_island_px, options.polygon_epsilon)
        if not polygons:
            continue
        bbox, area = _polygon_bbox_area(polygons)
        instances.append(
            Instance(
                category_index=category_index,
                score=float(score),
                bbox=bbox,
                area=area,
                polygons=polygons,
            )
        )
    return instances


def annotate_image(model: SAM3Model, path: Path, name: str, options: Options) -> ImageRecord:
    """Segment one image against every prompt and collect the instances found.

    SAM3 takes one text prompt per forward, so a multi-class run re-encodes the image
    once per prompt.
    """
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"unreadable image: {path}")
    height, width = image.shape[:2]

    instances: list[Instance] = []
    for category_index, prompt in enumerate(options.prompts):
        res = model(image, prompt=prompt)[0]
        instances.extend(build_instances(res, category_index, width, height, options))
    return ImageRecord(file_name=name, width=width, height=height, instances=instances)


def to_coco(records: list[ImageRecord], options: Options) -> dict:
    """Assemble a COCO document; `segmentation` is present only for `--task segment`."""
    images, annotations = [], []
    for image_id, record in enumerate(records, start=1):
        images.append(
            {
                "id": image_id,
                "file_name": record.file_name,
                "width": record.width,
                "height": record.height,
            }
        )
        for instance in record.instances:
            annotation = {
                "id": len(annotations) + 1,
                "image_id": image_id,
                "category_id": instance.category_index + 1,
                "bbox": [round(v, 2) for v in instance.bbox],
                "area": instance.area,
                "iscrowd": 0,
                "score": round(instance.score, 4),
            }
            if options.task == "segment":
                annotation["segmentation"] = [
                    [round(v, 2) for v in polygon] for polygon in instance.polygons
                ]
            annotations.append(annotation)
    return {
        "info": {
            "description": f"SAM3 {options.task} pre-annotations for {', '.join(options.prompts)}"
        },
        "images": images,
        "categories": [
            {"id": index + 1, "name": prompt, "supercategory": prompt}
            for index, prompt in enumerate(options.prompts)
        ],
        "annotations": annotations,
    }


def to_yolo_lines(record: ImageRecord, options: Options) -> list[str]:
    """Render one image's instances as YOLO label lines, one line per instance."""
    scale = np.array([record.width, record.height], np.float32)
    lines = []
    for instance in record.instances:
        if options.task == "segment":
            points = np.asarray(merge_islands(instance.polygons), np.float32).reshape(-1, 2)
        else:
            x, y, w, h = instance.bbox
            points = np.array([[x + w / 2, y + h / 2], [w, h]], np.float32)
        coords = np.clip(points / scale, 0.0, 1.0).ravel()
        lines.append(f"{instance.category_index} " + " ".join(f"{v:.6f}" for v in coords))
    return lines


def draw_record(image: np.ndarray, record: ImageRecord, options: Options) -> np.ndarray:
    """Overlay the geometry that gets written, so the vis validates the labels."""
    vis = image.copy()
    for instance in record.instances:
        color = MASK_COLORS[instance.category_index % len(MASK_COLORS)]
        if instance.polygons:
            # exactly the rings the writer emits: islands for COCO, one spliced ring for YOLO
            polygons = (
                [merge_islands(instance.polygons)]
                if options.format == "yolo"
                else instance.polygons
            )
            overlay = vis.copy()
            cv2.fillPoly(overlay, [np.asarray(p, np.int32).reshape(-1, 2) for p in polygons], color)
            cv2.addWeighted(overlay, MASK_ALPHA, vis, 1 - MASK_ALPHA, 0, vis)
        x, y, w, h = (int(round(v)) for v in instance.bbox)
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            vis,
            f"{options.prompts[instance.category_index]} {instance.score:.2f}",
            (x, max(y - 6, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return vis


def default_out(images: Path) -> Path:
    """Name the output folder after the image folder: `frames` -> `frames_labels`."""
    resolved = images.expanduser().resolve()
    return resolved.parent / f"{resolved.name}_labels"


def write_output(
    out: Path,
    records: list[ImageRecord],
    options: Options,
    retained: frozenset[Path] = frozenset(),
) -> Path:
    """Write the dataset and return the path to point a training loader at.

    `retained` names label files for images the run skipped rather than annotated, which
    the stale sweep below must leave alone.
    """
    out.mkdir(parents=True, exist_ok=True)
    if options.format == "coco":
        target = out / COCO_NAME
        with target.open("w") as handle:
            json.dump(to_coco(records, options), handle)
        return target

    class_list = out / "labels.txt"
    labels = {out / Path(record.file_name).with_suffix(".txt"): record for record in records}
    # A YOLO loader pairs an image with any `.txt` of the same stem, so a label an earlier
    # run left behind would be trained on as if this run had produced it.
    stale = [
        path
        for path in out.rglob("*.txt")
        if path != class_list and path not in labels and path not in retained
    ]
    for path in stale:
        path.unlink()
    if stale:
        print(f"removed {len(stale)} stale label file(s) from {out}")

    class_list.write_text("\n".join(options.prompts) + "\n")
    for label, record in labels.items():
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("".join(f"{line}\n" for line in to_yolo_lines(record, options)))
    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("images", type=Path, help="folder of images to annotate")
    parser.add_argument(
        "--prompt",
        dest="prompts",
        action="append",
        required=True,
        metavar="TEXT",
        help="SAM3 text prompt; repeat for a multi-class dataset (order sets class ids)",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        help="output folder (default: the image folder's name plus `_labels`, alongside it)",
    )
    parser.add_argument("--format", choices=("coco", "yolo"), default="coco")
    parser.add_argument(
        "--task",
        choices=("segment", "detect"),
        default="segment",
        help="segment writes polygons, detect writes bboxes only",
    )
    parser.add_argument("--recursive", action="store_true", help="descend into subfolders")
    parser.add_argument("--limit", type=int, help="annotate only the first N images")
    parser.add_argument("--visualize", action="store_true", help="also write overlays to `vis/`")
    parser.add_argument("--model", default="facebook/sam3", help="HF id or local weights path")
    parser.add_argument(
        "--conf", type=float, default=0.5, help="presence x per-object confidence threshold"
    )
    parser.add_argument("--mask-thresh", type=float, default=0.5, help="mask binarization cutoff")
    parser.add_argument(
        "--min-island-px",
        type=int,
        default=40,
        help="segment only: drop islands smaller than this (a whole object that small goes too)",
    )
    parser.add_argument(
        "--polygon-epsilon",
        type=float,
        default=0.002,
        help="Douglas-Peucker tolerance as a fraction of ring perimeter",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.images = args.images.expanduser()
    scanned = _iter_images(args.images, args.recursive)
    paths = scanned[: args.limit] if args.limit else scanned
    if not paths:
        print(f"no images under {args.images}", file=sys.stderr)
        return 1

    options = Options(
        prompts=[prompt.strip() for prompt in args.prompts],
        task=args.task,
        format=args.format,
        min_island_px=args.min_island_px,
        polygon_epsilon=args.polygon_epsilon,
    )
    out = args.out or default_out(args.images)
    vis_dir = out / "vis" if args.visualize else None
    print(f"{len(paths)} image(s) -> {out} [{', '.join(options.prompts)}]")

    model = SAM3Model(model_path=args.model, conf_thresh=args.conf, mask_threshold=args.mask_thresh)

    records: list[ImageRecord] = []
    failures: list[tuple[Path, Exception]] = []
    total = 0
    with tqdm(paths, unit="img") as pbar:
        for path in pbar:
            name = path.relative_to(args.images).as_posix()
            pbar.set_description(name)
            try:
                record = annotate_image(model, path, name, options)
            except Exception as exc:
                failures.append((path, exc))
                continue
            records.append(record)
            total += len(record.instances)
            pbar.set_postfix(found=len(record.instances), total=total)
            if vis_dir is not None:
                vis_path = (vis_dir / name).with_suffix(".jpg")
                vis_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(vis_path), draw_record(cv2.imread(str(path)), record, options))

    skipped = set(scanned) - set(paths)
    retained = frozenset(
        out / Path(path.relative_to(args.images).as_posix()).with_suffix(".txt") for path in skipped
    )
    target = write_output(out, records, options, retained)
    print(f"wrote {total} instance(s) across {len(records)} image(s) -> {target}")
    for path, exc in failures:
        print(f"FAILED {path}: {exc!r}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
