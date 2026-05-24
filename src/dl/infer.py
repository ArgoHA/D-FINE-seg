from pathlib import Path
from shutil import rmtree

import cv2
import hydra
import numpy as np
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from src.dl.dataset import read_image_hwc
from src.dl.utils import Visualizer, abs_xyxy_to_norm_xywh, get_latest_experiment_name
from src.infer.byte_track import ByteTrack, Detection
from src.infer.torch_model import Torch_model

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def figure_input_type(folder_path: Path):
    video_types = ["mp4", "avi", "mov", "mkv"]
    # .tif/.tiff intentionally excluded: cv2.imread mangles 4-channel TIFFs
    # (alpha pre-multiplication + photometric-tag swap). Convert to .npy first
    # (see src/etl/preprocess.py for a PIL-based TIFF->JPG path for 3-channel).
    img_types = ["jpg", "png", "jpeg", "npy"]

    for f in folder_path.iterdir():
        if f.suffix[1:].lower() in video_types:
            data_type = "video"
            break
        elif f.suffix[1:].lower() in img_types:
            data_type = "image"
            break
    logger.info(
        f"Inferencing on data type: {data_type}, path: {folder_path}",
    )
    return data_type


def visualize(img, boxes, labels, scores, output_path, img_path, label_to_name, masks=None):
    output_path.mkdir(parents=True, exist_ok=True)
    results = {"boxes": boxes, "labels": labels, "scores": scores}
    if masks is not None:
        results["masks"] = masks
    vis = Visualizer(n_classes=max(label_to_name.keys()) + 1, class_names=label_to_name)
    img = vis.draw(img, results)
    if len(boxes):
        cv2.imwrite((str(f"{output_path / Path(img_path).stem}.jpg")), img)


def save_yolo_annotations(res, output_path, img_path, img_shape):
    output_path.mkdir(parents=True, exist_ok=True)

    if len(res["boxes"]) == 0:
        return

    has_polys = "polys" in res and res["polys"] is not None and len(res["polys"]) > 0

    with open(output_path / f"{Path(img_path).stem}.txt", "a") as f:
        for idx, (class_id, box) in enumerate(zip(res["labels"], res["boxes"])):
            if has_polys:
                # YOLO segmentation format: class_id x1 y1 x2 y2 x3 y3 ...
                poly = res["polys"][idx]
                if len(poly) >= 3:  # Need at least 3 points for a valid polygon
                    norm_coords = []
                    for point in poly:
                        norm_coords.append(f"{point[0]:.6f}")
                        norm_coords.append(f"{point[1]:.6f}")
                    f.write(f"{int(class_id)} {' '.join(norm_coords)}\n")
            else:
                # YOLO detection format: class_id x_center y_center width height
                norm_box = abs_xyxy_to_norm_xywh(box[None], img_shape[0], img_shape[1])[0]
                f.write(
                    f"{int(class_id)} {norm_box[0]:.6f} {norm_box[1]:.6f} {norm_box[2]:.6f} {norm_box[3]:.6f}\n"
                )


def crops(or_img, res, paddings, output_path, output_stem):
    if isinstance(paddings["w"], float):
        paddings["w"] = int(or_img.shape[1] * paddings["w"])
    if isinstance(paddings["h"], float):
        paddings["h"] = int(or_img.shape[0] * paddings["h"])

    for crop_id, box in enumerate(res["boxes"]):
        x1, y1, x2, y2 = map(int, box.tolist())
        crop = or_img[
            max(y1 - paddings["h"], 0) : min(y2 + paddings["h"], or_img.shape[0]),
            max(x1 - paddings["w"], 0) : min(x2 + paddings["w"], or_img.shape[1]),
        ]

        (output_path / "crops").mkdir(parents=True, exist_ok=True)
        cv2.imwrite((str(output_path / "crops" / f"{output_stem}_{crop_id}.jpg")), crop)


def run_images(
    torch_model, folder_path, output_path, label_to_name, to_crop, paddings, conf_thresh
):
    batch = 0
    imag_paths = [img.name for img in folder_path.iterdir() if not str(img).startswith(".")]
    labels = set()
    for img_path in tqdm(imag_paths):
        img = read_image_hwc(folder_path / img_path)
        if img is None:
            logger.warning(f"Skipping unreadable image: {img_path}")
            continue
        or_img = img.copy()
        is_npy = Path(img_path).suffix.lower() == ".npy"
        raw_res = torch_model(img, bgr=not is_npy)

        # Convert torch tensors to numpy for saving/visualization
        res = {
            "boxes": raw_res[batch]["boxes"].cpu().numpy(),
            "labels": raw_res[batch]["labels"].cpu().numpy(),
            "scores": raw_res[batch]["scores"].cpu().numpy(),
        }
        if "masks" in raw_res[0]:
            res["masks"] = raw_res[batch]["masks"].cpu()
            res["polys"] = torch_model.mask2poly(res["masks"], img.shape)

        # visualization / crops only support 3-channel; slice for N>3.
        # cv2 saves in BGR; .npy stacks are RGB(+extras) by convention.
        vis_img = img[:, :, :3] if img.shape[2] > 3 else img
        crop_img = or_img[:, :, :3] if or_img.shape[2] > 3 else or_img
        if is_npy:
            vis_img = np.ascontiguousarray(vis_img[..., ::-1])
            crop_img = np.ascontiguousarray(crop_img[..., ::-1])

        visualize(
            img=vis_img,
            boxes=res["boxes"],
            labels=res["labels"],
            scores=res["scores"],
            output_path=output_path / "images",
            img_path=img_path,
            label_to_name=label_to_name,
            masks=res.get("masks", None),
        )

        for class_id in res["labels"]:
            labels.add(class_id)

        save_yolo_annotations(
            res=res, output_path=output_path / "labels", img_path=img_path, img_shape=img.shape
        )

        if to_crop:
            crops(crop_img, res, paddings, output_path, Path(img_path).stem)

    with open(output_path / "labels.txt", "w") as f:
        for class_id in labels:
            f.write(f"{label_to_name[int(class_id)]}\n")


def run_videos(
    torch_model, folder_path, output_path, label_to_name, to_crop, paddings, conf_thresh
):
    batch = 0
    vid_paths = [vid.name for vid in folder_path.iterdir() if not str(vid.name).startswith(".")]
    labels = set()
    for vid_path in vid_paths:
        vid = cv2.VideoCapture(str(folder_path / vid_path))
        total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        pbar = tqdm(total=total_frames, desc=vid_path, unit="frame")
        success, img = vid.read()
        idx = 0
        while success:
            idx += 1
            raw_res = torch_model(img)

            # Convert torch tensors to numpy for saving/visualization
            res = {
                "boxes": raw_res[batch]["boxes"].cpu().numpy(),
                "labels": raw_res[batch]["labels"].cpu().numpy(),
                "scores": raw_res[batch]["scores"].cpu().numpy(),
            }
            if "masks" in raw_res[0]:
                res["masks"] = raw_res[batch]["masks"].cpu()
                res["polys"] = torch_model.mask2poly(res["masks"], img.shape)

            frame_name = f"{Path(vid_path).stem}_frame_{idx}"
            visualize(
                img=img,
                boxes=res["boxes"],
                labels=res["labels"],
                scores=res["scores"],
                output_path=output_path / "images",
                img_path=frame_name,
                label_to_name=label_to_name,
                masks=res.get("masks", None),
            )

            for class_id in res["labels"]:
                labels.add(class_id)

            save_yolo_annotations(
                res=res,
                output_path=output_path / "labels",
                img_path=frame_name,
                img_shape=img.shape,
            )

            if to_crop:
                crops(img, res, paddings, output_path, frame_name)

            pbar.update(1)
            success, img = vid.read()
        pbar.close()
        vid.release()

    with open(output_path / "labels.txt", "w") as f:
        for class_id in labels:
            f.write(f"{label_to_name[int(class_id)]}\n")


def _run_video_tracked(torch_model, tracker, visualizer, video_path, output_path):
    vid = cv2.VideoCapture(str(video_path))
    if not vid.isOpened():
        logger.warning(f"Could not open {video_path}, skipping")
        return

    fps = vid.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_vid = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    pbar = tqdm(total=total_frames, desc=video_path.name, unit="frame")
    success, frame = vid.read()
    while success:
        raw_res = torch_model(frame)
        res = raw_res[0]

        boxes = res["boxes"].cpu().numpy()
        labels = res["labels"].cpu().numpy()
        scores = res["scores"].cpu().numpy()

        detections = [
            Detection(bbox=tuple(b.tolist()), score=float(s), cls_id=int(c))
            for b, c, s in zip(boxes, labels, scores)
        ]
        tracked = tracker.update(detections, frame_shape=(height, width))

        if tracked:
            tracked_results = {
                "track_ids": np.array([t[0] for t in tracked], dtype=int),
                "labels": np.array([t[1] for t in tracked], dtype=int),
                "boxes": np.array([t[2] for t in tracked], dtype=np.float64),
                "scores": np.array([t[3] for t in tracked], dtype=np.float64),
            }
        else:
            tracked_results = {
                "track_ids": np.zeros(0, dtype=int),
                "labels": np.zeros(0, dtype=int),
                "boxes": np.zeros((0, 4), dtype=np.float64),
                "scores": np.zeros(0, dtype=np.float64),
            }

        out_vid.write(visualizer.draw(frame, tracked_results))
        pbar.update(1)
        success, frame = vid.read()

    pbar.close()
    vid.release()
    out_vid.release()
    logger.info(f"Output video saved: {output_path}")


def run_videos_tracked(torch_model, folder_path, output_path, label_to_name, tracker_cfg):
    video_files = sorted(
        f
        for f in folder_path.iterdir()
        if f.suffix.lower() in VIDEO_EXTS and not f.name.startswith(".")
    )
    if not video_files:
        logger.error(f"No video files found in {folder_path}")
        return

    output_path.mkdir(parents=True, exist_ok=True)
    visualizer = Visualizer(n_classes=max(label_to_name.keys()) + 1, class_names=label_to_name)

    for video_path in video_files:
        # Fresh tracker per video so IDs don't bleed across unrelated clips.
        tracker = ByteTrack(
            track_thresh=tracker_cfg["track_thresh"],
            unmatched_thresh=tracker_cfg["unmatched_thresh"],
            detrack_thresh=tracker_cfg["detrack_thresh"],
            tracking_thresh=tracker_cfg["tracking_thresh"],
            track_buffer=tracker_cfg["track_buffer"],
            max_age=tracker_cfg["max_age"],
            min_hits=tracker_cfg["min_hits"],
            iou_weight=tracker_cfg["iou_weight"],
            drag=tracker_cfg["drag"],
            velocity_alpha=tracker_cfg["velocity_alpha"],
        )
        out_path = output_path / f"{video_path.stem}_tracked.mp4"
        logger.info(f"Processing: {video_path}")
        _run_video_tracked(torch_model, tracker, visualizer, video_path, out_path)


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig):
    cfg.exp = get_latest_experiment_name(cfg.exp, cfg.train.path_to_save)

    to_crop = cfg.infer.to_crop
    paddings = cfg.infer.paddings
    to_track = cfg.infer.get("to_track", True)

    folder_path = Path(str(cfg.train.path_to_test_data))
    data_type = figure_input_type(folder_path)

    # Tracking only applies to videos.
    use_tracking = to_track and data_type == "video"

    if use_tracking:
        # ByteTrack defaults — picked to exercise the two-stage association.
        tracker_cfg = {
            "track_thresh": float(cfg.train.conf_thresh),  # high/low pool split
            "unmatched_thresh": 0.7,  # min score to start a new track
            "detrack_thresh": 0.4,  # hard floor inside the tracker
            "tracking_thresh": 0.8,  # max match cost (iou_weight*(1-IoU)+...)
            "track_buffer": 30,
            "max_age": 0,
            "min_hits": 2,
            "iou_weight": 0.75,
            "drag": 0.85,
            "velocity_alpha": 0.6,
        }
        if "track" in cfg:
            for k, v in dict(cfg.track).items():
                tracker_cfg[k] = v
    else:
        tracker_cfg = None

    torch_model = Torch_model(
        model_name=cfg.model_name,
        model_path=Path(cfg.train.path_to_save) / "model.pt",
        n_outputs=len(cfg.train.label_to_name),
        input_width=cfg.train.img_size[1],
        input_height=cfg.train.img_size[0],
        conf_thresh=cfg.train.conf_thresh,
        rect=cfg.export.dynamic_input,
        enable_mask_head=cfg.task == "segment",
        channels=cfg.train.in_channels,
    )

    if data_type == "video" and cfg.train.in_channels != 3:
        raise ValueError(
            f"Video inference only supports 3-channel input, got in_channels={cfg.train.in_channels}"
        )

    output_path = Path(cfg.train.infer_path)
    if output_path.exists():
        rmtree(output_path)

    if data_type == "image":
        run_images(
            torch_model,
            folder_path,
            output_path,
            label_to_name=cfg.train.label_to_name,
            to_crop=to_crop,
            paddings=paddings,
            conf_thresh=cfg.train.conf_thresh,
        )
    elif data_type == "video":
        if use_tracking:
            run_videos_tracked(
                torch_model,
                folder_path,
                output_path,
                label_to_name=cfg.train.label_to_name,
                tracker_cfg=tracker_cfg,
            )
        else:
            run_videos(
                torch_model,
                folder_path,
                output_path,
                label_to_name=cfg.train.label_to_name,
                to_crop=to_crop,
                paddings=paddings,
                conf_thresh=cfg.train.conf_thresh,
            )


if __name__ == "__main__":
    main()
