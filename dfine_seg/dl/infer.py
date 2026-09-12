from pathlib import Path
from shutil import rmtree

import cv2
import hydra
import numpy as np
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from dfine_seg.config.resolve import CONFIG_NAME, config_dir
from dfine_seg.dl.dataset import read_image_hwc
from dfine_seg.dl.utils import (
    abs_xyxy_to_norm_xywh,
    get_latest_experiment_name,
)
from dfine_seg.viz import Visualizer, overlay_sem_seg, sem_seg_palette
from dfine_seg.infer.byte_track import ByteTrack, Detection
from dfine_seg.infer.torch_model import TorchModel

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".npy"}


def _files(folder_path, extensions):
    return sorted(
        path
        for path in folder_path.iterdir()
        if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in extensions
    )


def figure_input_type(folder_path: Path):
    kinds = {"image" for _ in _files(folder_path, IMAGE_EXTS)} | {
        "video" for _ in _files(folder_path, VIDEO_EXTS)
    }
    if len(kinds) != 1:
        found = "mixed image and video files" if kinds else "no supported files"
        raise ValueError(f"Expected one input type in {folder_path}, found {found}")
    data_type = kinds.pop()
    logger.info(f"Inferencing on data type: {data_type}, path: {folder_path}")
    return data_type


def _host_result(torch_model, raw_result, image_shape):
    result = {key: raw_result[key].cpu().numpy() for key in ("boxes", "labels", "scores")}
    if "masks" in raw_result:
        result["masks"] = raw_result["masks"].cpu()
        result["polys"] = torch_model.mask2poly(result["masks"], image_shape)
    return result


def _display_image(image, is_npy=False):
    image = image[..., :3]
    return np.ascontiguousarray(image[..., ::-1]) if is_npy else image


def visualize(visualizer, img, result, output_path, stem):
    if not len(result["boxes"]):
        return
    output_path.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path / f"{stem}.jpg"), visualizer.draw(img, result))


def _write_class_names(output_path, labels, label_to_name):
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "labels.txt", "w") as f:
        for class_id in sorted(labels):
            f.write(f"{label_to_name[int(class_id)]}\n")


def save_yolo_annotations(res, output_path, img_path, img_shape):
    output_path.mkdir(parents=True, exist_ok=True)

    if len(res["boxes"]) == 0:
        return

    has_polys = "polys" in res and res["polys"] is not None and len(res["polys"]) > 0

    with open(output_path / f"{Path(img_path).stem}.txt", "w") as f:
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
    pad_w = (
        int(or_img.shape[1] * paddings["w"]) if isinstance(paddings["w"], float) else paddings["w"]
    )
    pad_h = (
        int(or_img.shape[0] * paddings["h"]) if isinstance(paddings["h"], float) else paddings["h"]
    )

    for crop_id, box in enumerate(res["boxes"]):
        x1, y1, x2, y2 = map(int, box.tolist())
        crop = or_img[
            max(y1 - pad_h, 0) : min(y2 + pad_h, or_img.shape[0]),
            max(x1 - pad_w, 0) : min(x2 + pad_w, or_img.shape[1]),
        ]

        (output_path / "crops").mkdir(parents=True, exist_ok=True)
        cv2.imwrite((str(output_path / "crops" / f"{output_stem}_{crop_id}.jpg")), crop)


def run_images(torch_model, folder_path, output_path, label_to_name, to_crop, paddings):
    visualizer = Visualizer(n_classes=max(label_to_name) + 1, class_names=label_to_name)
    labels = set()
    for img_path in tqdm(_files(folder_path, IMAGE_EXTS)):
        img = read_image_hwc(img_path)
        if img is None:
            logger.warning(f"Skipping unreadable image: {img_path.name}")
            continue
        is_npy = img_path.suffix.lower() == ".npy"
        res = _host_result(torch_model, torch_model(img, bgr=not is_npy)[0], img.shape)
        display_img = _display_image(img, is_npy)
        visualize(visualizer, display_img, res, output_path / "images", img_path.stem)
        labels.update(res["labels"].tolist())

        save_yolo_annotations(
            res=res, output_path=output_path / "labels", img_path=img_path, img_shape=img.shape
        )

        if to_crop:
            crops(display_img, res, paddings, output_path, img_path.stem)

    _write_class_names(output_path, labels, label_to_name)


def run_images_sem_seg(torch_model, folder_path, output_path, label_to_name):
    """Overlay + raw label-map PNG per image; crops/YOLO txt are box-based -> skipped."""
    palette = sem_seg_palette(len(label_to_name))
    (output_path / "images").mkdir(parents=True, exist_ok=True)
    (output_path / "labels").mkdir(parents=True, exist_ok=True)
    labels = set()
    for img_path in tqdm(_files(folder_path, IMAGE_EXTS)):
        img = read_image_hwc(img_path)
        if img is None:
            logger.warning(f"Skipping unreadable image: {img_path.name}")
            continue
        is_npy = img_path.suffix.lower() == ".npy"
        label_map = torch_model(img, bgr=not is_npy)[0]["sem_seg"].cpu().numpy()

        vis_img = _display_image(img, is_npy)
        cv2.imwrite(
            str(output_path / "images" / f"{img_path.stem}.jpg"),
            overlay_sem_seg(vis_img, label_map, palette),
        )
        cv2.imwrite(str(output_path / "labels" / f"{img_path.stem}.png"), label_map)
        labels.update(np.unique(label_map).tolist())

    _write_class_names(output_path, labels, label_to_name)


def run_videos_sem_seg(torch_model, folder_path, output_path, label_to_name):
    """Per-frame overlay written to <stem>_sem_seg.mp4; tracking is box-based -> skipped."""
    palette = sem_seg_palette(len(label_to_name))
    output_path.mkdir(parents=True, exist_ok=True)
    for video_path in _files(folder_path, VIDEO_EXTS):
        vid = cv2.VideoCapture(str(video_path))
        if not vid.isOpened():
            logger.warning(f"Could not open {video_path}, skipping")
            continue
        fps = vid.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = output_path / f"{video_path.stem}_sem_seg.mp4"
        out_vid = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )

        pbar = tqdm(total=total_frames, desc=video_path.name, unit="frame")
        success, frame = vid.read()
        while success:
            label_map = torch_model(frame)[0]["sem_seg"].cpu().numpy()
            out_vid.write(overlay_sem_seg(frame, label_map, palette))
            pbar.update(1)
            success, frame = vid.read()
        pbar.close()
        vid.release()
        out_vid.release()
        logger.info(f"Output video saved: {out_path}")


def run_videos(torch_model, folder_path, output_path, label_to_name, to_crop, paddings):
    visualizer = Visualizer(n_classes=max(label_to_name) + 1, class_names=label_to_name)
    labels = set()
    for video_path in _files(folder_path, VIDEO_EXTS):
        vid = cv2.VideoCapture(str(video_path))
        if not vid.isOpened():
            logger.warning(f"Could not open {video_path}, skipping")
            continue
        total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        pbar = tqdm(total=total_frames, desc=video_path.name, unit="frame")
        success, img = vid.read()
        idx = 0
        while success:
            idx += 1
            res = _host_result(torch_model, torch_model(img)[0], img.shape)
            frame_name = f"{video_path.stem}_frame_{idx}"
            visualize(visualizer, img, res, output_path / "images", frame_name)
            labels.update(res["labels"].tolist())

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

    _write_class_names(output_path, labels, label_to_name)


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
        masks = res["masks"].cpu().numpy() if "masks" in res else None

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
            if masks is not None:
                # tracker emits the source detection index, so masks follow the boxes
                tracked_results["masks"] = masks[[t[4] for t in tracked]]
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
    video_files = _files(folder_path, VIDEO_EXTS)
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


@hydra.main(version_base=None, config_path=config_dir(), config_name=CONFIG_NAME)
def main(cfg: DictConfig):
    cfg.exp = get_latest_experiment_name(cfg.exp, cfg.train.path_to_save)

    to_crop = cfg.infer.to_crop
    paddings = cfg.infer.paddings
    to_track = cfg.infer.get("to_track", True)

    folder_path = Path(str(cfg.train.path_to_test_data))
    data_type = figure_input_type(folder_path)

    # Tracking only applies to videos (and is box-based, so never for sem_seg).
    use_tracking = to_track and data_type == "video" and cfg.task != "sem_seg"

    if use_tracking:
        # ByteTrack defaults - picked to exercise the two-stage association.
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

    torch_model = TorchModel(
        model_name=cfg.model_name,
        model_path=Path(cfg.train.path_to_save) / "model.pt",
        n_outputs=len(cfg.train.label_to_name),
        input_width=cfg.train.img_size[1],
        input_height=cfg.train.img_size[0],
        conf_thresh=cfg.train.conf_thresh,
        keep_ratio=cfg.train.keep_ratio,
        rect=cfg.export.dynamic_input,
        channels=cfg.train.in_channels,
        task=cfg.task,
    )

    if data_type == "video" and cfg.train.in_channels != 3:
        raise ValueError(
            f"Video inference only supports 3-channel input, got in_channels={cfg.train.in_channels}"
        )

    output_path = Path(cfg.train.infer_path)
    if output_path.exists():
        rmtree(output_path)

    if data_type == "image":
        if cfg.task == "sem_seg":
            run_images_sem_seg(
                torch_model, folder_path, output_path, label_to_name=cfg.train.label_to_name
            )
        else:
            run_images(
                torch_model,
                folder_path,
                output_path,
                label_to_name=cfg.train.label_to_name,
                to_crop=to_crop,
                paddings=paddings,
            )
    elif data_type == "video":
        if cfg.task == "sem_seg":
            run_videos_sem_seg(
                torch_model, folder_path, output_path, label_to_name=cfg.train.label_to_name
            )
        elif use_tracking:
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
            )


if __name__ == "__main__":
    main()
