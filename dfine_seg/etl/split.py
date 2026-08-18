import json
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
from loguru import logger
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from dfine_seg._config import CONFIG_NAME, config_dir

COCO_SRC = "coco.json"  # single-file COCO annotations, split into train/val(/test).json


def split_indices(n: int, train_split: float, val_split: float, seed: int, shuffle: bool) -> dict:
    """-> {'train': idxs, 'val': idxs[, 'test': idxs]}. Shared by the YOLO and COCO paths."""
    test_split = 1 - train_split - val_split
    if test_split <= 0.001:
        test_split = 0

    indices = np.arange(n)
    train_idxs, temp_idxs = train_test_split(
        indices, test_size=(1 - train_split), random_state=seed, shuffle=shuffle
    )

    if test_split:
        test_idxs, val_idxs = train_test_split(
            temp_idxs,
            test_size=(val_split / (val_split + test_split)),
            random_state=seed,
            shuffle=shuffle,
        )
    else:
        val_idxs = temp_idxs
        test_idxs = []

    splits = {"train": train_idxs, "val": val_idxs}
    if test_split:
        splits["test"] = test_idxs
    return splits


def split(
    data_path: Path,
    train_split: float,
    val_split: float,
    images_path: Path,
    ignore_negatives: bool,
    seed: int,
    shuffle: True,
) -> None:
    img_paths = [x.name for x in images_path.iterdir() if not str(x.name).startswith(".")]

    if not shuffle:
        img_paths.sort()

    if ignore_negatives:
        labels_path = images_path.parent / "labels"
        img_paths = [p for p in img_paths if (labels_path / f"{Path(p).stem}.txt").exists()]

    for split_name, idxs in split_indices(
        len(img_paths), train_split, val_split, seed, shuffle
    ).items():
        with open(data_path / f"{split_name}.csv", "w") as f:
            for idx in idxs:
                f.write(str(img_paths[idx]) + "\n")
        logger.info(f"{split_name}: {len(idxs)}")


def split_coco(
    data_path: Path,
    coco_path: Path,
    train_split: float,
    val_split: float,
    ignore_negatives: bool,
    seed: int,
    shuffle: bool,
) -> None:
    """Split one COCO json into train/val(/test).json - image-level, same ratios as the YOLO path.

    Each output keeps the source's non-image keys (categories, info, licenses) and only the
    annotations whose image_id survived, so every file is a standalone COCO dataset.
    """
    with open(coco_path, "r") as f:
        coco = json.load(f)

    img_to_anns = defaultdict(list)
    for ann in coco.get("annotations", []):
        img_to_anns[ann["image_id"]].append(ann)

    images = list(coco.get("images", []))
    if not shuffle:
        images.sort(key=lambda img: img["file_name"])

    if ignore_negatives:
        images = [img for img in images if img_to_anns.get(img["id"])]

    shared = {k: v for k, v in coco.items() if k not in ("images", "annotations")}

    for split_name, idxs in split_indices(
        len(images), train_split, val_split, seed, shuffle
    ).items():
        imgs = [images[i] for i in idxs]
        anns = [ann for img in imgs for ann in img_to_anns.get(img["id"], [])]
        out_path = data_path / f"{split_name}.json"
        with open(out_path, "w") as f:
            json.dump({**shared, "images": imgs, "annotations": anns}, f)
        logger.info(f"{split_name}: {len(imgs)} images, {len(anns)} annotations")


@hydra.main(version_base=None, config_path=config_dir(), config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    data_path = Path(cfg.train.data_path)

    if str(cfg.task).lower() == "sem_seg" and cfg.train.coco_dataset:
        raise ValueError("task=sem_seg expects PNG masks (labels/), not COCO JSON")

    if cfg.train.coco_dataset:
        coco_path = data_path / COCO_SRC
        if not coco_path.exists():
            raise FileNotFoundError(
                f"train.coco_dataset=True, so `make split` splits {coco_path}, which is missing. "
                "Put the single-file COCO annotations there, or skip `make split` if "
                f"{data_path}/train.json and val.json already exist."
            )
        split_coco(
            data_path,
            coco_path,
            cfg.split.train_split,
            cfg.split.val_split,
            cfg.split.ignore_negatives,
            cfg.train.seed,
            shuffle=cfg.split.shuffle,
        )
    else:
        split(
            data_path,
            cfg.split.train_split,
            cfg.split.val_split,
            data_path / "images",
            cfg.split.ignore_negatives,
            cfg.train.seed,
            shuffle=cfg.split.shuffle,
        )


if __name__ == "__main__":
    main()
