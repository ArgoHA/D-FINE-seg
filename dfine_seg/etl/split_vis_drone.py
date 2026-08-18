from pathlib import Path

import hydra
from omegaconf import DictConfig

from dfine_seg._config import CONFIG_NAME, config_dir


@hydra.main(version_base=None, config_path=config_dir(), config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    data_path = Path(cfg.train.data_path)
    # Hydra `+key=` additions, so the source dir isn't baked in (it used to ship in the wheel):
    #   python -m dfine_seg.etl.split_vis_drone +yolo_data_path=/abs/VisDrone2019-DET-test-dev
    if not cfg.get("yolo_data_path"):
        raise ValueError(
            "pass the source dataset dir: `+yolo_data_path=/abs/path` "
            "(and optionally `+split_name=test`)"
        )
    yolo_data_path = Path(cfg.yolo_data_path)
    split = cfg.get("split_name", "test")

    f_paths = []
    f_paths.extend(
        [x.name for x in (yolo_data_path / "images").iterdir() if not str(x.name).startswith(".")]
    )

    with open(data_path / f"{split}.csv", "w") as f:
        for f_path in f_paths:
            f.write(str(f_path) + "\n")


if __name__ == "__main__":
    main()
