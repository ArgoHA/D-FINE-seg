from pathlib import Path

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig) -> None:
    data_path = Path(cfg.train.data_path)
    yolo_data_path = Path("...")

    for split in ["train", "val"]:
        f_paths = []
        f_paths.extend(
            [
                x.name
                for x in (yolo_data_path / "images" / split).iterdir()
                if not str(x.name).startswith(".")
            ]
        )

        with open(data_path / f"{split}.csv", "w") as f:
            for f_path in f_paths:
                f.write(str(f_path) + "\n")


if __name__ == "__main__":
    main()
