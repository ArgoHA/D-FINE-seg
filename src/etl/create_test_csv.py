from pathlib import Path

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig) -> None:
    data_path = Path(cfg.train.path_to_test_data) / "images"
    img_paths = []
    img_paths.extend([x.name for x in data_path.iterdir() if not str(x.name).startswith(".")])

    with open(data_path.parent / "val.csv", "w") as f:
        for img_path in img_paths:
            f.write(str(img_path) + "\n")


if __name__ == "__main__":
    main()
