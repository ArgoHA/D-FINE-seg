import yaml
from loguru import logger

with open("config.yaml", "r") as cfg:
    try:
        cfg = yaml.safe_load(cfg)
    except yaml.YAMLError as exc:
        logger.error(exc)

img_size = cfg["train"]["img_size"]

label_to_name_mapping = cfg["train"]["label_to_name"]
name_to_label_mapping = {v: k for k, v in label_to_name_mapping.items()}
class_names = list(name_to_label_mapping.keys())
num_labels = len(class_names)

img_norms = ([0, 0, 0], [1, 1, 1])
