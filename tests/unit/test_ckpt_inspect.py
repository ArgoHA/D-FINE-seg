"""Checkpoint introspection: `TorchModel(path)` needs no other arguments.

Checkpoints are bare state_dicts, so size/task/num_classes come from key structure.
Fingerprints here are the ones measured on the 14 released checkpoints; synthetic
state_dicts keep this fast and weight-free.
"""

from pathlib import Path

import pytest
import torch
import yaml

from dfine_seg.api.ckpt import _FINGERPRINT, inspect, sibling_config

HIDDEN = {"n": 128, "s": 256, "m": 256, "l": 256, "x": 384}
N_BACKBONE = {"n": 312, "s": 312, "m": 442, "l": 400, "x": 650}


def fake_sd(size="s", task="detect", num_classes=80, in_channels=3):
    # -1: the stem key below is itself a backbone.* key and counts toward the fingerprint.
    sd = {f"backbone.k{i}": torch.zeros(1) for i in range(N_BACKBONE[size] - 1)}
    sd["backbone.stem.stem1.conv.weight"] = torch.zeros(16, in_channels, 3, 3)
    sd["encoder.input_proj.0.conv.weight"] = torch.zeros(HIDDEN[size], 8, 1, 1)
    if task == "sem_seg":
        sd["decoder.classifier.weight"] = torch.zeros(num_classes, 128, 1, 1)
    else:
        sd["decoder.enc_score_head.weight"] = torch.zeros(num_classes, HIDDEN[size])
        if task == "segment":
            sd["decoder.mask_decoder.lateral.0.weight"] = torch.zeros(1)
    return sd


def write(tmp_path, sd, name="model.pt"):
    p = tmp_path / name
    torch.save(sd, p)
    return p


def test_fingerprint_table_is_unique():
    assert len(set(_FINGERPRINT.values())) == len(_FINGERPRINT) == 5


def test_every_size_round_trips(tmp_path):
    for size in ("n", "s", "m", "l", "x"):
        info = inspect(write(tmp_path, fake_sd(size=size), f"{size}.pt"))
        assert info["model_name"] == size, size


def test_task_detection(tmp_path):
    for task in ("detect", "segment", "sem_seg"):
        info = inspect(write(tmp_path, fake_sd(task=task), f"{task}.pt"))
        assert info["task"] == task


def test_num_classes_from_head(tmp_path):
    assert inspect(write(tmp_path, fake_sd(num_classes=7)))["num_classes"] == 7
    p = write(tmp_path, fake_sd(task="sem_seg", num_classes=19), "sem.pt")
    assert inspect(p)["num_classes"] == 19


def test_in_channels_from_stem(tmp_path):
    assert inspect(write(tmp_path, fake_sd(in_channels=4)))["in_channels"] == 4


def test_unknown_architecture_raises(tmp_path):
    sd = fake_sd()
    sd["encoder.input_proj.0.conv.weight"] = torch.zeros(999, 8, 1, 1)
    p = write(tmp_path, sd, "weird.pt")
    try:
        inspect(p)
    except ValueError as e:
        assert "model_name" in str(e)
    else:
        raise AssertionError("expected ValueError for an unknown fingerprint")


def test_names_from_sibling_config(tmp_path):
    p = write(tmp_path, fake_sd(num_classes=2))
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"train": {"label_to_name": {0: "cat", 1: "dog"}}})
    )
    assert inspect(p)["names"] == {0: "cat", 1: "dog"}


def test_missing_sibling_config_is_not_fatal(tmp_path):
    assert inspect(write(tmp_path, fake_sd()))["names"] is None
    assert sibling_config(tmp_path / "nope.pt") == {}


def test_malformed_sibling_config_is_ignored(tmp_path):
    p = write(tmp_path, fake_sd())
    (tmp_path / "config.yaml").write_text("{{{ not yaml")
    assert inspect(p)["names"] is None


@pytest.mark.slow
@pytest.mark.parametrize("size", ["n", "s", "m", "l", "x"])
@pytest.mark.parametrize("task", ["detect", "segment"])
def test_fingerprints_match_released_checkpoints(size, task):
    """The synthetic fingerprints above are only valid if the real weights agree."""
    stem = f"dfine_seg_{size}_coco" if task == "segment" else f"dfine_{size}_coco"
    path = Path(__file__).resolve().parents[2] / "pretrained" / f"{stem}.pt"
    if not path.is_file():
        pytest.skip(f"{path.name} not downloaded")
    info = inspect(path)
    assert info["model_name"] == size
    assert info["task"] == task
    assert info["num_classes"] == 80
    assert info["in_channels"] == 3


# ---- preprocessing recovered from the sidecar config -------------------------


def test_img_size_and_keep_ratio_from_sibling_config(tmp_path):
    """Not in the weights, and getting them wrong is silent — so the config is read."""
    p = write(tmp_path, fake_sd(num_classes=19, task="sem_seg"))
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"train": {"img_size": [1024, 2048], "keep_ratio": True}})
    )
    info = inspect(p)
    assert info["img_size"] == (1024, 2048)
    assert info["keep_ratio"] is True


def test_preprocessing_is_none_without_a_config(tmp_path):
    info = inspect(write(tmp_path, fake_sd()))
    assert info["img_size"] is None and info["keep_ratio"] is None
