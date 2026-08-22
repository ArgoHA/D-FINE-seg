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


# ---- the meta envelope training writes ---------------------------------------


def meta(**over):
    m = {
        "dfine_seg_version": "0.3.0",
        "model_name": "s",
        "task": "detect",
        "num_classes": 2,
        "in_channels": 3,
        "label_to_name": {0: "cat", 1: "dog"},
        "img_size": [1024, 2048],
        "keep_ratio": True,
    }
    m.update(over)
    return m


def write_enveloped(tmp_path, sd, m, name="model.pt"):
    p = tmp_path / name
    torch.save({"model": sd, "meta": m}, p)
    return p


def test_meta_carries_names_and_preprocessing(tmp_path):
    """The whole point: a checkpoint moved away from its run dir stays self-describing."""
    info = inspect(write_enveloped(tmp_path, fake_sd(num_classes=2), meta()))
    assert info["names"] == {0: "cat", 1: "dog"}
    assert info["img_size"] == (1024, 2048)
    assert info["keep_ratio"] is True


def test_architecture_still_comes_from_the_weights(tmp_path):
    """meta is a claim; the weights are the thing being loaded, so they win."""
    sd = fake_sd(size="m", task="segment", num_classes=7, in_channels=4)
    info = inspect(write_enveloped(tmp_path, sd, meta(model_name="x", task="detect")))
    assert (info["model_name"], info["task"]) == ("m", "segment")
    assert (info["num_classes"], info["in_channels"]) == (7, 4)


def test_meta_model_name_rescues_an_unknown_fingerprint(tmp_path):
    sd = fake_sd()
    sd["encoder.input_proj.0.conv.weight"] = torch.zeros(999, 8, 1, 1)
    assert inspect(write_enveloped(tmp_path, sd, meta(model_name="xl")))["model_name"] == "xl"


def test_meta_wins_over_the_sidecar_config(tmp_path):
    p = write_enveloped(tmp_path, fake_sd(num_classes=2), meta())
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(
            {"train": {"label_to_name": {0: "stale"}, "img_size": [640, 640], "keep_ratio": False}}
        )
    )
    info = inspect(p)
    assert info["names"] == {0: "cat", 1: "dog"}
    assert info["img_size"] == (1024, 2048)
    assert info["keep_ratio"] is True


def test_sidecar_still_fills_in_when_meta_is_absent(tmp_path):
    """Every released checkpoint is a bare state_dict; nothing about them changes."""
    p = tmp_path / "model.pt"
    torch.save({"model": fake_sd(num_classes=2)}, p)  # envelope without a meta block
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"train": {"label_to_name": {0: "cat", 1: "dog"}, "keep_ratio": True}})
    )
    info = inspect(p)
    assert info["names"] == {0: "cat", 1: "dog"} and info["keep_ratio"] is True


def test_meta_survives_weights_only_load(tmp_path):
    """Plain python only - an OmegaConf node here would raise on every load."""
    p = write_enveloped(tmp_path, fake_sd(num_classes=2), meta())
    assert torch.load(p, weights_only=True)["meta"]["label_to_name"] == {0: "cat", 1: "dog"}


def test_writer_and_reader_agree(tmp_path):
    """The format has two ends in two modules; pin them to each other."""
    from omegaconf import OmegaConf

    from dfine_seg.dl.train import ckpt_meta
    from dfine_seg.model.utils import save_checkpoint

    cfg = OmegaConf.create(
        {
            "model_name": "m",
            "task": "sem_seg",
            "train": {
                "label_to_name": {0: "road", 1: "car"},
                "in_channels": 4,
                "img_size": [1024, 2048],
                "keep_ratio": True,
            },
        }
    )
    p = tmp_path / "model.pt"
    save_checkpoint(
        p, fake_sd(size="m", task="sem_seg", num_classes=2, in_channels=4), ckpt_meta(cfg)
    )

    info = inspect(p)
    assert info["model_name"] == "m" and info["task"] == "sem_seg"
    assert info["num_classes"] == 2 and info["in_channels"] == 4
    assert info["names"] == {0: "road", 1: "car"}
    assert info["img_size"] == (1024, 2048) and info["keep_ratio"] is True
