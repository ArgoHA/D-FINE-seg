"""`dfine` CLI: init writes a usable config, and commands refuse to run without one."""

import sys

import pytest
import yaml

from dfine_seg import cli


def run(monkeypatch, *argv) -> int:
    monkeypatch.setattr(sys, "argv", ["dfine", *argv])
    return cli.main()


@pytest.fixture
def clean_cwd(tmp_path, monkeypatch):
    """cwd with no config.yaml and no repo root on the search path."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "find_config", lambda: None)
    return tmp_path


def test_help_lists_every_command(monkeypatch, capsys):
    assert run(monkeypatch) == 0
    out = capsys.readouterr().out
    for command in [*cli.COMMANDS, "init", "predict", "demo", "version"]:
        assert command in out


def test_unknown_command_exits_nonzero(monkeypatch, capsys):
    assert run(monkeypatch, "trian") == 2
    assert "unknown command" in capsys.readouterr().err


def test_version(monkeypatch, capsys):
    from dfine_seg import __version__

    assert run(monkeypatch, "version") == 0
    assert capsys.readouterr().out.strip() == __version__


def test_init_writes_loadable_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert run(monkeypatch, "init") == 0
    cfg = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert cfg["task"] == "detect" and cfg["model_name"] == "s"
    assert cfg["train"]["root"] == "."


def test_init_refuses_overwrite_without_force(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("keep: me\n")
    assert run(monkeypatch, "init") == 1
    assert "--force" in capsys.readouterr().err
    assert (tmp_path / "config.yaml").read_text() == "keep: me\n"


def test_init_force_overwrites(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("keep: me\n")
    assert run(monkeypatch, "init", "--force") == 0
    assert "keep: me" not in (tmp_path / "config.yaml").read_text()


@pytest.mark.parametrize("task", ["segment", "sem_seg"])
def test_init_task_flag(tmp_path, monkeypatch, task):
    monkeypatch.chdir(tmp_path)
    assert run(monkeypatch, "init", "--task", task) == 0
    assert yaml.safe_load((tmp_path / "config.yaml").read_text())["task"] == task


@pytest.mark.parametrize("task", ["segment", "sem_seg"])
def test_init_mask_tasks_point_at_the_seg_checkpoint(tmp_path, monkeypatch, task):
    """Both train a MaskDecoder; the detection default leaves it at random init."""
    monkeypatch.chdir(tmp_path)
    assert run(monkeypatch, "init", "--task", task) == 0
    cfg = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert cfg["train"]["pretrained_model_path"] == "pretrained/dfine_seg_${model_name}_coco.pt"


def test_init_detect_keeps_the_detection_checkpoint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert run(monkeypatch, "init", "--task", "detect") == 0
    cfg = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert "dfine_seg_" not in cfg["train"]["pretrained_model_path"]


def test_set_key_keeps_the_inline_comment(tmp_path):
    assert cli._set_key("task: detect # a | b\n", "task", "sem_seg") == "task: sem_seg # a | b\n"


def test_set_key_raises_when_the_template_moves(tmp_path):
    """A substring replace would silently no-op instead; this must be loud."""
    with pytest.raises(KeyError):
        cli._set_key("other: 1\n", "task", "segment")


def test_init_model_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert run(monkeypatch, "init", "--model", "x") == 0
    assert yaml.safe_load((tmp_path / "config.yaml").read_text())["model_name"] == "x"


def test_init_writes_into_a_target_dir(tmp_path, monkeypatch):
    """-d takes a directory; the filename is always config.yaml so find_config() sees it."""
    monkeypatch.chdir(tmp_path)
    assert run(monkeypatch, "init", "-d", "proj") == 0
    assert (tmp_path / "proj" / "config.yaml").is_file()


def test_init_creates_missing_target_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert run(monkeypatch, "init", "-d", "a/b/c") == 0
    assert (tmp_path / "a/b/c/config.yaml").is_file()


def test_init_cannot_produce_an_undiscoverable_name(tmp_path, monkeypatch):
    """There is no way to ask for a filename other than config.yaml."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        run(monkeypatch, "init", "-o", "other.yaml")


def test_commands_refuse_to_run_without_config(clean_cwd, monkeypatch, capsys):
    for command in cli.COMMANDS:
        assert run(monkeypatch, command) == 1, command
        err = capsys.readouterr().err
        assert "dfine init" in err and cli.ENV_VAR in err


def test_ddp_launch_skipped_when_disabled(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"train": {"ddp": {"enabled": False}}}))
    monkeypatch.setattr(cli, "find_config", lambda: cfg)
    assert cli._ddp_launch([]) is None  # None = "not handled, fall through to in-process"


def test_ddp_launch_returns_the_childs_exit_code(tmp_path, monkeypatch):
    """A signal-killed child returns -signum; that must not read as "not handled"."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"train": {"ddp": {"enabled": True, "n_gpus": 2}}}))
    monkeypatch.setattr(cli, "find_config", lambda: cfg)
    monkeypatch.setattr(cli.shutil, "which", lambda _: "/usr/bin/torchrun")
    monkeypatch.setattr(cli.subprocess, "call", lambda cmd: -1)  # SIGHUP
    assert cli._ddp_launch([]) == -1


@pytest.mark.parametrize(
    "yaml_enabled, override, expect_torchrun",
    [
        (False, "train.ddp.enabled=True", True),
        (True, "train.ddp.enabled=False", False),
    ],
)
def test_ddp_launch_honours_cli_overrides(
    tmp_path, monkeypatch, yaml_enabled, override, expect_torchrun
):
    """The same overrides go on to Hydra, so they have to win over the yaml here too."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"train": {"ddp": {"enabled": yaml_enabled, "n_gpus": 2}}}))
    monkeypatch.setattr(cli, "find_config", lambda: cfg)
    monkeypatch.setattr(cli.shutil, "which", lambda _: "/usr/bin/torchrun")
    monkeypatch.setattr(cli.subprocess, "call", lambda cmd: 0)
    assert (cli._ddp_launch([override]) == 0) is expect_torchrun


def test_ddp_launch_n_gpus_override(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"train": {"ddp": {"enabled": True, "n_gpus": 2}}}))
    monkeypatch.setattr(cli, "find_config", lambda: cfg)
    monkeypatch.setattr(cli.shutil, "which", lambda _: "/usr/bin/torchrun")
    seen = {}
    monkeypatch.setattr(cli.subprocess, "call", lambda cmd: seen.setdefault("cmd", cmd) and 0)
    cli._ddp_launch(["train.ddp.n_gpus=8"])
    assert seen["cmd"][1] == "--nproc_per_node=8"


def test_ddp_launch_uses_torchrun_with_n_gpus(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"train": {"ddp": {"enabled": True, "n_gpus": 4}}}))
    monkeypatch.setattr(cli, "find_config", lambda: cfg)
    monkeypatch.setattr(cli.shutil, "which", lambda _: "/usr/bin/torchrun")
    seen = {}
    monkeypatch.setattr(cli.subprocess, "call", lambda cmd: seen.setdefault("cmd", cmd) and 0)
    cli._ddp_launch(["train.epochs=1"])
    assert seen["cmd"][:2] == ["torchrun", "--nproc_per_node=4"]
    assert seen["cmd"][-2:] == ["dfine_seg.dl.train", "train.epochs=1"]


def test_ddp_launch_errors_without_torchrun(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"train": {"ddp": {"enabled": True}}}))
    monkeypatch.setattr(cli, "find_config", lambda: cfg)
    monkeypatch.setattr(cli.shutil, "which", lambda _: None)
    assert cli._ddp_launch([]) == 1
    assert "torchrun is not on PATH" in capsys.readouterr().err


# ---- predict -----------------------------------------------------------------


class _FakeModel:
    """Stands in for a backend wrapper so predict is tested without loading weights."""

    task = "detect"
    names = {0: "cat"}

    def __call__(self, img, bgr=True):
        import torch

        return [
            {
                "labels": torch.tensor([0]),
                "boxes": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
                "scores": torch.tensor([0.9]),
            }
        ]


def _fake_load(monkeypatch, model=None):
    seen = {}

    def fake(spec, task=None, **kw):
        seen["spec"], seen["task"], seen["kw"] = spec, task, kw
        return model or _FakeModel()

    monkeypatch.setattr("dfine_seg.load_model", fake)
    return seen


def _an_image(tmp_path, name="a.jpg"):
    import cv2
    import numpy as np

    p = tmp_path / name
    cv2.imwrite(str(p), np.zeros((8, 8, 3), dtype=np.uint8))
    return p


def test_predict_missing_source(monkeypatch, capsys, tmp_path):
    assert run(monkeypatch, "predict", "s", str(tmp_path / "nope.jpg")) == 1
    assert "not found" in capsys.readouterr().err


def test_predict_empty_directory(monkeypatch, capsys, tmp_path):
    assert run(monkeypatch, "predict", "s", str(tmp_path)) == 1
    assert "no images" in capsys.readouterr().err


def test_predict_single_image_reports_detections(monkeypatch, capsys, tmp_path):
    _fake_load(monkeypatch)
    img = _an_image(tmp_path)
    assert run(monkeypatch, "predict", "s", str(img)) == 0
    out = capsys.readouterr().out
    assert "a.jpg: 1 objects" in out and "cat 0.90" in out


def test_predict_forwards_conf_task_and_device(monkeypatch, tmp_path):
    seen = _fake_load(monkeypatch)
    img = _an_image(tmp_path)
    run(
        monkeypatch,
        "predict",
        "s",
        str(img),
        "--conf",
        "0.3",
        "--task",
        "segment",
        "--device",
        "cpu",
    )
    assert seen["spec"] == "s" and seen["task"] == "segment"
    assert seen["kw"] == {"conf_thresh": 0.3, "device": "cpu"}


def test_predict_directory_picks_up_every_image(monkeypatch, capsys, tmp_path):
    _fake_load(monkeypatch)
    _an_image(tmp_path, "a.jpg")
    _an_image(tmp_path, "b.png")
    (tmp_path / "notes.txt").write_text("ignored")
    assert run(monkeypatch, "predict", "s", str(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "a.jpg" in out and "b.png" in out and "notes.txt" not in out


def test_predict_writes_annotated_output(monkeypatch, tmp_path):
    _fake_load(monkeypatch)
    img = _an_image(tmp_path)
    dest = tmp_path / "out"
    assert run(monkeypatch, "predict", "s", str(img), "-o", str(dest)) == 0
    assert (dest / "a.jpg").is_file()


def test_predict_writes_output_for_a_backend_without_task(monkeypatch, tmp_path):
    """`task` is a TorchModel attribute; graph wrappers don't have one."""

    class GraphModel:
        names = {0: "cat"}

        __call__ = _FakeModel.__call__

    assert not hasattr(GraphModel, "task")
    _fake_load(monkeypatch, GraphModel())
    img = _an_image(tmp_path)
    dest = tmp_path / "out"
    assert run(monkeypatch, "predict", "model.onnx", str(img), "-o", str(dest)) == 0
    assert (dest / "a.jpg").is_file()


# ---- demo -------------------------------------------------------------------


def _fake_demo(monkeypatch):
    """Stand in for dfine_seg.demo so no UI is built and gradio isn't needed."""
    import sys as _sys
    import types

    seen = {}
    mod = types.ModuleType("dfine_seg.demo")
    mod.main = lambda *a: seen.setdefault("args", a)
    monkeypatch.setitem(_sys.modules, "dfine_seg.demo", mod)
    return seen


def test_demo_defaults(monkeypatch):
    seen = _fake_demo(monkeypatch)
    assert run(monkeypatch, "demo") == 0
    # loopback, not 0.0.0.0: the page loads arbitrary local paths as models
    assert seen["args"] == ("s", "auto", "127.0.0.1", 7860, False)


def test_demo_forwards_model_and_server_flags(monkeypatch):
    seen = _fake_demo(monkeypatch)
    argv = ["demo", "runs/model.pt", "--task", "segment", "--host", "127.0.0.1", "--port", "8080"]
    assert run(monkeypatch, *argv) == 0
    assert seen["args"] == ("runs/model.pt", "segment", "127.0.0.1", 8080, False)


def test_demo_without_gradio_points_at_the_extra(monkeypatch, capsys):
    import sys as _sys

    monkeypatch.setitem(_sys.modules, "dfine_seg.demo", None)  # makes the import raise
    assert cli._demo([]) == 1
    assert "dfine-seg[demo]" in capsys.readouterr().err


def test_demo_needs_no_config(clean_cwd, monkeypatch):
    """It is config-free like predict/init — `demo` must not be in the Hydra command table."""
    assert "demo" not in cli.COMMANDS
    seen = _fake_demo(monkeypatch)
    assert run(monkeypatch, "demo") == 0 and "args" in seen


def test_predict_sem_seg_writes_label_map(monkeypatch, capsys, tmp_path):
    import numpy as np
    import torch

    class SemModel:
        task = "sem_seg"
        names = {0: "road", 1: "sky"}

        def __call__(self, img, bgr=True):
            m = np.zeros((8, 8), dtype=np.uint8)
            m[4:] = 1
            return [{"sem_seg": torch.from_numpy(m)}]

    _fake_load(monkeypatch, SemModel())
    img = _an_image(tmp_path)
    dest = tmp_path / "out"
    assert run(monkeypatch, "predict", "s", str(img), "-o", str(dest)) == 0
    assert (dest / "a.png").is_file()
    assert "2 classes" in capsys.readouterr().out
