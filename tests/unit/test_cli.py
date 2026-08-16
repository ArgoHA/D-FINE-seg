"""`dfine-seg` CLI: init writes a usable config, and commands refuse to run without one."""

import sys

import pytest
import yaml

from dfine_seg import cli


def run(monkeypatch, *argv) -> int:
    monkeypatch.setattr(sys, "argv", ["dfine-seg", *argv])
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
    for command in [*cli.COMMANDS, "init", "version"]:
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
        assert "dfine-seg init" in err and cli.ENV_VAR in err


def test_ddp_launch_skipped_when_disabled(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"train": {"ddp": {"enabled": False}}}))
    monkeypatch.setattr(cli, "find_config", lambda: cfg)
    assert cli._ddp_launch([]) == -1  # -1 = "not handled, fall through to in-process"


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
