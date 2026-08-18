"""Where `config.yaml` comes from: env var, cwd, repo root — and when that must fail."""

import pytest

from dfine_seg.config import resolve
from dfine_seg.config.resolve import ENV_VAR, find_config


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """cwd with no config, and no repo root reachable behind it."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.setattr(resolve, "_REPO_ROOT", tmp_path / "nowhere")
    return tmp_path


def test_cwd_wins(isolated):
    (isolated / "config.yaml").write_text("task: detect\n")
    assert find_config() == isolated / "config.yaml"


def test_none_when_nothing_is_found(isolated):
    assert find_config() is None


def test_env_var_wins_over_cwd(isolated, tmp_path_factory, monkeypatch):
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    (elsewhere / "config.yaml").write_text("task: segment\n")
    (isolated / "config.yaml").write_text("task: detect\n")
    monkeypatch.setenv(ENV_VAR, str(elsewhere))
    assert find_config() == elsewhere / "config.yaml"


def test_env_var_pointing_nowhere_raises(isolated, tmp_path_factory, monkeypatch):
    """Falling through to cwd would train against a different config than the one asked for."""
    empty = tmp_path_factory.mktemp("empty")
    (isolated / "config.yaml").write_text("task: detect\n")
    monkeypatch.setenv(ENV_VAR, str(empty))
    with pytest.raises(FileNotFoundError, match=ENV_VAR):
        find_config()


def test_cli_reports_a_bad_env_var_instead_of_crashing(
    isolated, tmp_path_factory, monkeypatch, capsys
):
    from dfine_seg import cli

    empty = tmp_path_factory.mktemp("empty2")
    monkeypatch.setenv(ENV_VAR, str(empty))
    assert cli._run("train", []) == 1
    assert ENV_VAR in capsys.readouterr().err
