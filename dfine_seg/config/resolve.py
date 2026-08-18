"""Locate the Hydra config dir for both the clone workflow and pip installs.

Clone: cwd is the repo root, so `./config.yaml` wins and behavior is unchanged.
Pip: `dfine init` writes `./config.yaml` from the packaged template below.
"""

import os
from pathlib import Path

ENV_VAR = "DFINE_SEG_CONFIG_DIR"
CONFIG_NAME = "config"

_HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = _HERE / "default.yaml"  # `dfine init` template, this file's neighbour
_REPO_ROOT = _HERE.parents[1]  # repo root for a clone / editable install


def find_config() -> Path | None:
    """First `config.yaml` on the search path, or None.

    A set-but-wrong $DFINE_SEG_CONFIG_DIR raises rather than falling through: training
    against a config other than the one you pointed at is worse than an error.
    """
    if env := os.environ.get(ENV_VAR):
        p = Path(env) / f"{CONFIG_NAME}.yaml"
        if not p.is_file():
            raise FileNotFoundError(f"${ENV_VAR}={env} holds no {CONFIG_NAME}.yaml")
        return p
    for d in (Path.cwd(), _REPO_ROOT):  # _REPO_ROOT: clone / editable installs only
        p = d / f"{CONFIG_NAME}.yaml"
        if p.is_file():
            return p
    return None


def config_dir() -> str:
    """Absolute dir for `@hydra.main(config_path=...)`.

    Deliberately never falls back to the packaged template - training against
    someone else's defaults is worse than Hydra's own "cannot find config" error.
    """
    found = find_config()
    return str(found.parent if found else Path.cwd())
