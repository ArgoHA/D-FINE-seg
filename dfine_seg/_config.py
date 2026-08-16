"""Locate the Hydra config dir for both the clone workflow and pip installs.

Clone: cwd is the repo root, so `./config.yaml` wins and behavior is unchanged.
Pip: `dfine-seg init` writes `./config.yaml` from the packaged template below.
"""

import os
from pathlib import Path

ENV_VAR = "DFINE_SEG_CONFIG_DIR"
CONFIG_NAME = "config"

_PKG_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = _PKG_ROOT / "config" / "default.yaml"  # `dfine-seg init` template
_REPO_ROOT = _PKG_ROOT.parent  # repo root for a clone / editable install


def _candidates() -> list[Path]:
    dirs = []
    if env := os.environ.get(ENV_VAR):
        dirs.append(Path(env))
    dirs.append(Path.cwd())
    dirs.append(_REPO_ROOT)  # only exists as a real dir for clone / editable installs
    return dirs


def find_config() -> Path | None:
    """First `config.yaml` on the search path, or None."""
    for d in _candidates():
        p = d / f"{CONFIG_NAME}.yaml"
        if p.is_file():
            return p
    return None


def config_dir() -> str:
    """Absolute dir for `@hydra.main(config_path=...)`.

    Deliberately never falls back to the packaged template — training against
    someone else's defaults is worse than Hydra's own "cannot find config" error.
    """
    found = find_config()
    return str(found.parent if found else Path.cwd())
