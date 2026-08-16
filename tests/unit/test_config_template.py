"""The packaged `dfine-seg init` template must not drift from the working config.

Root `config.yaml` is the live dev config (machine paths, real classes); the packaged
`dfine_seg/config/default.yaml` is its sanitized twin. They must stay structurally
identical, or a pip user gets a config missing keys the code reads.
"""

from pathlib import Path

import pytest
import yaml

from dfine_seg._config import DEFAULT_CONFIG

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_CONFIG = REPO_ROOT / "config.yaml"

# Keys deliberately different: paths/classes/logging are user-specific by design.
ALLOWED_VALUE_DIFFS = {
    "project_name",
    "exp_name",
    "model_name",
    "task",
    "train.root",
    "train.use_wandb",
    "train.label_to_name",
}


def _paths(node, prefix=""):
    """Recursive set of dotted key paths. Mapping values under label_to_name are opaque."""
    out = set()
    if isinstance(node, dict):
        for k, v in node.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            out.add(p)
            if p not in ALLOWED_VALUE_DIFFS:
                out |= _paths(v, p)
    return out


@pytest.fixture(scope="module")
def configs():
    if not ROOT_CONFIG.is_file():
        pytest.skip("root config.yaml absent (not a source checkout)")
    return yaml.safe_load(ROOT_CONFIG.read_text()), yaml.safe_load(DEFAULT_CONFIG.read_text())


def test_template_exists():
    assert DEFAULT_CONFIG.is_file(), f"packaged template missing at {DEFAULT_CONFIG}"


def test_same_key_structure(configs):
    root, tmpl = configs
    missing = _paths(root) - _paths(tmpl)
    extra = _paths(tmpl) - _paths(root)
    assert not missing, f"template is missing keys added to config.yaml: {sorted(missing)}"
    assert not extra, f"template has keys not in config.yaml: {sorted(extra)}"


def test_template_is_sanitized(configs):
    _, tmpl = configs
    assert tmpl["train"]["root"] == ".", "template must not carry an absolute machine path"
    assert tmpl["train"]["use_wandb"] is False, "template must not force a wandb login on install"
    assert "/home/" not in DEFAULT_CONFIG.read_text(), "template leaks a developer path"


def test_template_values_match_where_shared(configs):
    """Every key outside ALLOWED_VALUE_DIFFS must carry the same default."""
    root, tmpl = configs

    def walk(a, b, prefix=""):
        for k in a:
            p = f"{prefix}.{k}" if prefix else str(k)
            if p in ALLOWED_VALUE_DIFFS:
                continue
            if isinstance(a[k], dict):
                walk(a[k], b[k], p)
            else:
                assert a[k] == b[k], f"{p}: config.yaml={a[k]!r} template={b[k]!r}"

    walk(root, tmpl)
