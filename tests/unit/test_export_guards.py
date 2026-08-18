"""Export must refuse the setups that produce a broken artifact with a green parity check."""

import pytest
import torch

from tests.unit.test_ckpt_inspect import fake_sd

pytest.importorskip("pandas")  # dl/export pulls the training stack


@pytest.fixture
def ckpt(tmp_path):
    p = tmp_path / "dfine_s_coco.pt"
    torch.save(fake_sd(num_classes=80), p)
    return str(p)


def test_class_count_mismatch_is_refused(ckpt):
    """`dfine-seg init` ships 2 template classes: 80 -> 2 drops every score head."""
    from dfine_seg.dl.export import _check_pretrained_classes

    with pytest.raises(ValueError, match="80 classes but train.label_to_name has 2"):
        _check_pretrained_classes(ckpt, 2)


def test_matching_class_count_passes(ckpt):
    from dfine_seg.dl.export import _check_pretrained_classes

    _check_pretrained_classes(ckpt, 80)


def test_wrapped_checkpoint_formats_are_unwrapped(tmp_path):
    """Legacy checkpoints nest the weights under "model"/"ema"."""
    from dfine_seg.dl.export import _check_pretrained_classes

    p = tmp_path / "legacy.pt"
    torch.save({"model": fake_sd(num_classes=7)}, p)
    _check_pretrained_classes(str(p), 7)
    with pytest.raises(ValueError):
        _check_pretrained_classes(str(p), 8)


def test_latest_experiment_raises_when_nothing_was_trained(tmp_path):
    """Previously an unguarded iterdir(): FileNotFoundError with no explanation."""
    from dfine_seg.dl.utils import get_latest_experiment_name

    missing = tmp_path / "output" / "models" / "exp_2026-01-01"
    with pytest.raises(FileNotFoundError, match="no run matching 'exp_<date>'"):
        get_latest_experiment_name("exp_2026-01-01", str(missing))


def test_latest_experiment_ignores_undated_siblings(tmp_path):
    from dfine_seg.dl.utils import get_latest_experiment_name

    runs = tmp_path / "models"
    runs.mkdir()
    (runs / "scratch").mkdir()  # no underscore at all
    (runs / "exp_notadate").mkdir()
    (runs / "exp_2026-01-01").mkdir()
    (runs / "exp_2026-03-09").mkdir()
    assert get_latest_experiment_name("exp_2026-08-18", str(runs / "exp_2026-08-18")) == (
        "exp_2026-03-09"
    )


def test_latest_experiment_returns_the_dir_that_exists(tmp_path):
    from dfine_seg.dl.utils import get_latest_experiment_name

    here = tmp_path / "exp_2026-08-18"
    here.mkdir()
    assert get_latest_experiment_name("exp_2026-08-18", str(here)) == "exp_2026-08-18"
