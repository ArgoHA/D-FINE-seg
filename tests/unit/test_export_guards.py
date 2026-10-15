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
    """`dfine init` ships 2 template classes: 80 -> 2 drops every score head."""
    from dfine_seg.dl.export import _check_pretrained

    with pytest.raises(ValueError, match="80 classes but train.label_to_name has 2"):
        _check_pretrained(ckpt, "detect", 2, [640, 640])


def test_matching_class_count_passes(ckpt):
    from dfine_seg.dl.export import _check_pretrained

    _check_pretrained(ckpt, "detect", 80, [640, 640])


def test_task_mismatch_is_refused(tmp_path):
    """A sem-seg release exported with task=segment (19 classes matching) would export
    random detection heads - and parity would still pass."""
    from dfine_seg.dl.export import _check_pretrained

    p = tmp_path / "dfine_sem_s_cityscapes.pt"
    torch.save(fake_sd(task="sem_seg", num_classes=19), p)
    with pytest.raises(ValueError, match="sem_seg checkpoint but task is segment"):
        _check_pretrained(str(p), "segment", 19, [640, 640])
    with pytest.raises(ValueError, match="sem_seg checkpoint but task is detect"):
        _check_pretrained(str(p), "detect", 19, [640, 640])
    _check_pretrained(str(p), "sem_seg", 19, [640, 640])  # matching task passes


def test_detect_from_segment_checkpoint_is_allowed(tmp_path):
    """Segment weights carry the full det head: a detect export is not a broken artifact."""
    from dfine_seg.dl.export import _check_pretrained

    p = tmp_path / "dfine_seg_s_coco.pt"
    torch.save(fake_sd(task="segment", num_classes=80), p)
    _check_pretrained(str(p), "detect", 80, [640, 640])


def test_wrapped_checkpoint_formats_are_unwrapped(tmp_path):
    """Legacy checkpoints nest the weights under "model"/"ema"."""
    from dfine_seg.dl.export import _check_pretrained

    p = tmp_path / "legacy.pt"
    torch.save({"model": fake_sd(num_classes=7)}, p)
    _check_pretrained(str(p), "detect", 7, [640, 640])
    with pytest.raises(ValueError):
        _check_pretrained(str(p), "detect", 8, [640, 640])


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
