"""`Visualizer` - the public drawing entry point (`from dfine_seg import Visualizer`).

Synthetic arrays only: no weights, no model build. What is pinned here is the contract the
API promises - construction off a wrapper, one call for every task, and the guards that turn
the two easy mistakes (a result list, a 4-channel stack) into readable errors.
"""

import numpy as np
import pytest
import torch

from dfine_seg import Visualizer
from dfine_seg.viz import classes_from_model


class _Wrapper:
    """Stand-in for a wrapper from `load_model`; graph artifacts have no `n_outputs`."""

    def __init__(self, names=None, n_outputs=None):
        self.names = names
        if n_outputs is not None:
            self.n_outputs = n_outputs


def _img(h=64, w=64):
    return np.full((h, w, 3), 40, dtype=np.uint8)


def _boxes(labels=(0,)):
    return {
        "boxes": torch.tensor([[8.0, 8.0, 40.0, 40.0]] * len(labels)),
        "scores": torch.tensor([0.9] * len(labels)),
        "labels": torch.tensor(list(labels), dtype=torch.int64),
    }


# ── construction ────────────────────────────────────────────────────────
def test_classes_from_model_prefers_the_larger_source():
    assert classes_from_model(_Wrapper({0: "a", 1: "b"}, n_outputs=80)) == (80, {0: "a", 1: "b"})
    assert classes_from_model(_Wrapper({0: "a", 3: "d"})) == (4, {0: "a", 3: "d"})  # names only
    assert classes_from_model(_Wrapper()) == (0, {})  # nothing knowable


def test_init_reads_the_model():
    vis = Visualizer(_Wrapper({0: "cat", 1: "dog"}, n_outputs=2))
    assert vis.n_classes == 2
    assert vis.class_names == {0: "cat", 1: "dog"}


def test_init_falls_back_to_coco_when_nothing_is_knowable():
    assert Visualizer(_Wrapper()).n_classes == 80


def test_init_accepts_an_int_and_the_legacy_keywords():
    assert Visualizer(3).n_classes == 3
    vis = Visualizer(n_classes=2, class_names={0: "a", 1: "b"})  # every in-repo call site
    assert (vis.n_classes, vis.class_names[1]) == (2, "b")


def test_colors_run_violet_to_red():
    colors = Visualizer(5).colors  # BGR
    assert colors[0][0] > colors[0][2] and colors[-1][2] > colors[-1][0]


# ── boxes / masks ───────────────────────────────────────────────────────
def test_draw_returns_a_copy():
    img, vis = _img(), Visualizer(1)
    out = vis(img, _boxes())
    assert out is not img and not np.array_equal(out, img)
    assert img.max() == 40  # input untouched
    assert out.shape == img.shape and out.dtype == np.uint8


def test_empty_result_is_an_unchanged_copy():
    img = _img()
    out = Visualizer(1)(img, {"boxes": [], "scores": [], "labels": []})
    assert out is not img and np.array_equal(out, img)


def test_masks_are_drawn():
    img = _img()
    res = _boxes()
    res["masks"] = torch.zeros(1, 64, 64)
    res["masks"][0, 50:60, 50:60] = 1.0  # outside the box, so only the mask can paint it
    out = Visualizer(1)(img, res)
    assert not np.array_equal(out[50:60, 50:60], img[50:60, 50:60])


def test_minimize_skips_the_text():
    img, vis = _img(), Visualizer(1, class_names={0: "an intentionally long class name"})
    assert not np.array_equal(vis(img, _boxes()), vis(img, _boxes(), minimize=True))


# ── sem_seg ─────────────────────────────────────────────────────────────
def test_sem_seg_dispatch_leaves_ignore_pixels_alone():
    img = _img()
    label_map = np.full((64, 64), 255, dtype=np.uint8)  # ignore_index
    label_map[:32] = 1
    out = Visualizer(3)(img, {"sem_seg": torch.from_numpy(label_map)})
    assert not np.array_equal(out[:32], img[:32])
    assert np.array_equal(out[32:], img[32:])


def test_palette_is_built_once():
    vis = Visualizer(3)
    assert vis.palette is vis.palette and vis.palette.shape == (256, 3)


# ── guards ──────────────────────────────────────────────────────────────
def test_a_result_list_names_the_fix():
    with pytest.raises(TypeError, match=r"model\(img\)\[0\]"):
        Visualizer(1)(_img(), [_boxes()])


def test_extra_channels_name_the_fix():
    stack = np.zeros((64, 64, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match=r"img\[:, :, :3\]"):
        Visualizer(1)(stack, _boxes())
