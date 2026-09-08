"""SAM3Model prompt parsing + multi-prompt merge - stubbed processor, no weights needed."""

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")  # the [label] extra

from dfine_seg.infer.sam3_model import SAM3Model  # noqa: E402


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("person", ["person"]),
        ("car, person", ["car", "person"]),
        ("car,person", ["car", "person"]),
        ("person\ndog", ["person", "dog"]),
        (["car", "person"], ["car", "person"]),
        (["person, dog"], ["person", "dog"]),
        (["a,b", "c"], ["a", "b", "c"]),
        (None, ["object"]),
        ("", ["object"]),
        ("  ", ["object"]),
    ],
)
def test_parse_prompts(raw, expected):
    assert SAM3Model.parse_prompts(raw) == expected


# ─── multi-prompt merge ─────────────────────────────────────────────────
IMG = np.zeros((60, 80, 3), np.uint8)
MASK_RES = 32  # transformers leaves an empty prompt's masks at SAM3's own resolution


class _Inputs(dict):
    def to(self, *_):
        return self


class _Processor:
    """Mimics Sam3Processor: `hits` says how many detections each prompt returns."""

    def __init__(self, hits):
        self.hits = list(hits)

    def __call__(self, **_):
        return _Inputs()

    def _result(self, target_sizes, masks: bool):
        n = self.hits.pop(0)
        h, w = target_sizes[0] if n else (MASK_RES, MASK_RES)
        res = {"scores": torch.full((n,), 0.9), "boxes": torch.ones(n, 4)}
        if masks:
            res["masks"] = torch.ones(n, h, w, dtype=torch.long)
        return [res]

    def post_process_instance_segmentation(self, outputs, threshold, mask_threshold, target_sizes):
        return self._result(target_sizes, masks=True)

    def post_process_object_detection(self, outputs, threshold, target_sizes):
        return self._result(target_sizes, masks=False)


def _model(prompts, hits, task="segment"):
    m = SAM3Model.__new__(SAM3Model)
    m.device, m.conf_thresh, m.mask_threshold = "cpu", 0.5, 0.5
    m.task = task
    m.prompts = prompts
    m.processor = _Processor(hits)
    m.model = lambda **_: None
    return m


@pytest.mark.parametrize("task", ["segment", "detect"])
@pytest.mark.parametrize(
    ("hits", "labels"),
    [
        ((1, 2), [0, 1, 1]),
        ((0, 2), [1, 1]),  # an empty prompt used to break the cat: mask sizes disagree
        ((2, 0), [0, 0]),
        ((0, 0), []),
    ],
)
def test_merge_labels_by_prompt_index(hits, labels, task):
    res = _model(["car", "person"], hits, task)(IMG)[0]
    assert res["labels"].tolist() == labels
    assert res["boxes"].shape == (len(labels), 4)
    assert len(res["scores"]) == len(labels)
    if task == "detect":
        assert "masks" not in res  # detect never builds SAM3's full-resolution masks
    else:
        assert res["masks"].shape == (len(labels), *IMG.shape[:2])
