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


def test_parse_classes_groups_synonyms():
    names, prompts, cls = SAM3Model.parse_classes({"veg": ["tree", "bush, grass"], "sky": None})
    assert names == ["veg", "sky"]
    assert prompts == ["tree", "bush", "grass", "sky"]
    assert cls == [0, 0, 0, 1]
    assert SAM3Model.parse_classes("car, person") == (["car", "person"],) * 2 + ([0, 1],)


class _Inputs(dict):
    def __getattr__(self, key):
        return self[key]

    def to(self, *_):
        return self


class _Vision(dict):
    pass


class _Out:
    def __init__(self, n, q=2, res=4, fill=None):
        self.pred_logits = torch.zeros(n, q)
        self.presence_logits = torch.full((n, 1), 10.0)
        self.pred_masks = torch.full((n, q, res, res), -10.0) if fill is None else fill


class _Processor:
    """Mimics Sam3Processor: `hits` says how many detections each prompt returns."""

    def __init__(self, hits):
        self.hits = list(hits)

    def __call__(self, text=None, images=None, **_):
        if text is not None:
            n = len(text)
            return _Inputs(input_ids=torch.arange(n)[:, None], attention_mask=torch.ones(n, 1))
        return _Inputs(pixel_values=torch.zeros(1, 3, 8, 8))

    def _results(self, outputs, target_sizes, masks: bool):
        results = []
        for _ in range(len(outputs.pred_logits)):
            n = self.hits.pop(0) if self.hits else 0  # padded batch slots find nothing
            h, w = target_sizes[0] if n else (MASK_RES, MASK_RES)
            res = {"scores": torch.full((n,), 0.9), "boxes": torch.ones(n, 4)}
            if masks:
                res["masks"] = torch.ones(n, h, w, dtype=torch.long)
            results.append(res)
        return results

    def post_process_instance_segmentation(self, outputs, threshold, mask_threshold, target_sizes):
        return self._results(outputs, target_sizes, masks=True)

    def post_process_object_detection(self, outputs, threshold, target_sizes):
        return self._results(outputs, target_sizes, masks=False)


class _Net:
    """Mimics Sam3Model: batch = number of text embeddings; `out_fn(text_ids)` builds outputs."""

    dtype = torch.float32

    def __init__(self, out_fn=None):
        self.out_fn = out_fn or (lambda ids: _Out(len(ids)))
        self.batches = []

    def get_text_features(self, input_ids, attention_mask):
        return _Inputs(pooler_output=input_ids.float()[:, :, None])

    def get_vision_features(self, pixel_values):
        return _Vision(fpn_hidden_states=(torch.zeros(1, 2, 4, 4),))

    def __call__(self, vision_embeds, text_embeds, attention_mask):
        ids = text_embeds[:, 0, 0].long().tolist()
        assert vision_embeds["fpn_hidden_states"][0].shape[0] == len(ids)
        self.batches.append(ids)
        return self.out_fn(ids)


def _model(prompts, hits=(), task="segment", prompt_batch=20, net=None):
    m = SAM3Model.__new__(SAM3Model)
    m.device, m.conf_thresh, m.mask_threshold = "cpu", 0.5, 0.5
    m.task, m.prompt_batch, m.tiles = task, prompt_batch, 1
    m.prompts = prompts
    m.processor = _Processor(hits)
    m.model = net or _Net()
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


def test_balanced_padded_prompt_batches():
    """5 prompts at batch 2 -> 3 forwards of 2 (last padded); padded slots never surface."""
    net = _Net()
    m = _model(
        {"a": ["p0", "p1"], "b": ["p2", "p3", "p4"]}, (1, 1, 1, 1, 1), prompt_batch=2, net=net
    )
    res = m(IMG)[0]
    assert net.batches == [[0, 1], [2, 3], [4, 4]]
    assert res["labels"].tolist() == [0, 0, 1, 1, 1]


def _sem_out(ids):
    """Flat prompt k's single query covers mask column k of a 4x4 mask, at score 0.5."""
    masks = torch.full((len(ids), 1, 4, 4), -20.0)
    for i, k in enumerate(ids):
        masks[i, 0, :, k] = 20.0
    out = _Out(len(ids), q=1, fill=masks)
    return out


def test_sem_seg_whole_image_by_default():
    """Official SAM3 inference: one encode of the whole (squashed) image, no tiling."""
    net = _Net(_sem_out)
    m = _model({"x": ["p0", "p2"], "y": ["p1"], "z": ["p3"]}, task="sem_seg", net=net)
    res = m(np.zeros((40, 80, 3), np.uint8))[0]
    assert res["sem_seg"].shape == (40, 80)
    assert len(net.batches) == 1
    assert res["sem_seg"][20, 10::20].tolist() == [0, 0, 1, 2]  # 4 mask columns of 20 px


def test_sem_seg_max_over_class_prompts_and_tiles():
    net = _Net(_sem_out)
    m = _model({"x": ["p0", "p2"], "y": ["p1"], "z": ["p3"]}, task="sem_seg", net=net)
    m.tiles = 2
    img = np.zeros((40, 80, 3), np.uint8)  # two 40x40 tiles, one encode each
    res = m(img)[0]
    assert res["sem_seg"].shape == res["scores"].shape == (40, 80)
    assert len(net.batches) == 2
    # flat prompts p0 p2 p1 p3 -> classes x x y z; 4 mask columns of 10 px per 40-px tile
    row = res["sem_seg"][20, 5::10].tolist()
    assert row == [0, 0, 1, 2] * 2
    assert torch.allclose(
        res["scores"][20, 5::10], torch.full((8,), 0.5), atol=0.03
    )  # bilinear upsample
