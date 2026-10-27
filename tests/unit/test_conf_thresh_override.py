"""Pin the per-call conf_thresh override of the inference wrappers' postprocess."""

import numpy as np
import pytest
import torch

from dfine_seg.infer.onnx_model import ONNXModel
from dfine_seg.infer.torch_model import TorchModel

SCORES = [0.9, 0.6, 0.3, 0.7]
LABELS = [0, 1, 0, 1]
SIZES = [(64, 64)]


def _onnx(conf_thresh):
    m = ONNXModel.__new__(ONNXModel)
    m.conf_thresh = m._prep_conf(conf_thresh)
    m.sem_seg, m.has_masks, m.keep_ratio = False, False, False
    m.labels_to_use, m.apply_nms = [], False
    boxes = np.array([[[i * 10, 0, i * 10 + 5, 5] for i in range(4)]], dtype=np.float32)
    outputs = [np.array([LABELS]), boxes, np.array([SCORES], dtype=np.float32)]
    return lambda **kw: m._postprocess(outputs, SIZES, SIZES, **kw)[0]["scores"]


def _torch(conf_thresh):
    m = TorchModel.__new__(TorchModel)
    m.device, m.n_outputs, m.keep_ratio = "cpu", 2, False
    m.conf_thresh = m._prep_conf(conf_thresh)
    m.labels_to_use, m.apply_nms = [], False
    logits = torch.full((1, 4, 2), -20.0)  # sigmoid ~0 for the off-label class
    logits[0, range(4), LABELS] = torch.logit(torch.tensor(SCORES))
    boxes = torch.tensor([[[0.1 + i * 0.2, 0.5, 0.05, 0.05] for i in range(4)]])
    outputs = {"pred_logits": logits, "pred_boxes": boxes}
    return lambda **kw: m._preds_postprocess(outputs, SIZES, SIZES, **kw)[0]["scores"]


@pytest.mark.parametrize("build", [_onnx, _torch])
@pytest.mark.parametrize("init", [0.5, [0.5, 0.5]])
def test_call_threshold_overrides_init(build, init):
    run = build(init)
    assert sorted(run().tolist()) == pytest.approx([0.6, 0.7, 0.9])
    assert sorted(run(conf_thresh=0.2).tolist()) == pytest.approx(sorted(SCORES))
    assert sorted(run(conf_thresh=torch.tensor(0.65)).tolist()) == pytest.approx([0.7, 0.9])
    # per-class: class 0 at 0.95 drops 0.9 and 0.3; class 1 at 0.65 keeps only 0.7
    assert run(conf_thresh=[0.95, 0.65]).tolist() == pytest.approx([0.7])
    assert sorted(run().tolist()) == pytest.approx([0.6, 0.7, 0.9])  # override did not persist
