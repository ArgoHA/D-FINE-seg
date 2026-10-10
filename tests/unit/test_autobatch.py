import pytest
import torch

from dfine_seg.dl.utils import search_batch_size
from dfine_seg.dl.train import KDTeacher, training_forward


@pytest.mark.parametrize("limit", [0, 1, 2, 3, 7, 16, 31, 64])
def test_search_returns_largest_tested_batch(limit):
    tried = []

    def probe(batch):
        tried.append(batch)
        return batch <= limit

    assert search_batch_size(probe, maximum=64) == limit
    assert tried[0] == 1
    assert limit == 0 or limit in tried


def test_search_does_not_hide_unrelated_failures():
    def probe(batch):
        raise RuntimeError("CUDA illegal memory access")

    with pytest.raises(RuntimeError, match="illegal memory access"):
        search_batch_size(probe)


class FakeTeacher(torch.nn.Module):
    def __init__(self, task):
        super().__init__()
        self.task = task
        self.calls = []
        self.weight = torch.nn.Parameter(torch.tensor(2.0), requires_grad=False)

    def forward(self, x, targets=None):
        self.calls.append((len(x), tuple(x.shape[-2:]), torch.is_grad_enabled()))
        if self.task == "sem_seg":
            return {"sem_seg_logits_q": x[:, :1, ::4, ::4] * self.weight}
        out = {"pred_logits": x.mean((2, 3))[:, None], "up": self.weight, "reg_scale": self.weight}
        if self.task == "segment" and targets is None:
            out["pred_mask_logits"] = x[:, :1, ::4, ::4]
        return out


def make_teacher(task, chunk):
    teacher = KDTeacher.__new__(KDTeacher)
    teacher.task = task
    teacher.device = torch.device("cpu")
    teacher.dtype = torch.bfloat16
    teacher.amp_enabled = False
    teacher.batch_size = chunk
    teacher.img_size = (64, 96)
    teacher.model = FakeTeacher(task)
    return teacher


@pytest.mark.parametrize("task", ["sem_seg", "detect", "segment"])
@pytest.mark.parametrize("chunk", [1, 2, 4, 10])
def test_chunking_preserves_outputs_order_and_scalar_metadata(task, chunk):
    inputs = torch.randn(5, 3, 32, 64)
    targets = [{"masks": torch.empty(0)} for _ in range(5)]
    targets[0] = {"masks": torch.ones(1, 32, 64)}
    whole, chunked = make_teacher(task, None), make_teacher(task, chunk)
    expected, actual = whole(inputs, targets), chunked(inputs, targets)
    if task != "sem_seg":
        expected, actual = expected["kd_teacher"], actual["kd_teacher"]
    assert actual.keys() == expected.keys()
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key])
        assert not actual[key].requires_grad
    assert all(
        size <= chunk and shape == (64, 96) and not grad
        for size, shape, grad in chunked.model.calls
    )


def test_teacher_runs_before_student_and_only_student_gets_gradients():
    events = []
    student = torch.nn.Linear(3, 1)
    teacher = torch.nn.Linear(3, 1).requires_grad_(False)
    x = torch.randn(2, 3)

    def predict_teacher(inputs, targets):
        events.append("teacher")
        return {"teacher": teacher(inputs)}

    def predict_student(inputs, targets):
        events.append("student")
        return {"student": student(inputs)}

    def criterion(outputs, targets):
        return {"loss": (outputs["student"] - outputs["teacher"]).square().mean()}

    losses = training_forward(
        predict_student, criterion, predict_teacher, x, [], enabled=False, dtype=torch.bfloat16
    )
    losses["loss"].backward()
    assert events == ["teacher", "student"]
    assert student.weight.grad is not None
    assert teacher.weight.grad is None


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_teacher_chunk_fails_before_loading_checkpoint(value):
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {"task": "detect", "train": {"amp_enabled": False, "kd": {"teacher_batch": value}}}
    )
    with pytest.raises(ValueError, match="teacher_batch"):
        KDTeacher(cfg, torch.device("cpu"))


def test_probe_rejects_invalid_memory_fraction():
    from dfine_seg.dl.utils import auto_batch_size

    with pytest.raises(ValueError, match="target_fraction"):
        auto_batch_size(None, torch.device("cuda"), 1.2)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_probe_child_receives_recipe_without_changing_parent_config(tmp_path, monkeypatch, dtype):
    import json
    from pathlib import Path
    from types import SimpleNamespace

    from omegaconf import OmegaConf
    from dfine_seg.dl import utils

    cfg = OmegaConf.create(
        {
            "train": {
                "batch_size": -1,
                "amp_dtype": dtype,
                "use_ema": True,
                "b_accum_steps": 3,
                "kd": {"teacher": "teacher.pt", "teacher_batch": 2},
            }
        }
    )
    original = OmegaConf.to_container(cfg)

    def run(command, **kwargs):
        child = OmegaConf.load(command[3])
        assert OmegaConf.to_container(child) == original
        assert command[4] == "cuda:1"
        assert "PYTHONPATH" in kwargs["env"]
        Path(command[-1]).write_text(json.dumps({"batch": 7, "peak_reserved": 1234}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(utils.subprocess, "run", run)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda device: SimpleNamespace(total_memory=2**34)
    )
    assert utils.auto_batch_size(cfg, torch.device("cuda:1")) == 7
    assert OmegaConf.to_container(cfg) == original


def test_probe_propagates_child_failure_instead_of_returning_batch_one(monkeypatch):
    import json
    from pathlib import Path
    from types import SimpleNamespace

    from omegaconf import OmegaConf
    from dfine_seg.dl import utils

    def run(command, **kwargs):
        Path(command[-1]).write_text(json.dumps({"error": "Batch 1 does not fit"}))
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(utils.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="Batch 1 does not fit"):
        utils.auto_batch_size(OmegaConf.create({}), torch.device("cuda"))


def test_distributed_probe_uses_minimum_local_batch(monkeypatch):
    import json
    from pathlib import Path
    from types import SimpleNamespace

    from omegaconf import OmegaConf
    from dfine_seg.dl import utils

    def run(command, **kwargs):
        Path(command[-1]).write_text(json.dumps({"batch": 12, "peak_reserved": 1234}))
        return SimpleNamespace(returncode=0)

    def reduce(tensor, op):
        assert op == torch.distributed.ReduceOp.MIN
        assert tensor.item() == 12
        tensor.fill_(5)

    real_tensor = torch.tensor
    monkeypatch.setattr(utils.subprocess, "run", run)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda device: SimpleNamespace(total_memory=2**34)
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    monkeypatch.setattr(torch, "tensor", lambda value, **kwargs: real_tensor(value))
    assert utils.auto_batch_size(OmegaConf.create({}), torch.device("cuda")) == 5


def test_teacher_with_other_class_order_is_rejected(monkeypatch):
    from omegaconf import OmegaConf

    import dfine_seg.dl.train as train

    info = {"task": "detect", "num_classes": 2, "names": {0: "car", 1: "person"}}
    monkeypatch.setattr(train, "load_and_describe", lambda path: ({}, info))
    cfg = OmegaConf.create(
        {
            "task": "detect",
            "train": {
                "amp_enabled": False,
                "label_to_name": {0: "person", 1: "car"},
                "kd": {"teacher": "teacher.pt"},
            },
        }
    )
    with pytest.raises(ValueError, match="classes"):
        KDTeacher(cfg, torch.device("cpu"))
