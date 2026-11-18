import json

import numpy as np
import torch

from dfine_seg.dl.utils import append_history, get_vram_usage


def test_vram_usage_uses_torch_cuda_memory(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (3_000, 4_000))
    assert get_vram_usage() == 25


def test_vram_usage_is_zero_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert get_vram_usage() == 0


def test_append_history_writes_one_json_line_per_call(tmp_path):
    path = tmp_path / "run" / "history.jsonl"
    append_history(path, {"epoch": 1, "val": {"mAP_50": np.float32(0.5)}, "best": True})
    append_history(
        path, {"epoch": 2, "val": {"per_class": np.array([0.25, 0.5])}, "lr": torch.tensor(1e-4)}
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0] == {"epoch": 1, "val": {"mAP_50": 0.5}, "best": True}
    assert rows[1]["val"]["per_class"] == [0.25, 0.5]
    assert abs(rows[1]["lr"] - 1e-4) < 1e-9
