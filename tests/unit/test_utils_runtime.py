import torch

from dfine_seg.dl.utils import get_vram_usage


def test_vram_usage_uses_torch_cuda_memory(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (3_000, 4_000))
    assert get_vram_usage() == 25


def test_vram_usage_is_zero_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert get_vram_usage() == 0
