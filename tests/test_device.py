import os
import torch

from ctrl_freeq.utils.device import select_device, resolve_cpu_cores


def test_select_device_cpu():
    device, backend = select_device("cpu")
    assert backend == "cpu"
    assert device.type == "cpu"


def test_select_device_gpu_with_cuda_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    device, backend = select_device("gpu")
    assert backend == "cuda"
    assert device.type == "cuda"


def test_select_device_gpu_without_cuda(monkeypatch, caplog):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with caplog.at_level("WARNING"):
        device, backend = select_device("gpu")
        assert backend == "cpu"
        assert device.type == "cpu"
        assert any("falling back to CPU" in rec.message for rec in caplog.records)


def test_resolve_cpu_cores_default(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    # simulate different cpu counts
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    assert resolve_cpu_cores(None) == 7


def test_resolve_cpu_cores_clamp_high_low(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 4)
    assert resolve_cpu_cores(100) == 4
    assert resolve_cpu_cores(0) == 1
    assert resolve_cpu_cores(-3) == 1
    assert resolve_cpu_cores(3) == 3
