"""torch.profiler on paras devices: operators get device times (the
PrivateUse1 fallback, timed with the queues' native events)."""

import unittest

import torch
import torch_paras
from torch.profiler import ProfilerActivity, profile


def _device_time(evt):
    for name in ("device_time_total", "self_device_time_total", "privateuse1_time_total"):
        if hasattr(evt, name):
            return getattr(evt, name)
    return 0


def test_profiler_device_times(device):
    if not torch_paras.is_gpu_device(torch.device(device).index or 0):
        raise unittest.SkipTest("device times of a GPU paras device")
    acts = torch.profiler.supported_activities()
    if ProfilerActivity.PrivateUse1 not in acts:
        raise unittest.SkipTest("this PyTorch's profiler has no PrivateUse1 activity")
    x = torch.randn(1024, 1024, device=device)
    with torch.no_grad():
        (x @ x).sum()
        torch_paras.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]) as prof:
            for _ in range(3):
                y = x @ x
                z = y + 1
            torch_paras.synchronize()
    by = {e.key: e for e in prof.key_averages()}
    assert "aten::mm" in by and "aten::add" in by, sorted(by)[:20]
    mm, add = _device_time(by["aten::mm"]), _device_time(by["aten::add"])
    assert mm > 0 and add > 0, (mm, add)
    assert mm > add, (mm, add)
    assert by["aten::mm"].count >= 3, by["aten::mm"].count
