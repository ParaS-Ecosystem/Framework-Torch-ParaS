"""Record and replay (torch_paras.graphs): replays compute what eager code
computes on new inputs, vendor GEMMs and attention masks included; work
that needs the host is refused; captured memory stays the graph's own."""

import unittest

import torch
import torch.nn.functional as F
import torch_paras
from torch_paras import graphs

from common import assert_close


def _require_graphs(device):
    if not torch_paras.is_gpu_device(torch.device(device).index or 0):
        raise unittest.SkipTest("graphs capture GPU devices")
    if not graphs.is_supported():
        raise unittest.SkipTest("ParaS without record-and-replay command graphs")


def _block(x, w1, w2, gamma):
    h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * gamma
    h = F.silu(h @ w1) @ w2
    return torch.softmax(x + h, dim=-1)


def test_replay_matches_eager(device):
    _require_graphs(device)
    torch.manual_seed(0)
    w1 = torch.randn(64, 128, device=device) / 8
    w2 = torch.randn(128, 64, device=device) / 8
    gamma = torch.randn(64, device=device)
    static_x = torch.randn(16, 64, device=device)
    with torch.no_grad():
        _block(static_x, w1, w2, gamma)
        g = graphs.Graph()
        with graphs.graph(g, device=device):
            static_y = _block(static_x, w1, w2, gamma)
        for _ in range(3):
            x = torch.randn(16, 64, device=device)
            static_x.copy_(x)
            g.replay()
            assert_close(static_y, _block(x, w1, w2, gamma), "replay")
    g.reset()


def test_capture_refuses_host_work(device):
    _require_graphs(device)
    x = torch.randn(32, device=device)
    for what, work in (("item", lambda: (x * 2).sum().item()),
                       ("to cpu", lambda: (x + 1).cpu()),
                       ("synchronize", lambda: torch_paras.synchronize(torch.device(device)))):
        g = graphs.Graph()
        try:
            with graphs.graph(g, device=device):
                work()
        except Exception:
            pass
        else:
            raise AssertionError(f"capture accepted {what}")
        assert not graphs.is_capturing(device), what
        try:
            g.replay()
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"failed capture ({what}) left a replayable graph")
        assert_close((x * 3).cpu(), x.cpu() * 3, f"device usable after {what}")


def test_graph_memory_stays_private(device):
    _require_graphs(device)
    a = torch.randn(1024, device=device)
    g = graphs.Graph()
    with graphs.graph(g, device=device):
        tmp = a * 2
        out = tmp + 1
        del tmp
    others = [torch.full((1024,), 7.0, device=device) for _ in range(8)]
    a.copy_(torch.arange(1024.0, device=device))
    g.replay()
    assert_close(out, torch.arange(1024.0) * 2 + 1, "replay output")
    for o in others:
        assert_close(o, torch.full((1024,), 7.0), "eager tensors untouched by replay")
    before = torch_paras.memory_allocated(torch.device(device).index or 0)
    del out
    g.reset()
    after = torch_paras.memory_allocated(torch.device(device).index or 0)
    assert after < before, (before, after)


def test_attention_mask_replay(device):
    _require_graphs(device)
    torch.manual_seed(0)
    B, H, L, D = 1, 4, 96, 64
    q, k, v = (torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16) for _ in range(3))
    mask = torch.zeros(B, 1, L, L, device=device, dtype=torch.bfloat16)
    def masks():
        for cut in (L, 40, 7):
            m = torch.zeros(B, 1, L, L, dtype=torch.bfloat16)
            m[..., cut:] = torch.finfo(torch.bfloat16).min
            yield m
    with torch.no_grad():
        F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        g = graphs.Graph()
        with graphs.graph(g, device=device):
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        for m in masks():
            mask.copy_(m)
            g.replay()
            ref = F.scaled_dot_product_attention(q, k, v, attn_mask=m.to(device))
            assert_close(out.float(), ref.float(), "masked attention replay")
