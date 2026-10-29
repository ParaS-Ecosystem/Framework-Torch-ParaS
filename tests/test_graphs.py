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


class _Mlp(torch.nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.norm = torch.nn.RMSNorm(d)
        self.up = torch.nn.Linear(d, 4 * d)
        self.down = torch.nn.Linear(4 * d, d)

    def forward(self, x):
        for _ in range(3):
            x = x + self.down(F.silu(self.up(self.norm(x))))
        return x


def _replay_reports():
    return [r.replay for r in torch_paras.optimizer.reports() if r.replay]


def _reset():
    torch._dynamo.reset()
    torch_paras.optimizer.reset_reports()


def test_compiled_replay_matches_eager(device):
    _require_graphs(device)
    _reset()
    torch.manual_seed(0)
    m = _Mlp().to(device).eval()
    c = torch_paras.compile(m, graphs=True)
    with torch.no_grad():
        for _ in range(4):
            x = torch.randn(2, 64, device=device)
            assert torch.equal(c(x).cpu(), m(x).cpu())
    assert _replay_reports() == ["replayed"], _replay_reports()


def test_compiled_replay_under_inference_mode(device):
    _require_graphs(device)
    _reset()
    torch.manual_seed(0)
    m = _Mlp().to(device).eval()
    c = torch_paras.compile(m, graphs=True)
    with torch.inference_mode():
        x = torch.randn(2, 64, device=device)
        for _ in range(4):
            assert torch.equal(c(x).cpu(), m(x).cpu())
            x.mul_(1.5)
        for _ in range(2):
            y = torch.randn(2, 64, device=device)
            assert torch.equal(c(y).cpu(), m(y).cpu())
    assert _replay_reports() == ["replayed"], _replay_reports()


def test_compiled_replay_copies_mutated_inputs_back(device):
    _require_graphs(device)
    _reset()

    def f(x, acc):
        acc.add_(x * 3)
        return acc.sum(0) + x.mean()

    c = torch_paras.compile(f, graphs=True)
    acc, ref = torch.zeros(4, 8, device=device), torch.zeros(4, 8)
    with torch.no_grad():
        for _ in range(4):
            x = torch.randn(4, 8)
            out = c(x.to(device), acc)
            want = f(x, ref)
            assert_close(out, want, "output")
            assert_close(acc, ref, "mutated input")
    assert _replay_reports() == ["replayed"], _replay_reports()


def test_compiled_replay_moved_parameter(device):
    _require_graphs(device)
    _reset()
    torch.manual_seed(0)
    m = _Mlp().to(device).eval()
    c = torch_paras.compile(m, graphs=True)
    x = torch.randn(2, 64, device=device)
    with torch.no_grad():
        for _ in range(3):
            c(x)
        m.up.weight.data = m.up.weight.data * 2
        assert torch.equal(c(x).cpu(), m(x).cpu())


def test_compiled_replay_fallbacks(device):
    _require_graphs(device)
    torch.manual_seed(0)
    m = _Mlp().to(device)

    def check(fn, inputs, expected, grad=False):
        _reset()
        c = torch_paras.compile(fn, graphs=True)
        with torch.set_grad_enabled(grad):
            for x in inputs:
                torch.manual_seed(1)
                out = c(x)
                torch.manual_seed(1)
                ref = fn(x)
                assert_close(out, ref, expected)
        reasons = _replay_reports()
        assert reasons and all(expected in r for r in reasons), (expected, reasons)

    xs = [torch.randn(2, 64, device=device) for _ in range(3)]
    check(m, xs, "needs autograd", grad=True)
    check(lambda x: F.dropout(m(x), 0.5, training=True), xs, "random operators")
    check(lambda x: F.layer_norm(m(x), (64,)), xs, "replayed")
    eye = torch.eye(8, device=device) * 4
    check(lambda x: torch.linalg.inv(m(x).view(2, 8, 8) + eye), xs, "capture failed")
    _reset()
    c = torch_paras.compile(m, graphs=True, dynamic=True)
    with torch.no_grad():
        for n in (3, 5, 7):
            x = torch.randn(n, 64, device=device)
            assert_close(c(x), m(x), "dynamic")
    assert any("no piece without changing sizes" in r for r in _replay_reports()), _replay_reports()


def test_compiled_replay_static_pieces(device):
    _require_graphs(device)
    _reset()
    torch.manual_seed(0)
    pre, post = _Mlp().to(device).eval(), _Mlp().to(device).eval()

    def step(x, cache):
        q = pre(x)
        att = torch.softmax(q @ cache.transpose(0, 1) / 8, dim=-1) @ cache
        return post(att + x)

    c = torch_paras.compile(step, graphs=True)
    with torch.no_grad():
        for n in (3, 5, 7, 9, 11, 13, 15, 17):
            x, cache = torch.randn(1, 64, device=device), torch.randn(n, 64, device=device)
            assert torch.equal(c(x, cache).cpu(), step(x, cache).cpu()), n
    assert any("in 2 pieces" in r for r in _replay_reports()), _replay_reports()


def test_compiled_replay_attention_without_dropout(device):
    _require_graphs(device)
    _reset()
    torch.manual_seed(0)
    lin = torch.nn.Linear(64, 3 * 64).to(device)

    def attend(x):
        q, k, v = lin(x).view(1, 16, 3, 4, 16).permute(2, 0, 3, 1, 4)
        return F.scaled_dot_product_attention(q, k, v, is_causal=True).sum(1)

    c = torch_paras.compile(attend, graphs=True)
    with torch.no_grad():
        for _ in range(3):
            x = torch.randn(1, 16, 64, device=device)
            assert torch.equal(c(x).cpu(), attend(x).cpu())
    assert _replay_reports() == ["replayed"], _replay_reports()


def test_compiled_replay_piece_with_tuple_input(device):
    _require_graphs(device)
    _reset()
    torch.manual_seed(0)
    norm, post = torch.nn.RMSNorm(64).to(device), _Mlp().to(device).eval()

    def step(x, cache):
        h = norm(x)
        att = torch.softmax(h @ cache.transpose(0, 1), dim=-1) @ cache
        return post(att + h * 2 + torch.cat([h[..., :32] * 3, h[..., 32:]], dim=-1))

    c = torch_paras.compile(step, graphs=True)
    with torch.no_grad():
        for n in (3, 5, 7, 9, 11, 13, 15, 17):
            x, cache = torch.randn(1, 64, device=device), torch.randn(n, 64, device=device)
            assert torch.equal(c(x, cache).cpu(), step(x, cache).cpu()), n


def test_compiled_replay_pieces_with_steady_symbolic_size(device):
    _require_graphs(device)
    _reset()
    torch.manual_seed(0)
    pre, post = _Mlp().to(device).eval(), _Mlp().to(device).eval()

    def step(x, cache):
        q = pre(x)
        att = torch.softmax(q @ cache.transpose(0, 1) / 8, dim=-1) @ cache
        return post(att + x)

    c = torch_paras.compile(step, graphs=True, dynamic=True)
    from torch_paras.optimizer import replay
    replays = replay.stats["replays"]
    with torch.no_grad():
        for b in (2, 3):
            for n in range(3, 12):
                x, cache = torch.randn(b, 64, device=device), torch.randn(n, 64, device=device)
                assert torch.equal(c(x, cache).cpu(), step(x, cache).cpu()), (b, n)
    assert any("in 2 pieces" in r for r in _replay_reports()), _replay_reports()
    assert replay.stats["replays"] > replays, "no piece was replayed"


def test_free_of_released_pool_during_capture(device):
    _require_graphs(device)
    x = torch.randn(4096, device=device)
    with torch.no_grad():
        x * 2
        g1 = graphs.Graph()
        with graphs.graph(g1, device=device):
            y = x * 2
        g1.reset()
        g2 = graphs.Graph()
        with graphs.graph(g2, device=device):
            z = x + 1
            del y
        g2.replay()
        assert_close(z, x.cpu() + 1, "replay after a deferred free")
        g2.reset()
        del z
        torch_paras.empty_cache()
