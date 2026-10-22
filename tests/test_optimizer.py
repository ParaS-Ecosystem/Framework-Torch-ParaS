"""Graph optimizer (torch.compile backend "paras"): fusion passes, training
and inference parity with eager, residency reports and strict mode."""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_paras
from torch_paras import optimizer

from common import assert_close


def _reset():
    torch._dynamo.reset()
    optimizer.reset_reports()


def _fusions():
    total = {}
    for r in optimizer.reports():
        for k, v in r.fusions.items():
            total[k] = total.get(k, 0) + v
    return total


def _close(actual, expected, tol, name):
    actual = actual.detach().cpu().float()
    expected = expected.detach().cpu().float()
    assert actual.shape == expected.shape, f"{name}: {actual.shape} vs {expected.shape}"
    scale = expected.abs().max().clamp_min(1.0)
    err = ((actual - expected).abs().max() / scale).item()
    assert err <= tol, f"{name}: relative error {err:.3e} > {tol:.1e}"


class RMSNorm(nn.Module):
    def __init__(self, hidden, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden) + 0.1 * torch.randn(hidden))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class SwiGLU(nn.Module):
    def __init__(self, hidden, inter):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class Block(nn.Module):
    """Pre-norm decoder block: RMSNorm, RoPE attention with GQA, SwiGLU MLP."""

    def __init__(self, hidden=64, heads=4, kv_heads=2, inter=128):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, hidden // heads
        self.input_layernorm = RMSNorm(hidden)
        self.post_attention_layernorm = RMSNorm(hidden)
        self.q_proj = nn.Linear(hidden, heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(heads * self.head_dim, hidden, bias=False)
        self.mlp = SwiGLU(hidden, inter)

    def forward(self, x, cos, sin):
        b, s, _ = x.shape
        h = self.input_layernorm(x)
        q = self.q_proj(h).view(b, s, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        rep = self.heads // self.kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o_proj(a.transpose(1, 2).reshape(b, s, -1))
        return x + self.mlp(self.post_attention_layernorm(x))


def _rope(seq, head_dim, device, dtype):
    inv = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    freqs = torch.outer(torch.arange(seq, dtype=torch.float32), inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(device, dtype), emb.sin().to(device, dtype)


def test_backend_registered(device):
    assert "paras" in torch._dynamo.list_backends(exclude_tags=())


def test_rms_norm_fused_and_matches_eager(device):
    for dtype in (torch.float32, torch.bfloat16):
        for shape in ((3, 5, 48), (2, 128, 2048), (1, 1, 2048)):
            _reset()
            norm = RMSNorm(shape[-1]).to(device, dtype)
            x = torch.randn(*shape).to(device, dtype)
            ref = norm(x)
            out = torch.compile(norm, backend="paras")(x)
            assert _fusions() == {"rms_norm": 1}, (dtype, _fusions())
            assert torch.equal(out.cpu(), ref.cpu()), f"rms_norm {dtype} {shape} not bit-exact"


def test_rms_norm_not_fused_when_semantics_differ(device):
    _reset()
    norm = RMSNorm(32).to(device, torch.bfloat16)
    x = torch.randn(4, 32, device=device)
    ref = norm(x)
    out = torch.compile(norm, backend="paras")(x)
    assert _fusions() == {}, _fusions()
    assert_close(out, ref, "rms_norm mixed dtypes")

    class ColumnNorm(RMSNorm):
        def forward(self, h):
            hf = h.to(torch.float32)
            v = hf.pow(2).mean(0, keepdim=True)
            return self.weight * (hf * torch.rsqrt(v + self.variance_epsilon)).to(h.dtype)

    _reset()
    col = ColumnNorm(32).to(device)
    x = torch.randn(32, 32, device=device)
    out = torch.compile(col, backend="paras")(x)
    assert _fusions() == {}, _fusions()
    assert_close(out, col(x), "column norm")


def test_rms_norm_shared_intermediate_not_fused(device):
    class Leaky(RMSNorm):
        def forward(self, h):
            hf = h.to(torch.float32)
            var = hf.pow(2).mean(-1, keepdim=True)
            y = self.weight * (hf * torch.rsqrt(var + self.variance_epsilon)).to(h.dtype)
            return y, var

    _reset()
    m = Leaky(16).to(device)
    x = torch.randn(2, 16, device=device)
    y, var = torch.compile(m, backend="paras")(x)
    ry, rvar = m(x)
    assert _fusions() == {}, _fusions()
    assert_close(y, ry, "leaky y")
    assert_close(var, rvar, "leaky var")


def _silu_fusable(device, dtype):
    """SwiGLU is fused only where eager silu runs the ParaS kernel; where
    CLAP-DNN runs it in the vendor library it is deliberately left unfused."""
    C = torch_paras._C
    if not C.clap_dnn_enabled():
        return True
    index = torch.device(device).index or 0
    return not C.clap_dnn_activation_uses_vendor(index, str(dtype).replace("torch.", ""))


def test_swiglu_fused_and_matches_eager(device):
    for dtype in (torch.float32, torch.bfloat16):
        _reset()
        mlp = SwiGLU(32, 64).to(device, dtype)
        x = torch.randn(2, 7, 32).to(device, dtype)
        out = torch.compile(mlp, backend="paras")(x)
        expected = {"swiglu": 1} if _silu_fusable(device, dtype) else {}
        assert _fusions() == expected, (dtype, _fusions())
        assert torch.equal(out.cpu(), mlp(x).cpu()), f"swiglu {dtype} not bit-exact"


def test_swiglu_broadcast_not_fused(device):
    _reset()

    def f(g, u):
        return F.silu(g) * u

    g = torch.randn(4, 8, device=device)
    u = torch.randn(8, device=device)
    out = torch.compile(f, backend="paras")(g, u)
    assert _fusions() == {}, _fusions()
    assert_close(out, f(g, u), "silu broadcast")


def test_rotate_half_fused_and_exact(device):
    _reset()
    x = torch.randn(2, 3, 5, 16, device=device)
    out = torch.compile(rotate_half, backend="paras")(x)
    assert _fusions() == {"rotate_half": 1}, _fusions()
    assert_close(out, rotate_half(x), "rotate_half")

    _reset()

    def not_half(t):
        return torch.cat((-t[..., 4:], t[..., :4]), dim=-1)

    out = torch.compile(not_half, backend="paras")(x)
    assert _fusions() == {}, _fusions()
    assert_close(out, not_half(x), "not_half")


def test_block_inference_parity_and_reports(device):
    _reset()
    torch.manual_seed(0)
    block = Block().to(device)
    x = torch.randn(2, 9, 64, device=device)
    cos, sin = _rope(9, 16, device, torch.float32)
    with torch.no_grad():
        ref = block(x, cos, sin)
        out = torch.compile(block, backend="paras")(x, cos, sin)
    _close(out, ref, 1e-5, "block inference")
    expected = {"rms_norm": 2, "rotate_half": 2, "rope": 2, "add_rms_norm": 1}
    if _silu_fusable(device, torch.float32):
        expected["swiglu"] = 1
    assert _fusions() == expected, _fusions()
    for r in optimizer.reports():
        assert r.fully_on_paras, r.summary()


def test_block_training_parity(device):
    _reset()
    torch.manual_seed(0)
    eager = Block().to(device)
    compiled_model = copy.deepcopy(eager)
    compiled = torch.compile(compiled_model, backend="paras")
    opt_e = torch.optim.AdamW(eager.parameters(), lr=1e-3)
    opt_c = torch.optim.AdamW(compiled_model.parameters(), lr=1e-3)
    cos, sin = _rope(12, 16, device, torch.float32)
    for step in range(4):
        x = torch.randn(2, 12, 64, device=device)
        target = torch.randn(2, 12, 64, device=device)
        loss_e = F.mse_loss(eager(x, cos, sin), target)
        loss_c = F.mse_loss(compiled(x, cos, sin), target)
        _close(loss_c, loss_e, 1e-5, f"loss step {step}")
        opt_e.zero_grad()
        opt_c.zero_grad()
        loss_e.backward()
        loss_c.backward()
        for (name, pe), (_, pc) in zip(eager.named_parameters(), compiled_model.named_parameters()):
            assert pc.grad is not None, f"{name}: no gradient through the compiled graph"
            _close(pc.grad, pe.grad, 1e-4, f"grad {name} step {step}")
        opt_e.step()
        opt_c.step()
    for r in optimizer.reports():
        assert "backward" in r.residency, r.summary()
        assert r.fully_on_paras, r.summary()


def test_dynamic_sequence_lengths(device):
    _reset()
    torch.manual_seed(0)
    block = Block().to(device)
    compiled = torch.compile(block, backend="paras")
    for seq in (5, 11, 17, 6):
        x = torch.randn(1, seq, 64, device=device)
        cos, sin = _rope(seq, 16, device, torch.float32)
        with torch.no_grad():
            _close(compiled(x, cos, sin), block(x, cos, sin), 1e-5, f"seq {seq}")


def _expert_loop(experts):
    """An MoE's eager expert loop: Dynamo cannot trace the data-dependent
    branch, so each expert MLP is a nested frame with a varying token count."""
    def moe(x, counts):
        out, start = [], 0
        for i, n in enumerate(counts.cpu().numpy()):
            if n == 0:
                continue
            out.append(experts[i](x[start:start + n]))
            start += n
        return torch.cat(out)
    return moe


def test_small_dynamic_graph_runs_eagerly(device):
    torch.manual_seed(0)
    experts = nn.ModuleList([SwiGLU(16, 32) for _ in range(4)]).to(device)
    counts = torch.tensor([3, 0, 5, 2])
    x = torch.randn(int(counts.sum()), 16, device=device)
    for limit, eager_expected in ((None, True), (0, False)):
        _reset()
        moe = _expert_loop(experts)
        compiled = torch_paras.compile(moe, min_dynamic_graph_ops=limit)
        with torch.no_grad():
            for _ in range(3):
                out = compiled(x, counts)
        assert torch.equal(out.cpu(), moe(x, counts).cpu()), f"limit {limit}: result differs"
        eager = [r for r in optimizer.reports() if r.runs_eagerly == "small dynamic-shape graph"]
        assert bool(eager) == eager_expected, [r.summary() for r in optimizer.reports()]
    _reset()
    compiled = torch_paras.compile(_expert_loop(experts))
    with torch.no_grad():
        for n in ([3, 0, 5, 2], [1, 4, 2, 3], [2, 2, 2, 4], [5, 1, 1, 3]):
            compiled(x, torch.tensor(n))
        graphs = len(optimizer.reports())
        compiled(x, torch.tensor([4, 3, 2, 1]))
    assert len(optimizer.reports()) == graphs, "a skipped frame was compiled again"


def test_recompile_limit_runs_eagerly(device):
    """A frame whose guards specialise on a Python value (here a list's
    length, like a growing KV cache) keeps asking Dynamo for new graphs. Once
    Dynamo refuses one (recompile limit), the frame is handed back (its
    callees may still be compiled); a frame that needs exactly the limit
    keeps its graphs. Results are unchanged either way."""
    def helper(x):
        return F.silu(x) * x

    def f(x, items):
        return helper(x) * len(items)

    x = torch.randn(4, 8, device=device)
    eager_reason = "recompile limit reached"
    for lengths, at_limit, eager_expected in (
            (range(1, 4), None, False),
            (range(1, 8), None, True),
            (range(1, 8), False, False)):
        _reset()
        with torch._dynamo.config.patch(recompile_limit=3):
            compiled = torch_paras.compile(f, eager_at_recompile_limit=at_limit)
            for n in list(lengths) + [1, 2]:
                assert torch.equal(compiled(x, [0] * n).cpu(), f(x, [0] * n).cpu()), n
            f_graphs = sum(r.frame.startswith("f (") for r in optimizer.reports())
            if eager_expected:
                compiled(x, [0] * 9)
        eager = [r for r in optimizer.reports() if r.runs_eagerly == eager_reason]
        assert bool(eager) == eager_expected, (list(lengths), [r.summary() for r in optimizer.reports()])
        if eager_expected:
            assert sum(r.frame.startswith("f (") for r in optimizer.reports()) == f_graphs, \
                "a handed-back frame was compiled again"


class _Layer(nn.Module):
    """A transformer-style layer: specialises on its own index and breaks
    the graph (like an MoE's host read), so each call is its own frame."""

    def __init__(self, layer_idx):
        super().__init__()
        self.proj = nn.Linear(8, 8)
        self.proj.layer_idx = layer_idx

    def forward(self, x):
        x = self.proj(x) * (self.proj.layer_idx + 1)
        torch._dynamo.graph_break()
        return x + self.proj.layer_idx


class _Stack(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(i) for i in range(n)])
        self.experts = nn.ModuleList([nn.Linear(8, 8) for _ in range(n)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return sum(e(x) for e in self.experts)


def test_specialize_repeated_blocks(device):
    torch.manual_seed(0)
    x = torch.randn(3, 8, device=device)
    for specialize, eager_expected in ((True, False), (False, True)):
        _reset()
        stack = _Stack(4).to(device)
        ref = stack(x)
        with torch._dynamo.config.patch(recompile_limit=2):
            compiled = torch_paras.compile(stack, specialize_blocks=specialize)
            raised = torch._dynamo.config.recompile_limit >= 32
            for _ in range(3):
                out = compiled(x)
        assert raised == specialize, "recompile budget"
        assert torch.equal(out.cpu(), ref.cpu()), specialize
        own = ["forward" in vars(layer) for layer in stack.layers]
        assert own == [specialize] * 4, own
        assert not any("forward" in vars(e) for e in stack.experts), "experts differ only in weights"
        codes = {id(layer.forward.__code__) for layer in stack.layers}
        assert len(codes) == (4 if specialize else 1), len(codes)
        at_limit = [r for r in optimizer.reports() if r.runs_eagerly == "recompile limit reached"]
        assert bool(at_limit) == eager_expected, [r.summary() for r in optimizer.reports()]
        if specialize:
            assert optimizer.unspecialize(stack) == 4
            assert not any("forward" in vars(layer) for layer in stack.layers)


def test_identical_graphs_analysed_once(device):
    import sys
    residency = sys.modules["torch_paras.optimizer.residency"]
    sys.modules["torch_paras.optimizer.backend"]._analyses.clear()
    calls = []
    analyse = residency.analyse

    def counting(gm):
        calls.append(gm)
        return analyse(gm)

    def f(x):
        return torch.relu(x) * 2.0

    def g(x):
        return torch.special.zeta(x, x + 1.0)

    residency.analyse = counting
    try:
        x = torch.rand(8, device=device) + 1.0
        runs = []
        for inp in (x, x, torch.rand(5, device=device)):
            _reset()
            torch_paras.compile(f)(inp)
            runs.append((len(calls), optimizer.reports()[0].residency))
        assert runs[1] == runs[0], runs
        assert runs[2][0] > runs[1][0], runs
        _reset()
        torch_paras.compile(g)(x)
        n = len(calls)
        _reset()
        try:
            torch_paras.compile(g, strict=True)(x)
        except Exception as e:
            assert "special_zeta" in str(e), e
        else:
            raise AssertionError("strict mode accepted a reused analysis with a CPU fallback")
        assert len(calls) == n, "strict check re-traced"
    finally:
        residency.analyse = analyse


def test_strict_allows_tensor_constants(device):
    _reset()

    def f(x):
        return x * torch.tensor([2.0], device=x.device)

    x = torch.randn(4, device=device)
    out = torch_paras.compile(f, strict=True)(x)
    assert torch.equal(out.cpu(), f(x).cpu())
    assert all(r.fully_on_paras for r in optimizer.reports()), [r.summary() for r in optimizer.reports()]


def test_fusions_option(device):
    _reset()
    mlp = SwiGLU(16, 32).to(device)
    x = torch.randn(3, 16, device=device)
    out = torch_paras.compile(mlp, fusions=())(x)
    assert _fusions() == {}, _fusions()
    _close(out, mlp(x), 1e-5, "no fusions")
    try:
        torch_paras.compile(mlp, fusions=("nope",))
    except ValueError:
        pass
    else:
        raise AssertionError("unknown fusion name was accepted")


def test_strict_rejects_cpu_fallback(device):
    _reset()

    def f(x):
        return torch.special.zeta(x, x + 1.0)

    x = torch.rand(8, device=device) + 1.0
    try:
        torch_paras.compile(f, strict=True)(x)
    except Exception as e:
        assert "special_zeta" in str(e), e
    else:
        raise AssertionError("strict mode compiled a graph with a CPU-fallback op")
    _reset()
    out = torch_paras.compile(f)(x)
    assert any("special_zeta" in op for r in optimizer.reports() for op in r.cpu_fallback_ops)
    assert_close(out, f(x.cpu()), "zeta non-strict")


def test_compiled_training_stays_on_device(device):
    if not torch_paras.is_gpu_device(torch.device(device).index or 0):
        return
    _reset()
    torch.manual_seed(0)
    block = Block().to(device)
    compiled = torch.compile(block, backend="paras")
    x = torch.randn(2, 8, 64, device=device)
    cos, sin = _rope(8, 16, device, torch.float32)
    compiled(x, cos, sin).sum().backward()
    torch_paras.reset_runtime_stats()
    compiled(x, cos, sin).sum().backward()
    stats = torch_paras.runtime_stats()
    assert stats["cpu_fallbacks"] == {}, stats
    assert stats["device_to_host_copies"] == 0 and stats["host_to_device_copies"] == 0, stats


def test_fusion_keeps_unused_inplace_ops(device):
    _reset()

    def f(x, buf):
        buf.mul_(2.0)
        return rotate_half(x)

    x = torch.randn(2, 8, device=device)
    buf = torch.ones(3, device=device)
    out = torch.compile(f, backend="paras")(x, buf)
    assert _fusions() == {"rotate_half": 1}, _fusions()
    assert_close(out, rotate_half(x), "rotate_half")
    assert_close(buf, torch.full((3,), 2.0), "in-place mul_ survived")


def test_no_fusion_across_inplace_update(device):
    def f(x, w):
        var = x.pow(2).mean(-1, keepdim=True)
        x.add_(1.0)
        return w * (x * torch.rsqrt(var + 1e-6))

    _reset()
    x = torch.randn(4, 16, device=device)
    w = torch.randn(16, device=device)
    ref = f(x.clone(), w)
    out = torch.compile(f, backend="paras")(x.clone(), w)
    assert _fusions() == {}, _fusions()
    assert_close(out, ref, "rms_norm across x.add_")


def _repeat_kv(x, n_rep):
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def test_gqa_attention_fused_and_exact(device):
    torch.manual_seed(0)

    def attend(q, k, v, mask):
        return F.scaled_dot_product_attention(q, _repeat_kv(k, 4), _repeat_kv(v, 4), attn_mask=mask)

    with torch.no_grad():
        for Lq, Lk in ((1, 77), (64, 64)):
            _reset()
            q = torch.randn(1, 8, Lq, 64, device=device, dtype=torch.bfloat16)
            k = torch.randn(1, 2, Lk, 64, device=device, dtype=torch.bfloat16)
            v = torch.randn(1, 2, Lk, 64, device=device, dtype=torch.bfloat16)
            mask = torch.zeros(1, 1, Lq, Lk, device=device, dtype=torch.bfloat16)
            mask[..., Lk // 2:] = torch.finfo(torch.bfloat16).min
            ref = attend(q, k, v, mask)
            out = torch_paras.compile(attend)(q, k, v, mask)
            assert _fusions().get("gqa_attention") == 1, _fusions()
            assert torch.equal(out.cpu(), ref.cpu()), (Lq, Lk)


def test_sdpa_bool_mask_fused_and_exact(device):
    torch.manual_seed(0)

    def attend(q, k, v, mask):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)

    for Lq, Lk in ((1, 77), (64, 64)):
        q = torch.randn(1, 8, Lq, 64, device=device, dtype=torch.bfloat16)
        k = torch.randn(1, 2, Lk, 64, device=device, dtype=torch.bfloat16)
        v = torch.randn(1, 2, Lk, 64, device=device, dtype=torch.bfloat16)
        mask = torch.ones(1, 1, Lq, Lk, device=device, dtype=torch.bool).tril(Lk - Lq)
        mask[..., :3] = False
        with torch.no_grad():
            _reset()
            ref = attend(q, k, v, mask)
            out = torch_paras.compile(attend)(q, k, v, mask)
            assert _fusions().get("sdpa_bool_mask") == 1, _fusions()
            assert torch.equal(out.cpu(), ref.cpu()), (Lq, Lk)
    _reset()
    qg = q.clone().requires_grad_()
    out = torch_paras.compile(attend)(qg, k, v, mask)
    assert "sdpa_bool_mask" not in _fusions(), _fusions()
    assert torch.equal(out.detach().cpu(), attend(qg, k, v, mask).detach().cpu())



def _rotate(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def _apply_rope(q, k, cos, sin):
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return q * cos + _rotate(q) * sin, k * cos + _rotate(k) * sin


def _apply_partial_rope(q, cos, sin, rot=32):
    q_rot, q_pass = q[..., :rot], q[..., rot:]
    q_rot = q_rot * cos.unsqueeze(1) + _rotate(q_rot) * sin.unsqueeze(1)
    return torch.cat((q_rot, q_pass), dim=-1)


def test_rope_fused_and_exact(device):
    torch.manual_seed(0)
    for dtype in (torch.bfloat16, torch.float32):
        _reset()
        q = torch.randn(2, 8, 5, 64, device=device, dtype=dtype).transpose(1, 2)
        k = torch.randn(2, 2, 8, 64, device=device, dtype=dtype)
        cos = torch.randn(2, 8, 64, device=device, dtype=dtype)
        sin = torch.randn(2, 8, 64, device=device, dtype=dtype)
        with torch.no_grad():
            ref = _apply_rope(q, k, cos, sin)
            out = torch_paras.compile(_apply_rope)(q, k, cos, sin)
        assert _fusions().get("rope") == 2, _fusions()
        for o, r in zip(out, ref):
            assert torch.equal(o.cpu(), r.cpu()), dtype
        _reset()
        with torch.no_grad():
            ref = _apply_partial_rope(q, cos[..., :32], sin[..., :32])
            out = torch_paras.compile(_apply_partial_rope)(q, cos[..., :32], sin[..., :32])
        assert _fusions().get("rope") == 1, _fusions()
        assert torch.equal(out.cpu(), ref.cpu()), dtype


def test_rope_training_gradient(device):
    torch.manual_seed(0)
    _reset()
    q = torch.randn(1, 4, 6, 32, device=device, requires_grad=True)
    k = torch.randn(1, 4, 6, 32, device=device, requires_grad=True)
    cos, sin = torch.randn(1, 6, 32, device=device), torch.randn(1, 6, 32, device=device)
    loss = lambda f: sum((t * t).sum() for t in f(q, k, cos, sin))
    ref = torch.autograd.grad(loss(_apply_rope), (q, k))
    got = torch.autograd.grad(loss(torch_paras.compile(_apply_rope)), (q, k))
    assert _fusions().get("rope") == 2, _fusions()
    for g, r in zip(got, ref):
        _close(g, r, 1e-5, "rope grad")


class _ResidualNorm(nn.Module):
    """h = residual + x; (h, RMSNorm(h)) -- a decoder layer's residual add and
    the norm after it."""

    def __init__(self, hidden):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden))

    def forward(self, residual, x):
        h = residual + x
        hf = h.to(torch.float32)
        var = hf.pow(2).mean(-1, keepdim=True)
        norm = self.weight * (hf * torch.rsqrt(var + 1e-6)).to(h.dtype)
        return h * 2, norm


def test_add_rms_norm_fused_and_exact(device):
    torch.manual_seed(0)
    for dtype in (torch.bfloat16, torch.float32):
        for hidden in (48, 2048):
            _reset()
            m = _ResidualNorm(hidden).to(device=device, dtype=dtype)
            r = torch.randn(3, 4, hidden, device=device, dtype=dtype)
            x = torch.randn(3, 4, hidden, device=device, dtype=dtype)
            with torch.no_grad():
                ref = m(r, x)
                out = torch_paras.compile(m)(r, x)
            assert _fusions().get("add_rms_norm") == 1, _fusions()
            for o, e in zip(out, ref):
                assert torch.equal(o.cpu(), e.cpu()), (dtype, hidden)


def test_add_rms_norm_training_gradient(device):
    torch.manual_seed(0)
    _reset()
    m = _ResidualNorm(256).to(device)
    r = torch.randn(2, 3, 256, device=device, requires_grad=True)
    x = torch.randn(2, 3, 256, device=device, requires_grad=True)

    def loss(f):
        h, n = f(r, x)
        return (h * h).sum() + (n * n * 0.5).sum()

    ref = torch.autograd.grad(loss(m), (r, x, m.weight))
    got = torch.autograd.grad(loss(torch_paras.compile(m)), (r, x, m.weight))
    assert _fusions().get("add_rms_norm") == 1, _fusions()
    for g, e in zip(got, ref):
        _close(g, e, 1e-4, "add_rms_norm grad")


def test_fused_nodes_have_example_values(device):
    import sys
    passes = sys.modules["torch_paras.optimizer.passes"]
    seen = []

    def backend(gm, example_inputs):
        passes.run(gm, set(passes.PASSES))
        for n in gm.graph.nodes:
            if n.op == "call_function" and getattr(n.target, "namespace", "") == "paras":
                seen.append((str(n.target), n.meta.get("example_value") is not None))
        return gm.forward

    torch.manual_seed(0)
    m = _ResidualNorm(64).to(device)
    cos, sin = torch.randn(1, 3, 64, device=device), torch.randn(1, 3, 64, device=device)

    def step(res, x):
        h, n = m(res, x)
        q, k = _apply_rope(n.view(1, 3, 1, 64).transpose(1, 2), n.view(1, 3, 1, 64).transpose(1, 2), cos, sin)
        return h + q.sum() + k.sum()

    torch._dynamo.reset()
    with torch.no_grad():
        torch.compile(step, backend=backend)(torch.randn(1, 3, 64, device=device),
                                             torch.randn(1, 3, 64, device=device))
    names = {t for t, _ in seen}
    assert {"paras.add_rms_norm.default", "paras.rope.default"} <= names, seen
    assert all(ok for _, ok in seen), seen
