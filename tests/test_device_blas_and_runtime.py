"""Device GEMM through CLAP, execution telemetry / strict mode, and the
nonzero / scatter_add / argsort kernels, checked against CPU references."""

import torch
import torch_paras

from common import assert_close, check_op


def _is_gpu(device):
    return torch_paras.is_gpu_device(torch.device(device).index or 0)


def _assert_gemm_close(out, ref, k, name):
    """Half-precision GEMMs round once from an fp32 accumulator, so compare
    against an fp32 reference with a tolerance that scales with k."""
    out = out.cpu().float()
    ref = ref.float()
    assert out.shape == ref.shape, f"{name}: shape {out.shape} vs {ref.shape}"
    scale = ref.abs().max().clamp_min(1.0)
    err = (out - ref).abs().max() / scale
    assert err < 2e-2, f"{name}: max relative error {err:.3e}"


def test_mm_dtypes_and_layouts(device):
    for dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
        a = torch.randn(37, 53).to(dtype)
        b = torch.randn(53, 29).to(dtype)
        variants = {
            "plain": (a, b),
            "a.t": (a.t().contiguous().t(), b),
            "b.t": (a, b.t().contiguous().t()),
            "slices": (torch.randn(37, 80).to(dtype)[:, 5:58], torch.randn(60, 29).to(dtype)[3:56]),
        }
        for name, (x, y) in variants.items():
            out = torch.mm(x.to(device), y.to(device))
            ref = x.double() @ y.double()
            if dtype in (torch.float16, torch.bfloat16):
                _assert_gemm_close(out, ref, 53, f"mm {dtype} {name}")
            else:
                assert_close(out, ref.to(dtype), f"mm {dtype} {name}")


def test_mm_degenerate_shapes(device):
    for m, k, n in ((0, 4, 3), (3, 4, 0), (3, 0, 5), (1, 1, 1)):
        a = torch.randn(m, k)
        b = torch.randn(k, n)
        check_op(f"mm {m}x{k}x{n}", lambda: a @ b, lambda: a.to(device) @ b.to(device))


def test_addmm_alpha_beta_broadcast(device):
    a = torch.randn(16, 24)
    b = torch.randn(24, 8)
    for bias in (torch.randn(8), torch.randn(16, 1), torch.randn(16, 8)):
        check_op("addmm",
                 lambda: torch.addmm(bias, a, b, beta=0.5, alpha=2.0),
                 lambda: torch.addmm(bias.to(device), a.to(device), b.to(device),
                                     beta=0.5, alpha=2.0))


def test_addmm_beta_zero_ignores_nan(device):
    a = torch.randn(4, 5)
    b = torch.randn(5, 3)
    bias = torch.full((4, 3), float("nan"))
    out = torch.addmm(bias.to(device), a.to(device), b.to(device), beta=0.0)
    assert_close(out, a @ b, "addmm beta=0")


def test_addmm_inplace(device):
    a = torch.randn(6, 7)
    b = torch.randn(7, 6)
    c = torch.randn(6, 6)
    dc = c.to(device)
    dc.addmm_(a.to(device), b.to(device), beta=1.5, alpha=0.5)
    assert_close(dc, c.addmm(a, b, beta=1.5, alpha=0.5), "addmm_")


def test_bmm_layouts(device):
    a = torch.randn(3, 9, 11)
    b = torch.randn(3, 11, 7)
    check_op("bmm", lambda: torch.bmm(a, b), lambda: torch.bmm(a.to(device), b.to(device)))
    bt = torch.randn(3, 7, 11).transpose(1, 2)
    check_op("bmm b.transpose", lambda: torch.bmm(a, bt),
             lambda: torch.bmm(a.to(device), bt.to(device)))
    shared = torch.randn(1, 11, 7).expand(3, 11, 7)
    check_op("bmm broadcast", lambda: torch.bmm(a, shared),
             lambda: torch.bmm(a.to(device), shared.to(device)))


def test_bmm_dtypes(device):
    for dtype in (torch.float64, torch.float16, torch.bfloat16):
        a = torch.randn(3, 9, 40).to(dtype)
        bt = torch.randn(3, 7, 40).to(dtype).transpose(1, 2)
        shared = torch.randn(1, 40, 7).to(dtype).expand(3, 40, 7)
        for name, b in (("b.transpose", bt), ("broadcast", shared)):
            out = torch.bmm(a.to(device), b.to(device))
            ref = a.double() @ b.double()
            if dtype == torch.float64:
                assert_close(out, ref, f"bmm {dtype} {name}")
            else:
                _assert_gemm_close(out, ref, 40, f"bmm {dtype} {name}")
                assert out.dtype == dtype


def test_linear_forward_backward(device):
    for shape in ((5, 12), (2, 7, 12)):
        x = torch.randn(*shape, requires_grad=True)
        w = torch.randn(9, 12, requires_grad=True)
        bias = torch.randn(9, requires_grad=True)
        dx, dw, db = (t.detach().to(device).requires_grad_() for t in (x, w, bias))
        ref = torch.nn.functional.linear(x, w, bias)
        out = torch.nn.functional.linear(dx, dw, db)
        assert_close(out, ref, f"linear {shape}")
        g = torch.randn_like(ref)
        ref.backward(g)
        out.backward(g.to(device))
        assert_close(dx.grad, x.grad, f"linear grad_input {shape}")
        assert_close(dw.grad, w.grad, f"linear grad_weight {shape}")
        assert_close(db.grad, bias.grad, f"linear grad_bias {shape}")


def test_gemm_stays_on_device(device):
    if not _is_gpu(device) or not torch_paras._C.clap_blas_enabled():
        return
    x = torch.randn(64, 128, device=device, requires_grad=True)
    w = torch.randn(256, 128, device=device, requires_grad=True)
    torch_paras.reset_runtime_stats()
    y = torch.nn.functional.linear(x, w)
    y.sum().backward()
    torch.bmm(x.detach().view(4, 16, 128), w.detach().t().expand(4, 128, 256))
    for dtype in (torch.float64, torch.float16, torch.bfloat16):
        a = x.detach().to(dtype)
        b = w.detach().to(dtype)
        torch.mm(a, b.t())
        torch.bmm(a.view(4, 16, 128), b.t().expand(4, 128, 256))
    stats = torch_paras.runtime_stats()
    assert stats["cpu_fallbacks"] == {}, stats
    assert stats["device_to_host_copies"] == 0, stats
    assert stats["host_to_device_copies"] == 0, stats


def test_runtime_stats_count_transfers(device):
    torch_paras.reset_runtime_stats()
    t = torch.arange(10, dtype=torch.float32).to(device)
    t.cpu()
    float(t.sum())
    stats = torch_paras.runtime_stats()
    assert stats["host_to_device_copies"] == 1, stats
    assert stats["host_to_device_bytes"] == 40, stats
    assert stats["device_to_host_copies"] == 1, stats
    assert stats["scalar_reads"] >= 1, stats


def test_strict_mode_rejects_cpu_fallback(device):
    x = torch.rand(8, device=device) + 0.5
    was_strict = torch_paras.is_strict()
    try:
        torch_paras.set_strict(False)
        torch_paras.reset_runtime_stats()
        torch.special.zeta(x, x)
        fallbacks = torch_paras.runtime_stats()["cpu_fallbacks"]
        assert any(op.startswith("aten::special_zeta") for op in fallbacks), fallbacks
        torch_paras.set_strict(True)
        try:
            torch.special.zeta(x, x)
        except RuntimeError as e:
            assert "strict mode" in str(e), e
        else:
            raise AssertionError("strict mode did not reject a CPU fallback")
    finally:
        torch_paras.set_strict(was_strict)


def test_nonzero(device):
    cases = [
        torch.tensor([0.0, 1.5, 0.0, -2.0, float("nan")]),
        (torch.randn(7, 300) > 0.3),
        torch.randint(-2, 3, (4, 5, 6)),
        torch.randn(6, 8).t(),
        torch.zeros(5, 5),
        torch.tensor(3.0),
        torch.tensor(0.0),
        torch.empty(0, 4),
        torch.randn(100, 1000).to(torch.bfloat16),
        torch.tensor([1e-320, 0.0], dtype=torch.float64),
    ]
    for i, t in enumerate(cases):
        check_op(f"nonzero[{i}]", lambda: torch.nonzero(t), lambda: torch.nonzero(t.to(device)))


def test_boolean_mask_indexing_without_fallback(device):
    x = torch.randn(64, 32)
    mask = torch.randn(64) > 0
    dx, dmask = x.to(device), mask.to(device)
    torch_paras.reset_runtime_stats()
    out = dx[dmask]
    y = dx.clone()
    y[dmask] = 0.0
    assert torch_paras.runtime_stats()["cpu_fallbacks"] == {}, torch_paras.runtime_stats()
    assert_close(out, x[mask], "x[mask]")
    ref = x.clone()
    ref[mask] = 0.0
    assert_close(y, ref, "x[mask] = 0")


def test_scatter_add(device):
    for dtype in (torch.float32, torch.float64, torch.bfloat16, torch.int64, torch.int32):
        self_ = torch.zeros(4, 6, dtype=dtype)
        index = torch.randint(0, 4, (5, 6))
        src = (torch.randn(5, 6) * 4).to(dtype)
        check_op(f"scatter_add {dtype}",
                 lambda: self_.scatter_add(0, index, src),
                 lambda: self_.to(device).scatter_add(0, index.to(device), src.to(device)))
    self_ = torch.randn(3, 5)
    index = torch.randint(0, 5, (2, 3))
    src = torch.randn(4, 6)
    check_op("scatter_add src>index",
             lambda: self_.scatter_add(1, index, src),
             lambda: self_.to(device).scatter_add(1, index.to(device), src.to(device)))


def test_scatter_src_larger_than_index(device):
    self_ = torch.zeros(3, 5)
    index = torch.tensor([[0, 4, 2], [1, 3, 0]])
    src = torch.arange(24, dtype=torch.float32).view(4, 6)
    check_op("scatter src>index",
             lambda: self_.scatter(1, index, src),
             lambda: self_.to(device).scatter(1, index.to(device), src.to(device)))


def test_argsort_long_rows_nan_ties(device):
    values = torch.randn(3, 10000)
    values[0, ::7] = float("nan")
    values[1, :5000] = 1.0
    for descending in (False, True):
        check_op(f"argsort long desc={descending}",
                 lambda: torch.argsort(values, dim=1, descending=descending, stable=True),
                 lambda: torch.argsort(values.to(device), dim=1, descending=descending, stable=True))


def test_argsort_exact_integer_and_double_keys(device):
    big = torch.tensor([2**40 + 3, 2**40 + 1, 2**40 + 2, -(2**33)], dtype=torch.int64)
    check_op("argsort int64", lambda: torch.argsort(big, stable=True),
             lambda: torch.argsort(big.to(device), stable=True))
    close = torch.tensor([1.0 + 2e-16, 1.0, 1.0 + 4e-16], dtype=torch.float64)
    check_op("argsort float64", lambda: torch.argsort(close, stable=True),
             lambda: torch.argsort(close.to(device), stable=True))


def test_argsort_middle_dim_noncontiguous(device):
    t = torch.randn(4, 300, 3).transpose(0, 2)
    check_op("argsort dim1", lambda: torch.argsort(t, dim=1, stable=True),
             lambda: torch.argsort(t.to(device), dim=1, stable=True))


def test_autograd_through_composite_ops(device):
    cases = {
        "contiguous": lambda t: t.transpose(0, 1).contiguous(),
        "reshape": lambda t: t.reshape(-1, t.shape[-1]),
        "reshape_noncontig": lambda t: t.transpose(0, 1).reshape(-1),
        "repeat_interleave_int": lambda t: t.repeat_interleave(2, dim=1),
        "repeat_interleave_tensor": lambda t: t.repeat_interleave(
            torch.tensor([1, 2, 3, 1], device=t.device), dim=1),
        "rms_norm": lambda t: torch.nn.functional.rms_norm(t, (t.shape[-1],), eps=1e-6),
        "sdpa": lambda t: torch.nn.functional.scaled_dot_product_attention(
            t[None], t[None], t[None], is_causal=True),
    }
    was_strict = torch_paras.is_strict()
    torch_paras.set_strict(True)
    try:
        for name, f in cases.items():
            x = torch.randn(3, 4, 6)
            xc = x.clone().requires_grad_()
            xd = x.clone().to(device).requires_grad_()
            yc = f(xc)
            g = torch.randn_like(yc)
            yc.backward(g)
            yd = f(xd)
            assert yd.grad_fn is not None, f"{name}: no autograd graph"
            yd.backward(g.to(device))
            assert_close(yd, yc.detach(), f"{name} forward")
            assert_close(xd.grad, xc.grad, f"{name} grad")
    finally:
        torch_paras.set_strict(was_strict)


def test_repeat_interleave_counts(device):
    for reps in ([0, 3, 1, 2], [2, 2, 2], [0, 0]):
        r = torch.tensor(reps)
        check_op(f"repeat_interleave {reps}", lambda: torch.repeat_interleave(r),
                 lambda: torch.repeat_interleave(r.to(device)))


def test_index_add(device):
    for dtype in (torch.float32, torch.float64, torch.bfloat16, torch.int64):
        self_ = torch.zeros(5, 3, dtype=dtype)
        index = torch.tensor([0, 4, 0, 2])
        src = (torch.randn(4, 3) * 3).to(dtype)
        check_op(f"index_add {dtype}",
                 lambda: self_.index_add(0, index, src, alpha=2),
                 lambda: self_.to(device).index_add(0, index.to(device), src.to(device), alpha=2))
    self_ = torch.randn(2, 6, 3)
    index = torch.tensor([5, 1, 1])
    src = torch.randn(2, 3, 3)
    check_op("index_add dim1", lambda: self_.index_add(1, index, src),
             lambda: self_.to(device).index_add(1, index.to(device), src.to(device)))


def test_logical_and_inf_predicates(device):
    x = torch.tensor([0.0, -float("inf"), float("inf"), float("nan"), 2.0, -0.0])
    for name, f in (("logical_not", torch.logical_not), ("isneginf", torch.isneginf),
                    ("isposinf", torch.isposinf), ("any", torch.any), ("all", torch.all)):
        check_op(name, lambda: f(x), lambda: f(x.to(device)))
    i = torch.tensor([3, -7, 0, 9])
    j = torch.tensor([2, 2, 5, -4])
    check_op("floor_divide int", lambda: torch.floor_divide(i, j),
             lambda: torch.floor_divide(i.to(device), j.to(device)))
    a = torch.randn(4, 5)
    b = torch.randn(4, 5) + 3.0
    check_op("floor_divide float", lambda: a // b, lambda: a.to(device) // b.to(device))
    check_op("isin", lambda: torch.isin(i, torch.tensor([9, 3])),
             lambda: torch.isin(i.to(device), torch.tensor([9, 3]).to(device)))
    check_op("isin invert", lambda: torch.isin(i, torch.tensor([9, 3]), invert=True),
             lambda: torch.isin(i.to(device), torch.tensor([9, 3]).to(device), invert=True))
    m = torch.randn(4, 6, 3) > 0.2
    for d in (0, 1, -1):
        check_op(f"all dim {d}", lambda: m.all(d), lambda: m.to(device).all(d))
        check_op(f"any dim {d} keepdim", lambda: m.any(d, keepdim=True),
                 lambda: m.to(device).any(d, keepdim=True))
    check_op("all dims", lambda: m.all(dim=(0, 2)), lambda: m.to(device).all(dim=(0, 2)))
    cond = torch.randn(3, 4) > 0
    check_op("where out", lambda: torch.where(cond, a[:3, :4], b[:3, :4]),
             lambda: torch.where(cond.to(device), a[:3, :4].to(device), b[:3, :4].to(device)))
    e = torch.empty(0, 3)
    check_op("any empty", lambda: torch.any(e), lambda: torch.any(e.to(device)))
    check_op("all empty", lambda: torch.all(e), lambda: torch.all(e.to(device)))


def test_vector_norm_and_clip(device):
    g = torch.randn(7, 9)
    for ord_ in (2, 1, 0, float("inf"), -float("inf"), 3.0):
        check_op(f"vector_norm ord={ord_}", lambda: torch.linalg.vector_norm(g, ord_),
                 lambda: torch.linalg.vector_norm(g.to(device), ord_))
    check_op("vector_norm dim", lambda: torch.linalg.vector_norm(g, 2, dim=1, keepdim=True),
             lambda: torch.linalg.vector_norm(g.to(device), 2, dim=1, keepdim=True))
    params = [torch.nn.Parameter(torch.randn(5, 4)) for _ in range(3)]
    dparams = [torch.nn.Parameter(p.detach().to(device)) for p in params]
    for p, dp in zip(params, dparams):
        p.grad = torch.randn_like(p) * 5
        dp.grad = p.grad.to(device)
    n_ref = torch.nn.utils.clip_grad_norm_(params, 1.0)
    n_dev = torch.nn.utils.clip_grad_norm_(dparams, 1.0)
    assert_close(n_dev, n_ref, "clip_grad_norm total norm")
    for p, dp in zip(params, dparams):
        assert_close(dp.grad, p.grad, "clipped grad")


def test_sort_and_topk_on_device(device):
    x = torch.randn(3, 5000)
    x[1, ::11] = float("nan")
    torch_paras.reset_runtime_stats()
    for largest in (True, False):
        v, i = torch.topk(x.to(device), 17, dim=1, largest=largest)
        rv, ri = torch.topk(x, 17, dim=1, largest=largest)
        assert_close(v, rv, f"topk values largest={largest}")
        assert_close(torch.gather(x, 1, i.cpu()), rv, f"topk indices largest={largest}")
    sv, si = torch.sort(x.to(device), dim=1, descending=True, stable=True)
    rsv, rsi = torch.sort(x, dim=1, descending=True, stable=True)
    assert_close(sv, rsv, "sort values")
    assert_close(si, rsi, "sort indices")
    stats = torch_paras.runtime_stats()
    assert stats["cpu_fallbacks"] == {}, stats
    assert stats["host_to_device_copies"] == 3, stats
    assert stats["device_to_host_copies"] == 6, stats


def test_argsort_rank_and_bitonic_boundary(device):
    for n in (64, 8192, 8193):
        v = torch.randint(-50, 50, (2, n)).float()
        v[0, ::97] = float("nan")
        for descending in (False, True):
            check_op(f"argsort n={n} desc={descending}",
                     lambda: torch.argsort(v, dim=1, descending=descending, stable=True),
                     lambda: torch.argsort(v.to(device), dim=1, descending=descending, stable=True))


def test_reductions_long_rows(device):
    x = torch.randn(3, 70000)
    for name, f in (("sum", lambda t: t.sum(1)), ("mean", lambda t: t.mean(-1, keepdim=True)),
                    ("amax", lambda t: t.amax(1)), ("softmax", lambda t: torch.softmax(t, -1))):
        check_op(f"{name} long rows", lambda: f(x), lambda: f(x.to(device)))
    y = torch.randn(128, 2048)
    check_op("mean hidden", lambda: y.pow(2).mean(-1, keepdim=True),
             lambda: y.to(device).pow(2).mean(-1, keepdim=True))


def test_reductions_on_bool(device):
    m = torch.randn(6, 9) > 0.1
    for name, f in (("sum", lambda t: t.sum()), ("sum dim", lambda t: t.sum(-1)),
                    ("amax", lambda t: t.amax(0)), ("max", lambda t: t.max()),
                    ("min", lambda t: t.min()), ("mean of cast", lambda t: t.float().mean())):
        check_op(f"bool {name}", lambda: f(m), lambda: f(m.to(device)))


def test_bmm_out_into_strided_view(device):
    a = torch.randn(3, 5, 4)
    b = torch.randn(3, 4, 6)
    big = torch.zeros(5, 3, 6, device=device)
    out = big.transpose(0, 1)
    torch.bmm(a.to(device), b.to(device), out=out)
    torch.testing.assert_close(out.cpu(), torch.bmm(a, b), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(big.cpu().transpose(0, 1), torch.bmm(a, b), atol=1e-5, rtol=1e-5)


def test_nonzero_block_scan_sizes(device):
    torch.manual_seed(0)
    cases = [torch.rand(1536) > 0.98, torch.rand(1025) > 0.5, torch.rand(1024 * 1024 + 7) > 0.999,
             torch.rand(3, 700, 5) > 0.7, torch.zeros(4000, dtype=torch.bool), torch.ones(3000, dtype=torch.bool),
             torch.randint(-1, 2, (2000,)), (torch.randn(1500) > 1).to(torch.bfloat16)]
    for x in cases:
        assert torch.equal(x.to(device).nonzero().cpu(), x.nonzero()), (x.shape, x.dtype)
    m = torch.rand(1536) > 0.9
    r = torch.randn(1536, 64)
    assert torch.equal(r.to(device)[m.to(device)].cpu(), r[m])


def test_log_softmax_long_rows_and_backward(device):
    import torch.nn.functional as F
    torch.manual_seed(0)
    for dtype, tol in ((torch.float32, 1e-5), (torch.bfloat16, 2e-2)):
        x = (4 * torch.randn(3, 128008)).to(dtype).requires_grad_()
        dx = x.detach().to(device).requires_grad_()
        tgt = torch.randint(0, 128008, (3,))
        F.cross_entropy(x.float(), tgt).backward()
        F.cross_entropy(dx.float(), tgt.to(device)).backward()
        torch.testing.assert_close(F.log_softmax(dx, -1).cpu().float(), F.log_softmax(x, -1).float(), atol=tol, rtol=tol)
        torch.testing.assert_close(dx.grad.cpu().float(), x.grad.float(), atol=tol, rtol=tol)
    y = torch.randn(5, 40, 7)
    torch.testing.assert_close(F.log_softmax(y.to(device), 1).cpu(), F.log_softmax(y, 1), atol=1e-5, rtol=1e-5)


def test_silu_backward_functional(device):
    import torch.nn.functional as F
    x = torch.randn(64, 33, requires_grad=True)
    dx = x.detach().to(device).requires_grad_()
    g = torch.randn(64, 33)
    F.silu(x).backward(g)
    F.silu(dx).backward(g.to(device))
    torch.testing.assert_close(dx.grad.cpu(), x.grad, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(torch.ops.aten.silu_backward(g.to(device), dx.detach()).cpu(),
                               torch.ops.aten.silu_backward(g, x.detach()), atol=1e-6, rtol=1e-5)


def test_full_reductions_multilevel_and_vector_norm(device):
    torch.manual_seed(0)
    for n in (1000, 70_001, 5_000_003):
        x = torch.randn(n)
        dx = x.to(device)
        torch.testing.assert_close(dx.sum().cpu(), x.double().sum().float(), atol=1e-3, rtol=1e-5)
        torch.testing.assert_close(dx.mean().cpu(), x.double().mean().float(), atol=1e-6, rtol=1e-5)
    y = torch.randn(37, 129, 3)
    dy = y.to(device)
    for p in (1.0, 2.0, 3.0):
        for dim, keep in ((None, False), (1, False), ((0, 2), True)):
            ref = torch.linalg.vector_norm(y.double(), ord=p, dim=dim, keepdim=keep).float()
            got = torch.linalg.vector_norm(dy, ord=p, dim=dim, keepdim=keep).cpu()
            assert got.shape == ref.shape, (p, dim, keep, got.shape, ref.shape)
            torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)
    b = torch.randn(3000, 700).to(torch.bfloat16)
    nb = torch.linalg.vector_norm(b.to(device)).cpu()
    assert nb.dtype == torch.bfloat16
    torch.testing.assert_close(nb.float(), torch.linalg.vector_norm(b.double()).float(), atol=0.5, rtol=4e-3)
    t = torch.randn(64, 48).t()
    torch.testing.assert_close(torch.linalg.vector_norm(t.to(device)).cpu(),
                               torch.linalg.vector_norm(t.double()).float(), atol=1e-5, rtol=1e-5)


def test_foreach_norm_and_clip_grad_norm(device):
    torch.manual_seed(0)
    ts = [torch.randn(3_000_001), torch.randn(7, 5), torch.zeros(0), torch.randn(1_048_576), torch.randn(2)]
    for p in (2.0, 1.0, 3.0):
        got = torch._foreach_norm([t.to(device) for t in ts], p)
        for g, t in zip(got, ts):
            torch.testing.assert_close(g.cpu(), torch.linalg.vector_norm(t.double(), p).float(), atol=1e-4, rtol=1e-5)
    bf = [t.to(torch.bfloat16) for t in ts[:2]]
    got = torch._foreach_norm([t.to(device) for t in bf])
    assert got[0].dtype == torch.bfloat16
    torch.testing.assert_close(got[0].cpu().float(), torch.linalg.vector_norm(bf[0].double()).float(), atol=0.5, rtol=4e-3)
    mixed = torch._foreach_norm([ts[1].to(device), ts[1].double().to(device)])
    torch.testing.assert_close(mixed[1].cpu(), torch.linalg.vector_norm(ts[1].double()))
    ps = [torch.randn(1200, 1100, requires_grad=True), torch.randn(300, requires_grad=True)]
    dps = [q.detach().to(device).requires_grad_() for q in ps]
    for q, d in zip(ps, dps):
        q.grad = torch.randn_like(q)
        d.grad = q.grad.to(device)
    exact = torch.sqrt(sum((q.grad.double() ** 2).sum() for q in ps)).item()
    total = torch.nn.utils.clip_grad_norm_(ps, 1.0)
    dtotal = torch.nn.utils.clip_grad_norm_(dps, 1.0)
    assert abs(dtotal.item() - exact) <= 1e-6 * exact + 1e-4, (dtotal.item(), exact)
    torch.testing.assert_close(dtotal.cpu(), total, atol=0.1, rtol=1e-4)
    for q, d in zip(ps, dps):
        torch.testing.assert_close(d.grad.cpu(), q.grad, atol=1e-6, rtol=1e-4)


def _foreach_lists(device, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    sizes = [3 * 65536 + 5, 1_000_003, 7, 0, 1, 131072]
    base = torch.randn(1001, generator=g).to(dtype).to(device)
    out = [torch.randn(n, generator=g).to(dtype).to(device) for n in sizes]
    return out + [base[1:]]


def test_foreach_optimizer_ops_bit_exact(device):
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        xs = _foreach_lists(device, dtype, 0)
        ys = _foreach_lists(device, dtype, 1)
        zs = [t.abs() + 0.5 for t in _foreach_lists(device, dtype, 2)]
        coef = torch.tensor(0.37, device=device)
        sl = [0.5 + 0.25 * i for i in range(len(xs))]
        cases = {
            "mul_.Scalar": (lambda a: torch._foreach_mul_(a, 0.9), lambda t, i: t.mul_(0.9)),
            "mul_.Tensor": (lambda a: torch._foreach_mul_(a, coef), lambda t, i: t.mul_(coef)),
            "add_.List": (lambda a: torch._foreach_add_(a, ys, alpha=-1e-2), lambda t, i: t.add_(ys[i], alpha=-1e-2)),
            "add_.Scalar": (lambda a: torch._foreach_add_(a, 1e-3), lambda t, i: t.add_(1e-3)),
            "lerp_.Scalar": (lambda a: torch._foreach_lerp_(a, ys, 0.1), lambda t, i: t.lerp_(ys[i], 0.1)),
            "addcmul_.Scalar": (lambda a: torch._foreach_addcmul_(a, ys, zs, value=0.05),
                                lambda t, i: t.addcmul_(ys[i], zs[i], value=0.05)),
            "addcdiv_.ScalarList": (lambda a: torch._foreach_addcdiv_(a, ys, zs, sl),
                                    lambda t, i: t.addcdiv_(ys[i], zs[i], value=sl[i])),
            "div_.ScalarList": (lambda a: torch._foreach_div_(a, sl), lambda t, i: t.div_(sl[i])),
        }
        for name, (fused, single) in cases.items():
            got = [t.clone() for t in xs]
            ref = [t.clone() for t in xs]
            fused(got)
            for i, t in enumerate(ref):
                single(t, i)
            for i, (g_, r_) in enumerate(zip(got, ref)):
                assert torch.equal(g_.cpu(), r_.cpu()), (name, dtype, i)
        roots = torch._foreach_sqrt(zs)
        for r_, z in zip(roots, zs):
            assert torch.equal(r_.cpu(), z.sqrt().cpu()), ("sqrt", dtype)
    a = [torch.randn(70001), torch.randn(9)]
    d = [t.to(device) for t in a]
    torch._foreach_mul_(d, 0.9)
    torch._foreach_add_(d, d, alpha=0.5)
    for x, t in zip(d, a):
        torch.testing.assert_close(x.cpu(), t * 0.9 * 1.5, atol=1e-6, rtol=1e-6)
    mixed = [torch.ones(5, device=device), torch.ones(5, dtype=torch.float64, device=device)]
    torch._foreach_mul_(mixed, 3.0)
    assert all(torch.equal(m.cpu(), torch.full((5,), 3.0, dtype=m.dtype)) for m in mixed)


def test_foreach_optimizers_match_single_tensor(device):
    for make in (lambda p, fe: torch.optim.SGD(p, lr=1e-2, momentum=0.9, foreach=fe),
                 lambda p, fe: torch.optim.AdamW(p, lr=1e-2, weight_decay=0.1, foreach=fe)):
        for dtype in (torch.bfloat16, torch.float32):
            torch.manual_seed(0)
            init = [torch.randn(s).to(dtype) for s in ((257, 129), (65536 * 2 + 3,), (3,))]
            grads = [[torch.randn(t.shape).to(dtype) for t in init] for _ in range(3)]
            runs = []
            for fe in (True, False):
                ps = [t.clone().to(device).requires_grad_() for t in init]
                opt = make(ps, fe)
                for step in grads:
                    for p, g in zip(ps, step):
                        p.grad = g.to(device)
                    opt.step()
                runs.append([p.detach().cpu() for p in ps])
            for a, b in zip(*runs):
                assert torch.equal(a, b), (type(opt).__name__, dtype)


def test_argsort_bitonic_sizes_exact(device):
    torch.manual_seed(0)
    cases = [torch.randint(0, 64, (12288,)), torch.randint(0, 5, (4097,)), torch.randn(70001),
             torch.randint(-100, 100, (3, 5000)), torch.randn(2, 9000).to(torch.bfloat16)]
    nan = torch.randn(6000); nan[::97] = float("nan"); cases.append(nan)
    for x in cases:
        for desc in (False, True):
            got = torch.argsort(x.to(device), dim=-1, descending=desc, stable=True).cpu()
            ref = torch.argsort(x, dim=-1, descending=desc, stable=True)
            assert torch.equal(got, ref), (x.shape, x.dtype, desc)
