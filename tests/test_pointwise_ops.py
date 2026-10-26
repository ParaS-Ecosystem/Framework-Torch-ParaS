import torch
import torch_paras
import torch.nn.functional as F

from common import check_op, assert_close


def _pair(device):
    a = torch.randn(4, 5)
    b = torch.randn(4, 5)
    return a, b, a.to(device), b.to(device)


# --- unary ------------------------------------------------------------------

def test_relu_(device):
    a, _, da, _ = _pair(device)
    check_op("relu_", lambda: torch.relu_(a.clone()),
             lambda: torch.relu_(da.clone()))


def test_exp(device):
    a, _, da, _ = _pair(device)
    check_op("exp", lambda: torch.exp(a), lambda: torch.exp(da))


def test_log(device):
    a, _, da, _ = _pair(device)
    check_op("log", lambda: torch.log(torch.abs(a) + 1.0),
             lambda: torch.log(torch.abs(da) + 1.0))


def test_sqrt(device):
    a, _, da, _ = _pair(device)
    check_op("sqrt", lambda: torch.sqrt(torch.abs(a) + 1.0),
             lambda: torch.sqrt(torch.abs(da) + 1.0))


def test_sin(device):
    a, _, da, _ = _pair(device)
    check_op("sin", lambda: torch.sin(a), lambda: torch.sin(da))


def test_cos(device):
    a, _, da, _ = _pair(device)
    check_op("cos", lambda: torch.cos(a), lambda: torch.cos(da))


def test_sigmoid(device):
    a, _, da, _ = _pair(device)
    check_op("sigmoid", lambda: torch.sigmoid(a), lambda: torch.sigmoid(da))


def test_tanh(device):
    a, _, da, _ = _pair(device)
    check_op("tanh", lambda: torch.tanh(a), lambda: torch.tanh(da))


def test_abs(device):
    a, _, da, _ = _pair(device)
    check_op("abs", lambda: torch.abs(a), lambda: torch.abs(da))


def test_neg(device):
    a, _, da, _ = _pair(device)
    check_op("neg", lambda: torch.neg(a), lambda: torch.neg(da))


def test_reciprocal(device):
    a, _, da, _ = _pair(device)
    check_op("reciprocal", lambda: torch.reciprocal(a + 2.0),
             lambda: torch.reciprocal(da + 2.0))


def test_ceil(device):
    a, _, da, _ = _pair(device)
    check_op("ceil", lambda: torch.ceil(a), lambda: torch.ceil(da))


def test_round(device):
    a, _, da, _ = _pair(device)
    check_op("round", lambda: torch.round(a), lambda: torch.round(da))


def test_atan(device):
    a, _, da, _ = _pair(device)
    check_op("atan", lambda: torch.atan(a), lambda: torch.atan(da))


def test_gelu(device):
    a, _, da, _ = _pair(device)
    check_op("gelu", lambda: F.gelu(a), lambda: F.gelu(da))


def test_silu(device):
    a, _, da, _ = _pair(device)
    check_op("silu", lambda: F.silu(a), lambda: F.silu(da))


def test_log_sigmoid(device):
    a, _, da, _ = _pair(device)
    check_op("logsigmoid", lambda: F.logsigmoid(a), lambda: F.logsigmoid(da))


# --- binary -----------------------------------------------------------------

def test_add(device):
    a, b, da, db = _pair(device)
    check_op("add", lambda: a + b, lambda: da + db)


def test_sub(device):
    a, b, da, db = _pair(device)
    check_op("sub", lambda: a - b, lambda: da - db)


def test_mul(device):
    a, b, da, db = _pair(device)
    check_op("mul", lambda: a * b, lambda: da * db)


def test_div(device):
    a, b, da, db = _pair(device)
    check_op("div", lambda: (a + 3) / (b + 3), lambda: (da + 3) / (db + 3))


def test_maximum(device):
    a, b, da, db = _pair(device)
    check_op("maximum", lambda: torch.maximum(a, b),
             lambda: torch.maximum(da, db))


def test_minimum(device):
    a, b, da, db = _pair(device)
    check_op("minimum", lambda: torch.minimum(a, b),
             lambda: torch.minimum(da, db))


def test_pow(device):
    a, _, da, _ = _pair(device)
    check_op("pow", lambda: torch.pow(a, 2.0), lambda: torch.pow(da, 2.0))


def test_dot(device):
    a, b, da, db = _pair(device)
    check_op("dot", lambda: torch.dot(a.flatten(), b.flatten()),
             lambda: torch.dot(da.flatten(), db.flatten()))


def test_clamp(device):
    a, _, da, _ = _pair(device)
    check_op("clamp", lambda: torch.clamp(a, -1, 1),
             lambda: torch.clamp(da, -1, 1))


def test_clamp_min(device):
    a, _, da, _ = _pair(device)
    check_op("clamp_min", lambda: torch.clamp_min(a, 0.0),
             lambda: torch.clamp_min(da, 0.0))


# --- reductions ---------------------------------------------------------------

def test_mean(device):
    a, _, da, _ = _pair(device)
    check_op("mean", lambda: torch.mean(a), lambda: torch.mean(da))


def test_sum_dim(device):
    a, _, da, _ = _pair(device)
    check_op("sum.dim", lambda: torch.sum(a, dim=1),
             lambda: torch.sum(da, dim=1))


def test_amax(device):
    a, _, da, _ = _pair(device)
    check_op("amax", lambda: torch.amax(a, dim=1), lambda: torch.amax(da, dim=1))


def test_amin(device):
    a, _, da, _ = _pair(device)
    check_op("amin", lambda: torch.amin(a, dim=1), lambda: torch.amin(da, dim=1))


def test_argmax(device):
    a, _, da, _ = _pair(device)
    check_op("argmax", lambda: torch.argmax(a, dim=1),
             lambda: torch.argmax(da, dim=1))


def test_argmax_argmin_long_rows_ties_nan(device):
    torch.manual_seed(0)
    cases = [torch.randn(1, 128008), torch.randn(3, 70001).to(torch.bfloat16),
             torch.randint(-5, 5, (4, 9000)), torch.randn(2, 33, 5)]
    tie = torch.zeros(2, 5000); tie[:, 17] = 3.0; tie[:, 4000] = 3.0; cases.append(tie)
    nan = torch.randn(2, 6000); nan[0, 123] = float("nan"); nan[0, 5000] = float("nan")
    nan[1, 77] = float("inf"); cases.append(nan)
    cases.append(torch.full((3, 2000), float("-inf")))
    for x in cases:
        dx = x.to(device)
        for fn in (torch.argmax, torch.argmin):
            assert torch.equal(fn(dx, dim=-1).cpu(), fn(x, dim=-1)), (fn.__name__, x.shape, x.dtype)
            assert torch.equal(fn(dx, dim=-1, keepdim=True).cpu(), fn(x, dim=-1, keepdim=True))
            assert torch.equal(fn(dx).cpu(), fn(x)), (fn.__name__, "flat", x.shape)
            if x.dim() == 3:
                assert torch.equal(fn(dx, dim=1).cpu(), fn(x, dim=1))


# --- bitwise ------------------------------------------------------------------

def _bool_pair(device):
    a = torch.randint(0, 2, (4, 5), dtype=torch.bool)
    b = torch.randint(0, 2, (4, 5), dtype=torch.bool)
    return a, b, a.to(device), b.to(device)


def test_bitwise_and(device):
    a, b, da, db = _bool_pair(device)
    check_op("and", lambda: torch.bitwise_and(a, b),
             lambda: torch.bitwise_and(da, db))


def test_bitwise_or(device):
    a, b, da, db = _bool_pair(device)
    check_op("or", lambda: torch.bitwise_or(a, b),
             lambda: torch.bitwise_or(da, db))


def test_bitwise_xor(device):
    a, b, da, db = _bool_pair(device)
    check_op("xor", lambda: torch.bitwise_xor(a, b),
             lambda: torch.bitwise_xor(da, db))


def test_bitwise_not(device):
    a, _, da, _ = _bool_pair(device)
    check_op("not", lambda: torch.bitwise_not(a),
             lambda: torch.bitwise_not(da))


# --- softmax ------------------------------------------------------------------

def test_softmax_dim1(device):
    a, _, da, _ = _pair(device)
    check_op("softmax.dim1", lambda: torch.softmax(a, dim=1),
             lambda: torch.softmax(da, dim=1))


def test_softmax_dim0(device):
    a, _, da, _ = _pair(device)
    check_op("softmax.dim0", lambda: torch.softmax(a, dim=0),
             lambda: torch.softmax(da, dim=0))


def test_softmax_last_dim_3d(device):
    a = torch.randn(2, 3, 7)
    da = a.to(device)
    check_op("softmax.3d", lambda: torch.softmax(a, dim=-1),
             lambda: torch.softmax(da, dim=-1))


def test_softmax_negative_dim(device):
    a, _, da, _ = _pair(device)
    check_op("softmax.neg_dim", lambda: torch.softmax(a, dim=-1),
             lambda: torch.softmax(da, dim=-1))


def test_softmax_backward(device):
    a = torch.randn(4, 5, requires_grad=True)
    da = a.detach().to(device).requires_grad_(True)

    torch.softmax(a, dim=1).sum().backward()
    torch.softmax(da, dim=1).sum().backward()

    check_op("softmax.backward", lambda: a.grad, lambda: da.grad)


def test_softmax_backward_nontrivial_grad(device):
    a = torch.randn(3, 6, requires_grad=True)
    da = a.detach().to(device).requires_grad_(True)
    upstream = torch.randn(3, 6)

    torch.softmax(a, dim=1).backward(upstream)
    torch.softmax(da, dim=1).backward(upstream.to(device))

    check_op("softmax.backward_nontrivial", lambda: a.grad, lambda: da.grad)


# --- log_softmax --------------------------------------------------------------

def test_log_softmax_dim1(device):
    a, _, da, _ = _pair(device)
    check_op("log_softmax.dim1", lambda: torch.log_softmax(a, dim=1),
             lambda: torch.log_softmax(da, dim=1))


def test_log_softmax_last_dim_3d(device):
    a = torch.randn(2, 3, 7)
    da = a.to(device)
    check_op("log_softmax.3d", lambda: torch.log_softmax(a, dim=-1),
             lambda: torch.log_softmax(da, dim=-1))


def test_log_softmax_backward(device):
    a = torch.randn(4, 5, requires_grad=True)
    da = a.detach().to(device).requires_grad_(True)

    torch.log_softmax(a, dim=1).sum().backward()
    torch.log_softmax(da, dim=1).sum().backward()

    check_op("log_softmax.backward", lambda: a.grad, lambda: da.grad)


def test_log_softmax_backward_nontrivial_grad(device):
    a = torch.randn(3, 6, requires_grad=True)
    da = a.detach().to(device).requires_grad_(True)
    upstream = torch.randn(3, 6)

    torch.log_softmax(a, dim=1).backward(upstream)
    torch.log_softmax(da, dim=1).backward(upstream.to(device))

    check_op("log_softmax.backward_nontrivial", lambda: a.grad, lambda: da.grad)


def test_log_softmax_matches_cross_entropy(device):
    # This is the whole point of log_softmax: nn.CrossEntropyLoss calls it
    # directly rather than softmax()+log(), so verify the composed result
    # matches on-device too.
    x = torch.randn(4, 8, 6)
    target = torch.randint(0, 8, (4, 6))
    dx = x.to(device)
    dtarget = target.to(device)

    ref = torch.nn.functional.nll_loss(
        torch.log_softmax(x, dim=1), target)
    dev = torch.nn.functional.nll_loss(
        torch.log_softmax(dx, dim=1), dtarget)
    check_op("log_softmax.nll_loss_chain", lambda: ref, lambda: dev)


# --- cumsum -------------------------------------------------------------------

def test_cumsum_dim1(device):
    a, _, da, _ = _pair(device)
    check_op("cumsum.dim1", lambda: torch.cumsum(a, dim=1),
             lambda: torch.cumsum(da, dim=1))


def test_cumsum_dim0(device):
    a, _, da, _ = _pair(device)
    check_op("cumsum.dim0", lambda: torch.cumsum(a, dim=0),
             lambda: torch.cumsum(da, dim=0))


def test_cumsum_negative_dim(device):
    a = torch.randn(2, 3, 4)
    da = a.to(device)
    check_op("cumsum.neg_dim", lambda: torch.cumsum(a, dim=-1),
             lambda: torch.cumsum(da, dim=-1))


def test_cumsum_int64_exact(device):
    a = torch.tensor([2**62, 1, 1, 1, 1], dtype=torch.int64)
    da = a.to(device)
    check_op("cumsum.int64_exact", lambda: torch.cumsum(a, dim=0),
             lambda: torch.cumsum(da, dim=0))


def test_cumsum_bool_promotes_to_long(device):
    a = torch.tensor([True, False, True, True])
    da = a.to(device)
    dev_out = torch.cumsum(da, dim=0)
    assert dev_out.dtype == torch.int64
    check_op("cumsum.bool_promotion", lambda: torch.cumsum(a, dim=0),
             lambda: dev_out)


def test_cumsum_(device):
    a = torch.randn(4, 5)
    da = a.to(device)
    a.cumsum_(dim=1)
    da.cumsum_(dim=1)
    check_op("cumsum_", lambda: a, lambda: da)


# --- topk ---------------------------------------------------------------------

def test_topk_largest(device):
    a = torch.randn(3, 20)
    da = a.to(device)
    ref_v, ref_i = torch.topk(a, k=5, dim=1)
    dev_v, dev_i = torch.topk(da, k=5, dim=1)
    check_op("topk.largest.values", lambda: ref_v, lambda: dev_v)
    check_op("topk.largest.indices", lambda: ref_i, lambda: dev_i)


def test_topk_smallest(device):
    a = torch.randn(3, 20)
    da = a.to(device)
    ref_v, ref_i = torch.topk(a, k=4, dim=1, largest=False)
    dev_v, dev_i = torch.topk(da, k=4, dim=1, largest=False)
    check_op("topk.smallest.values", lambda: ref_v, lambda: dev_v)
    check_op("topk.smallest.indices", lambda: ref_i, lambda: dev_i)


def test_topk_dim0(device):
    a = torch.randn(10, 4)
    da = a.to(device)
    ref_v, ref_i = torch.topk(a, k=3, dim=0)
    dev_v, dev_i = torch.topk(da, k=3, dim=0)
    check_op("topk.dim0.values", lambda: ref_v, lambda: dev_v)
    check_op("topk.dim0.indices", lambda: ref_i, lambda: dev_i)


def test_topk_full_size(device):
    a = torch.randn(3, 20)
    da = a.to(device)
    ref_v, ref_i = torch.topk(a, k=20, dim=1)
    dev_v, dev_i = torch.topk(da, k=20, dim=1)
    check_op("topk.full.values", lambda: ref_v, lambda: dev_v)
    check_op("topk.full.indices", lambda: ref_i, lambda: dev_i)


def test_topk_values_only_with_ties(device):
    # PyTorch doesn't guarantee a particular tie-break index order for
    # duplicate values (only that values come back sorted), so we only
    # check the value sequence here, not exact indices.
    a = torch.tensor([[5.0, 3.0, 5.0, 5.0, 1.0, 3.0]])
    da = a.to(device)
    ref_v, _ = torch.topk(a, k=4, dim=1)
    dev_v, _ = torch.topk(da, k=4, dim=1)
    check_op("topk.ties.values", lambda: ref_v, lambda: dev_v)


# --- structure ----------------------------------------------------------------

def test_python_scalars_at_opmath_precision(device):
    torch.manual_seed(0)
    for dt in (torch.bfloat16, torch.float16):
        x = torch.randn(4099).to(dt)
        d = x.to(device)
        f = lambda v: float(torch.tensor(v, dtype=torch.float32))
        x64 = x.float()
        cases = {
            "mul": (d * 0.9, x64 * f(0.9)),
            "add": (d + 0.1, x64 + f(0.1)),
            "sub": (d - 0.3, x64 - f(0.3)),
            "div": (d / 3.0, x64 / f(3.0)),
            "rsub": (1.7 - d, f(1.7) - x64),
        }
        for name, (got, ref) in cases.items():
            assert torch.equal(got.cpu(), ref.to(dt)), (name, dt)
        old = x64 * float(torch.tensor(0.9, dtype=dt))
        assert not torch.equal((d * 0.9).cpu(), old.to(dt))
        m = d.clone(); m.mul_(0.9)
        assert torch.equal(m.cpu(), (d * 0.9).cpu())
    i = torch.arange(7, device=device)
    assert torch.equal((i * 2.5).cpu(), torch.arange(7) * 2.5)


def test_cat_one_launch_cases(device):
    torch.manual_seed(0)
    cases = [
        (torch.float32, [(5, 3), (2, 3), (0, 3), (7, 3)], 0),
        (torch.bfloat16, [(4, 6, 5), (4, 1, 5), (4, 3, 5)], 1),
        (torch.float64, [(3, 2), (3, 7)], -1),
        (torch.int64, [(2, 3, 1), (2, 3, 4)], -1),
        (torch.bool, [(3, 5), (3, 2), (3, 1)], 1),
        (torch.float32, [(int(n), 2048) for n in torch.randint(1, 13, (64,))], 0),
        (torch.float16, [(1, 3)] * 70, 0),
    ]
    for dtype, shapes, dim in cases:
        if dtype == torch.bool:
            ts = [torch.rand(s) > 0.5 for s in shapes]
        elif dtype.is_floating_point:
            ts = [torch.randn(s).to(dtype) for s in shapes]
        else:
            ts = [torch.randint(-9, 9, s) for s in shapes]
        got = torch.cat([t.to(device) for t in ts], dim=dim)
        assert torch.equal(got.cpu(), torch.cat(ts, dim=dim)), (dtype, dim)
    a = torch.randn(4, 6)
    b = torch.randn(3, 4).t()
    assert torch.equal(torch.cat([a.to(device), b.to(device)], dim=1).cpu(), torch.cat([a, b], dim=1))


def test_pow_short_exponents_exact(device):
    x = torch.randn(10007)
    d = x.to(device)
    assert torch.equal(d.pow(2).cpu(), (x.double() * x.double()).float())
    assert torch.equal(d.pow(1).cpu(), x)
    assert torch.equal(d.pow(-1).cpu(), (1.0 / x.double()).float())
    torch.testing.assert_close(d.pow(3).cpu(), x.double().pow(3).float(), atol=0, rtol=1e-6)


def test_cat(device):
    a, b, da, db = _pair(device)
    check_op("cat", lambda: torch.cat([a, b], dim=0),
             lambda: torch.cat([da, db], dim=0))


def test_mm(device):
    a = torch.randn(8, 16)
    b = torch.randn(16, 4)
    check_op("mm", lambda: a @ b, lambda: a.to(device) @ b.to(device))


def test_sort_ascending(device):
    a = torch.randn(100, dtype=torch.float32)
    cpu_vals, cpu_idxs = torch.sort(a)
    dev_vals, dev_idxs = torch.sort(a.to(device))
    assert_close(dev_vals, cpu_vals, "sort ascending values")
    assert_close(dev_idxs, cpu_idxs, "sort ascending indices")


def test_sort_descending(device):
    a = torch.randn(100, dtype=torch.float32)
    cpu_vals, cpu_idxs = torch.sort(a, descending=True)
    dev_vals, dev_idxs = torch.sort(a.to(device), descending=True)
    assert_close(dev_vals, cpu_vals, "sort descending values")
    assert_close(dev_idxs, cpu_idxs, "sort descending indices")


def test_sort_dim0(device):
    a = torch.randn(5, 8, dtype=torch.float32)
    cpu_vals, cpu_idxs = torch.sort(a, dim=0)
    dev_vals, dev_idxs = torch.sort(a.to(device), dim=0)
    assert_close(dev_vals, cpu_vals, "sort dim0 values")
    assert_close(dev_idxs, cpu_idxs, "sort dim0 indices")


def test_sort_stable(device):
    a = torch.tensor([3.0, 1.0, 2.0, 1.0, 3.0])
    cpu_vals, cpu_idxs = torch.sort(a, stable=True)
    dev_vals, dev_idxs = torch.sort(a.to(device), stable=True)
    assert_close(dev_vals, cpu_vals, "sort stable values")
    assert_close(dev_idxs, cpu_idxs, "sort stable indices")




def test_bmm(device):
    a = torch.randn(3, 8, 16)
    b = torch.randn(3, 16, 4)
    check_op("bmm", lambda: torch.bmm(a, b),
             lambda: torch.bmm(a.to(device), b.to(device)))


def test_addmm(device):
    self = torch.randn(8, 4)
    a = torch.randn(8, 16)
    b = torch.randn(16, 4)
    check_op("addmm",
             lambda: torch.addmm(self, a, b, beta=0.5, alpha=2.0),
             lambda: torch.addmm(self.to(device), a.to(device), b.to(device),
                                 beta=0.5, alpha=2.0))


def test_linear_forward(device):
    x = torch.randn(2, 5, 16)
    w = torch.randn(4, 16)
    b = torch.randn(4)
    check_op("linear", lambda: torch.nn.functional.linear(x, w, b),
             lambda: torch.nn.functional.linear(x.to(device), w.to(device),
                                                b.to(device)))


def test_sort(device):
    a = torch.randn(4, 7)
    check_op("sort", lambda: torch.sort(a, dim=-1, descending=True),
             lambda: torch.sort(a.to(device), dim=-1, descending=True))


def test_argsort(device):
    a = torch.randn(4, 7)
    check_op("argsort", lambda: torch.argsort(a, dim=-1),
             lambda: torch.argsort(a.to(device), dim=-1))





def test_safe_softmax_masked_rows_dtype_and_grad(device):
    x = torch.randn(2, 3, 7)
    x[0, 1, :] = float("-inf")
    x[1, 2, :4] = float("-inf")
    for dtype in (None, torch.float32):
        for src in (x, x.to(torch.bfloat16)):
            ref = torch.ops.aten._safe_softmax(src, -1, dtype)
            out = torch.ops.aten._safe_softmax(src.to(device), -1, dtype).cpu()
            assert out.dtype == ref.dtype
            assert torch.equal(out[0, 1], torch.zeros_like(out[0, 1]))
            tol = 1e-2 if src.dtype == torch.bfloat16 else 1e-6
            torch.testing.assert_close(out.float(), ref.float(), atol=tol, rtol=tol)
    xs = torch.randn(4, 9, requires_grad=True)
    dxs = xs.detach().to(device).requires_grad_()
    w = torch.randn(4, 9)
    (torch.ops.aten._safe_softmax(xs, -1, None) * w).sum().backward()
    (torch.ops.aten._safe_softmax(dxs, -1, None) * w.to(device)).sum().backward()
    torch.testing.assert_close(dxs.grad.cpu(), xs.grad, atol=1e-6, rtol=1e-5)


def test_vectorized_contiguous_path_matches_strided(device):
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        a = torch.randn(64, 515).to(device, dtype)
        b = torch.randn(64, 515).to(device, dtype)
        at, bt = a.t().contiguous().t(), b.t().contiguous().t()
        for name, fn in (("mul", torch.mul), ("add", torch.add), ("div", torch.div)):
            assert torch.equal(fn(a, b).cpu(), fn(at, bt).cpu()), (dtype, name)
        idx = torch.device(device).index or 0
        amd = torch_paras.is_gpu_device(idx) and torch_paras.device_kernel_caps(idx)["wave_size"] == 64
        unary = (("silu", torch.nn.functional.silu),) if amd else ()
        for name, fn in unary + (("exp", torch.exp), ("neg", torch.neg)):
            assert torch.equal(fn(a).cpu(), fn(at).cpu()), (dtype, name)


def test_transcendentals_at_fp32_opmath(device):
    x = torch.randn(4096) * 4
    for fn in (torch.nn.functional.silu, torch.exp, torch.sigmoid):
        got = fn(x.to(device)).cpu()
        ref = fn(x.double()).float()
        assert torch.allclose(got, ref, rtol=4e-7, atol=1e-30), fn


def test_softmax_rows_lengths(device):
    torch.manual_seed(0)
    for n in (7, 256, 300, 1000, 2048, 4096, 9000, 16384):
        for dtype, tol in ((torch.float32, 1e-6), (torch.bfloat16, 1e-2), (torch.float16, 1e-3)):
            x = (torch.randn(3, n) * 4).to(dtype)
            x[1, : n // 2] = float("-inf")
            got = torch.softmax(x.to(device), -1).cpu().float()
            ref = torch.softmax(x.float(), -1)
            assert torch.allclose(got, ref, atol=tol, rtol=tol), (n, dtype, (got - ref).abs().max())
    x = torch.randn(2, 4096)
    x[0] = float("-inf")
    assert torch.isnan(torch.softmax(x.to(device), -1).cpu()[0]).all()
    assert torch.equal(torch.ops.aten._safe_softmax(x.to(device), -1, None).cpu()[0], torch.zeros(4096))


def test_float_unary_ops_of_integer_inputs(device):
    x_int = torch.tensor([0, 1, 2, -3, 7])
    x_bool = torch.tensor([True, False, True, True, False])
    for f in (torch.exp, torch.exp2, torch.expm1, torch.log, torch.log1p, torch.log2, torch.sqrt, torch.rsqrt,
              torch.reciprocal, torch.sin, torch.cos, torch.tan, torch.tanh, torch.sigmoid, torch.atan,
              torch.erf, torch.erfc):
        for x in (x_int, x_bool):
            ref = f(x)
            got = f(x.to(device)).cpu()
            assert got.dtype == ref.dtype, (f.__name__, x.dtype, got.dtype)
            assert torch.allclose(got, ref, atol=1e-6, rtol=1e-5, equal_nan=True), (f.__name__, x.dtype, got, ref)


def test_vector_norm_of_0d_tensor(device):
    x = torch.tensor(-3.0)
    for kw in ({}, {"dim": 0}, {"dim": -1, "keepdim": True}, {"dim": 0, "ord": 1}):
        ref = torch.linalg.vector_norm(x, **kw)
        got = torch.linalg.vector_norm(x.to(device), **kw).cpu()
        assert got.shape == ref.shape and torch.allclose(got, ref), (kw, got, ref)
