"""Operators that used to fall back to the host: unary and binary math,
activations with their backward, scans, variance -- against PyTorch on the
CPU, with no CPU fallback."""

import torch
import torch.nn.functional as F
import torch_paras


def _no_fallback(fn, what):
    torch_paras.reset_runtime_stats()
    out = fn()
    fb = torch_paras.runtime_stats().get("cpu_fallbacks", {})
    assert not fb, (what, fb)
    return out


def _close(got, ref, what, tol=1e-5):
    got = got.cpu()
    assert got.dtype == ref.dtype and got.shape == ref.shape, (what, got.dtype, ref.dtype, got.shape, ref.shape)
    assert torch.allclose(got, ref, rtol=tol, atol=tol, equal_nan=True), (what, (got - ref).abs().nan_to_num().max())


def test_unary_math(device):
    torch.manual_seed(0)
    x = torch.randn(1000) * 3
    pos = x.abs() + 0.5
    unit = torch.rand(1000) * 1.98 - 0.99
    cases = {
        "erf": (torch.erf, x), "erfc": (torch.erfc, x), "erfinv": (torch.erfinv, unit),
        "expm1": (torch.expm1, x), "log1p": (torch.log1p, pos), "log2": (torch.log2, pos), "log10": (torch.log10, pos),
        "exp2": (torch.exp2, x), "tan": (torch.tan, x), "asin": (torch.asin, unit), "acos": (torch.acos, unit),
        "sinh": (torch.sinh, x), "cosh": (torch.cosh, x), "asinh": (torch.asinh, x), "acosh": (torch.acosh, pos + 1),
        "atanh": (torch.atanh, unit), "floor": (torch.floor, x), "trunc": (torch.trunc, x), "frac": (torch.frac, x),
        "lgamma": (torch.lgamma, pos), "digamma": (torch.digamma, x), "sign": (torch.sign, x),
    }
    for name, (fn, v) in cases.items():
        got = _no_fallback(lambda: fn(v.to(device)), name)
        _close(got, fn(v), name, tol=2e-5 if name in ("erfinv", "digamma", "lgamma") else 1e-5)
    xb = x.bfloat16()
    _close(_no_fallback(lambda: torch.erf(xb.to(device)), "erf bf16"), torch.erf(xb), "erf bf16", tol=1e-2)
    xi = torch.randint(-5, 5, (100,))
    _close(_no_fallback(lambda: torch.sign(xi.to(device)), "sign int"), torch.sign(xi), "sign int")


def test_binary_math(device):
    torch.manual_seed(0)
    a = torch.randn(500) * 4
    b = torch.randn(500) * 2
    b = torch.where(b.abs() < 0.1, torch.full_like(b, 0.5), b)
    for name, fn in (("atan2", torch.atan2), ("hypot", torch.hypot), ("copysign", torch.copysign),
                     ("fmod", torch.fmod), ("remainder", torch.remainder), ("logaddexp", torch.logaddexp),
                     ("logaddexp2", torch.logaddexp2), ("xlogy", lambda u, v: torch.xlogy(u, v.abs()))):
        got = _no_fallback(lambda: fn(a.to(device), b.to(device)), name)
        _close(got, fn(a, b), name)
    _close(_no_fallback(lambda: torch.remainder(a.to(device), 1.5), "remainder scalar"), torch.remainder(a, 1.5), "rem scalar")
    _close(_no_fallback(lambda: torch.fmod(a.to(device), -1.5), "fmod scalar"), torch.fmod(a, -1.5), "fmod scalar")
    inf = torch.tensor([float("inf"), -float("inf"), 1.0])
    _close(torch.logaddexp(inf.to(device), inf.to(device)), torch.logaddexp(inf, inf), "logaddexp inf")
    ai, bi = torch.randint(-20, 20, (200,)), torch.randint(1, 7, (200,)) * torch.randint(0, 2, (200,)).mul(2).sub(1)
    _close(torch.remainder(ai.to(device), bi.to(device)), torch.remainder(ai, bi), "remainder int")


def test_activations_and_backward(device):
    torch.manual_seed(0)
    x = torch.randn(300) * 3
    for name, fn in (("softplus", lambda t: F.softplus(t, beta=1.5, threshold=4)), ("elu", lambda t: F.elu(t, 0.7)),
                     ("selu", F.selu), ("celu", lambda t: F.celu(t, 1.3)), ("mish", F.mish)):
        ref_in = x.clone().requires_grad_()
        dev_in = x.to(device).requires_grad_()
        ref = fn(ref_in)
        got = _no_fallback(lambda: fn(dev_in), name)
        _close(got.detach(), ref.detach(), name)
        g = torch.randn_like(ref)
        ref.backward(g)
        _no_fallback(lambda: got.backward(g.to(device)), name + " backward")
        _close(dev_in.grad, ref_in.grad, name + " grad")


def test_scans(device):
    torch.manual_seed(0)
    x = torch.rand(4, 7, 5) + 0.5
    for dim in (0, 1, 2, -1):
        _close(_no_fallback(lambda: torch.cumprod(x.to(device), dim), "cumprod"), torch.cumprod(x, dim), "cumprod")
        _close(_no_fallback(lambda: torch.logcumsumexp(x.to(device), dim), "logcumsumexp"),
               torch.logcumsumexp(x, dim), "logcumsumexp")


def test_variance(device):
    torch.manual_seed(0)
    x = torch.randn(4, 6, 5) * 2 + 1
    for kw in (dict(), dict(dim=1), dict(dim=(0, 2), keepdim=True), dict(dim=2, correction=0), dict(dim=1, unbiased=False)):
        for fn in (torch.var, torch.std):
            _close(_no_fallback(lambda: fn(x.to(device), **kw), fn.__name__), fn(x, **kw), f"{fn.__name__} {kw}")
        for fn in (torch.var_mean, torch.std_mean):
            g = _no_fallback(lambda: fn(x.to(device), **kw), fn.__name__)
            r = fn(x, **kw)
            _close(g[0], r[0], f"{fn.__name__}[0] {kw}")
            _close(g[1], r[1], f"{fn.__name__}[1] {kw}")


def test_shape_and_selection(device):
    torch.manual_seed(0)
    x = torch.randn(3, 4, 5)
    xd = x.to(device)
    checks = {
        "flip": (lambda t: torch.flip(t, (0, 2))), "fliplr": torch.fliplr, "roll": (lambda t: torch.roll(t, (2, -1), (1, 2))),
        "roll flat": (lambda t: torch.roll(t, 7)), "rot90": (lambda t: torch.rot90(t, 1, (1, 2))),
        "unfold": (lambda t: t.unfold(2, 2, 2).contiguous()),
        "reflect1d": (lambda t: F.pad(t, (2, 3), mode="reflect")),
        "reflect2d": (lambda t: F.pad(t.unsqueeze(0), (1, 2, 3, 1), mode="reflect")),
        "nearest1d": (lambda t: F.interpolate(t, size=9, mode="nearest")),
        "cummax": (lambda t: torch.cummax(t, 1)[0]), "cummax idx": (lambda t: torch.cummax(t, 1)[1]),
        "cummin": (lambda t: torch.cummin(t, -1)[0]), "cummin idx": (lambda t: torch.cummin(t, -1)[1]),
        "kthvalue": (lambda t: torch.kthvalue(t, 3, 1)[0]), "kthvalue idx": (lambda t: torch.kthvalue(t, 3, 1)[1]),
        "median dim": (lambda t: torch.median(t, 2, keepdim=True)[0]), "median idx": (lambda t: torch.median(t, 2)[1]),
        "median": torch.median, "masked_select": (lambda t: torch.masked_select(t, t > 0.3)),
        "cross": (lambda t: torch.linalg.cross(t[:, :3], t[:, 1:], dim=1)),
    }
    for name, fn in checks.items():
        _close(_no_fallback(lambda: fn(xd), name), fn(x), name)
    nan = x.clone()
    nan[0, 1, 2] = float("nan")
    nan[1, 2] = float("nan")
    for name, fn in (("median nan", lambda t: torch.median(t, 2)[0]), ("nanmedian", lambda t: torch.nanmedian(t, 2)[0]),
                     ("nanmedian all", torch.nanmedian), ("nansum", lambda t: torch.nansum(t, 1)),
                     ("isnan", torch.isnan), ("cummax nan", lambda t: torch.cummax(t, 2)[0])):
        _close(_no_fallback(lambda: fn(nan.to(device)), name), fn(nan), name)
    a = x.clone().requires_grad_()
    ad = x.to(device).requires_grad_()
    F.pad(a, (2, 1), mode="reflect").pow(2).sum().backward()
    _no_fallback(lambda: F.pad(ad, (2, 1), mode="reflect").pow(2).sum().backward(), "reflect backward")
    _close(ad.grad, a.grad, "reflect grad")


def test_composites(device):
    torch.manual_seed(0)
    m, v, s = torch.randn(6, 5), torch.randn(5), torch.randn(6)
    b1, b2, c = torch.randn(3, 6, 4), torch.randn(3, 4, 5), torch.randn(3, 6, 5)
    a, b = torch.randn(40), torch.randn(40)
    a[3], b[7] = float("nan"), float("nan")
    lo, hi = torch.full((40,), -0.5), torch.full((40,), 0.7)
    w = torch.rand(40)
    d = lambda t: t.to(device)  # noqa: E731
    checks = [
        ("addmv", lambda f: torch.addmv(f(s), f(m), f(v), beta=0.5, alpha=2)),
        ("mv", lambda f: f(m) @ f(v)),
        ("baddbmm", lambda f: torch.baddbmm(f(c), f(b1), f(b2), beta=0.3, alpha=1.5)),
        ("addbmm", lambda f: torch.addbmm(f(c[0]), f(b1), f(b2))),
        ("vdot", lambda f: torch.vdot(f(a.nan_to_num()), f(b.nan_to_num()))),
        ("rsub", lambda f: torch.rsub(f(a), f(b), alpha=2)),
        ("clamp T", lambda f: torch.clamp(f(a), f(lo), f(hi))), ("clamp_min T", lambda f: torch.clamp_min(f(a), f(lo))),
        ("clamp_max T", lambda f: torch.clamp_max(f(a), f(hi))),
        ("fmax", lambda f: torch.fmax(f(a), f(b))), ("fmin", lambda f: torch.fmin(f(a), f(b))),
        ("maximum nan", lambda f: torch.maximum(f(a), f(b))), ("minimum nan", lambda f: torch.minimum(f(a), f(b))),
        ("pow TT", lambda f: torch.pow(f(a.abs()), f(b))),
        ("div trunc", lambda f: torch.div(f(a) * 5, f(b), rounding_mode="trunc")),
        ("div floor", lambda f: torch.div(f(a) * 5, f(b), rounding_mode="floor")),
        ("lerp T", lambda f: torch.lerp(f(a), f(b), f(w))),
        ("dist", lambda f: torch.dist(f(a.nan_to_num()), f(b.nan_to_num()), 3)),
        ("fill_ T", lambda f: f(a).clone().fill_(f(torch.tensor(2.5)))),
    ]
    for name, fn in checks:
        _close(_no_fallback(lambda: fn(d), name), fn(lambda t: t), name, tol=1e-4)
    ai, bi = torch.randint(-20, 20, (100,)), torch.randint(1, 6, (100,))
    _close(torch.div(ai.to(device), bi.to(device), rounding_mode="floor"), torch.div(ai, bi, rounding_mode="floor"), "floor int")
    for name, fn in (("logical_and", torch.logical_and), ("logical_or", torch.logical_or), ("logical_xor", torch.logical_xor)):
        u, v = torch.randint(0, 3, (50,)), torch.randn(50).round()
        _close(_no_fallback(lambda: fn(u.to(device), v.to(device)), name), fn(u, v), name)
        _close(_no_fallback(lambda: fn(u.to(device) > 0, v.to(device) > 0), name), fn(u > 0, v > 0), name)
    assert torch.equal(a.nan_to_num().to(device), a.nan_to_num().to(device).clone())
    assert not torch.equal(a.to(device), b.to(device))


def test_random_distributions(device):
    torch.manual_seed(0)
    n = 200_000
    for name, fill, mean, var in (
            ("exponential", lambda t: t.exponential_(2.0), 0.5, 0.25),
            ("geometric", lambda t: t.geometric_(0.25), 4.0, 12.0),
            ("log_normal", lambda t: t.log_normal_(0.0, 0.5), float(torch.tensor(0.125).exp()),
             float((torch.tensor(0.25).exp() - 1) * torch.tensor(0.25).exp()))):
        t = _no_fallback(lambda: fill(torch.empty(n, device=device)), name).cpu().double()
        assert abs(t.mean().item() - mean) < 0.02 * max(1.0, mean), (name, t.mean().item(), mean)
        assert abs(t.var().item() - var) < 0.05 * max(1.0, var), (name, t.var().item(), var)
    c = _no_fallback(lambda: torch.empty(n, device=device).cauchy_(1.0, 2.0), "cauchy").cpu()
    assert abs(c.median().item() - 1.0) < 0.05 and abs((c - 1).abs().median().item() - 2.0) < 0.05


def test_scalar_reductions_and_views(device):
    x = torch.tensor(1.5)
    xd = x.to(device)
    for name, fn in (("max dim", lambda t: torch.max(t, 0)[0]), ("min dim", lambda t: torch.min(t, 0, keepdim=True)[1]),
                     ("all dim", lambda t: torch.all(t, 0, keepdim=True)), ("any dim", lambda t: torch.any(t, 0)),
                     ("mean dim", lambda t: torch.mean(t, 0)), ("index_select", lambda t: t.index_select(0, torch.tensor([0], device=t.device))),
                     ("logsumexp", lambda t: torch.logsumexp(t, 0)), ("sum dim", lambda t: t.sum(0, keepdim=True))):
        _close(fn(xd), fn(x), name)
    v = torch.randn(4, device=device)
    assert v.t()._base is v and torch.transpose(v, 0, 0)._base is v
    assert v.squeeze(0)._base is v


def test_row_reductions_vector_path(device):
    torch.manual_seed(0)
    for dtype, tol in ((torch.float32, 1e-5), (torch.bfloat16, 2e-2), (torch.float16, 2e-3)):
        for shape in ((64, 2048), (3, 5, 256), (2, 16384), (5, 1000), (7, 1030)):
            x = torch.randn(shape).to(dtype)
            xd = x.to(device)
            for name, fn in (("sum", lambda t: t.sum(-1)), ("mean", lambda t: t.mean(-1, keepdim=True)),
                             ("amax", lambda t: t.amax(-1)), ("amin", lambda t: t.amin(-1)),
                             ("sum all", lambda t: t.sum())):
                got = fn(xd).cpu().float()
                ref = fn(x.float())
                assert torch.allclose(got, ref.to(dtype).float(), rtol=tol, atol=tol * shape[-1] ** 0.5), \
                    (name, dtype, shape, (got - ref).abs().max())
    nan = torch.randn(4, 2048)
    nan[1, 7] = float("nan")
    nan[2, 2047] = float("nan")
    for fn in (lambda t: t.amax(-1), lambda t: t.amin(-1), lambda t: t.amax(0)):
        _close(fn(nan.to(device)), fn(nan), "nan propagation")


def test_random_integers(device):
    torch.manual_seed(0)
    for dtype in (torch.int64, torch.int32, torch.uint8, torch.float32, torch.bfloat16):
        x = _no_fallback(lambda: torch.randint(3, 17, (20000,), device=device, dtype=dtype), "randint")
        c = x.cpu().to(torch.int64)
        assert c.min().item() == 3 and c.max().item() == 16, (dtype, c.min(), c.max())
        assert (torch.bincount(c - 3).float() / 20000 - 1 / 14).abs().max() < 0.02, dtype
    y = _no_fallback(lambda: torch.empty(5000, dtype=torch.int64, device=device).random_(10), "random_.to")
    assert y.min().item() >= 0 and y.max().item() <= 9
    b = _no_fallback(lambda: torch.empty(5000, dtype=torch.bool, device=device).random_(), "random_ bool")
    assert 0.4 < b.float().mean().item() < 0.6
