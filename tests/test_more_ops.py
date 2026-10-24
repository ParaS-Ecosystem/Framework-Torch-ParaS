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
