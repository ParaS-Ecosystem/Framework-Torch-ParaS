"""Quantised GEMMs on paras devices: _int_mm (int8 -> int32) and _scaled_mm
(fp8 with scales), and fp8 tensors' conversions -- against references on
the CPU, with no CPU fallback."""

import torch
import torch_paras

_FP8 = [t for t in (torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fnuz, torch.float8_e5m2fnuz)]


def _no_fallback(fn, what):
    torch_paras.reset_runtime_stats()
    out = fn()
    fb = torch_paras.runtime_stats().get("cpu_fallbacks", {})
    assert not fb, (what, fb)
    return out


def test_fp8_conversions(device):
    torch.manual_seed(0)
    x = torch.randn(1000) * 4
    for dt in _FP8:
        ref = x.to(dt)
        got = _no_fallback(lambda: x.to(device).to(dt), str(dt))
        assert got.dtype == dt
        assert torch.equal(got.cpu().float(), ref.float()), dt
        assert torch.equal(_no_fallback(lambda: got.to(torch.bfloat16), str(dt)).cpu(), ref.to(torch.bfloat16)), dt


def test_int_mm(device):
    torch.manual_seed(0)
    for m, k, n in ((64, 48, 40), (17, 128, 33), (128, 256, 64)):
        a = torch.randint(-128, 128, (m, k), dtype=torch.int8)
        b = torch.randint(-128, 128, (k, n), dtype=torch.int8)
        ref = (a.double() @ b.double()).to(torch.int32)
        for bd in (b.to(device), b.t().contiguous().to(device).t()):
            got = _no_fallback(lambda: torch._int_mm(a.to(device), bd), "_int_mm")
            assert got.dtype == torch.int32 and torch.equal(got.cpu(), ref), (m, k, n)
        out = torch.empty(m, n, dtype=torch.int32, device=device)
        torch._int_mm(a.to(device), b.to(device), out=out)
        assert torch.equal(out.cpu(), ref)


def test_scaled_mm(device):
    torch.manual_seed(0)
    m, k, n = 64, 96, 48
    av = torch.randint(-4, 5, (m, k)).float()
    bv = torch.randint(-4, 5, (k, n)).float()
    for dt in (torch.float8_e4m3fnuz, torch.float8_e4m3fn):
        a = av.to(dt).to(device)
        b = bv.t().contiguous().to(dt).to(device).t()
        sa, sb = torch.tensor(0.5, device=device), torch.tensor(2.0, device=device)
        raw = av @ bv
        for out_dtype in (torch.float32, torch.bfloat16):
            got = _no_fallback(lambda: torch._scaled_mm(a, b, sa, sb, out_dtype=out_dtype), "_scaled_mm")
            assert got.dtype == out_dtype and torch.equal(got.cpu(), (raw * 1.0).to(out_dtype)), (dt, out_dtype)
        bias = torch.randn(n).to(torch.bfloat16)
        got = torch._scaled_mm(a, b, sa, sb, bias=bias.to(device), out_dtype=torch.bfloat16)
        ref = (raw + bias.float()).to(torch.bfloat16)
        assert (got.cpu().float() - ref.float()).abs().max() <= 0.5, dt
        ra, rb = torch.rand(m, 1) + 0.5, torch.rand(1, n) + 0.5
        got = _no_fallback(lambda: torch._scaled_mm(a, b, ra.to(device), rb.to(device), out_dtype=torch.float32),
                           "_scaled_mm rowwise")
        assert torch.allclose(got.cpu(), raw * ra * rb, rtol=1e-5, atol=1e-4), dt
