import math

import torch
import torch.nn.functional as F

from common import assert_close


def _qkv(device, B=2, H=4, Lq=8, Lk=8, E=16, dtype=torch.float32):
    q = torch.randn(B, H, Lq, E, dtype=dtype)
    k = torch.randn(B, H, Lk, E, dtype=dtype)
    v = torch.randn(B, H, Lk, E, dtype=dtype)
    return q, k, v, q.to(device), k.to(device), v.to(device)


def _expect_raises(fn, exc_type=RuntimeError):
    try:
        fn()
    except exc_type:
        return
    except Exception as e:
        raise AssertionError(
            f"expected {exc_type.__name__}, got {type(e).__name__}: {e}"
        ) from e
    else:
        raise AssertionError(
            f"expected {exc_type.__name__} but no exception was raised"
        )


# -----------------------------------------------------------------------------
# scaled_dot_product_attention
# -----------------------------------------------------------------------------

def test_sdpa_basic(device):
    q, k, v, dq, dk, dv = _qkv(device)
    out_ref = F.scaled_dot_product_attention(q, k, v)
    out_dev = F.scaled_dot_product_attention(dq, dk, dv)
    assert_close(out_dev, out_ref, "sdpa_basic")


def test_sdpa_explicit_scale(device):
    q, k, v, dq, dk, dv = _qkv(device, E=16)
    scale = 1.0 / math.sqrt(32)  # deliberately non-default
    out_ref = F.scaled_dot_product_attention(q, k, v, scale=scale)
    out_dev = F.scaled_dot_product_attention(dq, dk, dv, scale=scale)
    assert_close(out_dev, out_ref, "sdpa_explicit_scale")


def test_sdpa_causal(device):
    q, k, v, dq, dk, dv = _qkv(device, Lq=8, Lk=8)
    out_ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out_dev = F.scaled_dot_product_attention(dq, dk, dv, is_causal=True)
    assert_close(out_dev, out_ref, "sdpa_causal")


def test_sdpa_causal_unequal_lengths(device):
    # Lk > Lq: causal masking has to offset correctly, not just be square.
    q, k, v, dq, dk, dv = _qkv(device, Lq=4, Lk=8)
    out_ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out_dev = F.scaled_dot_product_attention(dq, dk, dv, is_causal=True)
    assert_close(out_dev, out_ref, "sdpa_causal_unequal_lengths")


def test_sdpa_bool_mask(device):
    q, k, v, dq, dk, dv = _qkv(device)
    B, H, Lq, Lk = q.shape[0], q.shape[1], q.shape[2], k.shape[2]
    mask = torch.rand(B, H, Lq, Lk) > 0.3  # True = attend
    out_ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    out_dev = F.scaled_dot_product_attention(dq, dk, dv, attn_mask=mask.to(device))
    assert_close(out_dev, out_ref, "sdpa_bool_mask")


def test_sdpa_additive_mask(device):
    q, k, v, dq, dk, dv = _qkv(device)
    B, H, Lq, Lk = q.shape[0], q.shape[1], q.shape[2], k.shape[2]
    bias = torch.randn(B, H, Lq, Lk) * 0.1
    out_ref = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
    out_dev = F.scaled_dot_product_attention(dq, dk, dv, attn_mask=bias.to(device))
    assert_close(out_dev, out_ref, "sdpa_additive_mask")


def test_sdpa_gqa(device):
    # 4 query heads sharing 2 kv heads -- exercises the repeat_interleave
    # path in sdpa_math, called directly since F.scaled_dot_product_attention
    # may not expose enable_gqa on every torch version.
    # NOTE: scale and enable_gqa are keyword-only in the actual schema
    # (`*, float? scale=None, bool enable_gqa=False`) -- passing all 9
    # positionally raises "takes 7 positional argument(s) but 9 were given".
    B, Lq, Lk, E = 2, 8, 8, 16
    q = torch.randn(B, 4, Lq, E)
    k = torch.randn(B, 2, Lk, E)
    v = torch.randn(B, 2, Lk, E)
    dq, dk, dv = q.to(device), k.to(device), v.to(device)

    out_ref, _ = torch.ops.aten._scaled_dot_product_attention_math(
        q, k, v, None, 0.0, False, None, scale=None, enable_gqa=True)
    out_dev, _ = torch.ops.aten._scaled_dot_product_attention_math(
        dq, dk, dv, None, 0.0, False, None, scale=None, enable_gqa=True)
    assert_close(out_dev, out_ref, "sdpa_gqa")


def test_sdpa_mismatched_gqa_heads_raises(device):
    q = torch.randn(1, 5, 4, 8).to(device)   # 5 not a multiple of 2
    k = torch.randn(1, 2, 4, 8).to(device)
    v = torch.randn(1, 2, 4, 8).to(device)
    _expect_raises(lambda: torch.ops.aten._scaled_dot_product_attention_math(
        q, k, v, None, 0.0, False, None, scale=None, enable_gqa=True))


# -----------------------------------------------------------------------------
# rms_norm
# -----------------------------------------------------------------------------

def test_rms_norm_no_weight(device):
    x = torch.randn(4, 6, 32)
    dx = x.to(device)
    out_ref = torch.ops.aten.rms_norm(x, [32], None, None)
    out_dev = torch.ops.aten.rms_norm(dx, [32], None, None)
    assert_close(out_dev, out_ref, "rms_norm_no_weight")


def test_rms_norm_with_weight(device):
    x = torch.randn(4, 6, 32)
    w = torch.randn(32)
    dx, dw = x.to(device), w.to(device)
    out_ref = torch.ops.aten.rms_norm(x, [32], w, 1e-5)
    out_dev = torch.ops.aten.rms_norm(dx, [32], dw, 1e-5)
    assert_close(out_dev, out_ref, "rms_norm_with_weight")


def test_rms_norm_multi_dim_normalized_shape(device):
    x = torch.randn(3, 4, 5)
    dx = x.to(device)
    out_ref = torch.ops.aten.rms_norm(x, [4, 5], None, None)
    out_dev = torch.ops.aten.rms_norm(dx, [4, 5], None, None)
    assert_close(out_dev, out_ref, "rms_norm_multi_dim")


def test_rms_norm_shape_mismatch_raises(device):
    x = torch.randn(4, 6, 32).to(device)
    w = torch.randn(16).to(device)  # wrong size
    _expect_raises(lambda: torch.ops.aten.rms_norm(x, [32], w, None))


# -----------------------------------------------------------------------------
# swiglu
# -----------------------------------------------------------------------------

def test_swiglu(device):
    x = torch.randn(4, 32)
    gate = torch.randn(4, 32)
    dx, dgate = x.to(device), gate.to(device)
    out_ref = torch.ops.torch_paras.swiglu(x, gate)
    out_dev = torch.ops.torch_paras.swiglu(dx, dgate)
    assert_close(out_dev, out_ref, "swiglu")


def test_swiglu_matches_manual_silu_mul(device):
    # Cross-check against the decomposition it's built from, on-device.
    x = torch.randn(4, 32).to(device)
    gate = torch.randn(4, 32).to(device)
    out_ref = (x * F.silu(gate)).cpu()
    out_dev = torch.ops.torch_paras.swiglu(x, gate).cpu()
    assert_close(out_dev, out_ref, "swiglu_matches_manual")


# -----------------------------------------------------------------------------
# copy_  (only meaningful once csrc/ops/copy_ops.cpp is actually wired in --
# see the warning at the top of that file. Left in for when it's enabled;
# skip cleanly rather than fail the suite if aten::copy_ isn't routed here.)
# -----------------------------------------------------------------------------

def test_copy_same_shape(device):
    a = torch.randn(4, 5)
    src = torch.randn(4, 5)
    da, dsrc = a.to(device), src.to(device)
    out_ref = a.clone().copy_(src)
    out_dev = da.clone().copy_(dsrc)
    assert_close(out_dev, out_ref, "copy_same_shape")


def test_copy_broadcast(device):
    # NOTE: this exercises the kernel currently wired to aten::copy_ for
    # PrivateUse1 -- as of this test run that's tensor_ops.cpp's existing
    # implementation (csrc/ops/copy_ops.cpp from the sdpa/rms_norm patch
    # is still deliberately left out of CMakeLists.txt, see the warning at
    # the top of that file). That existing kernel raises "element count
    # mismatch" on a broadcasting copy_ rather than expanding src, so this
    # is currently expected to raise, not silently produce a wrong answer.
    # If/when copy_ops.cpp's broadcast-aware version replaces it, swap
    # this back to an assert_close() comparison against the CPU reference.
    a = torch.randn(4, 5).to(device)
    src = torch.randn(5).to(device)  # would broadcast over dim 0
    _expect_raises(lambda: a.clone().copy_(src))


def test_copy_dtype_cast(device):
    a = torch.randn(4, 5)  # fp32
    src = torch.randn(4, 5).half()
    da, dsrc = a.to(device), src.to(device)
    out_ref = a.clone().copy_(src)
    out_dev = da.clone().copy_(dsrc)
    assert_close(out_dev, out_ref, "copy_dtype_cast")


def test_copy_from_host(device):
    # host (plain CPU) tensor copied straight into a device tensor --
    # exercises the cross-device bounce path, not just same-device copy.
    a = torch.randn(4, 5).to(device)
    src_cpu = torch.randn(4, 5)
    a.copy_(src_cpu)
    assert_close(a.cpu(), src_cpu, "copy_from_host")

def _sdpa_ops(fn):
    """Runs fn, returning (result, names of the ATen ops it dispatched)."""
    from torch.utils._python_dispatch import TorchDispatchMode

    class Record(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.ops = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.ops.append(func.__name__.split(".")[0])
            return func(*args, **(kwargs or {}))

    rec = Record()
    with rec:
        out = fn()
    return out, rec.ops


def test_sdpa_fused_decode_path(device):
    torch.manual_seed(0)
    B, H, Lk, D = 2, 4, 37, 16
    for dtype, tol in ((torch.float32, 1e-5), (torch.bfloat16, 2e-2)):
        for Lq in (1, 5):
            q = torch.randn(B, Lq, H, D).transpose(1, 2).to(dtype)
            k = torch.randn(B, H, Lk, D).to(dtype)
            v = torch.randn(B, H, Lk, D).to(dtype)
            bmask = torch.rand(B, 1, Lq, Lk) > 0.3
            bmask[0, 0, 0] = False
            fmask = torch.randn(1, 1, Lq, Lk).to(dtype)
            for kw in ({}, {"attn_mask": bmask}, {"attn_mask": fmask},
                       {"is_causal": True}, {"scale": 0.3}):
                ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(),
                                                     **{a: (b.float() if a == "attn_mask" and b.dtype != torch.bool else b)
                                                        for a, b in kw.items()})
                if "attn_mask" in kw and kw["attn_mask"].dtype == torch.bool:
                    ref = torch.nan_to_num(ref, nan=0.0)
                dkw = {a: (b.to(device) if torch.is_tensor(b) else b) for a, b in kw.items()}
                out, ops = _sdpa_ops(lambda: F.scaled_dot_product_attention(
                    q.to(device), k.to(device), v.to(device), **dkw))
                assert "_scaled_dot_product_fused_attention_overrideable" in ops, ops
                assert out.dtype == dtype and out.shape == (B, H, Lq, D)
                torch.testing.assert_close(out.cpu().float(), ref, atol=tol, rtol=tol)


def test_sdpa_training_and_long_query_use_math(device):
    q = torch.randn(1, 2, 3, 8, requires_grad=True)
    k = torch.randn(1, 2, 6, 8, requires_grad=True)
    v = torch.randn(1, 2, 6, 8, requires_grad=True)
    dq, dk, dv = (t.detach().to(device).requires_grad_() for t in (q, k, v))
    out, ops = _sdpa_ops(lambda: F.scaled_dot_product_attention(dq, dk, dv, is_causal=True))
    assert "_scaled_dot_product_fused_attention_overrideable" not in ops
    F.scaled_dot_product_attention(q, k, v, is_causal=True).sum().backward()
    out.sum().backward()
    for a, b in ((dq, q), (dk, k), (dv, v)):
        torch.testing.assert_close(a.grad.cpu(), b.grad, atol=1e-5, rtol=1e-4)


def test_sdpa_fused_prefill_path(device):
    import torch_paras
    torch.manual_seed(0)
    gpu = torch_paras.is_gpu_device(torch.device(device).index or 0)
    B, H, L, D = 1, 3, 300, 32
    for dtype, tol in ((torch.float32, 1e-4), (torch.bfloat16, 3e-2)):
        q = torch.randn(B, L, H, D).transpose(1, 2).to(dtype)
        k = torch.randn(B, H, L, D).to(dtype)
        v = torch.randn(B, H, L, D).to(dtype)
        bmask = torch.rand(B, 1, L, L) > 0.2
        bmask[0, 0, 7] = False
        fmask = torch.triu(torch.full((L, L), float("-inf")), 1)[None, None].to(dtype)
        for kw in ({"is_causal": True}, {"attn_mask": bmask}, {"attn_mask": fmask}, {}):
            ref = F.scaled_dot_product_attention(
                q.float(), k.float(), v.float(),
                **{a: (b.float() if a == "attn_mask" and b.dtype != torch.bool else b) for a, b in kw.items()})
            ref = torch.nan_to_num(ref, nan=0.0)
            dkw = {a: (b.to(device) if torch.is_tensor(b) else b) for a, b in kw.items()}
            out, ops = _sdpa_ops(lambda: F.scaled_dot_product_attention(
                q.to(device), k.to(device), v.to(device), **dkw))
            fused = "_scaled_dot_product_fused_attention_overrideable" in ops
            assert fused == gpu, (fused, gpu, ops[:6])
            torch.testing.assert_close(out.cpu().float(), ref, atol=tol, rtol=tol)


def test_sdpa_fused_precision_matches_math(device):
    from torch.nn.attention import sdpa_kernel, SDPBackend
    torch.manual_seed(0)
    for L in (8, 300):
        q = 3 * torch.randn(1, 4, L, 64)
        k = 3 * torch.randn(1, 4, L, 64)
        v = torch.randn(1, 4, L, 64)
        mask = torch.triu(torch.full((L, L), torch.finfo(torch.bfloat16).min), 1)[None, None]
        ref = F.scaled_dot_product_attention(q.double(), k.double(), v.double(), attn_mask=mask.double())
        qb, kb, vb, mb = (t.to(torch.bfloat16).to(device) for t in (q, k, v, mask))
        err = lambda o: ((o.cpu().double() - ref).norm() / ref.norm()).item()
        fused = err(F.scaled_dot_product_attention(qb, kb, vb, attn_mask=mb))
        with sdpa_kernel(SDPBackend.MATH):
            math_err = err(F.scaled_dot_product_attention(qb, kb, vb, attn_mask=mb))
        assert fused <= 1.1 * math_err, (L, fused, math_err)


def test_sdpa_long_keys(device):
    import torch_paras
    torch.manual_seed(0)
    gpu = torch_paras.is_gpu_device(torch.device(device).index or 0)
    B, H, L, D = 1, 2, 5000, 64
    q = torch.randn(B, L, H, D).transpose(1, 2)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    for dtype, tol, want in ((torch.bfloat16, 2.5e-2, gpu), (torch.float16, 4e-3, gpu), (torch.float32, 1e-4, False)):
        out, ops = _sdpa_ops(lambda: F.scaled_dot_product_attention(
            q.to(dtype).to(device), k.to(dtype).to(device), v.to(dtype).to(device), is_causal=True))
        fused = "_scaled_dot_product_fused_attention_overrideable" in ops
        assert fused == want, (dtype, fused, ops[:6])
        torch.testing.assert_close(out.cpu().float(), ref, atol=tol, rtol=tol)


def test_sdpa_flash_kernel_cases(device):
    torch.manual_seed(0)
    cases = [(2, 3, 130, 130, 64), (1, 2, 70, 200, 128), (1, 4, 257, 257, 64)]
    for dtype, tol in ((torch.bfloat16, 2.5e-2), (torch.float16, 4e-3)):
        for B, H, Lq, Lk, D in cases:
            q = torch.randn(B, Lq, H, D).transpose(1, 2).to(dtype)
            k = torch.randn(B, H, Lk, D).to(dtype)
            v = torch.randn(B, H, Lk, D).to(dtype)
            bmask = torch.rand(B, 1, Lq, Lk) > 0.3
            bmask[0, 0, 5] = False
            fmask = torch.randn(B, 1, Lq, Lk).to(dtype)
            for kw in ({"is_causal": True}, {"attn_mask": bmask}, {"attn_mask": fmask}, {}, {"scale": 0.2}):
                ref = F.scaled_dot_product_attention(
                    q.double(), k.double(), v.double(),
                    **{n: (t.double() if n == "attn_mask" and t.dtype != torch.bool else t) for n, t in kw.items()})
                ref = torch.nan_to_num(ref, nan=0.0)
                dkw = {n: (t.to(device) if torch.is_tensor(t) else t) for n, t in kw.items()}
                out = F.scaled_dot_product_attention(q.to(device), k.to(device), v.to(device), **dkw)
                assert out.dtype == dtype and out.shape == (B, H, Lq, D)
                err = ((out.cpu().double() - ref).norm() / ref.norm()).item()
                assert err < tol, (dtype, B, H, Lq, Lk, D, list(kw), err)


def test_sdpa_flash_block_masks(device):
    torch.manual_seed(0)
    B, H, L, D = 1, 2, 300, 64
    for dtype in (torch.bfloat16, torch.float16):
        q = torch.randn(B, L, H, D).transpose(1, 2).to(dtype).to(device)
        k = torch.randn(B, H, L, D).to(dtype).to(device)
        v = torch.randn(B, H, L, D).to(dtype).to(device)
        causal = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        additive = torch.triu(torch.full((L, L), torch.finfo(dtype).min), 1)[None, None].to(dtype)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=additive.to(device))
        assert torch.equal(out.cpu(), causal.cpu()), dtype
        boolean = torch.ones(L, L, dtype=torch.bool).tril()[None, None]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=boolean.to(device))
        assert torch.equal(out.cpu(), causal.cpu()), dtype
        plain = F.scaled_dot_product_attention(q, k, v)
        zero = torch.zeros(B, 1, L, L, dtype=dtype, device=device)
        assert torch.equal(F.scaled_dot_product_attention(q, k, v, attn_mask=zero).cpu(), plain.cpu())
        pad = additive.clone()
        pad[..., :, :20] = torch.finfo(dtype).min
        ref = F.scaled_dot_product_attention(q.cpu().double(), k.cpu().double(), v.cpu().double(),
                                             attn_mask=pad.double())
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=pad.to(device))
        err = ((out.cpu().double() - ref).norm() / ref.norm()).item()
        assert err < 2.5e-2, (dtype, err)
        m = additive.to(device)
        F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        m.zero_()
        assert torch.equal(F.scaled_dot_product_attention(q, k, v, attn_mask=m).cpu(), plain.cpu())
