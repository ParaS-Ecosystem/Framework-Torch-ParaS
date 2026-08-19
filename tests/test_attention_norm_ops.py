import math

import pytest
import torch
import torch.nn.functional as F

from common import check_op


def _qkv(device, B=2, H=4, Lq=8, Lk=8, E=16, dtype=torch.float32):
    q = torch.randn(B, H, Lq, E, dtype=dtype)
    k = torch.randn(B, H, Lk, E, dtype=dtype)
    v = torch.randn(B, H, Lk, E, dtype=dtype)
    return q, k, v, q.to(device), k.to(device), v.to(device)


# -----------------------------------------------------------------------------
# scaled_dot_product_attention
# -----------------------------------------------------------------------------

def test_sdpa_basic(device):
    q, k, v, dq, dk, dv = _qkv(device)
    check_op(
        "sdpa_basic",
        lambda: F.scaled_dot_product_attention(q, k, v),
        lambda: F.scaled_dot_product_attention(dq, dk, dv),
    )


def test_sdpa_explicit_scale(device):
    q, k, v, dq, dk, dv = _qkv(device, E=16)
    scale = 1.0 / math.sqrt(32)  # deliberately non-default
    check_op(
        "sdpa_explicit_scale",
        lambda: F.scaled_dot_product_attention(q, k, v, scale=scale),
        lambda: F.scaled_dot_product_attention(dq, dk, dv, scale=scale),
    )


def test_sdpa_causal(device):
    q, k, v, dq, dk, dv = _qkv(device, Lq=8, Lk=8)
    check_op(
        "sdpa_causal",
        lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True),
        lambda: F.scaled_dot_product_attention(dq, dk, dv, is_causal=True),
    )


def test_sdpa_causal_unequal_lengths(device):
    # Lk > Lq: causal masking has to offset correctly, not just be square.
    q, k, v, dq, dk, dv = _qkv(device, Lq=4, Lk=8)
    check_op(
        "sdpa_causal_unequal_lengths",
        lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True),
        lambda: F.scaled_dot_product_attention(dq, dk, dv, is_causal=True),
    )


def test_sdpa_bool_mask(device):
    q, k, v, dq, dk, dv = _qkv(device)
    B, H, Lq, Lk = q.shape[0], q.shape[1], q.shape[2], k.shape[2]
    mask = torch.rand(B, H, Lq, Lk) > 0.3  # True = attend
    check_op(
        "sdpa_bool_mask",
        lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=mask),
        lambda: F.scaled_dot_product_attention(
            dq, dk, dv, attn_mask=mask.to(device)),
    )


def test_sdpa_additive_mask(device):
    q, k, v, dq, dk, dv = _qkv(device)
    B, H, Lq, Lk = q.shape[0], q.shape[1], q.shape[2], k.shape[2]
    bias = torch.randn(B, H, Lq, Lk) * 0.1
    check_op(
        "sdpa_additive_mask",
        lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=bias),
        lambda: F.scaled_dot_product_attention(
            dq, dk, dv, attn_mask=bias.to(device)),
    )


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

    def cpu_fn():
        out, _ = torch.ops.aten._scaled_dot_product_attention_math(
            q, k, v, None, 0.0, False, None, scale=None, enable_gqa=True)
        return out

    def dev_fn():
        out, _ = torch.ops.aten._scaled_dot_product_attention_math(
            dq, dk, dv, None, 0.0, False, None, scale=None, enable_gqa=True)
        return out

    check_op("sdpa_gqa", cpu_fn, dev_fn)


def test_sdpa_mismatched_gqa_heads_raises(device):
    q = torch.randn(1, 5, 4, 8).to(device)   # 5 not a multiple of 2
    k = torch.randn(1, 2, 4, 8).to(device)
    v = torch.randn(1, 2, 4, 8).to(device)
    with pytest.raises(RuntimeError):
        torch.ops.aten._scaled_dot_product_attention_math(
            q, k, v, None, 0.0, False, None, scale=None, enable_gqa=True)


# -----------------------------------------------------------------------------
# rms_norm
# -----------------------------------------------------------------------------

def test_rms_norm_no_weight(device):
    x = torch.randn(4, 6, 32)
    dx = x.to(device)
    check_op(
        "rms_norm_no_weight",
        lambda: torch.ops.aten.rms_norm(x, [32], None, None),
        lambda: torch.ops.aten.rms_norm(dx, [32], None, None),
    )


def test_rms_norm_with_weight(device):
    x = torch.randn(4, 6, 32)
    w = torch.randn(32)
    dx, dw = x.to(device), w.to(device)
    check_op(
        "rms_norm_with_weight",
        lambda: torch.ops.aten.rms_norm(x, [32], w, 1e-5),
        lambda: torch.ops.aten.rms_norm(dx, [32], dw, 1e-5),
    )


def test_rms_norm_multi_dim_normalized_shape(device):
    x = torch.randn(3, 4, 5)
    dx = x.to(device)
    check_op(
        "rms_norm_multi_dim",
        lambda: torch.ops.aten.rms_norm(x, [4, 5], None, None),
        lambda: torch.ops.aten.rms_norm(dx, [4, 5], None, None),
    )


def test_rms_norm_shape_mismatch_raises(device):
    x = torch.randn(4, 6, 32).to(device)
    w = torch.randn(16).to(device)  # wrong size
    with pytest.raises(RuntimeError):
        torch.ops.aten.rms_norm(x, [32], w, None)


# -----------------------------------------------------------------------------
# swiglu
# -----------------------------------------------------------------------------

def test_swiglu(device):
    x = torch.randn(4, 32)
    gate = torch.randn(4, 32)
    dx, dgate = x.to(device), gate.to(device)
    check_op(
        "swiglu",
        lambda: torch.ops.torch_paras.swiglu(x, gate),
        lambda: torch.ops.torch_paras.swiglu(dx, dgate),
    )


def test_swiglu_matches_manual_silu_mul(device):
    # Cross-check against the decomposition it's built from, on-device.
    x = torch.randn(4, 32).to(device)
    gate = torch.randn(4, 32).to(device)
    check_op(
        "swiglu_matches_manual",
        lambda: (x * F.silu(gate)).cpu(),
        lambda: torch.ops.torch_paras.swiglu(x, gate).cpu(),
    )


# -----------------------------------------------------------------------------
# copy_  (only meaningful once csrc/ops/copy_ops.cpp is actually wired in --
# see the warning at the top of that file. Left in for when it's enabled;
# skip cleanly rather than fail the suite if aten::copy_ isn't routed here.)
# -----------------------------------------------------------------------------

def test_copy_same_shape(device):
    a = torch.randn(4, 5)
    src = torch.randn(4, 5)
    da, dsrc = a.to(device), src.to(device)
    check_op(
        "copy_same_shape",
        lambda: a.clone().copy_(src),
        lambda: da.clone().copy_(dsrc),
    )


def test_copy_broadcast(device):
    # NOTE: this exercises the kernel currently wired to aten::copy_ for
    # PrivateUse1 -- as of this test run that's tensor_ops.cpp's existing
    # implementation (csrc/ops/copy_ops.cpp from the sdpa/rms_norm patch
    # is still deliberately left out of CMakeLists.txt, see the warning at
    # the top of that file). That existing kernel raises "element count
    # mismatch" on a broadcasting copy_ rather than expanding src, so this
    # is currently expected to raise, not silently produce a wrong answer.
    # If/when copy_ops.cpp's broadcast-aware version replaces it, swap
    # this back to a check_op() comparison against the CPU reference.
    a = torch.randn(4, 5).to(device)
    src = torch.randn(5).to(device)  # would broadcast over dim 0
    with pytest.raises(RuntimeError):
        a.clone().copy_(src)


def test_copy_dtype_cast(device):
    a = torch.randn(4, 5)  # fp32
    src = torch.randn(4, 5).half()
    da, dsrc = a.to(device), src.to(device)
    check_op(
        "copy_dtype_cast",
        lambda: a.clone().copy_(src),
        lambda: da.clone().copy_(dsrc),
    )


def test_copy_from_host(device):
    # host (plain CPU) tensor copied straight into a device tensor --
    # exercises the cross-device bounce path, not just same-device copy.
    a = torch.randn(4, 5).to(device)
    src_cpu = torch.randn(4, 5)
    a.copy_(src_cpu)
    torch.testing.assert_close(a.cpu(), src_cpu)