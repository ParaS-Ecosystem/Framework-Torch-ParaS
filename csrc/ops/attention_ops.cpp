// -----------------------------------------------------------------------------
// Copyright (c) 2026 Centre for Development of Advanced Computing (C-DAC)
//
// This file is part of Torch_ParaS, a component of the ParaS Ecosystem
//
// This library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License (LGPL)
// version 3 as published by the Free Software Foundation.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this library. If not, see <https://www.gnu.org/licenses/>.
// -----------------------------------------------------------------------------

#include "core/kernels.h"

#include <cmath>
#include <limits>

// NOTE ON WHERE THIS HOOKS IN
// ----------------------------------------------------------------------------
// In upstream PyTorch, `aten::scaled_dot_product_attention` is
// CompositeImplicitAutograd: it never reaches a per-backend dispatch table
// directly. It picks a backend (flash / mem-efficient / math) via
// `_fused_sdp_choice` and then calls that backend's op -- for anything
// without a fused kernel, that's `aten::_scaled_dot_product_attention_math`.
//
// That means one of two things needs to be true for PrivateUse1 (paras):
//   1. `_fused_sdp_choice` already falls back to Math for unrecognized
//      backends (it does, by default, for any device that isn't CUDA/MPS/
//      the handful of fused paths PyTorch knows about) -- in which case
//      registering `_scaled_dot_product_attention_math` below is
//      sufficient and `F.scaled_dot_product_attention(...)` "just works".
//   2. Something upstream of that (a custom `_fused_sdp_choice` override,
///     or an older/newer torch version with a different composite
//      boundary) intercepts before Math is chosen -- in which case you'd
//      also need to register the top-level op directly (kept below,
//      commented out, as a fallback).
//
// Verify which applies to your torch build with:
//   python -c "import torch; print(torch.ops.aten._scaled_dot_product_attention_math.default._schema)"
//   python -c "import torch; print(torch._C._dispatch_has_kernel_for_dispatch_key('aten::scaled_dot_product_attention', 'CompositeImplicitAutograd'))"
//
// Either way, this file introduces no new low-level SYCL loop -- it wires
// together kernels this tree already registers for PrivateUse1 (matmul,
// softmax, masked_fill, dropout, tril). It reproduces PyTorch's own math
// fallback, so it has the same O(Lq*Lk) memory profile as eager PyTorch's
// math path -- not a fused/flash kernel. Treat this as the correctness
// baseline; a fused kernel is future work behind the same schema.

namespace ptsycl {
namespace {

using at::Tensor;

// Real upstream schema (verify against your torch version -- this has
// drifted across releases, e.g. `enable_gqa` is recent):
//   _scaled_dot_product_attention_math(
//       Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None,
//       float dropout_p=0.0, bool is_causal=False, Tensor? dropout_mask=None,
//       *, float? scale=None, bool enable_gqa=False) -> (Tensor, Tensor)
// It returns (attn_output, attn_weights) -- callers that only want the
// output still get the weights tensor back and drop it on the Python side.
// `dropout_mask`, if passed, is used verbatim instead of sampling a new
// mask (needed so backward can recompute the same dropout pattern); we
// honor it here rather than silently ignoring it.
std::tuple<Tensor, Tensor> sdpa_math(
        const Tensor& query, const Tensor& key, const Tensor& value,
        const c10::optional<Tensor>& attn_mask, double dropout_p,
        bool is_causal, const c10::optional<Tensor>& dropout_mask,
        c10::optional<double> scale, bool enable_gqa) {
    PTSYCL_TRACE_OP("_scaled_dot_product_attention_math");
    TORCH_CHECK(query.dim() >= 3 && key.dim() >= 3 && value.dim() >= 3,
                "paras sdpa: query/key/value must be at least 3-D "
                "(..., seq, head_dim)");
    TORCH_CHECK(!(is_causal && attn_mask.has_value() && attn_mask->defined()),
                "paras sdpa: is_causal and attn_mask are mutually exclusive");

    Tensor k = key, v = value;
    if (enable_gqa && key.size(-3) != query.size(-3)) {
        // Grouped-query attention: repeat_interleave k/v heads up to
        // query's head count. Assumes head dim is -3 (..., H, L, E).
        TORCH_CHECK(query.size(-3) % key.size(-3) == 0,
                    "paras sdpa: query heads must be a multiple of "
                    "key/value heads for enable_gqa");
        const int64_t rep = query.size(-3) / key.size(-3);
        k = k.repeat_interleave(rep, /*dim=*/-3);
        v = v.repeat_interleave(rep, /*dim=*/-3);
    }

    const int64_t head_dim = query.size(-1);
    const double scale_factor = scale.has_value()
        ? *scale
        : 1.0 / std::sqrt(static_cast<double>(head_dim));

    // (..., Lq, E) @ (..., E, Lk) -> (..., Lq, Lk)
    Tensor scores = at::matmul(query, k.transpose(-2, -1)) * scale_factor;

    if (is_causal) {
        // PyTorch's own reference math fallback uses diagonal=0
        // unconditionally here -- NOT offset by (Lk - Lq). i.e. query
        // position i may attend to key positions [0, i] regardless of
        // how much longer the key sequence is than the query sequence.
        // (Getting this offset wrong is silently plausible-looking but
        // numerically wrong whenever Lq != Lk -- caught by
        // test_sdpa_causal_unequal_lengths.)
        const int64_t Lq = query.size(-2);
        const int64_t Lk = k.size(-2);
        Tensor allow = at::ones({Lq, Lk}, query.options().dtype(c10::kBool));
        allow = at::tril(allow, /*diagonal=*/0);
        scores = scores.masked_fill(allow.logical_not(),
                                     -std::numeric_limits<double>::infinity());
    } else if (attn_mask.has_value() && attn_mask->defined()) {
        if (attn_mask->scalar_type() == c10::kBool) {
            scores = scores.masked_fill(
                attn_mask->logical_not(),
                -std::numeric_limits<double>::infinity());
        } else {
            // additive mask (e.g. already -inf / bias values)
            scores = scores + *attn_mask;
        }
    }

    Tensor attn = at::softmax(scores, /*dim=*/-1);
    if (dropout_p > 0.0) {
        attn = dropout_mask.has_value() && dropout_mask->defined()
            ? attn * *dropout_mask / (1.0 - dropout_p)
            : at::dropout(attn, dropout_p, /*train=*/true);
    }
    Tensor out = at::matmul(attn, v);
    return {out, attn};
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("aten::_scaled_dot_product_attention_math",
           &ptsycl::sdpa_math);

    // Uncomment ONLY if the schema probe in the comment block above shows
    // your torch build does not already funnel scaled_dot_product_attention
    // to Math for an unrecognized backend. Registering both is harmless
    // (they're different op names) but redundant if _fused_sdp_choice
    // already lands on Math.
    //
    // m.impl("aten::scaled_dot_product_attention", &ptsycl::sdpa_math);
}

} // namespace ptsycl
