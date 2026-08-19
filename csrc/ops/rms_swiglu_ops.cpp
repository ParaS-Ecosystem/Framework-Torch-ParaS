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

// Both ops below are "wiring", not new low-level kernels: they compose
// primitives this tree already registers for PrivateUse1 (pow, mean,
// rsqrt, mul, silu -- all listed under the ~130 existing aten kernels in
// the README). Each call below redispatches through the operator table,
// so it lands on this backend's existing elementwise/reduction kernels
// automatically. If profiling later shows either is a hot path, replace
// the composition with a fused launch_flat kernel following the pattern
// in indexing_ops.cpp (a single-pass parallel reduction).

namespace ptsycl {
namespace {

using at::Tensor;

// rms_norm(Tensor input, int[] normalized_shape, Tensor? weight=None,
//          float? eps=None) -> Tensor
// Reduces over the trailing `normalized_shape.size()` dims, same
// convention as layer_norm. NOTE: real torch defaults `eps` per-dtype
// (see torch.finfo(dtype).eps-derived default in native_layer_norm) --
// 1e-6 below is a reasonable default for fp32/bf16 but check
// `torch.ops.aten.rms_norm.default._schema` / your torch version's
// actual default before relying on omitted eps for fp16.
Tensor rms_norm(const Tensor& input, at::IntArrayRef normalized_shape,
                 const c10::optional<Tensor>& weight,
                 c10::optional<double> eps) {
    PTSYCL_TRACE_OP("rms_norm");
    const int64_t ndim_norm = static_cast<int64_t>(normalized_shape.size());
    TORCH_CHECK(ndim_norm >= 1,
                "paras rms_norm: normalized_shape must be non-empty");
    TORCH_CHECK(input.dim() >= ndim_norm,
                "paras rms_norm: input has fewer dims than normalized_shape");
    for (int64_t i = 0; i < ndim_norm; ++i) {
        TORCH_CHECK(
            input.size(input.dim() - ndim_norm + i) == normalized_shape[i],
            "paras rms_norm: normalized_shape does not match the trailing "
            "dims of input");
    }

    std::vector<int64_t> reduce_dims;
    reduce_dims.reserve(ndim_norm);
    for (int64_t d = input.dim() - ndim_norm; d < input.dim(); ++d)
        reduce_dims.push_back(d);

    const double eps_val = eps.has_value() ? *eps : 1e-6;

    Tensor ms = input.pow(2).mean(reduce_dims, /*keepdim=*/true);
    Tensor inv_rms = at::rsqrt(ms + eps_val);
    Tensor out = input * inv_rms;
    if (weight.has_value() && weight->defined()) {
        TORCH_CHECK(weight->sizes().equals(normalized_shape),
                    "paras rms_norm: weight shape must equal normalized_shape");
        out = out * *weight;
    }
    return out;
}

// torch_paras::swiglu(Tensor x, Tensor gate) -> Tensor
// x * silu(gate)  (SiLU-gated linear unit, as in Llama/PaLM-style MLPs).
// Not a standard aten op, so it's registered under a `torch_paras`
// namespace rather than `aten::` -- call it as `torch.ops.torch_paras.swiglu`.
// x and gate broadcast per normal elementwise rules (typically same shape:
// both are the two halves of a chunked gate_up_proj output).
Tensor swiglu(const Tensor& x, const Tensor& gate) {
    PTSYCL_TRACE_OP("swiglu");
    return x * at::silu(gate);
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("aten::rms_norm", &ptsycl::rms_norm);
}

TORCH_LIBRARY_FRAGMENT(torch_paras, m) {
    m.def("swiglu(Tensor x, Tensor gate) -> Tensor");
}

TORCH_LIBRARY_IMPL(torch_paras, PrivateUse1, m) {
    m.impl("swiglu", &ptsycl::swiglu);
}

// swiglu's body is pure composition (calls at::silu / operator* and lets
// them redispatch on whatever device the tensors are already on), so the
// same function is correct as a CPU kernel too -- register it there as
// well rather than leaving CPU tensors with no kernel to fall back to.
// Useful for writing device-vs-CPU parity tests that call
// torch.ops.torch_paras.swiglu directly on both sides.
TORCH_LIBRARY_IMPL(torch_paras, CPU, m) {
    m.impl("swiglu", &ptsycl::swiglu);
}

} // namespace ptsycl
