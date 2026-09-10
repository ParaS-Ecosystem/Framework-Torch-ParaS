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

#include "core/kernel_utils.h"
namespace ptsycl {
namespace {

using at::Tensor;
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

TORCH_LIBRARY_IMPL(torch_paras, CPU, m) {
    m.impl("swiglu", &ptsycl::swiglu);
}

} // namespace ptsycl
