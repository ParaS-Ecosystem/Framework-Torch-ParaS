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

namespace ptsycl {
namespace {

using at::Tensor;

Tensor swiglu(const Tensor& x, const Tensor& gate) {
    PTSYCL_TRACE_OP("swiglu");
    return x * at::silu(gate);
}

} // namespace

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
