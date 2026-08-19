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
//
// !!! DO NOT ADD THIS FILE TO CMakeLists.txt YET !!!
//
// The project README states tensor lifecycle -- "allocation, strided
// layouts, views, copies in every direction (host/device, cross-device,
// dtype conversion), resize" -- is already implemented, which almost
// certainly means `aten::copy_` (and/or `aten::_copy_from`) is already
// registered for PrivateUse1 somewhere in csrc/ops/tensor_ops.cpp.
//
// `TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)` registering the same op name
// twice -- even from two different .cpp files -- aborts at library load
// time ("Tried to register multiple kernels for the same dispatch key").
//
// Before wiring this in:
//   grep -n "aten::copy_\|aten::_copy_from" csrc/ops/tensor_ops.cpp
// If it's already there, either delete this file, or move whatever this
// version does differently into tensor_ops.cpp's existing implementation
// instead of adding a second registration.
// -----------------------------------------------------------------------------

#include "core/kernels.h"

namespace ptsycl {
namespace {

using at::Tensor;

// copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)
Tensor& copy_(Tensor& self, const Tensor& src, bool /*non_blocking*/) {
    PTSYCL_TRACE_OP("copy_");
    TORCH_CHECK(at::is_expandable_to(src.sizes(), self.sizes()),
                "paras copy_: src of shape ", src.sizes(),
                " is not broadcastable to self of shape ", self.sizes());
    TORCH_CHECK(spec_supported(self) && spec_supported(src),
                "paras copy_: rank exceeds kernel limit");

    Tensor s = src.expand(self.sizes());

    // Cross-device transfer: bounce through host if src and self live on
    // different devices (paras CPU engine <-> paras GPU engine <-> a
    // non-paras device such as plain CPU). Mirrors to_compute() in
    // indexing_ops.cpp, generalized to go through c10::kCPU as the
    // common intermediate rather than assuming one side is already CPU.
    if (self.device() != s.device()) {
        Tensor host = s.device().is_cpu() ? s : s.to(c10::kCPU);
        s = self.device().is_cpu() ? host : host.to(self.device());
    }
    if (self.scalar_type() != s.scalar_type()) {
        s = s.to(self.scalar_type());
    }
    s = s.contiguous();

    auto& q = queue_for(self);
    const int64_t n = self.numel();
    if (n == 0) return self;

    const auto self_spec = make_spec(self);

    AT_DISPATCH_ALL_TYPES_AND3(
        c10::kBool, c10::kHalf, c10::kBFloat16, self.scalar_type(),
        "ptsycl_copy_", [&] {
            scalar_t*       pself = data_ptr<scalar_t>(self);
            const scalar_t* psrc  = data_ptr<scalar_t>(s);
            const auto sspec = self_spec;

            launch_flat(q, n, [=](std::size_t flat_) {
                const int64_t flat = static_cast<int64_t>(flat_);
                pself[sspec.index(flat)] = psrc[flat];
            });
        });
    return self;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("aten::copy_", &ptsycl::copy_);
}

} // namespace ptsycl
