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

#include <ATen/ATen.h>
#include <ATen/autocast_mode.h>
#include <torch/library.h>

namespace at::autocast {
namespace {

#define PTSYCL_KERNEL_LOWER(...) KERNEL_PRIVATEUSEONE(__VA_ARGS__, lower_precision_fp)
#define PTSYCL_KERNEL_FP32(...) KERNEL_PRIVATEUSEONE(__VA_ARGS__, fp32)
#define PTSYCL_KERNEL_FP32_SET_OPT_DTYPE(...) KERNEL_PRIVATEUSEONE(__VA_ARGS__, fp32_set_opt_dtype)
#define PTSYCL_KERNEL_PROMOTE(...) KERNEL_PRIVATEUSEONE(__VA_ARGS__, promote)

TORCH_LIBRARY_IMPL(_, AutocastPrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(aten, AutocastPrivateUse1, m) {
    AT_FORALL_LOWER_PRECISION_FP(PTSYCL_KERNEL_LOWER)
    AT_FORALL_FP32(PTSYCL_KERNEL_FP32)
    AT_FORALL_FP32_SET_OPT_DTYPE(PTSYCL_KERNEL_FP32_SET_OPT_DTYPE)
    AT_FORALL_DIFFERENT_REDISPATCH_SIGNATURE(KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_PRIVATEUSEONE)
    AT_FORALL_PROMOTE(PTSYCL_KERNEL_PROMOTE)
}

}
}
