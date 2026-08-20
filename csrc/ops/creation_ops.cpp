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
// Factory-function wiring: zeros/ones/full/*_like/arange, each built by
// composing at::empty(...) with fill_ (or copy_, for arange) -- no new
// low-level loop, same reasoning as rms_swiglu_ops.cpp. These are lower
// collision risk than matmul/copy_ (raw allocation is a core-runtime
// concern that would live in csrc/core/allocator.cpp, not here), but
// still worth a quick grep for `aten::zeros\|aten::full\|aten::arange`
// in tensor_ops.cpp before wiring this in, in case any were added as
// direct kernels rather than left to the CompositeImplicitAutograd
// fallback.
// -----------------------------------------------------------------------------

#include "core/kernels.h"

namespace ptsycl {
namespace {

using at::Tensor;

Tensor zeros(at::IntArrayRef size, c10::optional<c10::ScalarType> dtype,
             c10::optional<c10::Layout> layout, c10::optional<c10::Device> device,
             c10::optional<bool> pin_memory) {
    PTSYCL_TRACE_OP("zeros");
    Tensor out = at::empty(size, dtype, layout, device, pin_memory, c10::nullopt);
    return out.zero_();
}

Tensor ones(at::IntArrayRef size, c10::optional<c10::ScalarType> dtype,
            c10::optional<c10::Layout> layout, c10::optional<c10::Device> device,
            c10::optional<bool> pin_memory) {
    PTSYCL_TRACE_OP("ones");
    Tensor out = at::empty(size, dtype, layout, device, pin_memory, c10::nullopt);
    return out.fill_(1);
}

Tensor full(at::IntArrayRef size, const at::Scalar& fill_value,
            c10::optional<c10::ScalarType> dtype, c10::optional<c10::Layout> layout,
            c10::optional<c10::Device> device, c10::optional<bool> pin_memory) {
    PTSYCL_TRACE_OP("full");
    Tensor out = at::empty(size, dtype, layout, device, pin_memory, c10::nullopt);
    return out.fill_(fill_value);
}

Tensor zeros_like(const Tensor& self, c10::optional<c10::ScalarType> dtype,
                   c10::optional<c10::Layout> layout, c10::optional<c10::Device> device,
                   c10::optional<bool> pin_memory,
                   c10::optional<c10::MemoryFormat> memory_format) {
    PTSYCL_TRACE_OP("zeros_like");
    Tensor out = at::empty_like(self, dtype, layout, device, pin_memory, memory_format);
    return out.zero_();
}

Tensor ones_like(const Tensor& self, c10::optional<c10::ScalarType> dtype,
                  c10::optional<c10::Layout> layout, c10::optional<c10::Device> device,
                  c10::optional<bool> pin_memory,
                  c10::optional<c10::MemoryFormat> memory_format) {
    PTSYCL_TRACE_OP("ones_like");
    Tensor out = at::empty_like(self, dtype, layout, device, pin_memory, memory_format);
    return out.fill_(1);
}

Tensor full_like(const Tensor& self, const at::Scalar& fill_value,
                  c10::optional<c10::ScalarType> dtype, c10::optional<c10::Layout> layout,
                  c10::optional<c10::Device> device, c10::optional<bool> pin_memory,
                  c10::optional<c10::MemoryFormat> memory_format) {
    PTSYCL_TRACE_OP("full_like");
    Tensor out = at::empty_like(self, dtype, layout, device, pin_memory, memory_format);
    return out.fill_(fill_value);
}

// arange.start_step(Scalar start, Scalar end, Scalar step=1, ...) -> Tensor
// Built on host (the range is a tiny CPU-side computation regardless of
// output device) then a single copy_ moves it to the target device --
// reuses whatever copy_ this backend already has rather than writing a
// third arithmetic-progression kernel.
Tensor arange(const at::Scalar& start, const at::Scalar& end, const at::Scalar& step,
              c10::optional<c10::ScalarType> dtype, c10::optional<c10::Layout> layout,
              c10::optional<c10::Device> device, c10::optional<bool> pin_memory) {
    PTSYCL_TRACE_OP("arange.start_step");
    Tensor host = at::arange(start, end, step,
                              dtype.has_value() ? dtype : c10::optional<c10::ScalarType>(),
                              layout, c10::Device(c10::kCPU), pin_memory);
    Tensor out = at::empty(host.sizes(), host.options().device(
        device.has_value() ? *device : c10::Device(c10::DeviceType::PrivateUse1)));
    out.copy_(host);
    return out;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("aten::zeros", &ptsycl::zeros);
    m.impl("aten::ones", &ptsycl::ones);
    m.impl("aten::full", &ptsycl::full);
    m.impl("aten::zeros_like", &ptsycl::zeros_like);
    m.impl("aten::ones_like", &ptsycl::ones_like);
    m.impl("aten::full_like", &ptsycl::full_like);
    m.impl("aten::arange.start_step", &ptsycl::arange);
}

} // namespace ptsycl
