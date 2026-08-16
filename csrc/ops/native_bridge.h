// -----------------------------------------------------------------------------
// Copyright (c) 2026 Centre for Development of Advanced Computing (C-DAC)
//
// Part of Torch_ParaS. LGPL v3 (see LICENSE / core/common.h header).
// -----------------------------------------------------------------------------
#pragma once

// Native op bridge
// ----------------
// Instead of copying a paras tensor to the CPU, alias it (zero-copy) to a native
// ATen tensor on its *physical* device so heavy ops dispatch to the real backend
// kernels: cuDNN/cuBLAS/tensor cores on a paras GPU, native ATen on a paras CPU.
// The paras GPU allocation is a plain cudaMalloc pointer, so a CUDA tensor built
// over it with from_blob shares the same storage. The paras GPU Queue runs on
// the default CUDA stream (stream 0) -- the stream native ATen ops use by default
// -- so paras kernels, native ops, and the result copies are ordered by the
// stream itself; no cross-stream synchronization is needed at the hand-off.

#include <ATen/ATen.h>
#include <c10/core/Device.h>

#include "core/common.h"

namespace ptsycl {
namespace native_bridge {

// Alias a paras tensor to a native ATen tensor on its physical device.
inline at::Tensor as_native(const at::Tensor& t) {
    if (!is_paras_tensor(t)) return t;
    auto& q = queue_for(t);
    c10::Device native_dev = c10::Device(c10::kCPU);
#if defined(PTSYCL_BACKEND_CUDA)
    if (q.is_gpu()) native_dev = c10::Device(c10::kCUDA, q.native_id());
#endif
    return at::from_blob(
        t.data_ptr(), t.sizes(), t.strides(),
        /*deleter=*/[](void*) {},
        t.options().device(native_dev));
}

inline c10::optional<at::Tensor> opt_as_native(const c10::optional<at::Tensor>& t) {
    if (!t || !t->defined() || t->numel() == 0) return c10::nullopt;
    return as_native(*t);
}

// Copy a native result into a freshly-allocated paras tensor.
inline at::Tensor to_paras(const at::Tensor& native, c10::Device paras_dev) {
    at::Tensor nat = native.contiguous();
    at::Tensor out = at::empty(nat.sizes(), nat.options().device(paras_dev));
    auto& q = queue_for(paras_dev);
    q.copy(out.data_ptr(), nat.data_ptr(),
           static_cast<std::size_t>(nat.nbytes()), /*blocking=*/false);
    return out;
}

// Write a native result into a caller-provided paras `out` tensor (.out ops).
inline at::Tensor& copy_into_paras(const at::Tensor& native, at::Tensor& out) {
    at::Tensor nat = native.contiguous();
    auto& q = queue_for(out);
    q.copy(out.data_ptr(), nat.data_ptr(),
           static_cast<std::size_t>(nat.nbytes()), /*blocking=*/false);
    return out;
}

} // namespace native_bridge
} // namespace ptsycl
