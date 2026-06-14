#pragma once

// Shared definitions for op implementations.

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <torch/torch.h>

#include "compat/paras_compat.h"
#include "core/context.h"
#include "core/log.h"

namespace ptsycl {

constexpr c10::DeviceType kParasDevice = c10::DeviceType::PrivateUse1;

inline compat::Queue& queue_for(const c10::Device& dev) {
    return Context::instance().queue(static_cast<int>(dev.index()));
}

inline compat::Queue& queue_for(const at::Tensor& t) {
    return queue_for(t.device());
}

inline bool is_paras_tensor(const at::Tensor& t) {
    return t.device().type() == kParasDevice;
}


template <typename T>
inline T* data_ptr(const at::Tensor& t) {
    return static_cast<T*>(t.data_ptr());
}

} // namespace ptsycl
