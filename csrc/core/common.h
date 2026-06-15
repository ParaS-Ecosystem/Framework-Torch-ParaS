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
