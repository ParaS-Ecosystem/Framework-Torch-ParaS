// -----------------------------------------------------------------------------
// Copyright (c) 2026 Centre for Development of Advanced Computing (C-DAC)
//
// This file is part of Torch_ParaS, a component of the ParaS Ecosystem.
//
// This library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License (LGPL)
// version 3 as published by the Free Software Foundation.
// -----------------------------------------------------------------------------

#pragma once

#include <ATen/ATen.h>

#include <cstdint>
#include <string>

namespace ptsycl::clap_dnn {

// Supported data types and input limits for Sigmoid and SiLU:
//   - CUDA (cuDNN) / ROCm (MIOpen): Float32, Float16, BFloat16.
//   - CPU (oneDNN 3.14 on Xeon Gold 6132): Float32 and BFloat16 are supported.
//     Float16 is unsupported on this CPU/oneDNN setup and raises a RuntimeError
//     in dnnl_eltwise_forward_primitive_desc_create.
//   - Scalar (0-D), empty, Float64, and non-contiguous/strided tensors skip
//     CLAP and fall back to the ParaS unary kernel.
//   - Contiguous CUDA tensors with rank > 8 are rejected by cuDNN
//     TensorDescriptor (rank 1..8 limit).
//   - Backend overrides via CLAP_BACKEND or command-line flags (-cpu, -cuda,
//     -rocm, -gpu) must match the tensor's execution device.

bool try_sigmoid_forward(const at::Tensor& self, at::Tensor& out);
bool try_silu_forward(const at::Tensor& self, at::Tensor& out);

bool activation_uses_vendor(int device_index, const std::string& dtype);

bool is_enabled() noexcept;
void reset_stats() noexcept;
std::int64_t call_count() noexcept;
std::int64_t backend_call_count(const std::string& backend) noexcept;
std::int64_t fallback_count() noexcept;
std::string last_backend_name();

} // namespace ptsycl::clap_dnn
