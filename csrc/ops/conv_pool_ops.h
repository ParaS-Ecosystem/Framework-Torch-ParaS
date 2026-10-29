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

#include <array>
#include <tuple>

#include <ATen/ATen.h>

namespace ptsycl {

bool conv_device_supported(const at::Tensor& input, const at::Tensor& weight, bool transposed);
at::Tensor conv_forward_device(const at::Tensor& input, const at::Tensor& weight,
                               const c10::optional<at::Tensor>& bias, at::IntArrayRef stride,
                               at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups);
std::tuple<at::Tensor, at::Tensor, at::Tensor> conv_backward_device(
    const at::Tensor& grad_output, const at::Tensor& input, const at::Tensor& weight, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, std::array<bool, 3> mask);

at::Tensor conv_transposed_forward_device(const at::Tensor& input, const at::Tensor& weight,
                                          const c10::optional<at::Tensor>& bias, at::IntArrayRef stride,
                                          at::IntArrayRef padding, at::IntArrayRef output_padding,
                                          at::IntArrayRef dilation, int64_t groups);
std::tuple<at::Tensor, at::Tensor, at::Tensor> conv_transposed_backward_device(
    const at::Tensor& grad_output, const at::Tensor& input, const at::Tensor& weight, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation, int64_t groups,
    std::array<bool, 3> mask);

at::Tensor adaptive_avg_pool2d_device(const at::Tensor& self, at::IntArrayRef output_size);
at::Tensor adaptive_avg_pool2d_backward_device(const at::Tensor& grad_output, const at::Tensor& self);

std::tuple<at::Tensor&, at::Tensor&> max_pool2d_with_indices_out_device(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding,
    at::IntArrayRef dilation, bool ceil_mode, at::Tensor& out, at::Tensor& indices);
at::Tensor& max_pool2d_with_indices_backward_out_device(
    const at::Tensor& grad_output, const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const at::Tensor& indices,
    at::Tensor& grad_input);
at::Tensor& avg_pool2d_out_device(const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
                                  at::IntArrayRef padding, bool ceil_mode, bool count_include_pad,
                                  c10::optional<int64_t> divisor_override, at::Tensor& out);
at::Tensor& avg_pool2d_backward_out_device(const at::Tensor& grad_output, const at::Tensor& self,
                                           at::IntArrayRef kernel_size, at::IntArrayRef stride,
                                           at::IntArrayRef padding, bool ceil_mode, bool count_include_pad,
                                           c10::optional<int64_t> divisor_override, at::Tensor& grad_input);

at::Tensor& upsample_nearest2d_device(const at::Tensor& self, int64_t Ho, int64_t Wo,
                                      c10::optional<double> scales_h, c10::optional<double> scales_w,
                                      bool exact, at::Tensor& out);
at::Tensor& upsample_nearest2d_backward_device(const at::Tensor& grad_output, at::IntArrayRef input_size,
                                               c10::optional<double> scales_h, c10::optional<double> scales_w,
                                               bool exact, at::Tensor& grad_input);
at::Tensor& upsample_bilinear2d_device(const at::Tensor& self, int64_t Ho, int64_t Wo, bool align_corners,
                                       c10::optional<double> scales_h, c10::optional<double> scales_w,
                                       at::Tensor& out);
at::Tensor& upsample_bilinear2d_backward_device(const at::Tensor& grad_output, at::IntArrayRef input_size,
                                                bool align_corners, c10::optional<double> scales_h,
                                                c10::optional<double> scales_w, at::Tensor& grad_input);

}
