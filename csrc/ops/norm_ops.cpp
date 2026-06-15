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
#include <ATen/ATen.h>
#include <torch/torch.h>

namespace ptsycl {
namespace {

using namespace torch;

static Tensor to_cpu(const Tensor& t) {
    return t.contiguous().to(c10::Device(c10::kCPU));
}

static Tensor to_device(const Tensor& cpu_t, c10::Device dev) {
    Tensor result = at::empty(cpu_t.sizes(), at::device(dev).dtype(cpu_t.scalar_type()));
    auto& q = queue_for(result);
    q.copy(result.data_ptr(), cpu_t.contiguous().data_ptr(),
           static_cast<std::size_t>(cpu_t.contiguous().nbytes()),
           /*blocking=*/true);
    return result;
}

static c10::optional<Tensor> opt_to_cpu(const c10::optional<Tensor>& t) {
    if (!t || t->numel() == 0) return c10::nullopt;
    return to_cpu(*t);
}

static void copy_cpu_to_device_inplace(const Tensor& cpu_src, Tensor& dst) {
    Tensor c = cpu_src.contiguous();
    auto& q = queue_for(dst);
    q.copy(dst.data_ptr(), c.data_ptr(), static_cast<std::size_t>(c.nbytes()), true);
}

::std::tuple<Tensor, Tensor, Tensor> native_batch_norm(
    const Tensor& input,
    const c10::optional<Tensor>& weight,
    const c10::optional<Tensor>& bias,
    const c10::optional<Tensor>& running_mean,
    const c10::optional<Tensor>& running_var,
    bool training, double momentum, double eps)
{
    PTSYCL_TRACE_OP("native_batch_norm");
    bool weight_present = weight && weight->numel() > 0;
    bool bias_present   = bias   && bias->numel()   > 0;
    bool mean_present   = running_mean && running_mean->numel() > 0;
    bool var_present    = running_var  && running_var->numel()  > 0;
    TORCH_CHECK(weight_present == bias_present,
                "Can have affine or not affine but not partial");
    TORCH_CHECK(mean_present && var_present,
                "Running stats must be present");

    c10::Device dev = input.device();

    Tensor cpu_input  = to_cpu(input);
    auto   cpu_weight = opt_to_cpu(weight);
    auto   cpu_bias   = opt_to_cpu(bias);
    Tensor cpu_mean = to_cpu(*running_mean);
    Tensor cpu_var  = to_cpu(*running_var);

    auto result = at::native_batch_norm(cpu_input, cpu_weight, cpu_bias,
                                        cpu_mean, cpu_var,
                                        training, momentum, eps);
    auto cpu_out = std::get<0>(result);
    auto cpu_save_mean = std::get<1>(result);
    auto cpu_save_var = std::get<2>(result);

    if (mean_present)
        copy_cpu_to_device_inplace(cpu_mean, const_cast<Tensor&>(*running_mean));
    if (var_present)
        copy_cpu_to_device_inplace(cpu_var,  const_cast<Tensor&>(*running_var));

    Tensor out       = to_device(cpu_out, dev);
    Tensor save_mean = cpu_save_mean.defined() ? to_device(cpu_save_mean, dev) : Tensor{};
    Tensor save_var  = cpu_save_var.defined()  ? to_device(cpu_save_var,  dev) : Tensor{};

    return {out, save_mean, save_var};
}

::std::tuple<Tensor, Tensor, Tensor> native_batch_norm_backward(
    const Tensor& grad_out,
    const Tensor& input,
    const c10::optional<Tensor>& weight,
    const c10::optional<Tensor>& running_mean,
    const c10::optional<Tensor>& running_var,
    const c10::optional<Tensor>& save_mean,
    const c10::optional<Tensor>& save_var,
    bool train, double eps,
    ::std::array<bool, 3> output_mask)
{
    PTSYCL_TRACE_OP("native_batch_norm_backward");
    c10::Device dev = input.device();

    Tensor cpu_grad   = to_cpu(grad_out);
    Tensor cpu_input  = to_cpu(input);
    auto   cpu_weight = opt_to_cpu(weight);
    auto   cpu_rm     = opt_to_cpu(running_mean);
    auto   cpu_rv     = opt_to_cpu(running_var);
    auto   cpu_sm     = opt_to_cpu(save_mean);
    auto   cpu_sv     = opt_to_cpu(save_var);

    auto result = at::native_batch_norm_backward(
            cpu_grad, cpu_input, cpu_weight,
            cpu_rm, cpu_rv, cpu_sm, cpu_sv,
            train, eps, output_mask);

    auto cpu_dx = std::get<0>(result);
    auto cpu_dw = std::get<1>(result);
    auto cpu_db = std::get<2>(result);

    Tensor dx = cpu_dx.defined() ? to_device(cpu_dx, dev) : Tensor{};
    Tensor dw = cpu_dw.defined() ? to_device(cpu_dw, dev) : Tensor{};
    Tensor db = cpu_db.defined() ? to_device(cpu_db, dev) : Tensor{};

    return {dx, dw, db};
}

std::tuple<Tensor, Tensor, Tensor> native_layer_norm(
    const Tensor& input,
    c10::SymIntArrayRef normalized_shape,
    const c10::optional<Tensor>& weight,
    const c10::optional<Tensor>& bias,
    double eps)
{
    PTSYCL_TRACE_OP("native_layer_norm");
    c10::Device dev = input.device();

    Tensor cpu_input  = to_cpu(input);
    auto   cpu_weight = opt_to_cpu(weight);
    auto   cpu_bias   = opt_to_cpu(bias);

    std::vector<int64_t> ns;
    for (auto v : normalized_shape) ns.push_back(v.expect_int());
    c10::IntArrayRef norm_shape(ns.data(), ns.size());

    auto result = at::native_layer_norm(cpu_input, norm_shape, cpu_weight, cpu_bias, eps);
    auto cpu_out = std::get<0>(result);
    auto cpu_mean = std::get<1>(result);
    auto cpu_rstd = std::get<2>(result);

    Tensor out  = to_device(cpu_out,  dev);
    Tensor mean = to_device(cpu_mean, dev);
    Tensor rstd = to_device(cpu_rstd, dev);

    return {out, mean, rstd};
}

std::tuple<Tensor, Tensor, Tensor> native_layer_norm_backward(
    const Tensor& grad_out,
    const Tensor& input,
    c10::SymIntArrayRef normalized_shape,
    const Tensor& save_mean,
    const Tensor& save_rstd,
    const c10::optional<Tensor>& weight,
    const c10::optional<Tensor>& bias,
    ::std::array<bool, 3> output_mask)
{
    PTSYCL_TRACE_OP("native_layer_norm_backward");
    c10::Device dev = input.device();

    std::vector<int64_t> ns;
    for (auto v : normalized_shape) ns.push_back(v.expect_int());
    c10::IntArrayRef norm_shape(ns.data(), ns.size());

    Tensor cpu_grad  = to_cpu(grad_out);
    Tensor cpu_input = to_cpu(input);
    Tensor cpu_mean  = to_cpu(save_mean);
    Tensor cpu_rstd  = to_cpu(save_rstd);
    auto   cpu_w     = opt_to_cpu(weight);
    auto   cpu_b     = opt_to_cpu(bias);

    auto result = at::native_layer_norm_backward(
            cpu_grad, cpu_input, norm_shape,
            cpu_mean, cpu_rstd,
            cpu_w, cpu_b, output_mask);

    auto cpu_dx = std::get<0>(result);
    auto cpu_dw = std::get<1>(result);
    auto cpu_db = std::get<2>(result);

    Tensor dx = cpu_dx.defined() ? to_device(cpu_dx, dev) : Tensor{};
    Tensor dw = cpu_dw.defined() ? to_device(cpu_dw, dev) : Tensor{};
    Tensor db = cpu_db.defined() ? to_device(cpu_db, dev) : Tensor{};

    return {dx, dw, db};
}

} // namespace
} // namespace ptsycl

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("aten::native_batch_norm",          &ptsycl::native_batch_norm);
    m.impl("aten::native_batch_norm_backward", &ptsycl::native_batch_norm_backward);
    m.impl("aten::native_layer_norm",          &ptsycl::native_layer_norm);
    m.impl("aten::native_layer_norm_backward", &ptsycl::native_layer_norm_backward);
}
