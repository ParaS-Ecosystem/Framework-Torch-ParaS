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
#include "ops/native_bridge.h"
#include <ATen/ATen.h>
#include <torch/torch.h>

namespace ptsycl {
namespace {

using namespace torch;

// Norm and softmax run on the tensor's physical device over the paras
// allocation itself -- see ops/native_bridge.h.
using native_bridge::as_native;
using native_bridge::opt_as_native;
using native_bridge::to_paras;

static void copy_native_into_paras_inplace(const Tensor& src, Tensor& dst) {
    native_bridge::copy_into_paras(src, dst);
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

    Tensor n_input  = as_native(input);
    auto   n_weight = opt_as_native(weight);
    auto   n_bias   = opt_as_native(bias);
    Tensor n_mean = as_native(*running_mean);
    Tensor n_var  = as_native(*running_var);

    auto result = at::native_batch_norm(n_input, n_weight, n_bias,
                                        n_mean, n_var,
                                        training, momentum, eps);
    auto n_out = std::get<0>(result);
    auto n_save_mean = std::get<1>(result);
    auto n_save_var = std::get<2>(result);

    if (mean_present)
        copy_native_into_paras_inplace(n_mean, const_cast<Tensor&>(*running_mean));
    if (var_present)
        copy_native_into_paras_inplace(n_var,  const_cast<Tensor&>(*running_var));

    Tensor out       = to_paras(n_out, dev);
    Tensor save_mean = n_save_mean.defined() ? to_paras(n_save_mean, dev) : Tensor{};
    Tensor save_var  = n_save_var.defined()  ? to_paras(n_save_var,  dev) : Tensor{};

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

    Tensor n_grad   = as_native(grad_out);
    Tensor n_input  = as_native(input);
    auto   n_weight = opt_as_native(weight);
    auto   n_rm     = opt_as_native(running_mean);
    auto   n_rv     = opt_as_native(running_var);
    auto   n_sm     = opt_as_native(save_mean);
    auto   n_sv     = opt_as_native(save_var);

    auto result = at::native_batch_norm_backward(
            n_grad, n_input, n_weight,
            n_rm, n_rv, n_sm, n_sv,
            train, eps, output_mask);

    auto n_dx = std::get<0>(result);
    auto n_dw = std::get<1>(result);
    auto n_db = std::get<2>(result);

    Tensor dx = n_dx.defined() ? to_paras(n_dx, dev) : Tensor{};
    Tensor dw = n_dw.defined() ? to_paras(n_dw, dev) : Tensor{};
    Tensor db = n_db.defined() ? to_paras(n_db, dev) : Tensor{};

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

    Tensor n_input  = as_native(input);
    auto   n_weight = opt_as_native(weight);
    auto   n_bias   = opt_as_native(bias);

    std::vector<int64_t> ns;
    for (auto v : normalized_shape) ns.push_back(v.expect_int());
    c10::IntArrayRef norm_shape(ns.data(), ns.size());

    auto result = at::native_layer_norm(n_input, norm_shape, n_weight, n_bias, eps);
    auto n_out = std::get<0>(result);
    auto n_mean = std::get<1>(result);
    auto n_rstd = std::get<2>(result);

    Tensor out  = to_paras(n_out,  dev);
    Tensor mean = to_paras(n_mean, dev);
    Tensor rstd = to_paras(n_rstd, dev);

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

    Tensor n_grad  = as_native(grad_out);
    Tensor n_input = as_native(input);
    Tensor n_mean  = as_native(save_mean);
    Tensor n_rstd  = as_native(save_rstd);
    auto   n_w     = opt_as_native(weight);
    auto   n_b     = opt_as_native(bias);

    auto result = at::native_layer_norm_backward(
            n_grad, n_input, norm_shape,
            n_mean, n_rstd,
            n_w, n_b, output_mask);

    auto n_dx = std::get<0>(result);
    auto n_dw = std::get<1>(result);
    auto n_db = std::get<2>(result);

    Tensor dx = n_dx.defined() ? to_paras(n_dx, dev) : Tensor{};
    Tensor dw = n_dw.defined() ? to_paras(n_dw, dev) : Tensor{};
    Tensor db = n_db.defined() ? to_paras(n_db, dev) : Tensor{};

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
