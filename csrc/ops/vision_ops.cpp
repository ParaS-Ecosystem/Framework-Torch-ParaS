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
#include <ATen/ops/_native_multi_head_attention_cpu_dispatch.h>
#include <cmath>

namespace ptsycl {
namespace {

using namespace torch;
using torch::autograd::tensor_list;
using torch::autograd::AutogradContext;
using c10::Device;
using c10::DeviceType;

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



Tensor convolution_overrideable(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool transposed,
    IntArrayRef output_padding,
    int64_t groups)
{
    PTSYCL_TRACE_OP("convolution_overrideable");
    c10::Device dev = input.device();

    Tensor cpu_in = to_cpu(input);
    Tensor cpu_w  = to_cpu(weight);
    auto   cpu_b  = opt_to_cpu(bias);

    Tensor cpu_out = at::convolution(
        cpu_in, cpu_w, cpu_b,
        stride, padding, dilation,
        transposed, output_padding, groups);

    Tensor result = to_device(cpu_out, dev);
    return result;
}

::std::tuple<Tensor, Tensor, Tensor> convolution_backward_overrideable(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool transposed,
    IntArrayRef output_padding,
    int64_t groups,
    ::std::array<bool, 3> output_mask)
{
    PTSYCL_TRACE_OP("convolution_backward_overrideable");
    c10::Device dev = input.device();

    Tensor cpu_dy = to_cpu(grad_output);
    Tensor cpu_x  = to_cpu(input);
    Tensor cpu_w  = to_cpu(weight);

    auto result = at::convolution_backward(
        cpu_dy, cpu_x, cpu_w,
        c10::nullopt,
        stride, padding, dilation,
        transposed, output_padding, groups,
        output_mask);

    auto cpu_dx = std::get<0>(result);
    auto cpu_dw = std::get<1>(result);
    auto cpu_db = std::get<2>(result);

    Tensor dx = cpu_dx.defined() ? to_device(cpu_dx, dev) : Tensor{};
    Tensor dw = cpu_dw.defined() ? to_device(cpu_dw, dev) : Tensor{};
    Tensor db = cpu_db.defined() ? to_device(cpu_db, dev) : Tensor{};

    return {dx, dw, db};
}

Tensor _adaptive_avg_pool2d(const Tensor& self, IntArrayRef output_size)
{
    PTSYCL_TRACE_OP("_adaptive_avg_pool2d");
    Tensor self_c = self.contiguous();
    int64_t N = self_c.size(0), C = self_c.size(1);
    int64_t H = self_c.size(2), W = self_c.size(3);
    int64_t oh = output_size[0], ow = output_size[1];

    TORCH_CHECK(
        (oh == 1 && ow == 1) || (oh == H && ow == W),
        "Only global pooling (1×1) or identity (same size) supported");

    c10::Device dev = self.device();
    auto& q = queue_for(self);

    if (oh == H && ow == W) {
        Tensor result = at::empty(self_c.sizes(), at::device(dev).dtype(self_c.scalar_type()));
        q.copy(result.data_ptr(), self_c.data_ptr(),
               static_cast<std::size_t>(self_c.nbytes()), false);
        return result;
    }

    Tensor result = at::empty({N, C, 1, 1}, at::device(dev).dtype(self_c.scalar_type()));

    AT_DISPATCH_FLOATING_TYPES(self_c.scalar_type(), "adaptive_avg_pool", [&] {
        const scalar_t* X = data_ptr<scalar_t>(self_c);
        scalar_t*       Y = data_ptr<scalar_t>(result);
        int64_t hw = H * W;
        float   inv_hw = 1.0f / static_cast<float>(hw);
        launch_flat(q, N * C, [=](std::size_t i) {
            int64_t nc   = static_cast<int64_t>(i);
            const scalar_t* src = X + nc * hw;
            float acc = 0.0f;
            for (int64_t j = 0; j < hw; j++)
                acc += static_cast<float>(src[j]);
            Y[nc] = static_cast<scalar_t>(acc * inv_hw);
        });
    });

    return result;
}

Tensor _adaptive_avg_pool2d_backward(
    const Tensor& grad_output, const Tensor& self)
{
    PTSYCL_TRACE_OP("_adaptive_avg_pool2d_backward");
    Tensor self_c      = self.contiguous();
    Tensor grad_out_c  = grad_output.contiguous();
    int64_t N = self_c.size(0), C = self_c.size(1);
    int64_t H = self_c.size(2), W = self_c.size(3);
    int64_t oh = grad_out_c.size(2), ow = grad_out_c.size(3);

    TORCH_CHECK(
        (oh == 1 && ow == 1) || (grad_out_c.sizes() == self_c.sizes()),
        "Only global or identity adaptive avg pool backward supported");

    c10::Device dev = self.device();
    auto& q = queue_for(self);
    Tensor result = at::empty(self_c.sizes(), at::device(dev).dtype(self_c.scalar_type()));

    if (oh == H && ow == W) {
        q.copy(result.data_ptr(), grad_out_c.data_ptr(),
               static_cast<std::size_t>(grad_out_c.nbytes()), false);
        return result;
    }

    AT_DISPATCH_FLOATING_TYPES(self_c.scalar_type(), "adaptive_avg_pool_bwd", [&] {
        const scalar_t* dY = data_ptr<scalar_t>(grad_out_c);
        scalar_t*       dX = data_ptr<scalar_t>(result);
        int64_t hw = H * W;
        float   inv_hw = 1.0f / static_cast<float>(hw);
        launch_flat(q, N * C * H * W, [=](std::size_t i) {
            int64_t idx  = static_cast<int64_t>(i);
            int64_t nc   = idx / hw;
            dX[idx] = static_cast<scalar_t>(static_cast<float>(dY[nc]) * inv_hw);
        });
    });

    return result;
}

Tensor& avg_pool2d_out(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override,
    Tensor& out)
{
    PTSYCL_TRACE_OP("avg_pool2d.out");
    TORCH_CHECK(!divisor_override, "Divisor override is not implemented");
    Tensor cpu_in = to_cpu(self);
    IntArrayRef eff_stride = stride.empty() ? kernel_size : stride;
    Tensor cpu_out = at::avg_pool2d(cpu_in, kernel_size, eff_stride,
                                     padding, ceil_mode, count_include_pad,
                                     divisor_override);
    auto& q = queue_for(out);
    q.copy(out.data_ptr(), cpu_out.contiguous().data_ptr(),
           static_cast<std::size_t>(cpu_out.contiguous().nbytes()), true);
    return out;
}

Tensor& avg_pool2d_backward_out(
    const Tensor& grad_output,
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override,
    Tensor& grad_input)
{
    PTSYCL_TRACE_OP("avg_pool2d_backward.grad_input");
    TORCH_CHECK(!divisor_override, "Divisor override is not implemented");
    Tensor cpu_dy = to_cpu(grad_output);
    Tensor cpu_x  = to_cpu(self);
    IntArrayRef eff_stride = stride.empty() ? kernel_size : stride;
    Tensor cpu_dx = at::avg_pool2d_backward(
        cpu_dy, cpu_x, kernel_size, eff_stride, padding,
        ceil_mode, count_include_pad, divisor_override);
    auto& q = queue_for(grad_input);
    q.copy(grad_input.data_ptr(), cpu_dx.contiguous().data_ptr(),
           static_cast<std::size_t>(cpu_dx.contiguous().nbytes()), true);
    return grad_input;
}

class LinearFunction : public torch::autograd::Function<LinearFunction> {
public:
    static Tensor forward(AutogradContext* ctx,
                          const Tensor& input,
                          const Tensor& weight,
                          const c10::optional<Tensor>& bias)
    {
        at::AutoDispatchBelowADInplaceOrView g;
        c10::Device dev = input.device();

        Tensor cpu_x = to_cpu(input);
        Tensor cpu_w = to_cpu(weight);
        auto   cpu_b = opt_to_cpu(bias);

        Tensor cpu_out = at::linear(cpu_x, cpu_w,
                                     cpu_b.has_value() ? *cpu_b : Tensor{});
        Tensor result = to_device(cpu_out, dev);

        ctx->save_for_backward({input.contiguous(), weight});
        ctx->saved_data["has_bias"] = bias.has_value() && bias->numel() > 0;
        return result;
    }

    static tensor_list backward(AutogradContext* ctx,
                                 tensor_list grad_outputs)
    {
        Tensor input  = ctx->get_saved_variables()[0];
        Tensor weight = ctx->get_saved_variables()[1];
        bool has_bias = ctx->saved_data["has_bias"].toBool();
        c10::Device dev = input.device();

        Tensor dy    = grad_outputs[0].contiguous();
        Tensor cpu_dy = to_cpu(dy);
        Tensor cpu_x  = to_cpu(input);
        Tensor cpu_w  = to_cpu(weight);

        Tensor cpu_dx = at::mm(cpu_dy.view({-1, cpu_dy.size(-1)}), cpu_w);
        cpu_dx = cpu_dx.view(cpu_x.sizes());

        int64_t fi = cpu_w.size(1), fo = cpu_w.size(0);
        int64_t batch = cpu_x.numel() / fi;
        Tensor cpu_x2  = cpu_x.view({batch, fi});
        Tensor cpu_dy2 = cpu_dy.view({batch, fo});
        Tensor cpu_dw  = at::mm(cpu_dy2.t(), cpu_x2);

        Tensor dx = to_device(cpu_dx.contiguous(), dev);
        Tensor dw = to_device(cpu_dw.contiguous(), dev);
        Tensor db;
        if (has_bias) {
            Tensor cpu_db = cpu_dy2.sum(0);
            db = to_device(cpu_db.contiguous(), dev);
        }
        return {dx, dw, db};
    }
};

Tensor linear(const Tensor& input, const Tensor& weight,
              const c10::optional<Tensor>& bias)
{
    PTSYCL_TRACE_OP("linear");
    return LinearFunction::apply(input, weight, bias);
}

class MaxPool2dFunction : public torch::autograd::Function<MaxPool2dFunction> {
public:
    static Tensor forward(AutogradContext* ctx,
                          const Tensor& self,
                          IntArrayRef kernel_size,
                          IntArrayRef stride,
                          IntArrayRef padding,
                          IntArrayRef dilation,
                          bool ceil_mode)
    {
        at::AutoDispatchBelowADInplaceOrView g;
        c10::Device dev = self.device();

        Tensor cpu_in = to_cpu(self);
        Tensor cpu_out = at::max_pool2d(cpu_in, kernel_size, stride,
                                         padding, dilation, ceil_mode);
        Tensor result = to_device(cpu_out, dev);

        ctx->save_for_backward({self.contiguous()});
        ctx->saved_data["kernel_0"]  = kernel_size[0];
        ctx->saved_data["kernel_1"]  = kernel_size[1];
        ctx->saved_data["pad_0"]     = padding[0];
        ctx->saved_data["pad_1"]     = padding[1];
        ctx->saved_data["stride_0"]  = stride.empty() ? kernel_size[0] : stride[0];
        ctx->saved_data["stride_1"]  = stride.empty() ? kernel_size[1] : stride[1];
        ctx->saved_data["dil_0"]     = dilation[0];
        ctx->saved_data["dil_1"]     = dilation[1];
        ctx->saved_data["ceil_mode"] = ceil_mode;

        return result;
    }

    static tensor_list backward(AutogradContext* ctx,
                                 tensor_list grad_outputs)
    {
        Tensor input = ctx->get_saved_variables()[0];
        c10::Device dev = input.device();

        int64_t k0  = ctx->saved_data["kernel_0"].toInt();
        int64_t k1  = ctx->saved_data["kernel_1"].toInt();
        int64_t p0  = ctx->saved_data["pad_0"].toInt();
        int64_t p1  = ctx->saved_data["pad_1"].toInt();
        int64_t s0  = ctx->saved_data["stride_0"].toInt();
        int64_t s1  = ctx->saved_data["stride_1"].toInt();
        int64_t d0  = ctx->saved_data["dil_0"].toInt();
        int64_t d1  = ctx->saved_data["dil_1"].toInt();
        bool ceil_mode = ctx->saved_data["ceil_mode"].toBool();

        Tensor cpu_dy = to_cpu(grad_outputs[0]);
        Tensor cpu_x  = to_cpu(input);

        auto result_fwd = at::max_pool2d_with_indices(
            cpu_x, {k0, k1}, {s0, s1}, {p0, p1}, {d0, d1}, ceil_mode);
        auto cpu_idx = std::get<1>(result_fwd);

        Tensor cpu_dx = at::max_pool2d_with_indices_backward(
            cpu_dy, cpu_x, {k0, k1}, {s0, s1}, {p0, p1},
            {d0, d1}, ceil_mode, cpu_idx);

        Tensor dx = to_device(cpu_dx.contiguous(), dev);
        return {dx, Tensor{}, Tensor{}, Tensor{}, Tensor{}, Tensor{}};
    }
};

Tensor max_pool2d_autograd(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode)
{
    PTSYCL_TRACE_OP("max_pool2d");
    return MaxPool2dFunction::apply(self, kernel_size, stride,
                                    padding, dilation, ceil_mode);
}

Tensor& mm_out(const Tensor& self, const Tensor& mat2, Tensor& out)
{
    PTSYCL_TRACE_OP("mm.out");
    Tensor cpu_out = at::mm(to_cpu(self), to_cpu(mat2));
    auto& q = queue_for(out);
    q.copy(out.data_ptr(), cpu_out.contiguous().data_ptr(),
           static_cast<std::size_t>(cpu_out.contiguous().nbytes()), true);
    return out;
}

Tensor& bmm_out(const Tensor& self, const Tensor& mat2, Tensor& out)
{
    PTSYCL_TRACE_OP("bmm.out");
    Tensor cpu_out = at::bmm(to_cpu(self), to_cpu(mat2));
    auto& q = queue_for(out);
    q.copy(out.data_ptr(), cpu_out.contiguous().data_ptr(),
           static_cast<std::size_t>(cpu_out.contiguous().nbytes()), true);
    return out;
}

Tensor& addmm_out(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out)
{
    PTSYCL_TRACE_OP("addmm.out");
    Tensor cpu_out = at::addmm(to_cpu(self), to_cpu(mat1), to_cpu(mat2),
                                beta, alpha);
    auto& q = queue_for(out);
    q.copy(out.data_ptr(), cpu_out.contiguous().data_ptr(),
           static_cast<std::size_t>(cpu_out.contiguous().nbytes()), true);
    return out;
}

::std::tuple<Tensor&, Tensor&> _native_multi_head_attention_out(
    const Tensor& query, const Tensor& key, const Tensor& value,
    int64_t embed_dim, int64_t num_head,
    const Tensor& qkv_weight, const Tensor& qkv_bias,
    const Tensor& proj_weight, const Tensor& proj_bias,
    const ::std::optional<Tensor>& mask,
    bool need_weights, bool average_attn_weights,
    ::std::optional<int64_t> mask_type,
    Tensor& out0, Tensor& out1)
{
    PTSYCL_TRACE_OP("_native_multi_head_attention.out");
    auto r = at::cpu::_native_multi_head_attention(
        query, key, value, embed_dim, num_head,
        qkv_weight, qkv_bias, proj_weight, proj_bias,
        mask, need_weights, average_attn_weights, mask_type);
    out0.copy_(std::get<0>(r));
    out1.copy_(std::get<1>(r));
    return {out0, out1};
}

::std::tuple<Tensor, Tensor> _native_multi_head_attention(
    const Tensor& query, const Tensor& key, const Tensor& value,
    int64_t embed_dim, int64_t num_head,
    const Tensor& qkv_weight, const Tensor& qkv_bias,
    const Tensor& proj_weight, const Tensor& proj_bias,
    const ::std::optional<Tensor>& mask,
    bool need_weights, bool average_attn_weights,
    ::std::optional<int64_t> mask_type)
{
    PTSYCL_TRACE_OP("_native_multi_head_attention");
    return at::cpu::_native_multi_head_attention(
        query, key, value, embed_dim, num_head,
        qkv_weight, qkv_bias, proj_weight, proj_bias,
        mask, need_weights, average_attn_weights, mask_type);
}

std::tuple<Tensor, Tensor, Tensor> transform_bias_rescale_qkv(
    const Tensor& qkv, const Tensor& qkv_bias, int64_t num_head)
{
    PTSYCL_TRACE_OP("_transform_bias_rescale_qkv");
    auto qkv_ = qkv.is_nested()
        ? c10::MaybeOwned<Tensor>::owned(qkv.to_padded_tensor(0))
        : c10::MaybeOwned<Tensor>::borrowed(qkv);

    int64_t B        = qkv_->size(0);
    int64_t T        = qkv_->size(1);
    int64_t _3D      = qkv_->size(2);
    int64_t D        = _3D / 3;
    TORCH_CHECK(D % num_head == 0, "D must be divisible by num_head");
    TORCH_CHECK(_3D % 3 == 0,      "last dim must be divisible by 3");
    int64_t dim_per_head = D / num_head;

    const auto qkv_contig      = qkv_->expect_contiguous();
    const auto qkv_bias_contig = qkv_bias.expect_contiguous();

    Tensor q_k_v_same_order = at::empty(
        {B, T, 3, num_head, dim_per_head}, qkv_->options());

    auto& q = queue_for(*qkv_);

    int64_t BT  = B * T;
    double  scale = 1.0 / std::sqrt(static_cast<double>(dim_per_head));
    float   fscale = static_cast<float>(scale);
    int64_t fD  = D;

    AT_DISPATCH_FLOATING_TYPES(qkv_contig->scalar_type(), "transform_qkv", [&] {
        const scalar_t* QKV  = data_ptr<scalar_t>(*qkv_contig);
        const scalar_t* BIAS = data_ptr<scalar_t>(*qkv_bias_contig);
        scalar_t*       OUT  = data_ptr<scalar_t>(q_k_v_same_order);
        int64_t ld = _3D;

        launch_flat(q, BT * _3D, [=](std::size_t i) {
            int64_t bt  = static_cast<int64_t>(i) / ld;
            int64_t d1  = static_cast<int64_t>(i) % ld;
            float   s   = (d1 < fD) ? fscale : 1.0f;
            scalar_t val = (QKV[bt * ld + d1] + BIAS[d1])
                           * static_cast<scalar_t>(s);
            OUT[bt * ld + d1] = val;
        });
    });

    Tensor q_k_v = at::empty({3, B, num_head, T, dim_per_head},
                                  qkv_->options());
    Tensor tmp = q_k_v_same_order;
    tmp = torch::transpose(tmp, 1, 2);
    tmp = torch::transpose(tmp, 2, 3);
    tmp = torch::transpose(tmp, 0, 1);
    q_k_v.copy_(tmp);

    auto q_k_v_s = at::native::split(
        q_k_v.view({3 * B, num_head, T, dim_per_head}), B, 0);

    return {q_k_v_s[0], q_k_v_s[1], q_k_v_s[2]};
}

static Tensor& upsample_fwd_cpu(
    const Tensor& self,
    c10::SymIntArrayRef /*output_size*/,
    Tensor& out,
    std::function<Tensor(const Tensor&)> cpu_fn)
{
    Tensor cpu_in  = to_cpu(self);
    Tensor cpu_out = cpu_fn(cpu_in);
    auto& q = queue_for(out);
    q.copy(out.data_ptr(), cpu_out.contiguous().data_ptr(),
           static_cast<std::size_t>(cpu_out.contiguous().nbytes()), true);
    return out;
}

static Tensor& upsample_bwd_cpu(
    const Tensor& grad_output,
    Tensor& grad_input,
    std::function<Tensor(const Tensor&)> cpu_fn)
{
    Tensor cpu_dy  = to_cpu(grad_output);
    Tensor cpu_dx  = cpu_fn(cpu_dy);
    auto& q = queue_for(grad_input);
    q.copy(grad_input.data_ptr(), cpu_dx.contiguous().data_ptr(),
           static_cast<std::size_t>(cpu_dx.contiguous().nbytes()), true);
    return grad_input;
}

Tensor& upsample_nearest2d_out(
    const Tensor& self,
    c10::SymIntArrayRef output_size,
    ::std::optional<double> scales_h,
    ::std::optional<double> scales_w,
    Tensor& out)
{
    PTSYCL_TRACE_OP("upsample_nearest2d.out");
    std::vector<int64_t> os;
    for (auto v : output_size) os.push_back(v.expect_int());
    return upsample_fwd_cpu(self, output_size, out, [&](const Tensor& x) {
        return at::upsample_nearest2d(x, c10::IntArrayRef(os), scales_h, scales_w);
    });
}

Tensor& upsample_nearest2d_backward_out(
    const Tensor& grad_output,
    c10::SymIntArrayRef output_size,
    c10::SymIntArrayRef input_size,
    ::std::optional<double> scales_h,
    ::std::optional<double> scales_w,
    Tensor& grad_input)
{
    PTSYCL_TRACE_OP("upsample_nearest2d_backward.grad_input");
    std::vector<int64_t> os, is;
    for (auto v : output_size) os.push_back(v.expect_int());
    for (auto v : input_size)  is.push_back(v.expect_int());
    return upsample_bwd_cpu(grad_output, grad_input, [&](const Tensor& dy) {
        return at::upsample_nearest2d_backward(dy,
            c10::IntArrayRef(os), c10::IntArrayRef(is), scales_h, scales_w);
    });
}

Tensor& _upsample_nearest_exact2d_out(
    const Tensor& self,
    c10::SymIntArrayRef output_size,
    ::std::optional<double> scales_h,
    ::std::optional<double> scales_w,
    Tensor& out)
{
    PTSYCL_TRACE_OP("_upsample_nearest_exact2d.out");
    std::vector<int64_t> os;
    for (auto v : output_size) os.push_back(v.expect_int());
    return upsample_fwd_cpu(self, output_size, out, [&](const Tensor& x) {
        return at::_upsample_nearest_exact2d(x, c10::IntArrayRef(os),
                                              scales_h, scales_w);
    });
}

Tensor& _upsample_nearest_exact2d_backward_out(
    const Tensor& grad_output,
    c10::SymIntArrayRef output_size,
    c10::SymIntArrayRef input_size,
    ::std::optional<double> scales_h,
    ::std::optional<double> scales_w,
    Tensor& grad_input)
{
    PTSYCL_TRACE_OP("_upsample_nearest_exact2d_backward.grad_input");
    std::vector<int64_t> os, is;
    for (auto v : output_size) os.push_back(v.expect_int());
    for (auto v : input_size)  is.push_back(v.expect_int());
    return upsample_bwd_cpu(grad_output, grad_input, [&](const Tensor& dy) {
        return at::_upsample_nearest_exact2d_backward(dy,
            c10::IntArrayRef(os), c10::IntArrayRef(is), scales_h, scales_w);
    });
}

Tensor& upsample_bilinear2d_out(
    const Tensor& self,
    c10::SymIntArrayRef output_size,
    bool align_corners,
    ::std::optional<double> scales_h,
    ::std::optional<double> scales_w,
    Tensor& out)
{
    PTSYCL_TRACE_OP("upsample_bilinear2d.out");
    std::vector<int64_t> os;
    for (auto v : output_size) os.push_back(v.expect_int());
    return upsample_fwd_cpu(self, output_size, out, [&](const Tensor& x) {
        return at::upsample_bilinear2d(x, c10::IntArrayRef(os),
                                        align_corners, scales_h, scales_w);
    });
}

Tensor& upsample_bilinear2d_backward_out(
    const Tensor& grad_output,
    c10::SymIntArrayRef output_size,
    c10::SymIntArrayRef input_size,
    bool align_corners,
    ::std::optional<double> scales_h,
    ::std::optional<double> scales_w,
    Tensor& grad_input)
{
    PTSYCL_TRACE_OP("upsample_bilinear2d_backward.grad_input");
    std::vector<int64_t> os, is;
    for (auto v : output_size) os.push_back(v.expect_int());
    for (auto v : input_size)  is.push_back(v.expect_int());
    return upsample_bwd_cpu(grad_output, grad_input, [&](const Tensor& dy) {
        return at::upsample_bilinear2d_backward(dy,
            c10::IntArrayRef(os), c10::IntArrayRef(is),
            align_corners, scales_h, scales_w);
    });
}

} // namespace
} // namespace ptsycl

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("aten::convolution_overrideable",
           &ptsycl::convolution_overrideable);
    m.impl("aten::convolution_backward_overrideable",
           &ptsycl::convolution_backward_overrideable);
    m.impl("aten::_adaptive_avg_pool2d",
           &ptsycl::_adaptive_avg_pool2d);
    m.impl("aten::_adaptive_avg_pool2d_backward",
           &ptsycl::_adaptive_avg_pool2d_backward);
    m.impl("aten::avg_pool2d.out",
           &ptsycl::avg_pool2d_out);
    m.impl("aten::avg_pool2d_backward.grad_input",
           &ptsycl::avg_pool2d_backward_out);
    m.impl("aten::mm.out",
           &ptsycl::mm_out);
    m.impl("aten::bmm.out",
           &ptsycl::bmm_out);
    m.impl("aten::addmm.out",
           &ptsycl::addmm_out);
    m.impl("aten::_native_multi_head_attention.out",
           &ptsycl::_native_multi_head_attention_out);
    m.impl("aten::_native_multi_head_attention",
           &ptsycl::_native_multi_head_attention);
    m.impl("aten::_transform_bias_rescale_qkv",
           &ptsycl::transform_bias_rescale_qkv);
    m.impl("aten::upsample_nearest2d.out",
           &ptsycl::upsample_nearest2d_out);
    m.impl("aten::upsample_nearest2d_backward.grad_input",
           &ptsycl::upsample_nearest2d_backward_out);
    m.impl("aten::_upsample_nearest_exact2d.out",
           &ptsycl::_upsample_nearest_exact2d_out);
    m.impl("aten::_upsample_nearest_exact2d_backward.grad_input",
           &ptsycl::_upsample_nearest_exact2d_backward_out);
    m.impl("aten::upsample_bilinear2d.out",
           &ptsycl::upsample_bilinear2d_out);
    m.impl("aten::upsample_bilinear2d_backward.grad_input",
           &ptsycl::upsample_bilinear2d_backward_out);
}

TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
    m.impl("aten::linear",     &ptsycl::linear);
    m.impl("aten::max_pool2d", &ptsycl::max_pool2d_autograd);
}
