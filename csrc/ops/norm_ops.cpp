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
#include "core/telemetry.h"
#include <ATen/ATen.h>
#include <torch/torch.h>

namespace ptsycl {
namespace {

using namespace torch;

c10::ScalarType acc_type(c10::ScalarType t) {
    return t == c10::kDouble ? c10::kDouble : c10::kFloat;
}

#if defined(PTSYCL_BACKEND_SYCL)
template <typename T, typename A> class LayerNormRowsKernel;

template <typename T, typename A>
void layer_norm_rows(compat::Queue& q, const T* px, const T* pw, const T* pb, T* py, A* pmean,
                     A* prstd, int64_t M, int64_t N, double eps) {
    q.sycl_queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<double, 1> lds(sycl::range<1>(kReduceWG), cgh);
        cgh.parallel_for<LayerNormRowsKernel<T, A>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(M * kReduceWG)),
                              sycl::range<1>(kReduceWG)),
            [=](sycl::nd_item<1> it) {
                const int64_t r = static_cast<int64_t>(it.get_group(0));
                const T* x = px + r * N;
                const auto plus = [](double a, double b) { return a + b; };
                const double mean = reduce_row_in_group<double>(
                    it, &lds[0], r, N, 0.0, [=](int64_t, int64_t j) { return static_cast<double>(x[j]); },
                    plus) / static_cast<double>(N);
                sycl::group_barrier(it.get_group());
                const double var = reduce_row_in_group<double>(
                    it, &lds[0], r, N, 0.0,
                    [=](int64_t, int64_t j) {
                        const double d = static_cast<double>(x[j]) - mean;
                        return d * d;
                    },
                    plus) / static_cast<double>(N);
                const double rs = 1.0 / ::sqrt(var + eps);
                const int64_t l = static_cast<int64_t>(it.get_local_id(0));
                for (int64_t j = l; j < N; j += kReduceWG) {
                    double v = (static_cast<double>(x[j]) - mean) * rs;
                    if (pw != nullptr) v *= static_cast<double>(pw[j]);
                    if (pb != nullptr) v += static_cast<double>(pb[j]);
                    py[r * N + j] = static_cast<T>(v);
                }
                if (l == 0) {
                    pmean[r] = static_cast<A>(mean);
                    prstd[r] = static_cast<A>(rs);
                }
            });
    });
}
#endif

std::vector<int64_t> stat_shape(const Tensor& input, int64_t norm_dims) {
    std::vector<int64_t> s(input.sizes().begin(), input.sizes().end() - norm_dims);
    s.insert(s.end(), static_cast<std::size_t>(norm_dims), 1);
    return s;
}

std::tuple<Tensor, Tensor, Tensor> native_layer_norm(
    const Tensor& input,
    c10::SymIntArrayRef normalized_shape,
    const c10::optional<Tensor>& weight,
    const c10::optional<Tensor>& bias,
    double eps)
{
    PTSYCL_TRACE_OP("native_layer_norm");
    const int64_t nd = static_cast<int64_t>(normalized_shape.size());
    TORCH_CHECK(nd >= 1 && nd <= input.dim(), "native_layer_norm: bad normalized_shape");
    int64_t N = 1;
    for (int64_t d = 0; d < nd; ++d) {
        const int64_t s = normalized_shape[d].expect_int();
        TORCH_CHECK(input.size(input.dim() - nd + d) == s,
                    "native_layer_norm: normalized_shape does not match the input");
        N *= s;
    }
    const Tensor x = input.contiguous();
    const int64_t M = N > 0 ? x.numel() / N : 0;
    const auto at_ = acc_type(x.scalar_type());
    const bool has_w = weight && weight->defined(), has_b = bias && bias->defined();
    const Tensor w = has_w ? weight->to(x.scalar_type()).contiguous() : Tensor();
    const Tensor b = has_b ? bias->to(x.scalar_type()).contiguous() : Tensor();
    Tensor out = at::empty_like(x);
    Tensor mean = at::empty({M}, x.options().dtype(at_));
    Tensor rstd = at::empty({M}, x.options().dtype(at_));
    const auto shape = stat_shape(x, nd);
    if (M == 0 || N == 0) return {out, mean.view(shape), rstd.view(shape)};
#if defined(PTSYCL_BACKEND_SYCL)
    auto& q = queue_for(x);
    if (reduce_in_groups(q, N)) {
        AT_DISPATCH_FLOATING_TYPES_AND2(c10::kHalf, c10::kBFloat16, x.scalar_type(), "ptsycl_layer_norm", [&] {
            using A = std::conditional_t<std::is_same_v<scalar_t, double>, double, float>;
            layer_norm_rows<scalar_t, A>(q, data_ptr<scalar_t>(x), has_w ? data_ptr<scalar_t>(w) : nullptr,
                                         has_b ? data_ptr<scalar_t>(b) : nullptr, data_ptr<scalar_t>(out),
                                         data_ptr<A>(mean), data_ptr<A>(rstd), M, N, eps);
        });
        return {out, mean.view(shape), rstd.view(shape)};
    }
#endif
    const Tensor xf = x.view({M, N}).to(at_);
    const Tensor mu = xf.mean(1, true);
    const Tensor rs = ((xf - mu).square().mean(1, true) + eps).rsqrt();
    Tensor y = (xf - mu) * rs;
    if (has_w) y = y * w.view({1, N}).to(at_);
    if (has_b) y = y + b.view({1, N}).to(at_);
    out.copy_(y.view(x.sizes()));
    mean.copy_(mu.view({M}));
    rstd.copy_(rs.view({M}));
    return {out, mean.view(shape), rstd.view(shape)};
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
    int64_t N = 1;
    for (const auto& s : normalized_shape) N *= s.expect_int();
    const int64_t M = N > 0 ? input.numel() / N : 0;
    const auto at_ = acc_type(input.scalar_type());
    const bool has_w = weight && weight->defined();
    const Tensor x = input.contiguous().view({M, N}).to(at_);
    const Tensor dy = grad_out.contiguous().view({M, N}).to(at_);
    const Tensor mu = save_mean.reshape({M, 1}).to(at_);
    const Tensor rs = save_rstd.reshape({M, 1}).to(at_);
    const Tensor xhat = (x - mu) * rs;
    Tensor dx, dw, db;
    if (output_mask[0]) {
        const Tensor g = has_w ? dy * weight->reshape({1, N}).to(at_) : dy;
        dx = (rs * (g - g.mean(1, true) - xhat * (g * xhat).mean(1, true)))
                 .to(input.scalar_type()).view(input.sizes());
    }
    if (output_mask[1] && has_w)
        dw = (dy * xhat).sum(0).to(weight->scalar_type()).view(weight->sizes());
    if (output_mask[2] && bias && bias->defined())
        db = dy.sum(0).to(bias->scalar_type()).view(bias->sizes());
    return {dx, dw, db};
}

std::vector<int64_t> reduce_dims_but_channel(const Tensor& t) {
    std::vector<int64_t> d;
    for (int64_t i = 0; i < t.dim(); ++i)
        if (i != 1) d.push_back(i);
    return d;
}

std::vector<int64_t> channel_shape(const Tensor& t) {
    std::vector<int64_t> s(static_cast<std::size_t>(t.dim()), 1);
    s[1] = t.size(1);
    return s;
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
    TORCH_CHECK(input.dim() >= 2, "native_batch_norm: input needs a channel dimension");
    const bool has_rm = running_mean && running_mean->defined() && running_mean->numel() > 0;
    const bool has_rv = running_var && running_var->defined() && running_var->numel() > 0;
    TORCH_CHECK(training || (has_rm && has_rv), "native_batch_norm: evaluation needs running statistics");
    const auto at_ = acc_type(input.scalar_type());
    const auto dims = reduce_dims_but_channel(input);
    const auto cs = channel_shape(input);
    const Tensor xf = input.to(at_);
    Tensor mean, invstd, save_mean, save_invstd;
    if (training) {
        mean = xf.mean(dims);
        const Tensor var = (xf - mean.view(cs)).square().mean(dims);
        invstd = (var + eps).rsqrt();
        const int64_t n = input.numel() / input.size(1);
        at::NoGradGuard no_grad;
        if (has_rm)
            running_mean->mul_(1.0 - momentum).add_(mean.to(running_mean->scalar_type()), momentum);
        if (has_rv) {
            const double unbias = n > 1 ? static_cast<double>(n) / static_cast<double>(n - 1) : 1.0;
            running_var->mul_(1.0 - momentum).add_((var * unbias).to(running_var->scalar_type()), momentum);
        }
        save_mean = mean;
        save_invstd = invstd;
    } else {
        mean = running_mean->to(at_);
        invstd = (running_var->to(at_) + eps).rsqrt();
        save_mean = at::empty({0}, input.options().dtype(at_));
        save_invstd = at::empty({0}, input.options().dtype(at_));
    }
    Tensor y = (xf - mean.view(cs)) * invstd.view(cs);
    if (weight && weight->defined()) y = y * weight->to(at_).view(cs);
    if (bias && bias->defined()) y = y + bias->to(at_).view(cs);
    return {y.to(input.scalar_type()), save_mean, save_invstd};
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
    const auto at_ = acc_type(input.scalar_type());
    const auto dims = reduce_dims_but_channel(input);
    const auto cs = channel_shape(input);
    const Tensor x = input.to(at_);
    const Tensor dy = grad_out.to(at_);
    Tensor mean, invstd;
    if (train) {
        mean = save_mean->to(at_);
        invstd = save_var->to(at_);
    } else {
        mean = running_mean->to(at_);
        invstd = (running_var->to(at_) + eps).rsqrt();
    }
    const Tensor xhat = (x - mean.view(cs)) * invstd.view(cs);
    const Tensor sum_dy = dy.sum(dims);
    const Tensor sum_dy_xhat = (dy * xhat).sum(dims);
    const bool has_w = weight && weight->defined();
    Tensor dx, dw, db;
    if (output_mask[0]) {
        const Tensor scale = has_w ? invstd * weight->to(at_) : invstd;
        if (train) {
            const double n = static_cast<double>(input.numel() / input.size(1));
            dx = scale.view(cs) * (dy - sum_dy.view(cs) / n - xhat * sum_dy_xhat.view(cs) / n);
        } else {
            dx = dy * scale.view(cs);
        }
        dx = dx.to(input.scalar_type());
    }
    if (output_mask[1])
        dw = has_w ? sum_dy_xhat.to(weight->scalar_type()) : sum_dy_xhat.to(input.scalar_type());
    if (output_mask[2])
        db = has_w ? sum_dy.to(weight->scalar_type()) : sum_dy.to(input.scalar_type());
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
