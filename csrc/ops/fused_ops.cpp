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


#include <cmath>

#include "core/kernels.h"

// NOTE: these are registered under a new "paras::" op namespace rather than
// overriding aten::rms_norm. Upstream, aten::rms_norm has only a
// CompositeImplicitAutograd kernel (it decomposes into pow/mean/rsqrt/mul,
// each individually differentiable) and no dedicated backward op; replacing
// its PrivateUse1 forward with a single fused kernel would silently drop
// autograd for training, since AutogradPrivateUse1 is registered as a plain
// fallthrough elsewhere in this codebase. swiglu and rotate_half have no
// aten:: identity at all. All three are exposed with an explicit backward
// counterpart and a torch.autograd.Function wrapper on the Python side
// (python/torch_paras/nn.py) instead.

namespace ptsycl {
namespace {

using at::Tensor;

// -----------------------------------------------------------------------------
// rms_norm: normalizes over the last dimension only.
// -----------------------------------------------------------------------------
std::tuple<Tensor, Tensor> rms_norm(const Tensor& input, const Tensor& weight,
                                    double eps) {
    PTSYCL_TRACE_OP("paras::rms_norm");
    TORCH_CHECK(spec_supported(input), "paras rms_norm: rank exceeds kernel limit");

    Tensor in = input.contiguous();
    Tensor w  = weight.contiguous();
    const int64_t H    = in.size(-1);
    const int64_t rows = H > 0 ? in.numel() / H : 0;
    TORCH_CHECK(w.numel() == H,
                "paras rms_norm: weight size must match the last dimension");

    Tensor out  = at::empty_like(in);
    Tensor rstd = at::empty({rows}, in.options().dtype(c10::kFloat));
    if (rows == 0 || H == 0) return {out, rstd};

    auto& q = queue_for(in);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        c10::kHalf, c10::kBFloat16, in.scalar_type(), "ptsycl_rms_norm", [&] {
            const scalar_t* pin  = data_ptr<scalar_t>(in);
            const scalar_t* pw   = data_ptr<scalar_t>(w);
            scalar_t*       pout = data_ptr<scalar_t>(out);
            float*          pr   = data_ptr<float>(rstd);

            launch_flat(q, rows, [=](std::size_t r_) {
                const int64_t r = static_cast<int64_t>(r_);
                const scalar_t* row_in  = pin + r * H;
                scalar_t*       row_out = pout + r * H;

                double ss = 0.0;
                for (int64_t j = 0; j < H; ++j) {
                    const double v = static_cast<double>(row_in[j]);
                    ss += v * v;
                }
                const double rms = 1.0 / ::sqrt(ss / static_cast<double>(H) + eps);
                pr[r] = static_cast<float>(rms);

                for (int64_t j = 0; j < H; ++j) {
                    row_out[j] = static_cast<scalar_t>(
                        static_cast<double>(row_in[j]) * rms *
                        static_cast<double>(pw[j]));
                }
            });
        });
    return {out, rstd};
}

std::tuple<Tensor, Tensor> rms_norm_backward(const Tensor& grad_out,
                                             const Tensor& input,
                                             const Tensor& weight,
                                             const Tensor& rstd) {
    PTSYCL_TRACE_OP("paras::rms_norm_backward");
    Tensor go = grad_out.contiguous();
    Tensor in = input.contiguous();
    Tensor w  = weight.contiguous();
    const int64_t H    = in.size(-1);
    const int64_t rows = H > 0 ? in.numel() / H : 0;

    Tensor grad_input  = at::empty_like(in);
    Tensor grad_weight = at::zeros({H}, w.options());
    if (rows == 0 || H == 0) return {grad_input, grad_weight};

    auto& q = queue_for(in);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        c10::kHalf, c10::kBFloat16, in.scalar_type(), "ptsycl_rms_norm_bwd_dx",
        [&] {
            const scalar_t* pin = data_ptr<scalar_t>(in);
            const scalar_t* pgo = data_ptr<scalar_t>(go);
            const scalar_t* pw  = data_ptr<scalar_t>(w);
            const float*    pr  = data_ptr<float>(rstd);
            scalar_t*       pgi = data_ptr<scalar_t>(grad_input);

            launch_flat(q, rows, [=](std::size_t r_) {
                const int64_t r = static_cast<int64_t>(r_);
                const scalar_t* row_x = pin + r * H;
                const scalar_t* row_g = pgo + r * H;
                scalar_t*       row_i = pgi + r * H;
                const double rr = static_cast<double>(pr[r]);

                double dot = 0.0;
                for (int64_t j = 0; j < H; ++j)
                    dot += static_cast<double>(row_g[j]) *
                           static_cast<double>(pw[j]) *
                           static_cast<double>(row_x[j]);
                const double coef = rr * rr * rr / static_cast<double>(H);

                for (int64_t j = 0; j < H; ++j) {
                    const double gj = static_cast<double>(row_g[j]);
                    const double wj = static_cast<double>(pw[j]);
                    const double xj = static_cast<double>(row_x[j]);
                    row_i[j] = static_cast<scalar_t>(gj * wj * rr - coef * xj * dot);
                }
            });
        });

    AT_DISPATCH_FLOATING_TYPES_AND2(
        c10::kHalf, c10::kBFloat16, in.scalar_type(), "ptsycl_rms_norm_bwd_dw",
        [&] {
            const scalar_t* pin = data_ptr<scalar_t>(in);
            const scalar_t* pgo = data_ptr<scalar_t>(go);
            const float*    pr  = data_ptr<float>(rstd);
            scalar_t*       pgw = data_ptr<scalar_t>(grad_weight);

            launch_flat(q, H, [=](std::size_t j_) {
                const int64_t j = static_cast<int64_t>(j_);
                double acc = 0.0;
                for (int64_t r = 0; r < rows; ++r)
                    acc += static_cast<double>(pgo[r * H + j]) *
                           static_cast<double>(pin[r * H + j]) *
                           static_cast<double>(pr[r]);
                pgw[j] = static_cast<scalar_t>(acc);
            });
        });
    return {grad_input, grad_weight};
}

// -----------------------------------------------------------------------------
// swiglu: silu(gate) * up, fused into a single kernel each way.
// -----------------------------------------------------------------------------
Tensor swiglu(const Tensor& gate, const Tensor& up) {
    PTSYCL_TRACE_OP("paras::swiglu");
    TORCH_CHECK(gate.sizes() == up.sizes(),
                "paras swiglu: gate and up must have the same shape");
    TORCH_CHECK(spec_supported(gate) && spec_supported(up),
                "paras swiglu: rank exceeds kernel limit");

    Tensor out = at::empty_like(gate);
    auto& q = queue_for(out);
    const int64_t n = out.numel();
    if (n == 0) return out;
    Tensor ge = gate.expand_as(out);
    Tensor ue = up.expand_as(out);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        c10::kHalf, c10::kBFloat16, out.scalar_type(), "ptsycl_swiglu", [&] {
            const auto sg = make_spec(ge);
            const auto su = make_spec(ue);
            const auto so = make_spec(out);
            const scalar_t* pg = data_ptr<scalar_t>(ge);
            const scalar_t* pu = data_ptr<scalar_t>(ue);
            scalar_t*       po = data_ptr<scalar_t>(out);

            launch_flat(q, n, [=](std::size_t idx) {
                const int64_t i = static_cast<int64_t>(idx);
                const double g = static_cast<double>(pg[sg.index(i)]);
                const double s = g / (1.0 + ::exp(-g));
                po[so.index(i)] =
                    static_cast<scalar_t>(s * static_cast<double>(pu[su.index(i)]));
            });
        });
    return out;
}

std::tuple<Tensor, Tensor> swiglu_backward(const Tensor& grad_out,
                                           const Tensor& gate, const Tensor& up) {
    PTSYCL_TRACE_OP("paras::swiglu_backward");
    Tensor grad_gate = at::empty_like(gate);
    Tensor grad_up   = at::empty_like(up);
    auto& q = queue_for(grad_gate);
    const int64_t n = grad_out.numel();
    if (n == 0) return {grad_gate, grad_up};

    Tensor go = grad_out.contiguous();
    Tensor ge = gate.contiguous();
    Tensor ue = up.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        c10::kHalf, c10::kBFloat16, grad_out.scalar_type(), "ptsycl_swiglu_bwd",
        [&] {
            const scalar_t* pgo = data_ptr<scalar_t>(go);
            const scalar_t* pg  = data_ptr<scalar_t>(ge);
            const scalar_t* pu  = data_ptr<scalar_t>(ue);
            scalar_t*       pgg = data_ptr<scalar_t>(grad_gate);
            scalar_t*       pgu = data_ptr<scalar_t>(grad_up);

            launch_flat(q, n, [=](std::size_t idx) {
                const int64_t i = static_cast<int64_t>(idx);
                const double dy = static_cast<double>(pgo[i]);
                const double g  = static_cast<double>(pg[i]);
                const double u  = static_cast<double>(pu[i]);
                const double s  = 1.0 / (1.0 + ::exp(-g));
                pgu[i] = static_cast<scalar_t>(dy * g * s);
                pgg[i] = static_cast<scalar_t>(dy * u * (s * (1.0 + g * (1.0 - s))));
            });
        });
    return {grad_gate, grad_up};
}

// -----------------------------------------------------------------------------
// rotate_half: cat(-x2, x1, dim=-1) where x1, x2 = x.chunk(2, dim=-1).
// -----------------------------------------------------------------------------
Tensor rotate_half(const Tensor& x) {
    PTSYCL_TRACE_OP("paras::rotate_half");
    TORCH_CHECK(x.dim() >= 1, "paras rotate_half: input must have rank >= 1");
    TORCH_CHECK(spec_supported(x), "paras rotate_half: rank exceeds kernel limit");
    const int64_t D = x.size(-1);
    TORCH_CHECK(D % 2 == 0, "paras rotate_half: last dimension must be even");

    Tensor out = at::empty_like(x);
    auto& q = queue_for(out);
    const int64_t n = out.numel();
    if (n == 0) return out;
    const int64_t half = D / 2;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        c10::kHalf, c10::kBFloat16, x.scalar_type(), "ptsycl_rotate_half", [&] {
            const auto sx = make_spec(x);
            const auto so = make_spec(out);
            const scalar_t* px = data_ptr<scalar_t>(x);
            scalar_t*       po = data_ptr<scalar_t>(out);

            launch_flat(q, n, [=](std::size_t flat_) {
                const int64_t flat = static_cast<int64_t>(flat_);
                const int64_t col  = flat % D;
                const int64_t src_col  = col < half ? col + half : col - half;
                const int64_t src_flat = flat - col + src_col;
                const scalar_t v = px[sx.index(src_flat)];
                po[so.index(flat)] = col < half
                    ? static_cast<scalar_t>(-static_cast<double>(v))
                    : v;
            });
        });
    return out;
}

Tensor rotate_half_backward(const Tensor& grad_out) {
    PTSYCL_TRACE_OP("paras::rotate_half_backward");
    TORCH_CHECK(spec_supported(grad_out),
                "paras rotate_half_backward: rank exceeds kernel limit");
    const int64_t D = grad_out.size(-1);
    TORCH_CHECK(D % 2 == 0,
                "paras rotate_half_backward: last dimension must be even");

    Tensor grad_in = at::empty_like(grad_out);
    auto& q = queue_for(grad_in);
    const int64_t n = grad_in.numel();
    if (n == 0) return grad_in;
    const int64_t half = D / 2;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        c10::kHalf, c10::kBFloat16, grad_out.scalar_type(),
        "ptsycl_rotate_half_bwd", [&] {
            const auto sg = make_spec(grad_out);
            const auto so = make_spec(grad_in);
            const scalar_t* pg = data_ptr<scalar_t>(grad_out);
            scalar_t*       po = data_ptr<scalar_t>(grad_in);

            launch_flat(q, n, [=](std::size_t flat_) {
                const int64_t flat = static_cast<int64_t>(flat_);
                const int64_t col  = flat % D;
                const int64_t src_col  = col < half ? col + half : col - half;
                const int64_t src_flat = flat - col + src_col;
                const scalar_t v = pg[sg.index(src_flat)];
                po[so.index(flat)] = col < half
                    ? v
                    : static_cast<scalar_t>(-static_cast<double>(v));
            });
        });
    return grad_in;
}

} // namespace
} // namespace ptsycl

TORCH_LIBRARY(paras, m) {
    m.def("rms_norm(Tensor input, Tensor weight, float eps) -> (Tensor, Tensor)");
    m.def("rms_norm_backward(Tensor grad_out, Tensor input, Tensor weight, "
          "Tensor rstd) -> (Tensor, Tensor)");
    m.def("swiglu(Tensor gate, Tensor up) -> Tensor");
    m.def("swiglu_backward(Tensor grad_out, Tensor gate, Tensor up) -> "
          "(Tensor, Tensor)");
    m.def("rotate_half(Tensor x) -> Tensor");
    m.def("rotate_half_backward(Tensor grad_out) -> Tensor");
}

TORCH_LIBRARY_IMPL(paras, PrivateUse1, m) {
    m.impl("rms_norm", &ptsycl::rms_norm);
    m.impl("rms_norm_backward", &ptsycl::rms_norm_backward);
    m.impl("swiglu", &ptsycl::swiglu);
    m.impl("swiglu_backward", &ptsycl::swiglu_backward);
    m.impl("rotate_half", &ptsycl::rotate_half);
    m.impl("rotate_half_backward", &ptsycl::rotate_half_backward);
}
