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

#include <torch/torch.h>

#include "core/kernels.h"

// NOTE: these are registered under a new "paras::" op namespace rather than
// overriding aten::rms_norm. Upstream, aten::rms_norm has only a
// CompositeImplicitAutograd kernel (it decomposes into pow/mean/rsqrt/mul,
// each individually differentiable) and no dedicated backward op; replacing
// its PrivateUse1 forward with a single fused kernel would silently drop
// autograd for training, since AutogradPrivateUse1 is registered as a plain
// fallthrough elsewhere in this codebase. swiglu and rotate_half have no
// aten:: identity at all. All three are exposed with an explicit backward

namespace ptsycl {
namespace {

using at::Tensor;

#if defined(PTSYCL_BACKEND_SYCL)
template <typename T> class RmsNormRowsKernel;
#endif

// -----------------------------------------------------------------------------
// rms_norm: normalizes over the last dimension only.
// -----------------------------------------------------------------------------
std::tuple<Tensor, Tensor> rms_norm(const Tensor& input, const Tensor& weight,
                                    double eps) {
    PTSYCL_TRACE_OP("paras::rms_norm");
    TORCH_CHECK(spec_supported(input), "paras rms_norm: rank exceeds kernel limit");
    TORCH_CHECK(weight.scalar_type() == input.scalar_type(),
                "paras rms_norm: weight dtype ", weight.scalar_type(),
                " must match input dtype ", input.scalar_type());

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
    float* pr = data_ptr<float>(rstd);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        c10::kHalf, c10::kBFloat16, in.scalar_type(), "ptsycl_rms_norm", [&] {
            const scalar_t* pin  = data_ptr<scalar_t>(in);
            const scalar_t* pw   = data_ptr<scalar_t>(w);
            scalar_t*       pout = data_ptr<scalar_t>(out);
            const double inv_h = 1.0 / static_cast<double>(H);
            const double eps32 = static_cast<double>(static_cast<float>(eps));
            const auto square = [=](int64_t r, int64_t j) {
                const double x = static_cast<double>(
                    static_cast<float>(static_cast<double>(pin[r * H + j])));
                return static_cast<double>(static_cast<float>(x * x));
            };
            const auto plus = [](double a, double b) { return a + b; };
            const auto rstd_of = [=](double sum) {
                const float mean = static_cast<float>(
                    static_cast<double>(static_cast<float>(sum)) * inv_h);
                const float var = static_cast<float>(static_cast<double>(mean) + eps32);
                return static_cast<float>(1.0 / ::sqrt(static_cast<double>(var)));
            };
            const auto normalized = [=](int64_t i, float rs) {
                const float x = static_cast<float>(static_cast<double>(pin[i]));
                const float n = static_cast<float>(
                    static_cast<double>(x) * static_cast<double>(rs));
                const scalar_t nb = static_cast<scalar_t>(n);
                return static_cast<scalar_t>(
                    static_cast<double>(pw[i % H]) * static_cast<double>(nb));
            };
#if defined(PTSYCL_BACKEND_SYCL)
            if (reduce_in_groups(q, H)) {
                q.sycl_queue().submit([&](sycl::handler& cgh) {
                    sycl::local_accessor<double, 1> lds(sycl::range<1>(kReduceWG), cgh);
                    cgh.template parallel_for<RmsNormRowsKernel<scalar_t>>(
                        sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(rows * kReduceWG)),
                                          sycl::range<1>(kReduceWG)),
                        [=](sycl::nd_item<1> it) {
                            const int64_t r = static_cast<int64_t>(it.get_group(0));
                            const int l = static_cast<int>(it.get_local_id(0));
                            const float rs = rstd_of(
                                reduce_row_in_group<double>(it, &lds[0], r, H, 0.0, square, plus));
                            if (l == 0) pr[r] = rs;
                            for (int64_t j = l; j < H; j += kReduceWG)
                                pout[r * H + j] = normalized(r * H + j, rs);
                        });
                });
                return;
            }
#endif

            reduce_outer<double>(
                q, rows, H, 0.0, square, plus,
                [=](int64_t r, double sum) { pr[r] = rstd_of(sum); },
                rstd);

            launch_flat(q, rows * H, [=](std::size_t idx) {
                const int64_t i = static_cast<int64_t>(idx);
                pout[i] = normalized(i, pr[i / H]);
            });
        });
    return {out, rstd};
}

std::tuple<Tensor, Tensor> rms_norm_backward(const Tensor& grad_out,
                                             const Tensor& input,
                                             const Tensor& weight,
                                             const Tensor& rstd) {
    PTSYCL_TRACE_OP("paras::rms_norm_backward");
    TORCH_CHECK(grad_out.scalar_type() == input.scalar_type() &&
                    weight.scalar_type() == input.scalar_type(),
                "paras rms_norm_backward: grad_out, input and weight must share a dtype");
    Tensor go = grad_out.contiguous();
    Tensor in = input.contiguous();
    Tensor w  = weight.contiguous();
    const int64_t H    = in.size(-1);
    const int64_t rows = H > 0 ? in.numel() / H : 0;

    Tensor grad_input  = at::empty_like(in);
    Tensor grad_weight = at::zeros({H}, w.options());
    if (rows == 0 || H == 0) return {grad_input, grad_weight};

    auto& q = queue_for(in);
    Tensor dots = at::empty({rows}, in.options().dtype(c10::kDouble));
    double* pd = data_ptr<double>(dots);
    const float* pr = data_ptr<float>(rstd);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        c10::kHalf, c10::kBFloat16, in.scalar_type(), "ptsycl_rms_norm_bwd", [&] {
            const scalar_t* pin = data_ptr<scalar_t>(in);
            const scalar_t* pgo = data_ptr<scalar_t>(go);
            const scalar_t* pw  = data_ptr<scalar_t>(w);
            scalar_t*       pgi = data_ptr<scalar_t>(grad_input);
            scalar_t*       pgw = data_ptr<scalar_t>(grad_weight);

            reduce_outer<double>(
                q, rows, H, 0.0,
                [=](int64_t r, int64_t j) {
                    const int64_t i = r * H + j;
                    return static_cast<double>(pgo[i]) * static_cast<double>(pw[j]) *
                           static_cast<double>(pin[i]);
                },
                [](double a, double b) { return a + b; },
                [=](int64_t r, double dot) { pd[r] = dot; },
                dots);

            launch_flat(q, rows * H, [=](std::size_t idx) {
                const int64_t i = static_cast<int64_t>(idx);
                const int64_t r = i / H;
                const double rr = static_cast<double>(pr[r]);
                const double coef = rr * rr * rr / static_cast<double>(H);
                pgi[i] = static_cast<scalar_t>(
                    static_cast<double>(pgo[i]) * static_cast<double>(pw[i % H]) * rr -
                    coef * static_cast<double>(pin[i]) * pd[r]);
            });

            reduce_outer<double>(
                q, H, rows, 0.0,
                [=](int64_t j, int64_t r) {
                    const int64_t i = r * H + j;
                    return static_cast<double>(pgo[i]) * static_cast<double>(pin[i]) *
                           static_cast<double>(pr[r]);
                },
                [](double a, double b) { return a + b; },
                [=](int64_t j, double acc) { pgw[j] = static_cast<scalar_t>(acc); },
                grad_weight);
        });
    return {grad_input, grad_weight};
}

Tensor swiglu(const Tensor& gate, const Tensor& up) {
    PTSYCL_TRACE_OP("paras::swiglu");
    TORCH_CHECK(gate.sizes() == up.sizes(),
                "paras swiglu: gate and up must have the same shape");
    TORCH_CHECK(gate.scalar_type() == up.scalar_type(),
                "paras swiglu: gate and up must have the same dtype");
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
            const auto sg = make_flat_spec(ge);
            const auto su = make_flat_spec(ue);
            const auto so = make_flat_spec(out);
            const scalar_t* pg = data_ptr<scalar_t>(ge);
            const scalar_t* pu = data_ptr<scalar_t>(ue);
            scalar_t*       po = data_ptr<scalar_t>(out);

            launch_flat(q, n, [=](std::size_t idx) {
                const int64_t i = static_cast<int64_t>(idx);
                const double g = static_cast<double>(pg[sg.index(i)]);
                const scalar_t s = static_cast<scalar_t>(g / (1.0 + ::exp(-g)));
                po[so.index(i)] = static_cast<scalar_t>(
                    static_cast<double>(s) * static_cast<double>(pu[su.index(i)]));
            });
        });
    return out;
}

std::tuple<Tensor, Tensor> swiglu_backward(const Tensor& grad_out,
                                           const Tensor& gate, const Tensor& up) {
    PTSYCL_TRACE_OP("paras::swiglu_backward");
    TORCH_CHECK(grad_out.scalar_type() == gate.scalar_type() &&
                    up.scalar_type() == gate.scalar_type(),
                "paras swiglu_backward: grad_out, gate and up must share a dtype");
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
            const auto sx = make_flat_spec(x);
            const auto so = make_flat_spec(out);
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
            const auto sg = make_flat_spec(grad_out);
            const auto so = make_flat_spec(grad_in);
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

namespace ptsycl {
namespace {

using torch::autograd::AutogradContext;
using torch::autograd::variable_list;

template <typename Sig>
c10::TypedOperatorHandle<Sig> paras_op(const char* name) {
    return c10::Dispatcher::singleton().findSchemaOrThrow(name, "").typed<Sig>();
}

struct RmsNormFunction : public torch::autograd::Function<RmsNormFunction> {
    static variable_list forward(AutogradContext* ctx, const Tensor& input,
                                 const Tensor& weight, double eps) {
        static const auto op = paras_op<std::tuple<Tensor, Tensor>(
            const Tensor&, const Tensor&, double)>("paras::rms_norm");
        at::AutoDispatchBelowADInplaceOrView guard;
        auto [out, rstd] = op.call(input, weight, eps);
        ctx->save_for_backward({input, weight, rstd});
        ctx->mark_non_differentiable({rstd});
        return {out, rstd};
    }
    static variable_list backward(AutogradContext* ctx, variable_list grads) {
        static const auto op = paras_op<std::tuple<Tensor, Tensor>(
            const Tensor&, const Tensor&, const Tensor&, const Tensor&)>(
            "paras::rms_norm_backward");
        const auto saved = ctx->get_saved_variables();
        auto [grad_input, grad_weight] =
            op.call(grads[0].contiguous(), saved[0], saved[1], saved[2]);
        return {grad_input, grad_weight, Tensor()};
    }
};

struct SwigluFunction : public torch::autograd::Function<SwigluFunction> {
    static Tensor forward(AutogradContext* ctx, const Tensor& gate, const Tensor& up) {
        static const auto op =
            paras_op<Tensor(const Tensor&, const Tensor&)>("paras::swiglu");
        at::AutoDispatchBelowADInplaceOrView guard;
        ctx->save_for_backward({gate, up});
        return op.call(gate, up);
    }
    static variable_list backward(AutogradContext* ctx, variable_list grads) {
        static const auto op = paras_op<std::tuple<Tensor, Tensor>(
            const Tensor&, const Tensor&, const Tensor&)>("paras::swiglu_backward");
        const auto saved = ctx->get_saved_variables();
        auto [grad_gate, grad_up] = op.call(grads[0].contiguous(), saved[0], saved[1]);
        return {grad_gate, grad_up};
    }
};

struct RotateHalfFunction : public torch::autograd::Function<RotateHalfFunction> {
    static Tensor forward(AutogradContext*, const Tensor& x) {
        static const auto op = paras_op<Tensor(const Tensor&)>("paras::rotate_half");
        at::AutoDispatchBelowADInplaceOrView guard;
        return op.call(x);
    }
    static variable_list backward(AutogradContext*, variable_list grads) {
        static const auto op =
            paras_op<Tensor(const Tensor&)>("paras::rotate_half_backward");
        return {op.call(grads[0].contiguous())};
    }
};

std::tuple<Tensor, Tensor> rms_norm_autograd(const Tensor& input, const Tensor& weight,
                                             double eps) {
    auto out = RmsNormFunction::apply(input, weight, eps);
    return {out[0], out[1]};
}

Tensor swiglu_autograd(const Tensor& gate, const Tensor& up) {
    return SwigluFunction::apply(gate, up);
}

Tensor rotate_half_autograd(const Tensor& x) {
    return RotateHalfFunction::apply(x);
}

}
}

TORCH_LIBRARY_IMPL(paras, AutogradPrivateUse1, m) {
    m.impl("rms_norm", &ptsycl::rms_norm_autograd);
    m.impl("swiglu", &ptsycl::swiglu_autograd);
    m.impl("rotate_half", &ptsycl::rotate_half_autograd);
}
