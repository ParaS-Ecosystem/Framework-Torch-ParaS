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

namespace ptsycl {
namespace {

using at::Tensor;

// -----------------------------------------------------------------------------
// Elementwise launch helpers
// -----------------------------------------------------------------------------

template <typename Fn>
void unary_kernel(const Tensor& a, Tensor& out, Fn fn) {
    auto& q = queue_for(out);
    const int64_t n = out.numel();
    if (n == 0) return;
    Tensor ae = a.expand_as(out);

    AT_DISPATCH_ALL_TYPES_AND3(
        c10::kBool, c10::kHalf, c10::kBFloat16, ae.scalar_type(), "ptsycl_unary", [&] {
            const auto sa = make_spec(ae);
            const auto so = make_spec(out);
            const scalar_t* pa = data_ptr<scalar_t>(ae);
            scalar_t* po = data_ptr<scalar_t>(out);
            launch_flat(q, n, [=](std::size_t idx) {
                const int64_t i = static_cast<int64_t>(idx);
                po[so.index(i)] = static_cast<scalar_t>(
                    fn(static_cast<double>(pa[sa.index(i)])));
            });
        });
}


Tensor to_compute(const Tensor& t, c10::ScalarType dtype, c10::Device device) {
    if (t.scalar_type() == dtype && t.device() == device) return t;
    if (t.dim() == 0 && t.device().is_cpu())
        return at::full({}, t.item(), t.options().dtype(dtype).device(device));
    return t.to(t.options().dtype(dtype).device(device));
}

template <typename Fn>
void binary_kernel(const Tensor& a, const Tensor& b, Tensor& out, Fn fn) {
    auto& q = queue_for(out);
    const int64_t n = out.numel();
    if (n == 0) return;
    Tensor ae = to_compute(a, out.scalar_type(), out.device()).expand_as(out);
    Tensor be = to_compute(b, out.scalar_type(), out.device()).expand_as(out);

    AT_DISPATCH_ALL_TYPES_AND3(
        c10::kBool, c10::kHalf, c10::kBFloat16, ae.scalar_type(), "ptsycl_binary", [&] {
            const auto sa = make_spec(ae);
            const auto sb = make_spec(be);
            const auto so = make_spec(out);
            const scalar_t* pa = data_ptr<scalar_t>(ae);
            const scalar_t* pb = data_ptr<scalar_t>(be);
            scalar_t* po = data_ptr<scalar_t>(out);
            launch_flat(q, n, [=](std::size_t idx) {
                const int64_t i = static_cast<int64_t>(idx);
                po[so.index(i)] = static_cast<scalar_t>(
                    fn(static_cast<double>(pa[sa.index(i)]),
                       static_cast<double>(pb[sb.index(i)])));
            });
        });
}


template <typename Fn>
void compare_kernel(const Tensor& a, const Tensor& b, Tensor& out, Fn fn) {
    TORCH_CHECK(out.scalar_type() == c10::kBool,
                "paras comparison expects bool output, got ",
                out.scalar_type());
    auto& q = queue_for(out);
    const int64_t n = out.numel();
    if (n == 0) return;
    const auto common = at::result_type(a, b);
    Tensor ae = to_compute(a, common, out.device()).expand_as(out);
    Tensor be = to_compute(b, common, out.device()).expand_as(out);

    AT_DISPATCH_ALL_TYPES_AND3(
        c10::kBool, c10::kHalf, c10::kBFloat16, ae.scalar_type(), "ptsycl_compare", [&] {
            const auto sa = make_spec(ae);
            const auto sb = make_spec(be);
            const auto so = make_spec(out);
            const scalar_t* pa = data_ptr<scalar_t>(ae);
            const scalar_t* pb = data_ptr<scalar_t>(be);
            bool* po = data_ptr<bool>(out);
            launch_flat(q, n, [=](std::size_t idx) {
                const int64_t i = static_cast<int64_t>(idx);
                po[so.index(i)] = fn(static_cast<double>(pa[sa.index(i)]),
                                     static_cast<double>(pb[sb.index(i)]));
            });
        });
}

Tensor wrap_scalar(const c10::Scalar& s, const Tensor& like) {
    return at::full({}, s, like.options());
}

// -----------------------------------------------------------------------------
// Reductions
// -----------------------------------------------------------------------------

struct ReduceShape {
    StridedSpec kept; // indexes the input by output element
    StridedSpec red;  // indexes the reduction extent
    int64_t     out_n = 1;
    int64_t     red_n = 1;
};

ReduceShape split_dims(const Tensor& self, c10::ArrayRef<int64_t> dims) {
    bool reduce_flags[kMaxDims] = {};
    for (int64_t d : dims) {
        const int64_t wrapped = c10::maybe_wrap_dim(d, self.dim());
        reduce_flags[wrapped] = true;
    }
    ReduceShape rs;
    for (int d = 0; d < self.dim(); ++d) {
        if (reduce_flags[d]) {
            rs.red.sizes[rs.red.ndim]   = self.size(d);
            rs.red.strides[rs.red.ndim] = self.stride(d);
            ++rs.red.ndim;
            rs.red_n *= self.size(d);
        } else {
            rs.kept.sizes[rs.kept.ndim]   = self.size(d);
            rs.kept.strides[rs.kept.ndim] = self.stride(d);
            ++rs.kept.ndim;
            rs.out_n *= self.size(d);
        }
    }
    return rs;
}


Tensor squeeze_out(const Tensor& out, const Tensor& self,
                   c10::ArrayRef<int64_t> dims, bool keepdim) {
    if (!keepdim || dims.empty()) return out;
    std::vector<int64_t> wrapped(dims.begin(), dims.end());
    for (auto& d : wrapped) d = c10::maybe_wrap_dim(d, self.dim());
    std::sort(wrapped.begin(), wrapped.end(), std::greater<int64_t>());
    Tensor o = out;
    for (int64_t d : wrapped) o = o.squeeze(d);
    return o;
}

enum class ReduceKind { Sum, Prod, Min, Max };

template <typename in_t, typename acc_t>
void reduce_dims_typed(compat::Queue& q, const in_t* in,
                       const ReduceShape& rs, const Tensor& out_view,
                       ReduceKind kind) {
    const auto kept = rs.kept;
    const auto red  = rs.red;
    const int64_t red_n = rs.red_n;

    acc_t init{};
    switch (kind) {
        case ReduceKind::Sum:  init = acc_t(0); break;
        case ReduceKind::Prod: init = acc_t(1); break;
        case ReduceKind::Min:  init = std::numeric_limits<acc_t>::max(); break;
        case ReduceKind::Max:  init = std::numeric_limits<acc_t>::lowest(); break;
    }

    AT_DISPATCH_ALL_TYPES_AND2(
        c10::kHalf, c10::kBFloat16, out_view.scalar_type(),
        "ptsycl_reduce_store", [&] {
            using out_t = scalar_t;
            const auto so = make_spec(out_view);
            out_t* po = data_ptr<out_t>(out_view);
            const int k = static_cast<int>(kind);
            const auto combine = [=](acc_t a, acc_t b) {
                switch (k) {
                    case 0: return static_cast<acc_t>(a + b);
                    case 1: return static_cast<acc_t>(a * b);
                    case 2: return a < b ? a : b;
                    default: return a > b ? a : b;
                }
            };

            if (rs.out_n <= 64 && red_n >= (int64_t{1} << 16)) {
                for (int64_t o = 0; o < rs.out_n; ++o) {
                    const acc_t r = reduce_full<acc_t>(
                        q, red_n, init,
                        [=](int64_t j) {
                            return static_cast<acc_t>(
                                in[kept.index(o) + red.index(j)]);
                        },
                        combine);
                    // reduce_full synchronizes, so a host-side store is safe.
                    po[so.index(o)] = static_cast<out_t>(r);
                }
            } else {
                reduce_outer<acc_t>(
                    q, rs.out_n, red_n, init,
                    [=](int64_t o, int64_t j) {
                        return static_cast<acc_t>(
                            in[kept.index(o) + red.index(j)]);
                    },
                    combine,
                    [=](int64_t o, acc_t acc) {
                        po[so.index(o)] = static_cast<out_t>(acc);
                    });
            }
        });
}

void reduce_dims_op(const Tensor& self, c10::ArrayRef<int64_t> dims,
                    bool keepdim, Tensor& out, ReduceKind kind,
                    double post_scale = 1.0) {
    auto& q = queue_for(self);
    const auto rs = split_dims(self, dims);
    Tensor out_view = squeeze_out(out, self, dims, keepdim);
    TORCH_CHECK(out_view.numel() == rs.out_n,
                "paras reduce: unexpected output shape");

    AT_DISPATCH_ALL_TYPES_AND2(
        c10::kHalf, c10::kBFloat16, self.scalar_type(), "ptsycl_reduce", [&] {
            const scalar_t* in = data_ptr<scalar_t>(self);
            if (c10::isIntegralType(self.scalar_type(), /*bool=*/false)) {
                reduce_dims_typed<scalar_t, int64_t>(q, in, rs, out_view, kind);
            } else {
                reduce_dims_typed<scalar_t, double>(q, in, rs, out_view, kind);
            }
        });

    if (post_scale != 1.0) {
        Tensor ov = out_view;
        unary_kernel(ov, ov, [post_scale](double v) { return v * post_scale; });
    }
}

std::vector<int64_t> all_dims(const Tensor& t) {
    std::vector<int64_t> d(t.dim());
    for (int64_t i = 0; i < t.dim(); ++i) d[i] = i;
    return d;
}

// -----------------------------------------------------------------------------
// Elementwise op registrations
// -----------------------------------------------------------------------------

Tensor& add_out(const Tensor& self, const Tensor& other,
                const c10::Scalar& alpha, Tensor& out) {
    PTSYCL_TRACE_OP("add.out");
    const double a = alpha.toDouble();
    binary_kernel(self, other, out,
                  [a](double x, double y) { return x + a * y; });
    return out;
}

Tensor& sub_out(const Tensor& self, const Tensor& other,
                const c10::Scalar& alpha, Tensor& out) {
    PTSYCL_TRACE_OP("sub.out");
    const double a = alpha.toDouble();
    binary_kernel(self, other, out,
                  [a](double x, double y) { return x - a * y; });
    return out;
}

Tensor& mul_out(const Tensor& self, const Tensor& other, Tensor& out) {
    PTSYCL_TRACE_OP("mul.out");
    binary_kernel(self, other, out, [](double x, double y) { return x * y; });
    return out;
}

Tensor& mul_tensor_(Tensor& self, const Tensor& other) {
    PTSYCL_TRACE_OP("mul_.Tensor");
    binary_kernel(self, other, self, [](double x, double y) { return x * y; });
    return self;
}

Tensor& add_tensor_(Tensor& self, const Tensor& other, const c10::Scalar& alpha) {
    PTSYCL_TRACE_OP("add_.Tensor");
    const double a = alpha.toDouble();
    binary_kernel(self, other, self, [a](double x, double y) { return x + a * y; });
    return self;
}

Tensor& div_out(const Tensor& self, const Tensor& other, Tensor& out) {
    PTSYCL_TRACE_OP("div.out");
    binary_kernel(self, other, out, [](double x, double y) { return x / y; });
    return out;
}

Tensor& mul_scalar_(Tensor& self, const c10::Scalar& other) {
    PTSYCL_TRACE_OP("mul_.Scalar");
    const double w = other.toDouble();
    unary_kernel(self, self, [w](double v) { return v * w; });
    return self;
}

Tensor& maximum_out(const Tensor& self, const Tensor& other, Tensor& out) {
    binary_kernel(self, other, out,
                  [](double x, double y) { return x > y ? x : y; });
    return out;
}

Tensor& minimum_out(const Tensor& self, const Tensor& other, Tensor& out) {
    binary_kernel(self, other, out,
                  [](double x, double y) { return x < y ? x : y; });
    return out;
}

Tensor& addcmul_out(const Tensor& self, const Tensor& t1, const Tensor& t2,
                    const c10::Scalar& value, Tensor& out) {
    PTSYCL_TRACE_OP("addcmul.out");
    const double w = value.toDouble();
    // out = self + w * t1 * t2, fused as two broadcast launches.
    Tensor prod = at::empty_like(out);
    binary_kernel(t1, t2, prod, [w](double a, double b) { return w * a * b; });
    binary_kernel(self, prod, out, [](double a, double b) { return a + b; });
    return out;
}

Tensor& addcdiv_out(const Tensor& self, const Tensor& t1, const Tensor& t2,
                    const c10::Scalar& value, Tensor& out) {
    PTSYCL_TRACE_OP("addcdiv.out");
    const double w = value.toDouble();
    Tensor quot = at::empty_like(out);
    binary_kernel(t1, t2, quot, [w](double a, double b) { return w * a / b; });
    binary_kernel(self, quot, out, [](double a, double b) { return a + b; });
    return out;
}

// --- unary math ---------------------------------------------------------------
Tensor& exp_out(const Tensor& self, Tensor& out) {
    unary_kernel(self, out, [](double v) { return ::exp(v); });
    return out;
}
Tensor& log_out(const Tensor& self, Tensor& out) {
    unary_kernel(self, out, [](double v) { return ::log(v); });
    return out;
}
Tensor& sqrt_out(const Tensor& self, Tensor& out) {
    unary_kernel(self, out, [](double v) { return ::sqrt(v); });
    return out;
}
Tensor& sin_out(const Tensor& self, Tensor& out) {
    unary_kernel(self, out, [](double v) { return ::sin(v); });
    return out;
}
Tensor& cos_out(const Tensor& self, Tensor& out) {
    unary_kernel(self, out, [](double v) { return ::cos(v); });
    return out;
}
Tensor& neg_out(const Tensor& self, Tensor& out) {
    unary_kernel(self, out, [](double v) { return -v; });
    return out;
}
Tensor& rsqrt_out(const Tensor& self, Tensor& out) {
    unary_kernel(self, out, [](double v) { return 1.0 / ::sqrt(v); });
    return out;
}
Tensor& reciprocal_out(const Tensor& self, Tensor& out) {
    unary_kernel(self, out, [](double v) { return 1.0 / v; });
    return out;
}
Tensor& ceil_out(const Tensor& self, Tensor& out) {
    unary_kernel(self, out, [](double v) { return ::ceil(v); });
    return out;
}
Tensor& round_out(const Tensor& self, Tensor& out) {
    unary_kernel(self, out, [](double v) { return ::nearbyint(v); });
    return out;
}
Tensor& atan_out(const Tensor& self, Tensor& out) {
    unary_kernel(self, out, [](double v) { return ::atan(v); });
    return out;
}
Tensor& abs_out(const Tensor& self, Tensor& out) {
    unary_kernel(self, out, [](double v) { return v < 0 ? -v : v; });
    return out;
}
Tensor abs(const Tensor& self) {
    Tensor out = at::empty_like(self);
    return ptsycl::abs_out(self, out);
}
Tensor& sgn_out(const Tensor& self, Tensor& out) {
    unary_kernel(self, out,
                 [](double v) { return v > 0 ? 1.0 : (v < 0 ? -1.0 : 0.0); });
    return out;
}
Tensor& pow_scalar_out(const Tensor& self, const c10::Scalar& e, Tensor& out) {
    const double p = e.toDouble();
    unary_kernel(self, out, [p](double v) { return ::pow(v, p); });
    return out;
}
Tensor& lerp_scalar_out(const Tensor& self, const Tensor& end,
                        const c10::Scalar& weight, Tensor& out) {
    const double w = weight.toDouble();
    binary_kernel(self, end, out,
                  [w](double a, double b) { return a + w * (b - a); });
    return out;
}

// --- activations ----------------------------------------------------------------
Tensor relu(const Tensor& self) {
    Tensor out = at::empty_like(self);
    unary_kernel(self, out, [](double v) { return v > 0 ? v : 0.0; });
    return out;
}
Tensor& relu_(Tensor& self) {
    PTSYCL_TRACE_OP("relu_");
    unary_kernel(self, self, [](double v) { return v > 0 ? v : 0.0; });
    return self;
}

Tensor& sigmoid_out(const Tensor& self, Tensor& out) {
    unary_kernel(self, out, [](double v) { return 1.0 / (1.0 + ::exp(-v)); });
    return out;
}
Tensor sigmoid(const Tensor& self) {
    Tensor out = at::empty_like(self);
    return ptsycl::sigmoid_out(self, out);
}
Tensor& sigmoid_(Tensor& self) { return ptsycl::sigmoid_out(self, self); }
Tensor& sigmoid_backward_out(const Tensor& grad_output, const Tensor& output,
                             Tensor& grad_input) {
    binary_kernel(grad_output, output, grad_input,
                  [](double dy, double y) { return dy * y * (1.0 - y); });
    return grad_input;
}

Tensor& tanh_out(const Tensor& self, Tensor& out) {
    unary_kernel(self, out, [](double v) { return ::tanh(v); });
    return out;
}
Tensor tanh(const Tensor& self) {
    Tensor out = at::empty_like(self);
    return ptsycl::tanh_out(self, out);
}
Tensor& tanh_(Tensor& self) { return ptsycl::tanh_out(self, self); }
Tensor& tanh_backward_out(const Tensor& grad_output, const Tensor& output,
                          Tensor& grad_input) {
    binary_kernel(grad_output, output, grad_input,
                  [](double dy, double y) { return dy * (1.0 - y * y); });
    return grad_input;
}

Tensor& silu_out(const Tensor& self, Tensor& out) {
    unary_kernel(self, out,
                 [](double v) { return v / (1.0 + ::exp(-v)); });
    return out;
}
Tensor& silu_backward_out(const Tensor& grad_output, const Tensor& self,
                          Tensor& grad_input) {
    binary_kernel(grad_output, self, grad_input, [](double dy, double x) {
        const double s = 1.0 / (1.0 + ::exp(-x));
        return dy * s * (1.0 + x * (1.0 - s));
    });
    return grad_input;
}

Tensor hardtanh(const Tensor& self, const c10::Scalar& min_val,
                const c10::Scalar& max_val) {
    Tensor out = at::empty_like(self);
    const double lo = min_val.toDouble(), hi = max_val.toDouble();
    unary_kernel(self, out,
                 [lo, hi](double v) { return v < lo ? lo : (v > hi ? hi : v); });
    return out;
}
Tensor& hardtanh_(Tensor& self, const c10::Scalar& min_val,
                  const c10::Scalar& max_val) {
    const double lo = min_val.toDouble(), hi = max_val.toDouble();
    unary_kernel(self, self,
                 [lo, hi](double v) { return v < lo ? lo : (v > hi ? hi : v); });
    return self;
}
Tensor hardtanh_backward(const Tensor& grad_output, const Tensor& self,
                         const c10::Scalar& min_val,
                         const c10::Scalar& max_val) {
    Tensor out = at::empty_like(grad_output);
    const double lo = min_val.toDouble(), hi = max_val.toDouble();
    binary_kernel(grad_output, self, out, [lo, hi](double dy, double x) {
        return (x <= lo || x >= hi) ? 0.0 : dy;
    });
    return out;
}

Tensor& hardswish_(Tensor& self) {
    unary_kernel(self, self, [](double v) {
        const double r = v + 3.0;
        const double c = r < 0 ? 0.0 : (r > 6.0 ? 6.0 : r);
        return v * c / 6.0;
    });
    return self;
}
Tensor hardswish_backward(const Tensor& grad_output, const Tensor& self) {
    Tensor out = at::empty_like(grad_output);
    binary_kernel(grad_output, self, out, [](double dy, double x) {
        if (x < -3.0) return 0.0;
        if (x > 3.0) return dy;
        return dy * (x / 3.0 + 0.5);
    });
    return out;
}

Tensor& hardsigmoid_out(const Tensor& self, Tensor& out) {
    unary_kernel(self, out, [](double v) {
        const double r = v / 6.0 + 0.5;
        return r < 0 ? 0.0 : (r > 1.0 ? 1.0 : r);
    });
    return out;
}
Tensor& hardsigmoid_backward_out(const Tensor& grad_output, const Tensor& self,
                                 Tensor& grad_input) {
    binary_kernel(grad_output, self, grad_input, [](double dy, double x) {
        return (x > -3.0 && x < 3.0) ? dy / 6.0 : 0.0;
    });
    return grad_input;
}

Tensor& leaky_relu_out(const Tensor& self, const c10::Scalar& negative_slope,
                       Tensor& out) {
    const double slope = negative_slope.toDouble();
    unary_kernel(self, out,
                 [slope](double v) { return v >= 0 ? v : v * slope; });
    return out;
}
Tensor& leaky_relu_backward_out(const Tensor& grad_output, const Tensor& self,
                                const c10::Scalar& negative_slope,
                                bool self_is_result, Tensor& grad_input) {
    TORCH_CHECK(!self_is_result || negative_slope.toDouble() >= 0,
                "leaky_relu backward with negative slope on result tensor");
    const double slope = negative_slope.toDouble();
    binary_kernel(grad_output, self, grad_input, [slope](double dy, double x) {
        return x >= 0 ? dy : dy * slope;
    });
    return grad_input;
}

Tensor& threshold_backward_out(const Tensor& grad_output, const Tensor& self,
                               const c10::Scalar& threshold,
                               Tensor& grad_input) {
    const double th = threshold.toDouble();
    binary_kernel(grad_output, self, grad_input,
                  [th](double dy, double x) { return x > th ? dy : 0.0; });
    return grad_input;
}

Tensor& gelu_out(const Tensor& self, c10::string_view approximate,
                 Tensor& out) {
    const bool tanh_approx = (approximate == "tanh");
    unary_kernel(self, out, [tanh_approx](double x) {
        if (tanh_approx) {
            constexpr double k = 0.7978845608028654; // sqrt(2/pi)
            const double inner = k * (x + 0.044715 * x * x * x);
            return 0.5 * x * (1.0 + ::tanh(inner));
        }
        return 0.5 * x * (1.0 + ::erf(x * 0.7071067811865476));
    });
    return out;
}
Tensor& gelu_backward_out(const Tensor& grad_output, const Tensor& self,
                          c10::string_view approximate, Tensor& grad_input) {
    const bool tanh_approx = (approximate == "tanh");
    binary_kernel(grad_output, self, grad_input,
                  [tanh_approx](double dy, double x) {
        if (tanh_approx) {
            constexpr double k = 0.7978845608028654;
            const double x3    = x * x * x;
            const double inner = k * (x + 0.044715 * x3);
            const double t     = ::tanh(inner);
            const double dt    = (1.0 - t * t) * k * (1.0 + 3 * 0.044715 * x * x);
            return dy * (0.5 * (1.0 + t) + 0.5 * x * dt);
        }
        constexpr double inv_sqrt2    = 0.7071067811865476;
        constexpr double inv_sqrt2pi  = 0.3989422804014327;
        const double cdf = 0.5 * (1.0 + ::erf(x * inv_sqrt2));
        const double pdf = inv_sqrt2pi * ::exp(-0.5 * x * x);
        return dy * (cdf + x * pdf);
    });
    return grad_input;
}

Tensor logit(const Tensor& self, std::optional<double> eps) {
    Tensor out = at::empty_like(self);
    const double lo = eps ? *eps : -1.0;
    unary_kernel(self, out, [lo](double v) {
        double x = v;
        if (lo >= 0) {
            const double hi = 1.0 - lo;
            x = x < lo ? lo : (x > hi ? hi : x);
        }
        return ::log(x / (1.0 - x));
    });
    return out;
}
Tensor& logit_out(const Tensor& self, std::optional<double> eps, Tensor& out) {
    Tensor r = ptsycl::logit(self, eps);
    out.copy_(r);
    return out;
}

std::tuple<Tensor, Tensor> log_sigmoid_forward(const Tensor& self) {
    Tensor out = at::empty_like(self);
    Tensor buffer = at::empty_like(self); // CUDA-style: unused buffer
    unary_kernel(self, out, [](double x) {
        const double mn = x < 0 ? x : 0.0;
        return mn - ::log1p(::exp(-(x < 0 ? -x : x)));
    });
    buffer.zero_();
    return {out, buffer};
}
std::tuple<Tensor&, Tensor&> log_sigmoid_forward_out(const Tensor& self,
                                                     Tensor& output,
                                                     Tensor& buffer) {
    unary_kernel(self, output, [](double x) {
        const double mn = x < 0 ? x : 0.0;
        return mn - ::log1p(::exp(-(x < 0 ? -x : x)));
    });
    buffer.zero_();
    return {output, buffer};
}
Tensor log_sigmoid_backward(const Tensor& grad_output, const Tensor& self,
                            const Tensor& /*buffer*/) {
    Tensor out = at::empty_like(grad_output);
    binary_kernel(grad_output, self, out, [](double dy, double x) {
        return dy * (1.0 / (1.0 + ::exp(x)));
    });
    return out;
}
Tensor& log_sigmoid_backward_out(const Tensor& grad_output, const Tensor& self,
                                 const Tensor& /*buffer*/,
                                 Tensor& grad_input) {
    binary_kernel(grad_output, self, grad_input, [](double dy, double x) {
        return dy * (1.0 / (1.0 + ::exp(x)));
    });
    return grad_input;
}

Tensor& clamp_out(const Tensor& self, const std::optional<c10::Scalar>& min,
                  const std::optional<c10::Scalar>& max, Tensor& out) {
    const bool has_lo = min.has_value(), has_hi = max.has_value();
    const double lo = has_lo ? min->toDouble() : 0.0;
    const double hi = has_hi ? max->toDouble() : 0.0;
    unary_kernel(self, out, [=](double v) {
        if (has_lo && v < lo) v = lo;
        if (has_hi && v > hi) v = hi;
        return v;
    });
    return out;
}
Tensor& clamp_min_out(const Tensor& self, const c10::Scalar& min, Tensor& out) {
    const double lo = min.toDouble();
    unary_kernel(self, out, [lo](double v) { return v < lo ? lo : v; });
    return out;
}

// --- comparisons ----------------------------------------------------------------
#define PTSYCL_COMPARE(name, expr)                                            \
    Tensor& name##_tensor_out(const Tensor& self, const Tensor& other,        \
                              Tensor& out) {                                  \
        compare_kernel(self, other, out,                                      \
                       [](double x, double y) { return expr; });              \
        return out;                                                           \
    }                                                                         \
    Tensor& name##_scalar_out(const Tensor& self, const c10::Scalar& other,   \
                              Tensor& out) {                                  \
        Tensor o = wrap_scalar(other, self);                                  \
        compare_kernel(self, o, out,                                          \
                       [](double x, double y) { return expr; });              \
        return out;                                                           \
    }

PTSYCL_COMPARE(ne, x != y)
PTSYCL_COMPARE(eq, x == y)
PTSYCL_COMPARE(lt, x < y)
PTSYCL_COMPARE(gt, x > y)
PTSYCL_COMPARE(le, x <= y)
PTSYCL_COMPARE(ge, x >= y)
#undef PTSYCL_COMPARE

// --- bitwise (integral / bool) ---------------------------------------------------
template <typename Fn>
void bitwise_kernel(const Tensor& a, const Tensor& b, Tensor& out, Fn fn) {
    auto& q = queue_for(out);
    const int64_t n = out.numel();
    if (n == 0) return;
    Tensor ae = to_compute(a, out.scalar_type(), out.device()).expand_as(out);
    Tensor be = to_compute(b, out.scalar_type(), out.device()).expand_as(out);
    AT_DISPATCH_INTEGRAL_TYPES_AND(
        c10::kBool, out.scalar_type(), "ptsycl_bitwise", [&] {
            const auto sa = make_spec(ae);
            const auto sb = make_spec(be);
            const auto so = make_spec(out);
            const scalar_t* pa = data_ptr<scalar_t>(ae);
            const scalar_t* pb = data_ptr<scalar_t>(be);
            scalar_t* po = data_ptr<scalar_t>(out);
            launch_flat(q, n, [=](std::size_t idx) {
                const int64_t i = static_cast<int64_t>(idx);
                po[so.index(i)] = static_cast<scalar_t>(
                    fn(static_cast<int64_t>(pa[sa.index(i)]),
                       static_cast<int64_t>(pb[sb.index(i)])));
            });
        });
}

Tensor& bitwise_and_out(const Tensor& a, const Tensor& b, Tensor& out) {
    bitwise_kernel(a, b, out, [](int64_t x, int64_t y) { return x & y; });
    return out;
}
Tensor& bitwise_or_out(const Tensor& a, const Tensor& b, Tensor& out) {
    bitwise_kernel(a, b, out, [](int64_t x, int64_t y) { return x | y; });
    return out;
}
Tensor& bitwise_xor_out(const Tensor& a, const Tensor& b, Tensor& out) {
    bitwise_kernel(a, b, out, [](int64_t x, int64_t y) { return x ^ y; });
    return out;
}
Tensor& bitwise_not_out(const Tensor& self, Tensor& out) {
    // ~ on bool is logical negation (ATen semantics), not integer complement.
    if (out.scalar_type() == c10::kBool) {
        bitwise_kernel(self, self, out,
                       [](int64_t x, int64_t) { return x == 0 ? 1 : 0; });
    } else {
        bitwise_kernel(self, self, out, [](int64_t x, int64_t) { return ~x; });
    }
    return out;
}

// --- arange ----------------------------------------------------------------------
Tensor& arange_start_out(const c10::Scalar& start, const c10::Scalar& end,
                         const c10::Scalar& step, Tensor& out) {
    PTSYCL_TRACE_OP("arange.start_out");
    auto& q = queue_for(out);
    const double s0 = start.toDouble();
    const double st = step.toDouble();
    const int64_t n =
        static_cast<int64_t>(::ceil((end.toDouble() - s0) / st));
    TORCH_CHECK(n >= 0, "arange: invalid range");
    if (out.numel() != n)
        out.resize_({n});

    AT_DISPATCH_ALL_TYPES_AND2(
        c10::kHalf, c10::kBFloat16, out.scalar_type(), "ptsycl_arange", [&] {
            scalar_t* po = data_ptr<scalar_t>(out);
            launch_flat(q, n, [=](std::size_t i) {
                po[i] = static_cast<scalar_t>(s0 + st * static_cast<double>(i));
            });
        });
    return out;
}

// --- reductions --------------------------------------------------------------------
Tensor& sum_out(const Tensor& self, at::OptionalIntArrayRef dim, bool keepdim,
                std::optional<c10::ScalarType> /*dtype*/, Tensor& out) {
    PTSYCL_TRACE_OP("sum.IntList_out");
    auto dims = dim.has_value() && !dim->empty()
                    ? std::vector<int64_t>(dim->begin(), dim->end())
                    : all_dims(self);
    reduce_dims_op(self, dims, keepdim, out, ReduceKind::Sum);
    return out;
}

Tensor& mean_out(const Tensor& self, at::OptionalIntArrayRef dim, bool keepdim,
                 std::optional<c10::ScalarType> /*dtype*/, Tensor& out) {
    PTSYCL_TRACE_OP("mean.out");
    auto dims = dim.has_value() && !dim->empty()
                    ? std::vector<int64_t>(dim->begin(), dim->end())
                    : all_dims(self);
    int64_t count = 1;
    for (auto d : dims) count *= self.size(c10::maybe_wrap_dim(d, self.dim()));
    reduce_dims_op(self, dims, keepdim, out, ReduceKind::Sum,
                   count > 0 ? 1.0 / static_cast<double>(count) : 0.0);
    return out;
}

Tensor& prod_out(const Tensor& self, int64_t dim, bool keepdim,
                 std::optional<c10::ScalarType> /*dtype*/, Tensor& out) {
    PTSYCL_TRACE_OP("prod.int_out");
    reduce_dims_op(self, {dim}, keepdim, out, ReduceKind::Prod);
    return out;
}

Tensor& amax_out(const Tensor& self, c10::IntArrayRef dim, bool keepdim,
                 Tensor& out) {
    auto dims = dim.empty() ? all_dims(self)
                            : std::vector<int64_t>(dim.begin(), dim.end());
    reduce_dims_op(self, dims, keepdim, out, ReduceKind::Max);
    return out;
}

Tensor& amin_out(const Tensor& self, c10::IntArrayRef dim, bool keepdim,
                 Tensor& out) {
    auto dims = dim.empty() ? all_dims(self)
                            : std::vector<int64_t>(dim.begin(), dim.end());
    reduce_dims_op(self, dims, keepdim, out, ReduceKind::Min);
    return out;
}

Tensor reduce_all_to_scalar(const Tensor& self, ReduceKind kind) {
    auto& q = queue_for(self);
    Tensor self_c = self.contiguous();
    const int64_t n = self_c.numel();
    TORCH_CHECK(n > 0, "reduction over empty tensor is not defined");
    Tensor out = at::empty({}, self.options());

    AT_DISPATCH_ALL_TYPES_AND2(
        c10::kHalf, c10::kBFloat16, self.scalar_type(), "ptsycl_reduce_all",
        [&] {
            const scalar_t* in = data_ptr<scalar_t>(self_c);
            const int k = static_cast<int>(kind);
            double init;
            switch (kind) {
                case ReduceKind::Sum:  init = 0.0; break;
                case ReduceKind::Prod: init = 1.0; break;
                case ReduceKind::Min:  init = std::numeric_limits<double>::infinity(); break;
                default:               init = -std::numeric_limits<double>::infinity(); break;
            }
            const double r = reduce_full<double>(
                q, n, init,
                [=](int64_t i) { return static_cast<double>(in[i]); },
                [=](double a, double b) {
                    switch (k) {
                        case 0: return a + b;
                        case 1: return a * b;
                        case 2: return a < b ? a : b;
                        default: return a > b ? a : b;
                    }
                });
            scalar_t* po = data_ptr<scalar_t>(out);
            q.synchronize();
            *po = static_cast<scalar_t>(r);
        });
    return out;
}

Tensor min(const Tensor& self) { return reduce_all_to_scalar(self, ReduceKind::Min); }
Tensor max(const Tensor& self) { return reduce_all_to_scalar(self, ReduceKind::Max); }

Tensor dot(const Tensor& self, const Tensor& other) {
    PTSYCL_TRACE_OP("dot");
    TORCH_CHECK(self.dim() == 1 && other.dim() == 1 &&
                    self.numel() == other.numel(),
                "dot: expected 1-D tensors of equal length");
    auto& q = queue_for(self);
    Tensor a = self.contiguous(), b = other.contiguous();
    Tensor out = at::empty({}, self.options());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        c10::kHalf, c10::kBFloat16, self.scalar_type(), "ptsycl_dot", [&] {
            const scalar_t* pa = data_ptr<scalar_t>(a);
            const scalar_t* pb = data_ptr<scalar_t>(b);
            const double r = reduce_full<double>(
                q, a.numel(), 0.0,
                [=](int64_t i) {
                    return static_cast<double>(pa[i]) *
                           static_cast<double>(pb[i]);
                },
                [](double x, double y) { return x + y; });
            q.synchronize();
            *data_ptr<scalar_t>(out) = static_cast<scalar_t>(r);
        });
    return out;
}

Tensor& argmax_out(const Tensor& self, std::optional<int64_t> dim,
                   bool keepdim, Tensor& out) {
    PTSYCL_TRACE_OP("argmax.out");
    TORCH_CHECK(out.scalar_type() == c10::kLong,
                "argmax output must be int64");
    auto& q = queue_for(self);

    Tensor src = dim.has_value() ? self : self.reshape({self.numel()});
    const int64_t d = dim.has_value()
                          ? c10::maybe_wrap_dim(*dim, src.dim())
                          : 0;
    const auto rs = split_dims(src, {d});
    Tensor out_view = dim.has_value()
                          ? squeeze_out(out, src, {d}, keepdim)
                          : out;
    TORCH_CHECK(out_view.numel() == rs.out_n,
                "argmax: unexpected output shape");

    AT_DISPATCH_ALL_TYPES_AND2(
        c10::kHalf, c10::kBFloat16, self.scalar_type(), "ptsycl_argmax", [&] {
            const scalar_t* in = data_ptr<scalar_t>(src);
            const auto kept = rs.kept;
            const auto red  = rs.red;
            const auto so   = make_spec(out_view);
            int64_t* po = data_ptr<int64_t>(out_view);
            const int64_t red_n = rs.red_n;
            launch_flat(q, rs.out_n, [=](std::size_t o) {
                const int64_t base = kept.index(static_cast<int64_t>(o));
                double best = -std::numeric_limits<double>::infinity();
                int64_t best_j = 0;
                for (int64_t j = 0; j < red_n; ++j) {
                    const double v =
                        static_cast<double>(in[base + red.index(j)]);
                    if (v > best) { best = v; best_j = j; }
                }
                po[so.index(static_cast<int64_t>(o))] = best_j;
            });
        });
    return out;
}

// --- softmax --------------------------------------------------------------------
// _softmax(self, dim, half_to_float): numerically-stable softmax along one
// dim. half_to_float means self is fp16/bf16 and the result must be fp32;
// otherwise output dtype matches input dtype. Reuses the same kept/red
// StridedSpec split as the reductions above, computed separately for the
// input and output tensors since they can have different strides (and, in
// the half_to_float case, different dtypes/sizes-in-bytes).
Tensor& softmax_out(const Tensor& self, int64_t dim, bool /*half_to_float*/,
                    Tensor& out) {
    PTSYCL_TRACE_OP("_softmax.out");
    auto& q = queue_for(self);
    const int64_t d = c10::maybe_wrap_dim(dim, self.dim());
    const auto rs_in  = split_dims(self, {d});
    const auto rs_out = split_dims(out, {d});
    TORCH_CHECK(rs_in.out_n == rs_out.out_n && rs_in.red_n == rs_out.red_n,
                "paras softmax: output shape must match input shape");

    AT_DISPATCH_FLOATING_TYPES_AND2(
        c10::kHalf, c10::kBFloat16, self.scalar_type(), "ptsycl_softmax_in",
        [&] {
            using in_t = scalar_t;
            const in_t* pin = data_ptr<in_t>(self);
            const auto kin = rs_in.kept;
            const auto rin = rs_in.red;
            const int64_t red_n = rs_in.red_n;
            const int64_t out_n = rs_in.out_n;

            AT_DISPATCH_FLOATING_TYPES_AND2(
                c10::kHalf, c10::kBFloat16, out.scalar_type(),
                "ptsycl_softmax_out", [&] {
                    using out_t = scalar_t;
                    out_t* pout = data_ptr<out_t>(out);
                    const auto kout = rs_out.kept;
                    const auto rout = rs_out.red;

                    launch_flat(q, out_n, [=](std::size_t o_) {
                        const int64_t o = static_cast<int64_t>(o_);
                        const int64_t base_in  = kin.index(o);
                        const int64_t base_out = kout.index(o);

                        double m = -std::numeric_limits<double>::infinity();
                        for (int64_t j = 0; j < red_n; ++j) {
                            const double v = static_cast<double>(
                                pin[base_in + rin.index(j)]);
                            if (v > m) m = v;
                        }
                        double sum = 0.0;
                        for (int64_t j = 0; j < red_n; ++j) {
                            const double v = static_cast<double>(
                                pin[base_in + rin.index(j)]);
                            sum += std::exp(v - m);
                        }
                        for (int64_t j = 0; j < red_n; ++j) {
                            const double v = static_cast<double>(
                                pin[base_in + rin.index(j)]);
                            pout[base_out + rout.index(j)] =
                                static_cast<out_t>(std::exp(v - m) / sum);
                        }
                    });
                });
        });
    return out;
}

Tensor softmax(const Tensor& self, int64_t dim, bool half_to_float) {
    PTSYCL_TRACE_OP("_softmax");
    const auto out_dtype =
        half_to_float ? c10::kFloat : self.scalar_type();
    Tensor out = at::empty(self.sizes(), self.options().dtype(out_dtype));
    return ptsycl::softmax_out(self, dim, half_to_float, out);
}

// d(softmax)/dx = y * (dy - sum(dy * y, dim, keepdim=True)); input_dtype is
// the dtype of the original (pre-upcast) input and is what grad_input must
// be written back as.
Tensor& softmax_backward_out(const Tensor& grad_output, const Tensor& output,
                             int64_t dim, c10::ScalarType /*input_dtype*/,
                             Tensor& grad_input) {
    PTSYCL_TRACE_OP("_softmax_backward_data.out");
    auto& q = queue_for(grad_output);
    const int64_t d = c10::maybe_wrap_dim(dim, grad_output.dim());
    const auto rs_go = split_dims(grad_output, {d});
    const auto rs_y  = split_dims(output, {d});
    const auto rs_gi = split_dims(grad_input, {d});
    TORCH_CHECK(rs_go.out_n == rs_y.out_n && rs_go.red_n == rs_y.red_n &&
                    rs_go.out_n == rs_gi.out_n &&
                    rs_go.red_n == rs_gi.red_n,
                "paras softmax backward: shape mismatch");

    AT_DISPATCH_FLOATING_TYPES_AND2(
        c10::kHalf, c10::kBFloat16, grad_output.scalar_type(),
        "ptsycl_softmax_bwd_in", [&] {
            using acc_t = scalar_t;
            const acc_t* pgo = data_ptr<acc_t>(grad_output);
            const acc_t* py  = data_ptr<acc_t>(output);
            const auto kgo = rs_go.kept;
            const auto rgo = rs_go.red;
            const auto ky  = rs_y.kept;
            const auto ry  = rs_y.red;
            const int64_t red_n = rs_go.red_n;
            const int64_t out_n = rs_go.out_n;

            AT_DISPATCH_FLOATING_TYPES_AND2(
                c10::kHalf, c10::kBFloat16, grad_input.scalar_type(),
                "ptsycl_softmax_bwd_out", [&] {
                    using gi_t = scalar_t;
                    gi_t* pgi = data_ptr<gi_t>(grad_input);
                    const auto kgi = rs_gi.kept;
                    const auto rgi = rs_gi.red;

                    launch_flat(q, out_n, [=](std::size_t o_) {
                        const int64_t o = static_cast<int64_t>(o_);
                        const int64_t base_go = kgo.index(o);
                        const int64_t base_y  = ky.index(o);
                        const int64_t base_gi = kgi.index(o);

                        double dot = 0.0;
                        for (int64_t j = 0; j < red_n; ++j) {
                            const double dy = static_cast<double>(
                                pgo[base_go + rgo.index(j)]);
                            const double y = static_cast<double>(
                                py[base_y + ry.index(j)]);
                            dot += dy * y;
                        }
                        for (int64_t j = 0; j < red_n; ++j) {
                            const double dy = static_cast<double>(
                                pgo[base_go + rgo.index(j)]);
                            const double y = static_cast<double>(
                                py[base_y + ry.index(j)]);
                            pgi[base_gi + rgi.index(j)] =
                                static_cast<gi_t>(y * (dy - dot));
                        }
                    });
                });
        });
    return grad_input;
}

Tensor softmax_backward(const Tensor& grad_output, const Tensor& output,
                        int64_t dim, c10::ScalarType input_dtype) {
    PTSYCL_TRACE_OP("_softmax_backward_data");
    Tensor grad_input =
        at::empty(grad_output.sizes(), grad_output.options().dtype(input_dtype));
    return ptsycl::softmax_backward_out(grad_output, output, dim,
                                        input_dtype, grad_input);
}

// --- log_softmax ------------------------------------------------------------
// y = x - m - log(sum(exp(x - m))), m = max(x); same numerically-stable
// shape as softmax above, and the same split_dims/dual-dtype-dispatch
// approach for the half_to_float upcast path.
Tensor& log_softmax_out(const Tensor& self, int64_t dim,
                        bool /*half_to_float*/, Tensor& out) {
    PTSYCL_TRACE_OP("_log_softmax.out");
    auto& q = queue_for(self);
    const int64_t d = c10::maybe_wrap_dim(dim, self.dim());
    const auto rs_in  = split_dims(self, {d});
    const auto rs_out = split_dims(out, {d});
    TORCH_CHECK(rs_in.out_n == rs_out.out_n && rs_in.red_n == rs_out.red_n,
                "paras log_softmax: output shape must match input shape");

    AT_DISPATCH_FLOATING_TYPES_AND2(
        c10::kHalf, c10::kBFloat16, self.scalar_type(),
        "ptsycl_log_softmax_in", [&] {
            using in_t = scalar_t;
            const in_t* pin = data_ptr<in_t>(self);
            const auto kin = rs_in.kept;
            const auto rin = rs_in.red;
            const int64_t red_n = rs_in.red_n;
            const int64_t out_n = rs_in.out_n;

            AT_DISPATCH_FLOATING_TYPES_AND2(
                c10::kHalf, c10::kBFloat16, out.scalar_type(),
                "ptsycl_log_softmax_out", [&] {
                    using out_t = scalar_t;
                    out_t* pout = data_ptr<out_t>(out);
                    const auto kout = rs_out.kept;
                    const auto rout = rs_out.red;

                    launch_flat(q, out_n, [=](std::size_t o_) {
                        const int64_t o = static_cast<int64_t>(o_);
                        const int64_t base_in  = kin.index(o);
                        const int64_t base_out = kout.index(o);

                        double m = -std::numeric_limits<double>::infinity();
                        for (int64_t j = 0; j < red_n; ++j) {
                            const double v = static_cast<double>(
                                pin[base_in + rin.index(j)]);
                            if (v > m) m = v;
                        }
                        double sum = 0.0;
                        for (int64_t j = 0; j < red_n; ++j) {
                            const double v = static_cast<double>(
                                pin[base_in + rin.index(j)]);
                            sum += std::exp(v - m);
                        }
                        const double log_sum = std::log(sum);
                        for (int64_t j = 0; j < red_n; ++j) {
                            const double v = static_cast<double>(
                                pin[base_in + rin.index(j)]);
                            pout[base_out + rout.index(j)] =
                                static_cast<out_t>(v - m - log_sum);
                        }
                    });
                });
        });
    return out;
}

Tensor log_softmax(const Tensor& self, int64_t dim, bool half_to_float) {
    PTSYCL_TRACE_OP("_log_softmax");
    const auto out_dtype = half_to_float ? c10::kFloat : self.scalar_type();
    Tensor out = at::empty(self.sizes(), self.options().dtype(out_dtype));
    return ptsycl::log_softmax_out(self, dim, half_to_float, out);
}

// d(log_softmax)/dx = dy - exp(y) * sum(dy, dim, keepdim=True), where y is
// the forward's output (already log_softmax'd).
Tensor& log_softmax_backward_out(const Tensor& grad_output,
                                 const Tensor& output, int64_t dim,
                                 c10::ScalarType /*input_dtype*/,
                                 Tensor& grad_input) {
    PTSYCL_TRACE_OP("_log_softmax_backward_data.out");
    auto& q = queue_for(grad_output);
    const int64_t d = c10::maybe_wrap_dim(dim, grad_output.dim());
    const auto rs_go = split_dims(grad_output, {d});
    const auto rs_y  = split_dims(output, {d});
    const auto rs_gi = split_dims(grad_input, {d});
    TORCH_CHECK(rs_go.out_n == rs_y.out_n && rs_go.red_n == rs_y.red_n &&
                    rs_go.out_n == rs_gi.out_n &&
                    rs_go.red_n == rs_gi.red_n,
                "paras log_softmax backward: shape mismatch");

    AT_DISPATCH_FLOATING_TYPES_AND2(
        c10::kHalf, c10::kBFloat16, grad_output.scalar_type(),
        "ptsycl_log_softmax_bwd_in", [&] {
            using acc_t = scalar_t;
            const acc_t* pgo = data_ptr<acc_t>(grad_output);
            const acc_t* py  = data_ptr<acc_t>(output);
            const auto kgo = rs_go.kept;
            const auto rgo = rs_go.red;
            const auto ky  = rs_y.kept;
            const auto ry  = rs_y.red;
            const int64_t red_n = rs_go.red_n;
            const int64_t out_n = rs_go.out_n;

            AT_DISPATCH_FLOATING_TYPES_AND2(
                c10::kHalf, c10::kBFloat16, grad_input.scalar_type(),
                "ptsycl_log_softmax_bwd_out", [&] {
                    using gi_t = scalar_t;
                    gi_t* pgi = data_ptr<gi_t>(grad_input);
                    const auto kgi = rs_gi.kept;
                    const auto rgi = rs_gi.red;

                    launch_flat(q, out_n, [=](std::size_t o_) {
                        const int64_t o = static_cast<int64_t>(o_);
                        const int64_t base_go = kgo.index(o);
                        const int64_t base_y  = ky.index(o);
                        const int64_t base_gi = kgi.index(o);

                        double sum_dy = 0.0;
                        for (int64_t j = 0; j < red_n; ++j) {
                            sum_dy += static_cast<double>(
                                pgo[base_go + rgo.index(j)]);
                        }
                        for (int64_t j = 0; j < red_n; ++j) {
                            const double dy = static_cast<double>(
                                pgo[base_go + rgo.index(j)]);
                            const double y = static_cast<double>(
                                py[base_y + ry.index(j)]);
                            pgi[base_gi + rgi.index(j)] = static_cast<gi_t>(
                                dy - std::exp(y) * sum_dy);
                        }
                    });
                });
        });
    return grad_input;
}

Tensor log_softmax_backward(const Tensor& grad_output, const Tensor& output,
                            int64_t dim, c10::ScalarType input_dtype) {
    PTSYCL_TRACE_OP("_log_softmax_backward_data");
    Tensor grad_input = at::empty(grad_output.sizes(),
                                  grad_output.options().dtype(input_dtype));
    return ptsycl::log_softmax_backward_out(grad_output, output, dim,
                                            input_dtype, grad_input);
}

// --- cumsum ---------------------------------------------------------------------
// Row-parallel prefix sum: one thread per kept-position runs a sequential
// scan over its red_n elements. Not a work-efficient parallel scan, but
// correct and simple; a proper Blelloch-style scan is a reasonable
// follow-up if a profiled workload needs it.
c10::ScalarType cumsum_out_dtype(const Tensor& self,
                                 c10::optional<c10::ScalarType> dtype) {
    if (dtype.has_value()) return *dtype;
    if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/true) &&
        self.scalar_type() != c10::kLong)
        return c10::kLong;
    return self.scalar_type();
}

Tensor& cumsum_out(const Tensor& self, int64_t dim,
                   c10::optional<c10::ScalarType> /*dtype*/, Tensor& out) {
    PTSYCL_TRACE_OP("cumsum.out");
    auto& q = queue_for(self);
    const int64_t d = c10::maybe_wrap_dim(dim, self.dim());
    const auto rs_in  = split_dims(self, {d});
    const auto rs_out = split_dims(out, {d});
    TORCH_CHECK(rs_in.out_n == rs_out.out_n && rs_in.red_n == rs_out.red_n,
                "paras cumsum: output shape must match input shape");

    AT_DISPATCH_ALL_TYPES_AND3(
        c10::kBool, c10::kHalf, c10::kBFloat16, self.scalar_type(),
        "ptsycl_cumsum_in", [&] {
            using in_t = scalar_t;
            const in_t* pin = data_ptr<in_t>(self);
            const auto kin = rs_in.kept;
            const auto rin = rs_in.red;
            const int64_t red_n = rs_in.red_n;
            const int64_t out_n = rs_in.out_n;

            AT_DISPATCH_ALL_TYPES_AND3(
                c10::kBool, c10::kHalf, c10::kBFloat16, out.scalar_type(),
                "ptsycl_cumsum_out", [&] {
                    using out_t = scalar_t;
                    out_t* pout = data_ptr<out_t>(out);
                    const auto kout = rs_out.kept;
                    const auto rout = rs_out.red;

                    launch_flat(q, out_n, [=](std::size_t o_) {
                        const int64_t o = static_cast<int64_t>(o_);
                        const int64_t base_in  = kin.index(o);
                        const int64_t base_out = kout.index(o);

                        out_t acc = out_t(0);
                        for (int64_t j = 0; j < red_n; ++j) {
                            acc = static_cast<out_t>(
                                acc + static_cast<out_t>(
                                          pin[base_in + rin.index(j)]));
                            pout[base_out + rout.index(j)] = acc;
                        }
                    });
                });
        });
    return out;
}

Tensor cumsum(const Tensor& self, int64_t dim,
             c10::optional<c10::ScalarType> dtype) {
    PTSYCL_TRACE_OP("cumsum");
    Tensor out = at::empty(self.sizes(),
                           self.options().dtype(cumsum_out_dtype(self, dtype)));
    return ptsycl::cumsum_out(self, dim, dtype, out);
}

Tensor& cumsum_(Tensor& self, int64_t dim,
               c10::optional<c10::ScalarType> dtype) {
    PTSYCL_TRACE_OP("cumsum_");
    TORCH_CHECK(!dtype.has_value() || *dtype == self.scalar_type(),
                "paras cumsum_: in-place cumsum cannot change dtype");
    return ptsycl::cumsum_out(self, dim, dtype, self);
}

// --- topk -------------------------------------------------------------------
// Per-row repeated-selection top-k: each of the k output slots does a full
// O(red_n) scan bounded by the previous slot's (value, index) threshold, so
// no per-thread buffer larger than O(1) is needed -- unlike sorting the
// whole row, this scales fine to vocab-sized reduction dims (e.g. logits).
// Ties break toward the lower index (matching CPU's documented stable
// order). Output is always produced sorted, which is a valid choice
// regardless of the `sorted` flag since PyTorch doesn't guarantee any
// particular order when sorted=false.
std::tuple<Tensor&, Tensor&> topk_out(const Tensor& self, c10::SymInt k_sym,
                                      int64_t dim, bool largest,
                                      bool /*sorted*/, Tensor& values,
                                      Tensor& indices) {
    PTSYCL_TRACE_OP("topk.values");
    auto& q = queue_for(self);
    const int64_t k = k_sym.guard_int(__FILE__, __LINE__);
    const int64_t d = c10::maybe_wrap_dim(dim, self.dim());
    TORCH_CHECK(k >= 0 && k <= self.size(d), "paras topk: k out of range");

    const auto rs_in  = split_dims(self, {d});
    const auto rs_val = split_dims(values, {d});
    const auto rs_idx = split_dims(indices, {d});
    TORCH_CHECK(rs_in.out_n == rs_val.out_n && rs_val.out_n == rs_idx.out_n &&
                    rs_val.red_n == k && rs_idx.red_n == k,
                "paras topk: unexpected output shape");

    const int64_t red_n = rs_in.red_n;
    const int64_t out_n = rs_in.out_n;
    if (out_n == 0 || k == 0) return {values, indices};

    AT_DISPATCH_ALL_TYPES_AND2(
        c10::kHalf, c10::kBFloat16, self.scalar_type(), "ptsycl_topk", [&] {
            const scalar_t* pin  = data_ptr<scalar_t>(self);
            scalar_t*       pval = data_ptr<scalar_t>(values);
            int64_t*        pidx = data_ptr<int64_t>(indices);
            const auto kin  = rs_in.kept;
            const auto rin  = rs_in.red;
            const auto kval = rs_val.kept;
            const auto rval = rs_val.red;
            const auto kidx = rs_idx.kept;
            const auto ridx = rs_idx.red;

            launch_flat(q, out_n, [=](std::size_t o_) {
                const int64_t o = static_cast<int64_t>(o_);
                const int64_t base_in  = kin.index(o);
                const int64_t base_val = kval.index(o);
                const int64_t base_idx = kidx.index(o);

                double  prev_val = 0.0;
                int64_t prev_idx = -1;
                bool    has_prev = false;

                for (int64_t s = 0; s < k; ++s) {
                    double  best_val = 0.0;
                    int64_t best_idx = -1;
                    bool    has_best = false;

                    for (int64_t j = 0; j < red_n; ++j) {
                        const double v = static_cast<double>(
                            pin[base_in + rin.index(j)]);

                        bool within_bound;
                        if (!has_prev) {
                            within_bound = true;
                        } else if (largest) {
                            within_bound =
                                (v < prev_val) || (v == prev_val && j > prev_idx);
                        } else {
                            within_bound =
                                (v > prev_val) || (v == prev_val && j > prev_idx);
                        }
                        if (!within_bound) continue;

                        bool better;
                        if (!has_best) {
                            better = true;
                        } else if (largest) {
                            better = (v > best_val) ||
                                     (v == best_val && j < best_idx);
                        } else {
                            better = (v < best_val) ||
                                     (v == best_val && j < best_idx);
                        }
                        if (better) {
                            best_val = v;
                            best_idx = j;
                            has_best = true;
                        }
                    }
                    pval[base_val + rval.index(s)] = static_cast<scalar_t>(best_val);
                    pidx[base_idx + ridx.index(s)] = best_idx;
                    prev_val = best_val;
                    prev_idx = best_idx;
                    has_prev = true;
                }
            });
        });
    return {values, indices};
}

std::tuple<Tensor, Tensor> topk(const Tensor& self, c10::SymInt k,
                                int64_t dim, bool largest, bool sorted) {
    PTSYCL_TRACE_OP("topk");
    const int64_t d  = c10::maybe_wrap_dim(dim, self.dim());
    const int64_t kk = k.guard_int(__FILE__, __LINE__);
    std::vector<int64_t> sizes(self.sizes().begin(), self.sizes().end());
    sizes[d] = kk;
    Tensor values  = at::empty(sizes, self.options());
    Tensor indices = at::empty(sizes, self.options().dtype(c10::kLong));
    ptsycl::topk_out(self, k, dim, largest, sorted, values, indices);
    return {values, indices};
}

std::tuple<Tensor, Tensor> max_dim(const Tensor& self, int64_t dim, bool keepdim) {
    PTSYCL_TRACE_OP("max.dim");
    auto [val, idx] = ptsycl::topk(self, c10::SymInt(1), dim, /*largest=*/true, /*sorted=*/true);
    if (!keepdim) {
        val = val.squeeze(dim);
        idx = idx.squeeze(dim);
    }
    return {val, idx};
}

std::tuple<Tensor, Tensor> min_dim(const Tensor& self, int64_t dim, bool keepdim) {
    PTSYCL_TRACE_OP("min.dim");
    auto [val, idx] = ptsycl::topk(self, c10::SymInt(1), dim, /*largest=*/false, /*sorted=*/true);
    if (!keepdim) {
        val = val.squeeze(dim);
        idx = idx.squeeze(dim);
    }
    return {val, idx};
}

std::tuple<Tensor&, Tensor&> max_dim_out(const Tensor& self, int64_t dim, bool keepdim, Tensor& values, Tensor& indices) {
    PTSYCL_TRACE_OP("max.dim_max");
    auto [v, i] = ptsycl::max_dim(self, dim, keepdim);
    values.copy_(v);
    indices.copy_(i);
    return {values, indices};
}

std::tuple<Tensor&, Tensor&> min_dim_out(const Tensor& self, int64_t dim, bool keepdim, Tensor& values, Tensor& indices) {
    PTSYCL_TRACE_OP("min.dim_min");
    auto [v, i] = ptsycl::min_dim(self, dim, keepdim);
    values.copy_(v);
    indices.copy_(i);
    return {values, indices};
}

std::tuple<Tensor&, Tensor&> sort_out(
    const Tensor& self,
    c10::optional<bool> stable_opt,
    int64_t dim, bool descending,
    Tensor& values, Tensor& indices) {
    PTSYCL_TRACE_OP("sort.values_stable");
    auto& q = queue_for(self);
    const int64_t d = c10::maybe_wrap_dim(dim, self.dim());

    const auto rs_in  = split_dims(self,    {d});
    const auto rs_val = split_dims(values,  {d});
    const auto rs_idx = split_dims(indices, {d});
    TORCH_CHECK(rs_in.out_n == rs_val.out_n &&
                rs_val.out_n == rs_idx.out_n &&
                rs_in.red_n == rs_val.red_n &&
                rs_val.red_n == rs_idx.red_n,
                "paras sort: shape mismatch");

    const int64_t red_n = rs_in.red_n;
    const int64_t out_n = rs_in.out_n;
    if (out_n == 0 || red_n == 0) return {values, indices};

    AT_DISPATCH_ALL_TYPES_AND2(
        c10::kHalf, c10::kBFloat16, self.scalar_type(),
        "ptsycl_sort", [&] {
            const scalar_t* pin  = data_ptr<scalar_t>(self);
            scalar_t*       pval = data_ptr<scalar_t>(values);
            int64_t*        pidx = data_ptr<int64_t>(indices);
            const auto kin  = rs_in.kept;
            const auto rin  = rs_in.red;
            const auto kval = rs_val.kept;
            const auto rval = rs_val.red;
            const auto kidx = rs_idx.kept;
            const auto ridx = rs_idx.red;

            launch_flat(q, out_n, [=](std::size_t o_) {
                const int64_t o = static_cast<int64_t>(o_);
                const int64_t base_in  = kin.index(o);
                const int64_t base_val = kval.index(o);
                const int64_t base_idx = kidx.index(o);

                double  prev_val = 0.0;
                int64_t prev_idx = -1;
                bool    has_prev = false;

                for (int64_t s = 0; s < red_n; ++s) {
                    double  best_val = 0.0;
                    int64_t best_idx = -1;
                    bool    has_best = false;

                    for (int64_t j = 0; j < red_n; ++j) {
                        const double v = static_cast<double>(
                            pin[base_in + rin.index(j)]);

                        bool within_bound;
                        if (!has_prev) {
                            within_bound = true;
                        } else if (descending) {
                            within_bound = (v < prev_val) ||
                                           (v == prev_val && j > prev_idx);
                        } else {
                            within_bound = (v > prev_val) ||
                                           (v == prev_val && j > prev_idx);
                        }
                        if (!within_bound) continue;

                        bool better;
                        if (!has_best) {
                            better = true;
                        } else if (descending) {
                            better = (v > best_val) ||
                                     (v == best_val && j < best_idx);
                        } else {
                            better = (v < best_val) ||
                                     (v == best_val && j < best_idx);
                        }
                        if (better) {
                            best_val = v;
                            best_idx = j;
                            has_best = true;
                        }
                    }

                    pval[base_val + rval.index(s)] =
                        static_cast<scalar_t>(best_val);
                    pidx[base_idx + ridx.index(s)] = best_idx;
                    prev_val = best_val;
                    prev_idx = best_idx;
                    has_prev = true;
                }
            });
        });

    return {values, indices};
}

std::tuple<Tensor, Tensor> sort_stable(
    const Tensor& self,
    c10::optional<bool> stable,
    int64_t dim, bool descending) {
    PTSYCL_TRACE_OP("sort.stable");
  
    auto values  = at::empty_like(self);
    auto indices = at::empty(self.sizes(),
                             self.options().dtype(c10::kLong));
    ptsycl::sort_out(self, stable, dim, descending,
                     values, indices);
    return {values, indices};
}

std::tuple<Tensor, Tensor> sort_default(
    const Tensor& self,
    int64_t dim, bool descending) {
    PTSYCL_TRACE_OP("sort");
    return ptsycl::sort_stable(self, /*stable=*/false, dim, descending);
}

std::tuple<Tensor&, Tensor&> sort_out_default(
    const Tensor& self,
    int64_t dim, bool descending,
    Tensor& values, Tensor& indices) {
    PTSYCL_TRACE_OP("sort.values");
    return ptsycl::sort_out(self, /*stable=*/false, dim, descending, values, indices);
}

// --- concatenation ------------------------------------------------------------------
Tensor& cat_out(const at::ITensorListRef& tensors, int64_t dim, Tensor& out) {
    PTSYCL_TRACE_OP("cat.out");
    auto materialized = tensors.materialize();
    TORCH_CHECK(!materialized.empty(), "cat: expected non-empty tensor list");
    const int64_t wrapped = c10::maybe_wrap_dim(dim, out.dim());

    int64_t offset = 0;
    for (const Tensor& t : materialized) {
        if (t.numel() == 0) continue;
        const int64_t len = t.size(wrapped);
        Tensor slice = out.narrow(wrapped, offset, len);
        slice.copy_(t); // routes through _copy_from's strided machinery
        offset += len;
    }
    return out;
}

Tensor cat(const at::ITensorListRef& tensors, int64_t dim) {
    auto materialized = tensors.materialize();
    TORCH_CHECK(!materialized.empty(), "cat: expected non-empty tensor list");
    std::vector<int64_t> shape(materialized[0].get().sizes().begin(),
                               materialized[0].get().sizes().end());
    const int64_t wrapped =
        c10::maybe_wrap_dim(dim, materialized[0].get().dim());
    shape[wrapped] = 0;
    for (const Tensor& t : materialized) shape[wrapped] += t.size(wrapped);
    Tensor out = at::empty(shape, materialized[0].get().options());
    ptsycl::cat_out(tensors, dim, out);
    return out;
}



// --- sort / argsort -----------------------------------------------------------------
// A full sort is topk with k == the dimension's size; topk_out's tie-break
// (earlier index wins) already makes it a stable order, so sort's `stable`
// flag doesn't change the kernel used.
// std::tuple<Tensor&, Tensor&> sort_out(const Tensor& self, int64_t dim,
//                                       bool descending, Tensor& values,
//                                       Tensor& indices) {
//     PTSYCL_TRACE_OP("sort.values");
//     const int64_t d = c10::maybe_wrap_dim(dim, self.dim());
//     const int64_t n = self.size(d);
//     return ptsycl::topk_out(self, c10::SymInt(n), d, descending,
//                             /*sorted=*/true, values, indices);
// }

} // namespace
} // namespace ptsycl

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("aten::add.out", &ptsycl::add_out);
    m.impl("aten::sub.out", &ptsycl::sub_out);
    m.impl("aten::mul.out", &ptsycl::mul_out);
    m.impl("aten::div.out", &ptsycl::div_out);
    m.impl("aten::mul_.Tensor", &ptsycl::mul_tensor_);
    m.impl("aten::add_.Tensor", &ptsycl::add_tensor_);
    m.impl("aten::mul_.Scalar", &ptsycl::mul_scalar_);
    m.impl("aten::maximum.out", &ptsycl::maximum_out);
    m.impl("aten::minimum.out", &ptsycl::minimum_out);
    m.impl("aten::addcmul.out", &ptsycl::addcmul_out);
    m.impl("aten::addcdiv.out", &ptsycl::addcdiv_out);

    m.impl("aten::exp.out", &ptsycl::exp_out);
    m.impl("aten::log.out", &ptsycl::log_out);
    m.impl("aten::sqrt.out", &ptsycl::sqrt_out);
    m.impl("aten::sin.out", &ptsycl::sin_out);
    m.impl("aten::cos.out", &ptsycl::cos_out);
    m.impl("aten::neg.out", &ptsycl::neg_out);
    m.impl("aten::rsqrt.out", &ptsycl::rsqrt_out);
    m.impl("aten::reciprocal.out", &ptsycl::reciprocal_out);
    m.impl("aten::ceil.out", &ptsycl::ceil_out);
    m.impl("aten::round.out", &ptsycl::round_out);
    m.impl("aten::atan.out", &ptsycl::atan_out);
    m.impl("aten::abs", &ptsycl::abs);
    m.impl("aten::abs.out", &ptsycl::abs_out);
    m.impl("aten::sgn.out", &ptsycl::sgn_out);
    m.impl("aten::pow.Tensor_Scalar_out", &ptsycl::pow_scalar_out);
    m.impl("aten::lerp.Scalar_out", &ptsycl::lerp_scalar_out);

    m.impl("aten::relu", &ptsycl::relu);
    m.impl("aten::relu_", &ptsycl::relu_);
    m.impl("aten::sigmoid", &ptsycl::sigmoid);
    m.impl("aten::sigmoid_", &ptsycl::sigmoid_);
    m.impl("aten::sigmoid.out", &ptsycl::sigmoid_out);
    m.impl("aten::sigmoid_backward.grad_input", &ptsycl::sigmoid_backward_out);
    m.impl("aten::tanh", &ptsycl::tanh);
    m.impl("aten::tanh_", &ptsycl::tanh_);
    m.impl("aten::tanh.out", &ptsycl::tanh_out);
    m.impl("aten::tanh_backward.grad_input", &ptsycl::tanh_backward_out);
    m.impl("aten::silu.out", &ptsycl::silu_out);
    m.impl("aten::silu_backward.grad_input", &ptsycl::silu_backward_out);
    m.impl("aten::hardtanh", &ptsycl::hardtanh);
    m.impl("aten::hardtanh_", &ptsycl::hardtanh_);
    m.impl("aten::hardtanh_backward", &ptsycl::hardtanh_backward);
    m.impl("aten::hardswish_", &ptsycl::hardswish_);
    m.impl("aten::hardswish_backward", &ptsycl::hardswish_backward);
    m.impl("aten::hardsigmoid.out", &ptsycl::hardsigmoid_out);
    m.impl("aten::hardsigmoid_backward.grad_input",
           &ptsycl::hardsigmoid_backward_out);
    m.impl("aten::leaky_relu.out", &ptsycl::leaky_relu_out);
    m.impl("aten::leaky_relu_backward.grad_input",
           &ptsycl::leaky_relu_backward_out);
    m.impl("aten::threshold_backward.grad_input",
           &ptsycl::threshold_backward_out);
    m.impl("aten::gelu.out", &ptsycl::gelu_out);
    m.impl("aten::gelu_backward.grad_input", &ptsycl::gelu_backward_out);
    m.impl("aten::logit", &ptsycl::logit);
    m.impl("aten::logit.out", &ptsycl::logit_out);
    m.impl("aten::log_sigmoid_forward", &ptsycl::log_sigmoid_forward);
    m.impl("aten::log_sigmoid_forward.output",
           &ptsycl::log_sigmoid_forward_out);
    m.impl("aten::log_sigmoid_backward", &ptsycl::log_sigmoid_backward);
    m.impl("aten::log_sigmoid_backward.grad_input",
           &ptsycl::log_sigmoid_backward_out);
    m.impl("aten::clamp.out", &ptsycl::clamp_out);
    m.impl("aten::clamp_min.out", &ptsycl::clamp_min_out);

    m.impl("aten::ne.Tensor_out", &ptsycl::ne_tensor_out);
    m.impl("aten::eq.Tensor_out", &ptsycl::eq_tensor_out);
    m.impl("aten::lt.Tensor_out", &ptsycl::lt_tensor_out);
    m.impl("aten::gt.Tensor_out", &ptsycl::gt_tensor_out);
    m.impl("aten::le.Tensor_out", &ptsycl::le_tensor_out);
    m.impl("aten::ge.Tensor_out", &ptsycl::ge_tensor_out);
    m.impl("aten::ne.Scalar_out", &ptsycl::ne_scalar_out);
    m.impl("aten::eq.Scalar_out", &ptsycl::eq_scalar_out);
    m.impl("aten::lt.Scalar_out", &ptsycl::lt_scalar_out);
    m.impl("aten::gt.Scalar_out", &ptsycl::gt_scalar_out);
    m.impl("aten::le.Scalar_out", &ptsycl::le_scalar_out);
    m.impl("aten::ge.Scalar_out", &ptsycl::ge_scalar_out);

    m.impl("aten::bitwise_and.Tensor_out", &ptsycl::bitwise_and_out);
    m.impl("aten::bitwise_or.Tensor_out", &ptsycl::bitwise_or_out);
    m.impl("aten::bitwise_xor.Tensor_out", &ptsycl::bitwise_xor_out);
    m.impl("aten::bitwise_not.out", &ptsycl::bitwise_not_out);

    m.impl("aten::arange.start_out", &ptsycl::arange_start_out);

    m.impl("aten::sum.IntList_out", &ptsycl::sum_out);
    m.impl("aten::mean.out", &ptsycl::mean_out);
    m.impl("aten::prod.int_out", &ptsycl::prod_out);
    m.impl("aten::amax.out", &ptsycl::amax_out);
    m.impl("aten::amin.out", &ptsycl::amin_out);
    m.impl("aten::argmax.out", &ptsycl::argmax_out);
    m.impl("aten::min", &ptsycl::min);
    m.impl("aten::max", &ptsycl::max);
    m.impl("aten::dot", &ptsycl::dot);

    m.impl("aten::cat.out", &ptsycl::cat_out);
    m.impl("aten::cat", &ptsycl::cat);
    m.impl("aten::_softmax", &ptsycl::softmax);
    m.impl("aten::_softmax.out", &ptsycl::softmax_out);
    m.impl("aten::_softmax_backward_data", &ptsycl::softmax_backward);
    m.impl("aten::_softmax_backward_data.out",
           &ptsycl::softmax_backward_out);
    m.impl("aten::_log_softmax", &ptsycl::log_softmax);
    m.impl("aten::_log_softmax.out", &ptsycl::log_softmax_out);
    m.impl("aten::_log_softmax_backward_data", &ptsycl::log_softmax_backward);
    m.impl("aten::_log_softmax_backward_data.out",
           &ptsycl::log_softmax_backward_out);
    m.impl("aten::cumsum", &ptsycl::cumsum);
    m.impl("aten::cumsum.out", &ptsycl::cumsum_out);
    m.impl("aten::cumsum_", &ptsycl::cumsum_);
    m.impl("aten::topk", &ptsycl::topk);
    m.impl("aten::topk.values", &ptsycl::topk_out);
    m.impl("aten::max.dim", &ptsycl::max_dim);
    m.impl("aten::max.dim_max", &ptsycl::max_dim_out);
    m.impl("aten::min.dim", &ptsycl::min_dim);
    m.impl("aten::min.dim_min", &ptsycl::min_dim_out);
    m.impl("aten::sort", &ptsycl::sort_default);
    m.impl("aten::sort.stable", &ptsycl::sort_stable);
    m.impl("aten::sort.values", &ptsycl::sort_out_default);
    m.impl("aten::sort.values_stable", &ptsycl::sort_out);

}
