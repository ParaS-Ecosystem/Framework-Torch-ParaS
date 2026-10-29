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
#include <limits>
#include <vector>

#include <ATen/ATen.h>
#include <torch/torch.h>

#include "core/kernels.h"
#include "core/telemetry.h"

namespace ptsycl {
namespace {

using at::Tensor;

template <typename T>
void remap_kernel(compat::Queue& q, const T* in, T* out, StridedSpec src, int64_t n, int ndim,
                  const int64_t* flip_or_shift, bool is_roll) {
    struct PerDim { int64_t v[kMaxDims]; } a{};
    for (int d = 0; d < ndim; ++d) a.v[d] = flip_or_shift[d];
    launch_flat(q, n, [=](std::size_t idx) {
        int64_t rem = static_cast<int64_t>(idx), off = 0;
        for (int d = ndim - 1; d >= 0; --d) {
            const int64_t sz = src.sizes[d];
            int64_t c = rem % sz;
            rem /= sz;
            if (is_roll) {
                c -= a.v[d];
                if (c < 0) c += sz;
            } else if (a.v[d]) {
                c = sz - 1 - c;
            }
            off += c * src.strides[d];
        }
        out[idx] = in[off];
    });
}

Tensor remap(const Tensor& self, const std::vector<int64_t>& per_dim, bool is_roll) {
    Tensor out = at::empty(self.sizes(), self.options());
    if (self.numel() == 0 || self.dim() == 0) return out.copy_(self);
    auto& q = queue_for(self);
    AT_DISPATCH_ALL_TYPES_AND3(c10::kBool, c10::kHalf, c10::kBFloat16, self.scalar_type(), "ptsycl_remap", [&] {
        remap_kernel<scalar_t>(q, data_ptr<scalar_t>(self), data_ptr<scalar_t>(out), make_spec(self), self.numel(),
                               static_cast<int>(self.dim()), per_dim.data(), is_roll);
    });
    return out;
}

Tensor flip(const Tensor& self, at::IntArrayRef dims) {
    PTSYCL_TRACE_OP("flip");
    std::vector<int64_t> f(static_cast<std::size_t>(std::max<int64_t>(self.dim(), 1)), 0);
    for (int64_t d : dims) f[c10::maybe_wrap_dim(d, self.dim())] = 1;
    return remap(self, f, false);
}

Tensor roll(const Tensor& self, c10::SymIntArrayRef shifts_sym, at::IntArrayRef dims) {
    PTSYCL_TRACE_OP("roll");
    const auto shifts = C10_AS_INTARRAYREF_SLOW(shifts_sym);
    if (dims.empty()) {
        TORCH_CHECK(shifts.size() == 1, "roll: one shift without dims");
        return ptsycl::roll(self.flatten(), shifts_sym, {0}).view(self.sizes());
    }
    TORCH_CHECK(shifts.size() == dims.size(), "roll: shifts and dims must have the same length");
    std::vector<int64_t> s(static_cast<std::size_t>(std::max<int64_t>(self.dim(), 1)), 0);
    for (std::size_t i = 0; i < dims.size(); ++i) {
        const int64_t d = c10::maybe_wrap_dim(dims[i], self.dim());
        const int64_t sz = self.size(d);
        if (sz) s[d] = (((s[d] + shifts[i]) % sz) + sz) % sz;
    }
    return remap(self, s, true);
}

Tensor unfold(const Tensor& self, int64_t dimension, int64_t size, int64_t step) {
    PTSYCL_TRACE_OP("unfold");
    const int64_t d = c10::maybe_wrap_dim(dimension, std::max<int64_t>(self.dim(), 1));
    const int64_t len = self.dim() == 0 ? 1 : self.size(d);
    TORCH_CHECK(size <= len && step > 0, "unfold: invalid size or step");
    std::vector<int64_t> sizes(self.sizes().begin(), self.sizes().end());
    std::vector<int64_t> strides(self.strides().begin(), self.strides().end());
    const int64_t st = self.dim() == 0 ? 1 : strides[d];
    if (self.dim() > 0) {
        sizes[d] = (len - size) / step + 1;
        strides[d] = st * step;
    }
    sizes.push_back(size);
    strides.push_back(st);
    return self.as_strided(sizes, strides, self.storage_offset());
}

Tensor reflect_index(int64_t len, int64_t left, int64_t right, const Tensor& like) {
    std::vector<int64_t> idx(static_cast<std::size_t>(len + left + right));
    for (int64_t j = 0; j < len + left + right; ++j) {
        int64_t s = j - left;
        if (s < 0) s = -s;
        if (s >= len) s = 2 * (len - 1) - s;
        idx[j] = s;
    }
    return at::tensor(idx, at::kLong).to(like.device());
}

Tensor reflection_pad(const Tensor& self, at::IntArrayRef pad) {
    Tensor y = self;
    for (std::size_t i = 0; i < pad.size() / 2; ++i) {
        const int64_t d = self.dim() - 1 - static_cast<int64_t>(i);
        TORCH_CHECK(pad[2 * i] < self.size(d) && pad[2 * i + 1] < self.size(d),
                    "reflection_pad: padding must be smaller than the input dimension");
        y = y.index_select(d, reflect_index(self.size(d), pad[2 * i], pad[2 * i + 1], self));
    }
    return y;
}

Tensor reflection_pad_backward(const Tensor& grad_output, const Tensor& self, at::IntArrayRef pad) {
    Tensor g = grad_output;
    for (std::size_t i = pad.size() / 2; i-- > 0;) {
        const int64_t d = self.dim() - 1 - static_cast<int64_t>(i);
        std::vector<int64_t> s(g.sizes().begin(), g.sizes().end());
        s[d] = self.size(d);
        g = at::zeros(s, g.options()).index_add_(d, reflect_index(self.size(d), pad[2 * i], pad[2 * i + 1], self), g);
    }
    return g;
}

#define PTSYCL_REFLECTION_PAD(n)                                                                          \
    Tensor& reflection_pad##n##d_out(const Tensor& self, c10::SymIntArrayRef pad, Tensor& out) {          \
        PTSYCL_TRACE_OP("reflection_pad" #n "d.out");                                                    \
        Tensor y = reflection_pad(self, C10_AS_INTARRAYREF_SLOW(pad));                                    \
        out.resize_(y.sizes());                                                                          \
        return out.copy_(y);                                                                             \
    }                                                                                                    \
    Tensor& reflection_pad##n##d_backward_out(const Tensor& grad_output, const Tensor& self,              \
                                              c10::SymIntArrayRef pad, Tensor& grad_input) {             \
        PTSYCL_TRACE_OP("reflection_pad" #n "d_backward.grad_input");                                    \
        Tensor g = reflection_pad_backward(grad_output, self, C10_AS_INTARRAYREF_SLOW(pad));              \
        grad_input.resize_(g.sizes());                                                                   \
        return grad_input.copy_(g);                                                                      \
    }
PTSYCL_REFLECTION_PAD(1)
PTSYCL_REFLECTION_PAD(2)
Tensor reflection_pad2d(const Tensor& self, c10::SymIntArrayRef pad) {
    PTSYCL_TRACE_OP("reflection_pad2d");
    return reflection_pad(self, C10_AS_INTARRAYREF_SLOW(pad));
}
Tensor reflection_pad2d_backward(const Tensor& grad_output, const Tensor& self, c10::SymIntArrayRef pad) {
    PTSYCL_TRACE_OP("reflection_pad2d_backward");
    return reflection_pad_backward(grad_output, self, C10_AS_INTARRAYREF_SLOW(pad));
}
#undef PTSYCL_REFLECTION_PAD

Tensor& upsample_nearest1d_out(const Tensor& self, c10::SymIntArrayRef output_size, std::optional<double> scale,
                               Tensor& out) {
    PTSYCL_TRACE_OP("upsample_nearest1d.out");
    const int64_t L = C10_AS_INTARRAYREF_SLOW(output_size)[0];
    Tensor y = at::upsample_nearest2d(self.unsqueeze(2), {1, L}, 1.0, scale).squeeze(2);
    out.resize_(y.sizes());
    return out.copy_(y);
}

Tensor& upsample_nearest1d_backward_out(const Tensor& grad_output, c10::SymIntArrayRef output_size,
                                        c10::SymIntArrayRef input_size, std::optional<double> scale,
                                        Tensor& grad_input) {
    PTSYCL_TRACE_OP("upsample_nearest1d_backward.grad_input");
    const auto in = C10_AS_INTARRAYREF_SLOW(input_size);
    const int64_t L = C10_AS_INTARRAYREF_SLOW(output_size)[0];
    Tensor g = at::upsample_nearest2d_backward(grad_output.unsqueeze(2), {1, L}, {in[0], in[1], 1, in[2]}, 1.0, scale)
                   .squeeze(2);
    grad_input.resize_(g.sizes());
    return grad_input.copy_(g);
}

template <bool kMax>
void cum_extreme(const Tensor& self, Tensor& values, Tensor& indices, int64_t dim) {
    if (self.numel() == 0) return;
    const int64_t d = c10::maybe_wrap_dim(dim, std::max<int64_t>(self.dim(), 1));
    const Tensor x = self.dim() == 0 ? self.view({1, 1}) : self.movedim(d, -1).contiguous().view({-1, self.size(d)});
    const int64_t rows = x.size(0), n = x.size(1);
    Tensor v = at::empty_like(x), ix = at::empty(x.sizes(), x.options().dtype(at::kLong));
    auto& q = queue_for(self);
    AT_DISPATCH_ALL_TYPES_AND3(c10::kBool, c10::kHalf, c10::kBFloat16, self.scalar_type(), "ptsycl_cum_extreme", [&] {
        const scalar_t* px = data_ptr<scalar_t>(x);
        scalar_t* pv = data_ptr<scalar_t>(v);
        int64_t* pi = data_ptr<int64_t>(ix);
        launch_flat(q, rows, [=](std::size_t r) {
            const int64_t base = static_cast<int64_t>(r) * n;
            double best = static_cast<double>(px[base]);
            int64_t bi = 0;
            for (int64_t j = 0; j < n; ++j) {
                const double xv = static_cast<double>(px[base + j]);
                const bool better = kMax ? xv >= best : xv <= best;
                if (!std::isnan(best) && (std::isnan(xv) || better)) {
                    best = xv;
                    bi = j;
                }
                pv[base + j] = static_cast<scalar_t>(best);
                pi[base + j] = bi;
            }
        });
    });
    if (self.dim() == 0) {
        values.copy_(v.view(c10::IntArrayRef{}));
        indices.copy_(ix.view(c10::IntArrayRef{}));
        return;
    }
    std::vector<int64_t> moved(self.sizes().begin(), self.sizes().end());
    moved.erase(moved.begin() + d);
    moved.push_back(self.size(d));
    values.copy_(v.view(moved).movedim(-1, d));
    indices.copy_(ix.view(moved).movedim(-1, d));
}

void cummax_helper(const Tensor& self, Tensor& values, Tensor& indices, int64_t dim) {
    PTSYCL_TRACE_OP("_cummax_helper");
    cum_extreme<true>(self, values, indices, dim);
}
void cummin_helper(const Tensor& self, Tensor& values, Tensor& indices, int64_t dim) {
    PTSYCL_TRACE_OP("_cummin_helper");
    cum_extreme<false>(self, values, indices, dim);
}

std::tuple<Tensor, Tensor> take_sorted(const Tensor& self, int64_t d, const Tensor& pos, bool keepdim) {
    auto [sv, si] = at::sort(self, true, d, false);
    Tensor v = sv.gather(d, pos), i = si.gather(d, pos);
    if (!keepdim) {
        v = v.squeeze(d);
        i = i.squeeze(d);
    }
    return {v, i};
}

std::vector<int64_t> keep_shape(const Tensor& self, int64_t d) {
    std::vector<int64_t> s(self.sizes().begin(), self.sizes().end());
    s[d] = 1;
    return s;
}

std::tuple<Tensor&, Tensor&> write2(const std::tuple<Tensor, Tensor>& r, Tensor& values, Tensor& indices) {
    values.resize_(std::get<0>(r).sizes());
    indices.resize_(std::get<1>(r).sizes());
    values.copy_(std::get<0>(r));
    indices.copy_(std::get<1>(r));
    return {values, indices};
}

std::tuple<Tensor&, Tensor&> kthvalue_out(const Tensor& self, int64_t k, int64_t dim, bool keepdim, Tensor& values,
                                          Tensor& indices) {
    PTSYCL_TRACE_OP("kthvalue.values");
    if (self.dim() == 0) {
        TORCH_CHECK(k == 1, "kthvalue: k out of range");
        return write2({self.clone(), at::zeros({}, self.options().dtype(at::kLong))}, values, indices);
    }
    const int64_t d = c10::maybe_wrap_dim(dim, self.dim());
    TORCH_CHECK(k >= 1 && k <= self.size(d), "kthvalue: k out of range");
    const Tensor pos = at::full(keep_shape(self, d), k - 1, self.options().dtype(at::kLong));
    return write2(take_sorted(self, d, pos, keepdim), values, indices);
}

std::tuple<Tensor, Tensor> median_impl(const Tensor& self, int64_t dim, bool keepdim, bool ignore_nan) {
    if (self.dim() == 0) return {self.clone(), at::zeros({}, self.options().dtype(at::kLong))};
    const int64_t d = c10::maybe_wrap_dim(dim, self.dim());
    TORCH_CHECK(self.size(d) > 0, "median: empty dimension");
    const int64_t n = self.size(d);
    Tensor pos;
    if (self.is_floating_point()) {
        const Tensor numbers = self.isnan().logical_not().sum(d, true);
        pos = ignore_nan ? (numbers - 1).clamp_min(0).div(2, "floor")
                         : at::where(numbers < n, numbers, at::full_like(numbers, (n - 1) / 2));
    } else {
        pos = at::full(keep_shape(self, d), (n - 1) / 2, self.options().dtype(at::kLong));
    }
    return take_sorted(self, d, pos.to(at::kLong), keepdim);
}

std::tuple<Tensor&, Tensor&> median_dim_out(const Tensor& self, int64_t dim, bool keepdim, Tensor& values,
                                            Tensor& indices) {
    PTSYCL_TRACE_OP("median.dim_values");
    return write2(median_impl(self, dim, keepdim, false), values, indices);
}
std::tuple<Tensor&, Tensor&> nanmedian_dim_out(const Tensor& self, int64_t dim, bool keepdim, Tensor& values,
                                               Tensor& indices) {
    PTSYCL_TRACE_OP("nanmedian.dim_values");
    return write2(median_impl(self, dim, keepdim, true), values, indices);
}
Tensor median(const Tensor& self) {
    PTSYCL_TRACE_OP("median");
    return std::get<0>(median_impl(self.flatten(), 0, false, false));
}
Tensor nanmedian(const Tensor& self) {
    PTSYCL_TRACE_OP("nanmedian");
    return std::get<0>(median_impl(self.flatten(), 0, false, true));
}

Tensor& write(const Tensor& r, Tensor& out) {
    out.resize_(r.sizes());
    return out.copy_(r);
}

Tensor isnan(const Tensor& self) {
    PTSYCL_TRACE_OP("isnan");
    if (!self.is_floating_point()) return at::zeros(self.sizes(), self.options().dtype(at::kBool));
    return at::ne(self, self);
}

Tensor rsub(const Tensor& self, const Tensor& other, const c10::Scalar& alpha) {
    PTSYCL_TRACE_OP("rsub.Tensor");
    return at::sub(other, self, alpha);
}

Tensor& clamp_tensor_out(const Tensor& self, const std::optional<Tensor>& min, const std::optional<Tensor>& max,
                         Tensor& out) {
    PTSYCL_TRACE_OP("clamp.Tensor_out");
    Tensor r = self;
    if (min && min->defined()) r = at::maximum(r, *min);
    if (max && max->defined()) r = at::minimum(r, *max);
    return write(r.to(out.scalar_type()), out);
}
Tensor& clamp_min_tensor_out(const Tensor& self, const Tensor& min, Tensor& out) {
    PTSYCL_TRACE_OP("clamp_min.Tensor_out");
    return write(at::maximum(self, min).to(out.scalar_type()), out);
}
Tensor& clamp_max_tensor_out(const Tensor& self, const Tensor& max, Tensor& out) {
    PTSYCL_TRACE_OP("clamp_max.Tensor_out");
    return write(at::minimum(self, max).to(out.scalar_type()), out);
}

Tensor nansum(const Tensor& self, at::OptionalIntArrayRef dim, bool keepdim, std::optional<c10::ScalarType> dtype) {
    PTSYCL_TRACE_OP("nansum");
    const Tensor x = self.is_floating_point() ? at::where(self.isnan(), at::zeros({}, self.options()), self) : self;
    return at::sum(x, dim, keepdim, dtype);
}

Tensor scaled_add(const Tensor& self, Tensor prod, const c10::Scalar& beta, const c10::Scalar& alpha) {
    if (alpha.toDouble() != 1.0) prod = prod.mul(alpha);
    if (beta.toDouble() == 0.0) return prod;
    return beta.toDouble() == 1.0 ? prod.add(self) : prod.add(self.mul(beta));
}

Tensor& addmv_out(const Tensor& self, const Tensor& mat, const Tensor& vec, const c10::Scalar& beta,
                  const c10::Scalar& alpha, Tensor& out) {
    PTSYCL_TRACE_OP("addmv.out");
    return write(scaled_add(self, at::mm(mat, vec.unsqueeze(1)).squeeze(1), beta, alpha), out);
}

Tensor& baddbmm_out(const Tensor& self, const Tensor& b1, const Tensor& b2, const c10::Scalar& beta,
                    const c10::Scalar& alpha, Tensor& out) {
    PTSYCL_TRACE_OP("baddbmm.out");
    return write(scaled_add(self, at::bmm(b1, b2), beta, alpha), out);
}

Tensor addbmm(const Tensor& self, const Tensor& b1, const Tensor& b2, const c10::Scalar& beta,
              const c10::Scalar& alpha) {
    PTSYCL_TRACE_OP("addbmm");
    return scaled_add(self, at::bmm(b1, b2).sum(0), beta, alpha);
}
Tensor& addbmm_out(const Tensor& self, const Tensor& b1, const Tensor& b2, const c10::Scalar& beta,
                   const c10::Scalar& alpha, Tensor& out) {
    return write(ptsycl::addbmm(self, b1, b2, beta, alpha), out);
}

Tensor vdot(const Tensor& self, const Tensor& other) {
    PTSYCL_TRACE_OP("vdot");
    TORCH_CHECK(!self.is_complex(), "paras vdot: complex tensors are not supported");
    return at::dot(self, other);
}

bool equal(const Tensor& self, const Tensor& other) {
    PTSYCL_TRACE_OP("equal");
    if (!self.sizes().equals(other.sizes())) return false;
    if (self.numel() == 0) return true;
    return at::eq(self, other).all().item<bool>();
}

Tensor& lerp_tensor_out(const Tensor& self, const Tensor& end, const Tensor& weight, Tensor& out) {
    PTSYCL_TRACE_OP("lerp.Tensor_out");
    const Tensor diff = end - self;
    return write(at::where(weight.abs() < 0.5, self + weight * diff, end - diff * (1 - weight)), out);
}

Tensor& fill_tensor_(Tensor& self, const Tensor& value) {
    PTSYCL_TRACE_OP("fill_.Tensor");
    TORCH_CHECK(value.dim() == 0, "fill_ only supports a 0-dimensional value tensor");
    return self.copy_(value.to(self.device(), self.scalar_type()).expand(self.sizes()));
}

Tensor masked_select(const Tensor& self, const Tensor& mask) {
    PTSYCL_TRACE_OP("masked_select");
    auto b = at::broadcast_tensors({self, mask});
    const Tensor idx = at::nonzero(b[1].flatten()).squeeze(1);
    return b[0].flatten().index_select(0, idx);
}

Tensor& linalg_cross_out(const Tensor& self, const Tensor& other, int64_t dim, Tensor& out) {
    PTSYCL_TRACE_OP("linalg_cross.out");
    auto b = at::broadcast_tensors({self, other});
    const int64_t d = c10::maybe_wrap_dim(dim, b[0].dim());
    TORCH_CHECK(b[0].size(d) == 3, "linalg.cross: dimension ", dim, " must have size 3");
    auto c = [&](const Tensor& t, int64_t i) { return t.select(d, i); };
    const Tensor r = at::stack({c(b[0], 1) * c(b[1], 2) - c(b[0], 2) * c(b[1], 1),
                                c(b[0], 2) * c(b[1], 0) - c(b[0], 0) * c(b[1], 2),
                                c(b[0], 0) * c(b[1], 1) - c(b[0], 1) * c(b[1], 0)}, d);
    return write(r, out);
}

Tensor& norm_out(const Tensor& self, const std::optional<c10::Scalar>& p, at::IntArrayRef dim, bool keepdim,
                 Tensor& out) {
    PTSYCL_TRACE_OP("norm.out");
    return write(at::linalg_vector_norm(self, p.value_or(2.0), dim, keepdim, std::nullopt), out);
}

}
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("aten::flip", &ptsycl::flip);
    m.impl("aten::roll", &ptsycl::roll);
    m.impl("aten::unfold", &ptsycl::unfold);
    m.impl("aten::reflection_pad1d.out", &ptsycl::reflection_pad1d_out);
    m.impl("aten::reflection_pad2d.out", &ptsycl::reflection_pad2d_out);
    m.impl("aten::reflection_pad2d", &ptsycl::reflection_pad2d);
    m.impl("aten::reflection_pad2d_backward", &ptsycl::reflection_pad2d_backward);
    m.impl("aten::reflection_pad1d_backward.grad_input", &ptsycl::reflection_pad1d_backward_out);
    m.impl("aten::reflection_pad2d_backward.grad_input", &ptsycl::reflection_pad2d_backward_out);
    m.impl("aten::upsample_nearest1d.out", &ptsycl::upsample_nearest1d_out);
    m.impl("aten::upsample_nearest1d_backward.grad_input", &ptsycl::upsample_nearest1d_backward_out);
    m.impl("aten::_cummax_helper", &ptsycl::cummax_helper);
    m.impl("aten::_cummin_helper", &ptsycl::cummin_helper);
    m.impl("aten::kthvalue.values", &ptsycl::kthvalue_out);
    m.impl("aten::median.dim_values", &ptsycl::median_dim_out);
    m.impl("aten::nanmedian.dim_values", &ptsycl::nanmedian_dim_out);
    m.impl("aten::median", &ptsycl::median);
    m.impl("aten::nanmedian", &ptsycl::nanmedian);
    m.impl("aten::isnan", &ptsycl::isnan);
    m.impl("aten::rsub.Tensor", &ptsycl::rsub);
    m.impl("aten::clamp.Tensor_out", &ptsycl::clamp_tensor_out);
    m.impl("aten::clamp_min.Tensor_out", &ptsycl::clamp_min_tensor_out);
    m.impl("aten::clamp_max.Tensor_out", &ptsycl::clamp_max_tensor_out);
    m.impl("aten::nansum", &ptsycl::nansum);
    m.impl("aten::addmv.out", &ptsycl::addmv_out);
    m.impl("aten::baddbmm.out", &ptsycl::baddbmm_out);
    m.impl("aten::addbmm", &ptsycl::addbmm);
    m.impl("aten::addbmm.out", &ptsycl::addbmm_out);
    m.impl("aten::vdot", &ptsycl::vdot);
    m.impl("aten::equal", &ptsycl::equal);
    m.impl("aten::lerp.Tensor_out", &ptsycl::lerp_tensor_out);
    m.impl("aten::fill_.Tensor", &ptsycl::fill_tensor_);
    m.impl("aten::masked_select", &ptsycl::masked_select);
    m.impl("aten::linalg_cross.out", &ptsycl::linalg_cross_out);
    m.impl("aten::norm.out", &ptsycl::norm_out);
}
