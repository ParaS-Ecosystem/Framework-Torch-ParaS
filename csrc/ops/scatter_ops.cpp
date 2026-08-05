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

// NOTE: masked_fill.Scalar/.Tensor (out-of-place) and scatter.src/.value
// (out-of-place, non-structured-delegate entry) are CompositeExplicitAutograd
// or structured_delegate upstream -- they decompose into clone()/empty() plus
// the in-place or .out kernels below. We only implement the true per-backend
// entry points (masked_fill_.Scalar/.Tensor, scatter.*_out) directly and
// provide thin functional/in-place wrappers ourselves for clarity and to
// avoid depending on this backend's composite dispatch resolution for a new
// op family, matching the pattern already used for sigmoid/tanh/softmax.

namespace ptsycl {
namespace {

using at::Tensor;

Tensor to_long(const Tensor& t) {
    return t.scalar_type() == c10::kLong ? t : t.to(c10::kLong);
}

// -----------------------------------------------------------------------------
// masked_fill_
// -----------------------------------------------------------------------------
Tensor& masked_fill__scalar(Tensor& self, const Tensor& mask,
                            const c10::Scalar& value) {
    PTSYCL_TRACE_OP("masked_fill_.Scalar");
    TORCH_CHECK(mask.scalar_type() == c10::kBool,
                "paras masked_fill_: mask must be a bool tensor");
    TORCH_CHECK(spec_supported(self), "paras masked_fill_: rank exceeds kernel limit");

    auto& q = queue_for(self);
    const int64_t n = self.numel();
    if (n == 0) return self;
    Tensor me = mask.expand_as(self);
    TORCH_CHECK(spec_supported(me), "paras masked_fill_: rank exceeds kernel limit");

    AT_DISPATCH_ALL_TYPES_AND3(
        c10::kBool, c10::kHalf, c10::kBFloat16, self.scalar_type(),
        "ptsycl_masked_fill", [&] {
            scalar_t*   p  = data_ptr<scalar_t>(self);
            const bool* pm = data_ptr<bool>(me);
            const auto sself = make_spec(self);
            const auto smask = make_spec(me);
            const scalar_t v = value.to<scalar_t>();

            launch_flat(q, n, [=](std::size_t flat_) {
                const int64_t i = static_cast<int64_t>(flat_);
                if (pm[smask.index(i)]) p[sself.index(i)] = v;
            });
        });
    return self;
}

Tensor& masked_fill__tensor(Tensor& self, const Tensor& mask,
                           const Tensor& value) {
    PTSYCL_TRACE_OP("masked_fill_.Tensor");
    TORCH_CHECK(value.dim() == 0,
                "paras masked_fill_: value must be a 0-dim tensor");
    return ptsycl::masked_fill__scalar(self, mask, value.item());
}

// -----------------------------------------------------------------------------
// scatter
// -----------------------------------------------------------------------------
// index shares self's rank; index.size(d) <= self.size(d) for every d.
// out (== self, or a copy of it) is overwritten only at the positions index
// picks out. Like upstream, results with duplicate indices along `dim` are
// unspecified (racing writes, no atomics) -- this matches real CPU/CUDA
// scatter semantics, which document the non-reduce form as such too.
Tensor& scatter_src_out(const Tensor& self, int64_t dim, const Tensor& index,
                        const Tensor& src, Tensor& out) {
    PTSYCL_TRACE_OP("scatter.src_out");
    TORCH_CHECK(index.dim() == self.dim(),
                "paras scatter: index must have the same rank as self");
    TORCH_CHECK(spec_supported(self) && spec_supported(index) &&
                    spec_supported(src) && spec_supported(out),
                "paras scatter: rank exceeds kernel limit");

    auto& q = queue_for(self);
    const int64_t d = c10::maybe_wrap_dim(dim, self.dim());
    Tensor idx = to_long(index);

    if (out.data_ptr() != self.data_ptr()) out.copy_(self);

    const auto idx_spec = make_spec(idx);
    const auto out_spec = make_spec(out);
    const auto src_spec = make_spec(src);
    const int     ndim = idx_spec.ndim;
    const int64_t n    = idx.numel();
    const int64_t out_dim_stride = out.stride(d);
    if (n == 0) return out;

    AT_DISPATCH_ALL_TYPES_AND3(
        c10::kBool, c10::kHalf, c10::kBFloat16, self.scalar_type(),
        "ptsycl_scatter_src", [&] {
            scalar_t*       pout = data_ptr<scalar_t>(out);
            const scalar_t* psrc = data_ptr<scalar_t>(src);
            const int64_t*  pidx = data_ptr<int64_t>(idx);

            launch_flat(q, n, [=](std::size_t flat_) {
                const int64_t flat    = static_cast<int64_t>(flat_);
                const int64_t idx_off = idx_spec.index(flat);
                const int64_t src_off = src_spec.index(flat);

                int64_t rem = flat;
                int64_t out_off = 0;
                for (int dd = ndim - 1; dd >= 0; --dd) {
                    const int64_t c = rem % idx_spec.sizes[dd];
                    rem /= idx_spec.sizes[dd];
                    if (dd != d) out_off += c * out_spec.strides[dd];
                }
                const int64_t sel = pidx[idx_off];
                out_off += sel * out_dim_stride;
                pout[out_off] = psrc[src_off];
            });
        });
    return out;
}

Tensor& scatter_value_out(const Tensor& self, int64_t dim, const Tensor& index,
                          const c10::Scalar& value, Tensor& out) {
    PTSYCL_TRACE_OP("scatter.value_out");
    TORCH_CHECK(index.dim() == self.dim(),
                "paras scatter: index must have the same rank as self");
    TORCH_CHECK(spec_supported(self) && spec_supported(index) &&
                    spec_supported(out),
                "paras scatter: rank exceeds kernel limit");

    auto& q = queue_for(self);
    const int64_t d = c10::maybe_wrap_dim(dim, self.dim());
    Tensor idx = to_long(index);

    if (out.data_ptr() != self.data_ptr()) out.copy_(self);

    const auto idx_spec = make_spec(idx);
    const auto out_spec = make_spec(out);
    const int     ndim = idx_spec.ndim;
    const int64_t n    = idx.numel();
    const int64_t out_dim_stride = out.stride(d);
    if (n == 0) return out;

    AT_DISPATCH_ALL_TYPES_AND3(
        c10::kBool, c10::kHalf, c10::kBFloat16, self.scalar_type(),
        "ptsycl_scatter_value", [&] {
            scalar_t*      pout = data_ptr<scalar_t>(out);
            const int64_t* pidx = data_ptr<int64_t>(idx);
            const scalar_t v    = value.to<scalar_t>();

            launch_flat(q, n, [=](std::size_t flat_) {
                const int64_t flat    = static_cast<int64_t>(flat_);
                const int64_t idx_off = idx_spec.index(flat);

                int64_t rem = flat;
                int64_t out_off = 0;
                for (int dd = ndim - 1; dd >= 0; --dd) {
                    const int64_t c = rem % idx_spec.sizes[dd];
                    rem /= idx_spec.sizes[dd];
                    if (dd != d) out_off += c * out_spec.strides[dd];
                }
                const int64_t sel = pidx[idx_off];
                out_off += sel * out_dim_stride;
                pout[out_off] = v;
            });
        });
    return out;
}

Tensor scatter_src(const Tensor& self, int64_t dim, const Tensor& index,
                   const Tensor& src) {
    PTSYCL_TRACE_OP("scatter.src");
    Tensor out = self.clone();
    return ptsycl::scatter_src_out(self, dim, index, src, out);
}

Tensor scatter_value(const Tensor& self, int64_t dim, const Tensor& index,
                     const c10::Scalar& value) {
    PTSYCL_TRACE_OP("scatter.value");
    Tensor out = self.clone();
    return ptsycl::scatter_value_out(self, dim, index, value, out);
}

Tensor& scatter_src_(Tensor& self, int64_t dim, const Tensor& index,
                     const Tensor& src) {
    PTSYCL_TRACE_OP("scatter_.src");
    return ptsycl::scatter_src_out(self, dim, index, src, self);
}

Tensor& scatter_value_(Tensor& self, int64_t dim, const Tensor& index,
                      const c10::Scalar& value) {
    PTSYCL_TRACE_OP("scatter_.value");
    return ptsycl::scatter_value_out(self, dim, index, value, self);
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("aten::masked_fill_.Scalar", &ptsycl::masked_fill__scalar);
    m.impl("aten::masked_fill_.Tensor", &ptsycl::masked_fill__tensor);

    m.impl("aten::scatter.src", &ptsycl::scatter_src);
    m.impl("aten::scatter.src_out", &ptsycl::scatter_src_out);
    m.impl("aten::scatter_.src", &ptsycl::scatter_src_);
    m.impl("aten::scatter.value", &ptsycl::scatter_value);
    m.impl("aten::scatter.value_out", &ptsycl::scatter_value_out);
    m.impl("aten::scatter_.value", &ptsycl::scatter_value_);
}

} // namespace ptsycl
