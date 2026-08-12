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

namespace ptsycl {
namespace {

using at::Tensor;

Tensor to_long(const Tensor& t) {
    return t.scalar_type() == c10::kLong ? t : t.to(c10::kLong);
}

Tensor to_compute(const Tensor& t, c10::ScalarType dtype, c10::Device device) {
    if (t.scalar_type() == dtype && t.device() == device) return t;
    if (t.dim() == 0 && t.device().is_cpu())
        return at::full({}, t.item(), t.options().dtype(dtype).device(device));
    return t.to(t.options().dtype(dtype).device(device));
}

// -----------------------------------------------------------------------------
// gather
// -----------------------------------------------------------------------------
Tensor& gather_out(const Tensor& self, int64_t dim, const Tensor& index,
                   bool /*sparse_grad*/, Tensor& out) {
    PTSYCL_TRACE_OP("gather.out");
    TORCH_CHECK(index.dim() == self.dim(),
                "paras gather: index must have the same rank as self");
    TORCH_CHECK(spec_supported(self) && spec_supported(index) &&
                    spec_supported(out),
                "paras gather: rank exceeds kernel limit");

    auto& q = queue_for(self);
    const int64_t d = c10::maybe_wrap_dim(dim, self.dim());
    Tensor idx = to_long(index);

    const auto idx_spec  = make_spec(idx);
    const auto self_spec = make_spec(self);
    const auto out_spec  = make_spec(out);
    const int     ndim = idx_spec.ndim;
    const int64_t n    = idx.numel();
    const int64_t self_dim_stride = self.stride(d);
    if (n == 0) return out;

    AT_DISPATCH_ALL_TYPES_AND3(
        c10::kBool, c10::kHalf, c10::kBFloat16, self.scalar_type(),
        "ptsycl_gather", [&] {
            const scalar_t* pself = data_ptr<scalar_t>(self);
            scalar_t*       pout  = data_ptr<scalar_t>(out);
            const int64_t*  pidx  = data_ptr<int64_t>(idx);

            launch_flat(q, n, [=](std::size_t flat_) {
                const int64_t flat    = static_cast<int64_t>(flat_);
                const int64_t idx_off = idx_spec.index(flat);
                const int64_t out_off = out_spec.index(flat);

                int64_t rem = flat;
                int64_t self_off = 0;
                for (int dd = ndim - 1; dd >= 0; --dd) {
                    const int64_t c = rem % idx_spec.sizes[dd];
                    rem /= idx_spec.sizes[dd];
                    self_off += (dd == d) ? pidx[idx_off] * self_dim_stride
                                          : c * self_spec.strides[dd];
                }
                pout[out_off] = pself[self_off];
            });
        });
    return out;
}

Tensor gather(const Tensor& self, int64_t dim, const Tensor& index,
             bool sparse_grad) {
    PTSYCL_TRACE_OP("gather");
    Tensor out = at::empty(index.sizes(), self.options());
    return ptsycl::gather_out(self, dim, index, sparse_grad, out);
}

// -----------------------------------------------------------------------------
// index_select
// -----------------------------------------------------------------------------
Tensor& index_select_out(const Tensor& self, int64_t dim, const Tensor& index,
                         Tensor& out) {
    PTSYCL_TRACE_OP("index_select.out");
    TORCH_CHECK(index.dim() <= 1,
                "paras index_select: index must be 0-D or 1-D");
    TORCH_CHECK(spec_supported(self) && spec_supported(out),
                "paras index_select: rank exceeds kernel limit");

    auto& q = queue_for(self);
    const int64_t d = c10::maybe_wrap_dim(dim, self.dim());
    Tensor idx = to_long(index.dim() == 0 ? index.unsqueeze(0) : index)
                    .contiguous();
    const int64_t n = out.numel();
    if (n == 0) return out;

    const auto self_spec = make_spec(self);
    const auto out_spec  = make_spec(out);
    const int64_t self_dim_stride = self.stride(d);
    const int     ndim = out_spec.ndim;

    AT_DISPATCH_ALL_TYPES_AND3(
        c10::kBool, c10::kHalf, c10::kBFloat16, self.scalar_type(),
        "ptsycl_index_select", [&] {
            const scalar_t* pself = data_ptr<scalar_t>(self);
            scalar_t*       pout  = data_ptr<scalar_t>(out);
            const int64_t*  pidx  = data_ptr<int64_t>(idx);

            launch_flat(q, n, [=](std::size_t flat_) {
                const int64_t flat    = static_cast<int64_t>(flat_);
                const int64_t out_off = out_spec.index(flat);

                int64_t rem = flat;
                int64_t self_off = 0;
                int64_t sel = 0;
                for (int dd = ndim - 1; dd >= 0; --dd) {
                    const int64_t c = rem % out_spec.sizes[dd];
                    rem /= out_spec.sizes[dd];
                    if (dd == d) sel = c;
                    else self_off += c * self_spec.strides[dd];
                }
                self_off += pidx[sel] * self_dim_stride;
                pout[out_off] = pself[self_off];
            });
        });
    return out;
}

Tensor index_select(const Tensor& self, int64_t dim, const Tensor& index) {
    PTSYCL_TRACE_OP("index_select");
    const int64_t d = c10::maybe_wrap_dim(dim, self.dim());
    auto sizes = self.sizes().vec();
    sizes[d] = index.dim() == 0 ? 1 : index.numel();
    Tensor out = at::empty(sizes, self.options());
    return ptsycl::index_select_out(self, dim, index, out);
}

// -----------------------------------------------------------------------------
// index_copy_
// -----------------------------------------------------------------------------
Tensor& index_copy_(Tensor& self, int64_t dim, const Tensor& index,
                    const Tensor& source) {
    PTSYCL_TRACE_OP("index_copy_");
    TORCH_CHECK(spec_supported(self) && spec_supported(source),
                "paras index_copy_: rank exceeds kernel limit");

    auto& q = queue_for(self);
    const int64_t d = c10::maybe_wrap_dim(dim, self.dim());
    Tensor idx = to_long(index).contiguous();
    const int64_t n = source.numel();
    if (n == 0) return self;

    const auto self_spec = make_spec(self);
    const auto src_spec  = make_spec(source);
    const int64_t self_dim_stride = self.stride(d);
    const int     ndim = src_spec.ndim;

    AT_DISPATCH_ALL_TYPES_AND3(
        c10::kBool, c10::kHalf, c10::kBFloat16, self.scalar_type(),
        "ptsycl_index_copy", [&] {
            scalar_t*       pself = data_ptr<scalar_t>(self);
            const scalar_t* psrc  = data_ptr<scalar_t>(source);
            const int64_t*  pidx  = data_ptr<int64_t>(idx);

            launch_flat(q, n, [=](std::size_t flat_) {
                const int64_t flat    = static_cast<int64_t>(flat_);
                const int64_t src_off = src_spec.index(flat);

                int64_t rem = flat;
                int64_t self_off = 0;
                int64_t sel = 0;
                for (int dd = ndim - 1; dd >= 0; --dd) {
                    const int64_t c = rem % src_spec.sizes[dd];
                    rem /= src_spec.sizes[dd];
                    if (dd == d) sel = c;
                    else self_off += c * self_spec.strides[dd];
                }
                self_off += pidx[sel] * self_dim_stride;
                pself[self_off] = psrc[src_off];
            });
        });
    return self;
}

Tensor index_copy(const Tensor& self, int64_t dim, const Tensor& index,
                  const Tensor& source) {
    PTSYCL_TRACE_OP("index_copy");
    Tensor out = self.clone();
    return ptsycl::index_copy_(out, dim, index, source);
}

// -----------------------------------------------------------------------------
// Advanced indexing (index.Tensor / index_put_) -- supports a run of
// leading, non-null Long index tensors broadcast against each other, with
// the remaining trailing dims taken in full.
// -----------------------------------------------------------------------------
struct IndexPtrs {
    const int64_t* p[kMaxDims]{};
};

std::vector<int64_t> broadcast_shape(const std::vector<Tensor>& ts) {
    std::vector<int64_t> shape = ts[0].sizes().vec();
    for (std::size_t i = 1; i < ts.size(); ++i)
        shape = at::infer_size(shape, ts[i].sizes());
    return shape;
}

struct AdvancedIndex {
    std::vector<Tensor> idx;
    int64_t k = 0;
    std::vector<int64_t> bshape;
    int64_t bn = 1;
};

AdvancedIndex prepare_advanced_index(
    const Tensor& self, const c10::List<c10::optional<Tensor>>& indices) {
    TORCH_CHECK(static_cast<int64_t>(indices.size()) <= self.dim(),
                "paras index: too many indices for tensor");
    int64_t k = 0;
    std::vector<Tensor> raw;
    for (std::size_t i = 0; i < indices.size(); ++i) {
        c10::optional<Tensor> oi = indices[i];
        if (!oi.has_value() || !oi->defined()) break;
        raw.push_back(to_long(*oi));
        ++k;
    }
    for (int64_t i = k; i < static_cast<int64_t>(indices.size()); ++i) {
        c10::optional<Tensor> oi = indices[i];
        TORCH_CHECK(!oi.has_value() || !oi->defined(),
                    "paras index: only leading contiguous advanced "
                    "indices are supported");
    }
    TORCH_CHECK(k > 0, "paras index: at least one index tensor is required");

    AdvancedIndex ai;
    ai.k      = k;
    ai.bshape = broadcast_shape(raw);
    ai.bn     = 1;
    for (int64_t s : ai.bshape) ai.bn *= s;
    for (auto& t : raw) ai.idx.push_back(t.expand(ai.bshape).contiguous());
    return ai;
}

Tensor index_tensor(const Tensor& self,
                    const c10::List<c10::optional<Tensor>>& indices) {
    PTSYCL_TRACE_OP("index.Tensor");
    TORCH_CHECK(spec_supported(self), "paras index: rank exceeds kernel limit");

    auto ai = prepare_advanced_index(self, indices);
    auto& q = queue_for(self);

    std::vector<int64_t> out_sizes = ai.bshape;
    for (int64_t d = ai.k; d < self.dim(); ++d)
        out_sizes.push_back(self.size(d));
    Tensor out = at::empty(out_sizes, self.options());

    const int64_t n = out.numel();
    if (n == 0) return out;
    const int64_t tail_n = n / (ai.bn > 0 ? ai.bn : 1);

    const auto self_spec = make_spec(self);
    IndexPtrs pidx;
    for (int64_t i = 0; i < ai.k; ++i) pidx.p[i] = data_ptr<int64_t>(ai.idx[i]);
    const int64_t k = ai.k;

    AT_DISPATCH_ALL_TYPES_AND3(
        c10::kBool, c10::kHalf, c10::kBFloat16, self.scalar_type(),
        "ptsycl_index", [&] {
            const scalar_t* pself = data_ptr<scalar_t>(self);
            scalar_t*       pout  = data_ptr<scalar_t>(out);
            const auto sspec = self_spec;

            launch_flat(q, n, [=](std::size_t flat_) {
                const int64_t flat = static_cast<int64_t>(flat_);
                const int64_t b    = flat / tail_n;
                const int64_t t    = flat % tail_n;

                int64_t self_off = 0;
                for (int64_t i = 0; i < k; ++i)
                    self_off += pidx.p[i][b] * sspec.strides[i];

                int64_t rem = t;
                for (int dd = sspec.ndim - 1; dd >= static_cast<int>(k); --dd) {
                    const int64_t c = rem % sspec.sizes[dd];
                    rem /= sspec.sizes[dd];
                    self_off += c * sspec.strides[dd];
                }
                pout[flat] = pself[self_off];
            });
        });
    return out;
}

Tensor& index_put_(Tensor& self, const c10::List<c10::optional<Tensor>>& indices,
                   const Tensor& values, bool accumulate) {
    PTSYCL_TRACE_OP("index_put_");
    TORCH_CHECK(spec_supported(self),
                "paras index_put_: rank exceeds kernel limit");

    auto ai = prepare_advanced_index(self, indices);
    auto& q = queue_for(self);

    std::vector<int64_t> vshape = ai.bshape;
    for (int64_t d = ai.k; d < self.dim(); ++d) vshape.push_back(self.size(d));
    Tensor v = values.expand(vshape).contiguous();

    const int64_t n = v.numel();
    if (n == 0) return self;
    const int64_t tail_n = n / (ai.bn > 0 ? ai.bn : 1);

    const auto self_spec = make_spec(self);
    IndexPtrs pidx;
    for (int64_t i = 0; i < ai.k; ++i) pidx.p[i] = data_ptr<int64_t>(ai.idx[i]);
    const int64_t k = ai.k;

    AT_DISPATCH_ALL_TYPES_AND3(
        c10::kBool, c10::kHalf, c10::kBFloat16, self.scalar_type(),
        "ptsycl_index_put", [&] {
            scalar_t*       pself = data_ptr<scalar_t>(self);
            const scalar_t* pval  = data_ptr<scalar_t>(v);
            const auto sspec = self_spec;

            launch_flat(q, n, [=](std::size_t flat_) {
                const int64_t flat = static_cast<int64_t>(flat_);
                const int64_t b    = flat / tail_n;
                const int64_t t    = flat % tail_n;

                int64_t self_off = 0;
                for (int64_t i = 0; i < k; ++i)
                    self_off += pidx.p[i][b] * sspec.strides[i];

                int64_t rem = t;
                for (int dd = sspec.ndim - 1; dd >= static_cast<int>(k); --dd) {
                    const int64_t c = rem % sspec.sizes[dd];
                    rem /= sspec.sizes[dd];
                    self_off += c * sspec.strides[dd];
                }
                if (accumulate) pself[self_off] = pself[self_off] + pval[flat];
                else pself[self_off] = pval[flat];
            });
        });
    return self;
}

Tensor index_put(const Tensor& self,
                 const c10::List<c10::optional<Tensor>>& indices,
                 const Tensor& values, bool accumulate) {
    PTSYCL_TRACE_OP("index_put");
    Tensor out = self.clone();
    return ptsycl::index_put_(out, indices, values, accumulate);
}

// -----------------------------------------------------------------------------
// where.self
// -----------------------------------------------------------------------------
Tensor where_self(const Tensor& condition, const Tensor& self,
                  const Tensor& other) {
    PTSYCL_TRACE_OP("where.self");
    TORCH_CHECK(condition.scalar_type() == c10::kBool,
                "paras where: condition must be a bool tensor");

    const auto out_dtype = at::result_type(self, other);
    auto out_sizes =
        at::infer_size(at::infer_size(condition.sizes(), self.sizes()),
                       other.sizes());
    Tensor out = at::empty(out_sizes, self.options().dtype(out_dtype));

    auto& q = queue_for(out);
    const int64_t n = out.numel();
    if (n == 0) return out;

    Tensor ce = condition.expand(out_sizes);
    Tensor ae = to_compute(self, out_dtype, out.device()).expand(out_sizes);
    Tensor be = to_compute(other, out_dtype, out.device()).expand(out_sizes);

    AT_DISPATCH_ALL_TYPES_AND3(
        c10::kBool, c10::kHalf, c10::kBFloat16, out_dtype, "ptsycl_where", [&] {
            const auto sc = make_spec(ce);
            const auto sa = make_spec(ae);
            const auto sb = make_spec(be);
            const auto so = make_spec(out);
            const bool*     pc = data_ptr<bool>(ce);
            const scalar_t* pa = data_ptr<scalar_t>(ae);
            const scalar_t* pb = data_ptr<scalar_t>(be);
            scalar_t*       po = data_ptr<scalar_t>(out);

            launch_flat(q, n, [=](std::size_t flat_) {
                const int64_t i = static_cast<int64_t>(flat_);
                po[so.index(i)] =
                    pc[sc.index(i)] ? pa[sa.index(i)] : pb[sb.index(i)];
            });
        });
    return out;
}

// -----------------------------------------------------------------------------
// triu / tril
// -----------------------------------------------------------------------------
template <typename Cmp>
Tensor& triu_tril_out(const Tensor& self, int64_t diagonal, Tensor& out,
                      Cmp keep) {
    TORCH_CHECK(self.dim() >= 2,
                "paras triu/tril: input must have at least 2 dimensions");
    TORCH_CHECK(spec_supported(self) && spec_supported(out),
                "paras triu/tril: rank exceeds kernel limit");

    auto& q = queue_for(self);
    const int64_t n    = out.numel();
    if (n == 0) return out;
    const int64_t rows = self.size(-2);
    const int64_t cols = self.size(-1);
    Tensor se = self.expand_as(out);

    AT_DISPATCH_ALL_TYPES_AND3(
        c10::kBool, c10::kHalf, c10::kBFloat16, self.scalar_type(),
        "ptsycl_triu_tril", [&] {
            const auto ss = make_spec(se);
            const auto so = make_spec(out);
            const scalar_t* pin  = data_ptr<scalar_t>(se);
            scalar_t*       pout = data_ptr<scalar_t>(out);

            launch_flat(q, n, [=](std::size_t flat_) {
                const int64_t flat = static_cast<int64_t>(flat_);
                const int64_t col  = flat % cols;
                const int64_t row  = (flat / cols) % rows;
                const scalar_t v   = pin[ss.index(flat)];
                pout[so.index(flat)] = keep(row, col, diagonal) ? v : scalar_t(0);
            });
        });
    return out;
}

Tensor& triu_out(const Tensor& self, int64_t diagonal, Tensor& out) {
    PTSYCL_TRACE_OP("triu.out");
    return triu_tril_out(
        self, diagonal, out,
        [](int64_t row, int64_t col, int64_t diag) { return col - row >= diag; });
}

Tensor& tril_out(const Tensor& self, int64_t diagonal, Tensor& out) {
    PTSYCL_TRACE_OP("tril.out");
    return triu_tril_out(
        self, diagonal, out,
        [](int64_t row, int64_t col, int64_t diag) { return col - row <= diag; });
}

Tensor triu(const Tensor& self, int64_t diagonal) {
    PTSYCL_TRACE_OP("triu");
    Tensor out = at::empty_like(self);
    return ptsycl::triu_out(self, diagonal, out);
}

Tensor& triu_(Tensor& self, int64_t diagonal) {
    PTSYCL_TRACE_OP("triu_");
    return ptsycl::triu_out(self, diagonal, self);
}

Tensor tril(const Tensor& self, int64_t diagonal) {
    PTSYCL_TRACE_OP("tril");
    Tensor out = at::empty_like(self);
    return ptsycl::tril_out(self, diagonal, out);
}

Tensor& tril_(Tensor& self, int64_t diagonal) {
    PTSYCL_TRACE_OP("tril_");
    return ptsycl::tril_out(self, diagonal, self);
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("aten::gather", &ptsycl::gather);
    m.impl("aten::gather.out", &ptsycl::gather_out);

    m.impl("aten::index_select", &ptsycl::index_select);
    m.impl("aten::index_select.out", &ptsycl::index_select_out);

    m.impl("aten::index_copy", &ptsycl::index_copy);
    m.impl("aten::index_copy_", &ptsycl::index_copy_);

    m.impl("aten::index.Tensor", &ptsycl::index_tensor);
    m.impl("aten::index_put", &ptsycl::index_put);
    m.impl("aten::index_put_", &ptsycl::index_put_);

    m.impl("aten::where.self", &ptsycl::where_self);

    m.impl("aten::triu", &ptsycl::triu);
    m.impl("aten::triu_", &ptsycl::triu_);
    m.impl("aten::triu.out", &ptsycl::triu_out);
    m.impl("aten::tril", &ptsycl::tril);
    m.impl("aten::tril_", &ptsycl::tril_);
    m.impl("aten::tril.out", &ptsycl::tril_out);
}

} // namespace ptsycl
