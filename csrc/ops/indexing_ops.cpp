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
// Advanced indexing (index.Tensor / index_put_)
// -----------------------------------------------------------------------------
struct AdvancedIndexInfo {
    int self_ndim = 0;
    int64_t self_sizes[kMaxDims]{};
    int64_t self_strides[kMaxDims]{};

    int out_ndim = 0;
    int64_t out_sizes[kMaxDims]{};

    int b_ndim = 0;
    int64_t b_sizes[kMaxDims]{};
    int64_t b_strides[kMaxDims]{};

    bool is_indexed[kMaxDims]{};
    int idx_map[kMaxDims]{};
    int unindexed_out_dim[kMaxDims]{};

    bool is_bdim[kMaxDims]{};
    int b_dim_idx[kMaxDims]{};

    int num_indexed = 0;
    const int64_t* pidx[kMaxDims]{};

    PTSYCL_HOST_DEVICE inline int64_t compute_self_offset(int64_t flat) const {
        int64_t rem = flat;
        int64_t out_coord[kMaxDims];
        for (int d = out_ndim - 1; d >= 0; --d) {
            out_coord[d] = rem % out_sizes[d];
            rem /= out_sizes[d];
        }

        int64_t b_offset = 0;
        for (int o = 0; o < out_ndim; ++o) {
            if (is_bdim[o]) {
                int b_d = b_dim_idx[o];
                b_offset += out_coord[o] * b_strides[b_d];
            }
        }

        int64_t self_offset = 0;
        for (int d = 0; d < self_ndim; ++d) {
            int64_t coord_d;
            if (is_indexed[d]) {
                int idx_i = idx_map[d];
                int64_t raw_val = pidx[idx_i][b_offset];
                if (raw_val < 0) raw_val += self_sizes[d];
                coord_d = raw_val;
            } else {
                int out_d = unindexed_out_dim[d];
                coord_d = out_coord[out_d];
            }
            self_offset += coord_d * self_strides[d];
        }
        return self_offset;
    }
};

std::vector<int64_t> broadcast_shape(const std::vector<Tensor>& ts) {
    if (ts.empty()) return {};
    std::vector<int64_t> shape = ts[0].sizes().vec();
    for (std::size_t i = 1; i < ts.size(); ++i)
        shape = at::infer_size(shape, ts[i].sizes());
    return shape;
}

struct PreparedIndex {
    AdvancedIndexInfo info;
    std::vector<Tensor> expanded_indices;
    std::vector<int64_t> out_sizes;
    bool no_op = false;
};

PreparedIndex prepare_advanced_indexing(
    const Tensor& self, const c10::List<c10::optional<Tensor>>& orig_indices) {
    TORCH_CHECK(spec_supported(self), "paras index: rank exceeds kernel limit");

    std::vector<c10::optional<Tensor>> indices;
    for (std::size_t i = 0; i < orig_indices.size(); ++i) {
        c10::optional<Tensor> oi = orig_indices[i];
        if (oi.has_value() && oi->defined() &&
            (oi->scalar_type() == c10::kBool || oi->scalar_type() == c10::kByte)) {
            auto nonzeros = oi->nonzero().unbind(1);
            for (auto& nz : nonzeros) {
                indices.push_back(nz);
            }
        } else {
            indices.push_back(oi);
        }
    }

    TORCH_CHECK(static_cast<int64_t>(indices.size()) <= self.dim(),
                "paras index: too many indices for tensor");

    const int self_dim = static_cast<int>(self.dim());
    std::vector<int> indexed_dims;
    std::vector<Tensor> raw_indices;

    for (int d = 0; d < self_dim; ++d) {
        if (d < static_cast<int>(indices.size())) {
            c10::optional<Tensor> oi = indices[d];
            if (oi.has_value() && oi->defined()) {
                indexed_dims.push_back(d);
                raw_indices.push_back(to_long(*oi));
            }
        }
    }

    PreparedIndex res;
    if (indexed_dims.empty()) {
        res.no_op = true;
        res.out_sizes = self.sizes().vec();
        return res;
    }

    const int num_indexed = static_cast<int>(indexed_dims.size());
    std::vector<int64_t> bshape = broadcast_shape(raw_indices);
    const int b_dim = static_cast<int>(bshape.size());

    for (int i = 0; i < num_indexed; ++i) {
        res.expanded_indices.push_back(raw_indices[i].expand(bshape).contiguous());
    }

    const int first_idx = indexed_dims.front();
    const int last_idx = indexed_dims.back();
    const bool is_contiguous = (last_idx - first_idx + 1 == num_indexed);

    std::vector<int64_t> out_sizes;
    AdvancedIndexInfo& info = res.info;
    info.self_ndim = self_dim;
    for (int d = 0; d < self_dim; ++d) {
        info.self_sizes[d] = self.size(d);
        info.self_strides[d] = self.stride(d);
        info.is_indexed[d] = false;
        info.idx_map[d] = -1;
        info.unindexed_out_dim[d] = -1;
    }

    for (int i = 0; i < num_indexed; ++i) {
        int d = indexed_dims[i];
        info.is_indexed[d] = true;
        info.idx_map[d] = i;
    }

    info.b_ndim = b_dim;
    for (int d = 0; d < b_dim; ++d) {
        info.b_sizes[d] = bshape[d];
    }
    if (b_dim > 0) {
        info.b_strides[b_dim - 1] = 1;
        for (int d = b_dim - 2; d >= 0; --d) {
            info.b_strides[d] = info.b_strides[d + 1] * info.b_sizes[d + 1];
        }
    }

    if (is_contiguous) {
        for (int d = 0; d < first_idx; ++d) {
            info.unindexed_out_dim[d] = static_cast<int>(out_sizes.size());
            out_sizes.push_back(self.size(d));
        }
        int b_start_out_dim = static_cast<int>(out_sizes.size());
        for (int d = 0; d < b_dim; ++d) {
            out_sizes.push_back(bshape[d]);
        }
        for (int d = last_idx + 1; d < self_dim; ++d) {
            info.unindexed_out_dim[d] = static_cast<int>(out_sizes.size());
            out_sizes.push_back(self.size(d));
        }
        info.out_ndim = static_cast<int>(out_sizes.size());
        for (int o = 0; o < info.out_ndim; ++o) {
            info.out_sizes[o] = out_sizes[o];
            info.is_bdim[o] = (o >= b_start_out_dim && o < b_start_out_dim + b_dim);
            info.b_dim_idx[o] = info.is_bdim[o] ? (o - b_start_out_dim) : -1;
        }
    } else {
        for (int d = 0; d < b_dim; ++d) {
            out_sizes.push_back(bshape[d]);
        }
        for (int d = 0; d < self_dim; ++d) {
            if (!info.is_indexed[d]) {
                info.unindexed_out_dim[d] = static_cast<int>(out_sizes.size());
                out_sizes.push_back(self.size(d));
            }
        }
        info.out_ndim = static_cast<int>(out_sizes.size());
        for (int o = 0; o < info.out_ndim; ++o) {
            info.out_sizes[o] = out_sizes[o];
            info.is_bdim[o] = (o < b_dim);
            info.b_dim_idx[o] = (o < b_dim) ? o : -1;
        }
    }

    TORCH_CHECK(info.out_ndim <= kMaxDims,
                "paras index: result rank exceeds kernel limit");

    info.num_indexed = num_indexed;
    for (int i = 0; i < num_indexed; ++i) {
        info.pidx[i] = data_ptr<int64_t>(res.expanded_indices[i]);
    }
    res.out_sizes = out_sizes;
    return res;
}

Tensor index_tensor(const Tensor& self,
                    const c10::List<c10::optional<Tensor>>& indices) {
    PTSYCL_TRACE_OP("index.Tensor");
    auto prep = prepare_advanced_indexing(self, indices);
    if (prep.no_op) return self.clone();

    Tensor out = at::empty(prep.out_sizes, self.options());
    const int64_t n = out.numel();
    if (n == 0) return out;

    auto& q = queue_for(self);
    const auto info = prep.info;

    AT_DISPATCH_ALL_TYPES_AND3(
        c10::kBool, c10::kHalf, c10::kBFloat16, self.scalar_type(),
        "ptsycl_index", [&] {
            const scalar_t* pself = data_ptr<scalar_t>(self);
            scalar_t*       pout  = data_ptr<scalar_t>(out);

            launch_flat(q, n, [=](std::size_t flat_) {
                const int64_t flat = static_cast<int64_t>(flat_);
                const int64_t self_off = info.compute_self_offset(flat);
                pout[flat] = pself[self_off];
            });
        });
    return out;
}

Tensor& index_put_(Tensor& self, const c10::List<c10::optional<Tensor>>& indices,
                   const Tensor& values, bool accumulate) {
    PTSYCL_TRACE_OP("index_put_");
    auto prep = prepare_advanced_indexing(self, indices);
    if (prep.no_op) {
        if (accumulate) {
            self.add_(values);
        } else {
            self.copy_(values);
        }
        return self;
    }

    Tensor v = values.expand(prep.out_sizes).contiguous();
    const int64_t n = v.numel();
    if (n == 0) return self;

    auto& q = queue_for(self);
    const auto info = prep.info;

    AT_DISPATCH_ALL_TYPES_AND3(
        c10::kBool, c10::kHalf, c10::kBFloat16, self.scalar_type(),
        "ptsycl_index_put", [&] {
            scalar_t*       pself = data_ptr<scalar_t>(self);
            const scalar_t* pval  = data_ptr<scalar_t>(v);

            launch_flat(q, n, [=](std::size_t flat_) {
                const int64_t flat = static_cast<int64_t>(flat_);
                const int64_t self_off = info.compute_self_offset(flat);
                if (accumulate) {
                    atomic_add(&pself[self_off], pval[flat]);
                } else {
                    pself[self_off] = pval[flat];
                }
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