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

#include <ATen/native/Resize.h>

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

template <typename W>
void copy_index_rows(compat::Queue& q, const char* src, char* dst, const int64_t* idx,
                     int64_t rows, int64_t row_bytes, int64_t dim0, bool gather) {
    const int64_t words = row_bytes / static_cast<int64_t>(sizeof(W));
    const W* s = reinterpret_cast<const W*>(src);
    W* d = reinterpret_cast<W*>(dst);
    launch_flat(q, rows * words, [=](std::size_t flat_) {
        const int64_t flat = static_cast<int64_t>(flat_);
        const int64_t r = flat / words, w = flat % words;
        int64_t j = idx[r];
        if (j < 0) j += dim0;
        if (gather) d[r * words + w] = s[j * words + w];
        else        d[j * words + w] = s[r * words + w];
    });
}

struct alignas(16) Word16 { uint32_t v[4]; };

bool index_rows_applies(const Tensor& self, const PreparedIndex& prep) {
    const auto& info = prep.info;
    return info.num_indexed == 1 && info.is_indexed[0] && info.b_ndim == 1 &&
           self.dim() >= 1 && self.is_contiguous();
}

void index_rows(const Tensor& self, const PreparedIndex& prep, const Tensor& other, bool gather) {
    const int64_t rows = prep.expanded_indices[0].numel();
    const int64_t dim0 = self.size(0);
    const int64_t row_bytes = (dim0 == 0 ? 0 : self.numel() / dim0) * self.element_size();
    if (rows == 0 || row_bytes == 0) return;
    auto& q = queue_for(self);
    const int64_t* pidx = data_ptr<int64_t>(prep.expanded_indices[0]);
    const char* src = static_cast<const char*>(gather ? self.data_ptr() : other.data_ptr());
    char* dst = static_cast<char*>(gather ? other.data_ptr() : self.data_ptr());
    const auto aligned = [&](std::size_t a) {
        return row_bytes % static_cast<int64_t>(a) == 0 &&
               reinterpret_cast<std::uintptr_t>(src) % a == 0 &&
               reinterpret_cast<std::uintptr_t>(dst) % a == 0;
    };
    if (aligned(16))
        copy_index_rows<Word16>(q, src, dst, pidx, rows, row_bytes, dim0, gather);
    else if (aligned(4))
        copy_index_rows<uint32_t>(q, src, dst, pidx, rows, row_bytes, dim0, gather);
    else
        copy_index_rows<uint8_t>(q, src, dst, pidx, rows, row_bytes, dim0, gather);
}

Tensor index_tensor(const Tensor& self,
                    const c10::List<c10::optional<Tensor>>& indices) {
    PTSYCL_TRACE_OP("index.Tensor");
    auto prep = prepare_advanced_indexing(self, indices);
    if (prep.no_op) return self.clone();

    Tensor out = at::empty(prep.out_sizes, self.options());
    const int64_t n = out.numel();
    if (n == 0) return out;
    if (index_rows_applies(self, prep)) {
        index_rows(self, prep, out, true);
        return out;
    }

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

// -----------------------------------------------------------------------------
// index_put_
// -----------------------------------------------------------------------------
// Split into two dispatches rather than one: the plain-store path
// (accumulate=false) never touches atomic_add and keeps the full
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

    const c10::ScalarType st = self.scalar_type();
    if (accumulate && (st == c10::kHalf || st == c10::kBFloat16 ||
                       st == c10::kChar || st == c10::kByte || st == c10::kShort)) {
        const c10::ScalarType acc_t =
            (st == c10::kHalf || st == c10::kBFloat16) ? c10::kFloat : c10::kLong;
        Tensor acc = self.to(acc_t);
        ptsycl::index_put_(acc, indices, values.to(acc_t), true);
        self.copy_(acc);
        return self;
    }

    at::assert_no_internal_overlap(self);

    Tensor v = values.expand(prep.out_sizes).contiguous();
    const int64_t n = v.numel();
    if (n == 0) return self;

    auto& q = queue_for(self);
    const auto info = prep.info;

    if (!accumulate && index_rows_applies(self, prep)) {
        index_rows(self, prep, v, false);
        return self;
    }
    if (!accumulate) {
        AT_DISPATCH_ALL_TYPES_AND3(
            c10::kBool, c10::kHalf, c10::kBFloat16, self.scalar_type(),
            "ptsycl_index_put", [&] {
                scalar_t*       pself = data_ptr<scalar_t>(self);
                const scalar_t* pval  = data_ptr<scalar_t>(v);

                launch_flat(q, n, [=](std::size_t flat_) {
                    const int64_t flat = static_cast<int64_t>(flat_);
                    const int64_t self_off = info.compute_self_offset(flat);
                    pself[self_off] = pval[flat];
                });
            });
        return self;
    }

#define PTSYCL_INDEX_PUT_ATOMIC_SAFE_TYPES(_)                                \
    _(int32_t, c10::kInt)                                                    \
    _(int64_t, c10::kLong)                                                   \
    _(float,   c10::kFloat)                                                  \
    _(double,  c10::kDouble)                                                 \
    _(bool,    c10::kBool)

    switch (self.scalar_type()) {
#define CASE(T, ST)                                                          \
    case ST: {                                                              \
        T*       pself = data_ptr<T>(self);                                 \
        const T* pval  = data_ptr<T>(v);                                    \
        launch_flat(q, n, [=](std::size_t flat_) {                          \
            const int64_t flat = static_cast<int64_t>(flat_);               \
            const int64_t self_off = info.compute_self_offset(flat);        \
            atomic_add(&pself[self_off], pval[flat]);                       \
        });                                                                 \
        break;                                                              \
    }
        PTSYCL_INDEX_PUT_ATOMIC_SAFE_TYPES(CASE)
#undef CASE
        default:
            TORCH_CHECK(false, "paras index_put_ (accumulate=True): unsupported dtype ",
                        self.scalar_type());
    }
#undef PTSYCL_INDEX_PUT_ATOMIC_SAFE_TYPES
    return self;
}

Tensor& index_put_impl_(Tensor& self, const c10::List<c10::optional<Tensor>>& indices,
                        const Tensor& values, bool accumulate, bool ) {
    PTSYCL_TRACE_OP("_index_put_impl_");
    return ptsycl::index_put_(self, indices, values, accumulate);
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
Tensor& where_self_out(const Tensor& condition, const Tensor& self,
                       const Tensor& other, Tensor& out) {
    PTSYCL_TRACE_OP("where.self_out");
    TORCH_CHECK(condition.scalar_type() == c10::kBool,
                "paras where: condition must be a bool tensor");

    const auto out_dtype = at::result_type(self, other);
    TORCH_CHECK(out.scalar_type() == out_dtype,
                "paras where: out dtype ", out.scalar_type(),
                " does not match the result type ", out_dtype);
    auto out_sizes =
        at::infer_size(at::infer_size(condition.sizes(), self.sizes()),
                       other.sizes());
    at::native::resize_output(out, out_sizes);

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

Tensor where_self(const Tensor& condition, const Tensor& self,
                  const Tensor& other) {
    PTSYCL_TRACE_OP("where.self");
    Tensor out = at::empty({0}, self.options().dtype(at::result_type(self, other)));
    return ptsycl::where_self_out(condition, self, other, out);
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

constexpr int64_t kNonzeroChunk = 256;

#if defined(PTSYCL_BACKEND_SYCL) && PARAS_HIP_BACKEND
constexpr int64_t kScanLanes = 64;
constexpr int64_t kScanPerLane = 16;
constexpr int64_t kScanBlock = kScanLanes * kScanPerLane;

template <typename T> class ScanBlockTotals;
template <typename T> class ScanAddOffsets;
template <typename T> class NonzeroCount;
template <typename T> class NonzeroWrite;

void exclusive_scan_device(compat::Queue& q, Tensor& v) {
    const int64_t n = v.numel();
    if (n == 0) return;
    int64_t* pv = data_ptr<int64_t>(v);
    const int64_t blocks = (n + kScanBlock - 1) / kScanBlock;
    Tensor totals = at::empty({blocks}, v.options());
    int64_t* pt = data_ptr<int64_t>(totals);
    q.sycl_queue().template parallel_for<ScanBlockTotals<int64_t>>(
        sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(blocks * kScanLanes)),
                          sycl::range<1>(static_cast<std::size_t>(kScanLanes))),
        [=](sycl::nd_item<1> it) {
            const auto sg = it.get_sub_group();
            const int64_t blk = static_cast<int64_t>(it.get_group(0));
            const int64_t lane = static_cast<int64_t>(it.get_local_id(0));
            const int64_t base = blk * kScanBlock + lane * kScanPerLane;
            int64_t vals[kScanPerLane];
            int64_t c = 0;
#pragma unroll
            for (int e = 0; e < kScanPerLane; ++e) {
                vals[e] = base + e < n ? pv[base + e] : 0;
                c += vals[e];
            }
            const int64_t incl = sycl::inclusive_scan_over_group(sg, c, sycl::plus<int64_t>());
            int64_t run = incl - c;
#pragma unroll
            for (int e = 0; e < kScanPerLane; ++e) {
                if (base + e < n) pv[base + e] = run;
                run += vals[e];
            }
            if (lane == kScanLanes - 1) pt[blk] = incl;
        });
    if (blocks == 1) return;
    exclusive_scan_device(q, totals);
    q.sycl_queue().template parallel_for<ScanAddOffsets<int64_t>>(
        sycl::range<1>(static_cast<std::size_t>(n)), [=](sycl::id<1> i) {
            pv[i[0]] += pt[static_cast<int64_t>(i[0]) / kScanBlock];
        });
}

template <typename scalar_t>
PTSYCL_HOST_DEVICE inline bool is_nonzero(scalar_t x) {
    if constexpr (std::is_same_v<scalar_t, double>) return x != 0.0;
    else return static_cast<float>(x) != 0.0f;
}

void nonzero_gpu(compat::Queue& q, const Tensor& self, Tensor& out) {
    const int64_t n = self.numel();
    const int ndim = static_cast<int>(self.dim());
    const int64_t blocks = (n + kScanBlock - 1) / kScanBlock;
    Tensor offsets = at::empty({blocks + 1}, self.options().dtype(c10::kLong));
    int64_t* poff = data_ptr<int64_t>(offsets);
    const auto flat = make_flat_spec(self);
    const auto spec = make_spec(self);

    AT_DISPATCH_ALL_TYPES_AND3(
        c10::kBool, c10::kHalf, c10::kBFloat16, self.scalar_type(), "ptsycl_nonzero_count", [&] {
            const scalar_t* p = data_ptr<scalar_t>(self);
            q.sycl_queue().template parallel_for<NonzeroCount<scalar_t>>(
                sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(blocks * kScanLanes)),
                                  sycl::range<1>(static_cast<std::size_t>(kScanLanes))),
                [=](sycl::nd_item<1> it) {
                    const auto sg = it.get_sub_group();
                    const int64_t blk = static_cast<int64_t>(it.get_group(0));
                    const int64_t base = blk * kScanBlock +
                                         static_cast<int64_t>(it.get_local_id(0)) * kScanPerLane;
                    int64_t c = 0;
_Pragma("unroll")
                    for (int e = 0; e < kScanPerLane; ++e)
                        if (base + e < n && is_nonzero(p[flat.index(base + e)])) ++c;
                    c = sycl::reduce_over_group(sg, c, sycl::plus<int64_t>());
                    if (it.get_local_id(0) == 0) poff[blk] = c;
                    if (blk == 0 && it.get_local_id(0) == 0) poff[blocks] = 0;
                });
        });
    exclusive_scan_device(q, offsets);
    const int64_t total = offsets[blocks].item<int64_t>();
    out.resize_({total, ndim});
    if (total == 0 || ndim == 0) return;

    Tensor dst = out.is_contiguous() ? out : at::empty({total, ndim}, out.options());
    int64_t* pdst = data_ptr<int64_t>(dst);
    AT_DISPATCH_ALL_TYPES_AND3(
        c10::kBool, c10::kHalf, c10::kBFloat16, self.scalar_type(), "ptsycl_nonzero_write", [&] {
            const scalar_t* p = data_ptr<scalar_t>(self);
            q.sycl_queue().template parallel_for<NonzeroWrite<scalar_t>>(
                sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(blocks * kScanLanes)),
                                  sycl::range<1>(static_cast<std::size_t>(kScanLanes))),
                [=](sycl::nd_item<1> it) {
                    const auto sg = it.get_sub_group();
                    const int64_t blk = static_cast<int64_t>(it.get_group(0));
                    const int64_t base = blk * kScanBlock +
                                         static_cast<int64_t>(it.get_local_id(0)) * kScanPerLane;
                    bool nz[kScanPerLane];
                    int64_t c = 0;
_Pragma("unroll")
                    for (int e = 0; e < kScanPerLane; ++e) {
                        nz[e] = base + e < n && is_nonzero(p[flat.index(base + e)]);
                        c += nz[e] ? 1 : 0;
                    }
                    int64_t row = poff[blk] +
                                  sycl::inclusive_scan_over_group(sg, c, sycl::plus<int64_t>()) - c;
_Pragma("unroll")
                    for (int e = 0; e < kScanPerLane; ++e) {
                        if (!nz[e]) continue;
                        int64_t rem = base + e;
                        for (int dd = ndim - 1; dd >= 0; --dd) {
                            pdst[row * ndim + dd] = rem % spec.sizes[dd];
                            rem /= spec.sizes[dd];
                        }
                        ++row;
                    }
                });
        });
    if (!dst.is_same(out)) out.copy_(dst);
}
#endif

Tensor& nonzero_out(const Tensor& self, Tensor& out) {
    PTSYCL_TRACE_OP("nonzero.out");
    TORCH_CHECK(out.scalar_type() == c10::kLong,
                "paras nonzero: out must be an int64 tensor");
    TORCH_CHECK(spec_supported(self), "paras nonzero: rank exceeds kernel limit");

    const int64_t n = self.numel();
    const int ndim = static_cast<int>(self.dim());
    if (n == 0) {
        out.resize_({0, ndim});
        return out;
    }

    auto& q = queue_for(self);
#if defined(PTSYCL_BACKEND_SYCL) && PARAS_HIP_BACKEND
    if (q.is_gpu()) {
        nonzero_gpu(q, self, out);
        return out;
    }
#endif
    const int64_t chunks = (n + kNonzeroChunk - 1) / kNonzeroChunk;
    Tensor offsets = at::empty({chunks + 1}, self.options().dtype(c10::kLong));
    int64_t* poff = data_ptr<int64_t>(offsets);
    const auto spec = make_spec(self);

    AT_DISPATCH_ALL_TYPES_AND3(
        c10::kBool, c10::kHalf, c10::kBFloat16, self.scalar_type(),
        "ptsycl_nonzero_count", [&] {
            const scalar_t* p = data_ptr<scalar_t>(self);
            launch_flat(q, chunks, [=](std::size_t c_) {
                const int64_t c = static_cast<int64_t>(c_);
                const int64_t begin = c * kNonzeroChunk;
                const int64_t end = begin + kNonzeroChunk < n ? begin + kNonzeroChunk : n;
                int64_t count = 0;
                for (int64_t i = begin; i < end; ++i)
                    if constexpr (std::is_same_v<scalar_t, double>)
                        count += p[spec.index(i)] != 0.0 ? 1 : 0;
                    else
                        count += static_cast<float>(p[spec.index(i)]) != 0.0f ? 1 : 0;
                poff[c + 1] = count;
            });
        });
    launch_flat(q, 1, [=](std::size_t) {
        poff[0] = 0;
        for (int64_t c = 1; c <= chunks; ++c) poff[c] += poff[c - 1];
    });

    const int64_t total = offsets[chunks].item<int64_t>();
    out.resize_({total, ndim});
    if (total == 0 || ndim == 0) return out;

    Tensor dst = out.is_contiguous() ? out : at::empty({total, ndim}, out.options());
    int64_t* pdst = data_ptr<int64_t>(dst);
    AT_DISPATCH_ALL_TYPES_AND3(
        c10::kBool, c10::kHalf, c10::kBFloat16, self.scalar_type(),
        "ptsycl_nonzero_write", [&] {
            const scalar_t* p = data_ptr<scalar_t>(self);
            launch_flat(q, chunks, [=](std::size_t c_) {
                const int64_t c = static_cast<int64_t>(c_);
                const int64_t begin = c * kNonzeroChunk;
                const int64_t end = begin + kNonzeroChunk < n ? begin + kNonzeroChunk : n;
                int64_t row = poff[c];
                for (int64_t i = begin; i < end; ++i) {
                    bool nonzero_elem;
                    if constexpr (std::is_same_v<scalar_t, double>)
                        nonzero_elem = p[spec.index(i)] != 0.0;
                    else
                        nonzero_elem = static_cast<float>(p[spec.index(i)]) != 0.0f;
                    if (!nonzero_elem) continue;
                    int64_t rem = i;
                    for (int dd = ndim - 1; dd >= 0; --dd) {
                        pdst[row * ndim + dd] = rem % spec.sizes[dd];
                        rem /= spec.sizes[dd];
                    }
                    ++row;
                }
            });
        });
    if (!dst.is_same(out)) out.copy_(dst);
    return out;
}

Tensor nonzero(const Tensor& self) {
    PTSYCL_TRACE_OP("nonzero");
    Tensor out = at::empty({0, self.dim()}, self.options().dtype(c10::kLong));
    return ptsycl::nonzero_out(self, out);
}

Tensor repeat_interleave_counts(const Tensor& repeats,
                                std::optional<c10::SymInt> output_size) {
    PTSYCL_TRACE_OP("repeat_interleave.Tensor");
    TORCH_CHECK(repeats.dim() == 1, "repeat_interleave: repeats must be 1-D");
    TORCH_CHECK(repeats.scalar_type() == c10::kLong || repeats.scalar_type() == c10::kInt,
                "repeat_interleave: repeats must be an int32 or int64 tensor");
    Tensor rep = to_long(repeats).contiguous();
    const int64_t n = rep.numel();
    if (n == 0) return at::empty({0}, rep.options());
    TORCH_CHECK(rep.min().item<int64_t>() >= 0, "repeat_interleave: repeats can not be negative");

    Tensor ends = rep.cumsum(0);
    const int64_t total = output_size.has_value()
        ? output_size->expect_int()
        : ends[n - 1].item<int64_t>();
    Tensor out = at::empty({total}, rep.options());
    if (total == 0) return out;

    auto& q = queue_for(rep);
    const int64_t* prep = data_ptr<int64_t>(rep);
    const int64_t* pend = data_ptr<int64_t>(ends);
    int64_t*       pout = data_ptr<int64_t>(out);
    launch_flat(q, n, [=](std::size_t i_) {
        const int64_t i = static_cast<int64_t>(i_);
        const int64_t end = pend[i] < total ? pend[i] : total;
        for (int64_t j = pend[i] - prep[i]; j < end; ++j) pout[j] = i;
    });
    return out;
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
    m.impl("aten::_index_put_impl_", &ptsycl::index_put_impl_);

    m.impl("aten::where.self", &ptsycl::where_self);
    m.impl("aten::where.self_out", &ptsycl::where_self_out);

    m.impl("aten::triu", &ptsycl::triu);
    m.impl("aten::triu_", &ptsycl::triu_);
    m.impl("aten::triu.out", &ptsycl::triu_out);
    m.impl("aten::tril", &ptsycl::tril);
    m.impl("aten::tril_", &ptsycl::tril_);
    m.impl("aten::tril.out", &ptsycl::tril_out);

    m.impl("aten::nonzero", &ptsycl::nonzero);
    m.impl("aten::nonzero.out", &ptsycl::nonzero_out);

    m.impl("aten::repeat_interleave.Tensor", &ptsycl::repeat_interleave_counts);
}

} // namespace ptsycl