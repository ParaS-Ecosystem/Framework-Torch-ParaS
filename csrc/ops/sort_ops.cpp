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
//
// A NOTE ON WHY THIS FILE LOOKS DIFFERENT FROM THE OTHERS
// ----------------------------------------------------------------------------
// Every other kernel in this tree (gather, index_select, where, triu/tril,
// mm/bmm) does one independent unit of work per output element -- exactly
// what launch_flat(q, n, ...) is for. Sorting a row means one thread has
// to reason about the *entire row* relative to itself, which is a
// different shape of parallelism (bitonic/radix network across threads
// in a workgroup, or a library sort), not something safely improvised
// against launch_flat without knowing whether this queue type supports
// shared/local memory or cross-thread sync within a launch.
//
// So this file deliberately splits by flavor:
//   - PTSYCL_BACKEND_CPU: launch_flat still runs one thread per *row*
//     (not per element), and each thread does a real std::stable_sort /
//     std::sort with a heap-allocated buffer -- perfectly fine, this is
//     plain host C++ under the kem threadpool, same as any OpenMP kernel.
//   - PTSYCL_BACKEND_CUDA / PTSYCL_BACKEND_HIP: heap allocation and
//     std::sort inside device code are not something I can assume this
//     toolchain supports. What's below is a bounded insertion sort over
//     a fixed-size *local* array (no heap, no exceptions), capped by
//     PTSYCL_ARGSORT_MAX_LEN and enforced with a TORCH_CHECK. This is a
//     correctness stopgap, O(len^2) per row and one row per thread, not
//     a real GPU sort -- if `self.size(dim)` exceeds the cap, or if rows
//     are large enough that per-row O(len^2) actually matters for your
//     workloads, this needs a proper parallel sort (thrust::sort_by_key
//     if CUDA thrust is linkable here, or a bitonic sort using whatever
//     shared-memory primitive ParaS exposes) before it's production-
//     ready. Grep for an existing aten::sort registration first, too --
//     if one exists, it may already have solved this properly and
//     argsort should be wired to call it (via at::sort) instead of
//     duplicating a sort implementation here.
// -----------------------------------------------------------------------------

#include "core/kernels.h"

#include <algorithm>
#include <numeric>
#include <vector>

#ifndef PTSYCL_ARGSORT_MAX_LEN
#define PTSYCL_ARGSORT_MAX_LEN 4096
#endif

namespace ptsycl {
namespace {

using at::Tensor;

// Row layout shared by both flavors: given `self` and a (already
// wrapped) `dim`, a "row" is one 1-D slice along `dim`; there are
// self.numel() / self.size(dim) rows. Row `r`'s element `e` (0-based
// along dim) lives at self_spec base offset (computed by decomposing
// `r` over every dim except `dim`) plus `e * self.stride(dim)`.
struct RowLayout {
    int64_t dim_size;
    int64_t dim_stride;
    int64_t num_rows;
};

RowLayout make_row_layout(const Tensor& self, int64_t dim) {
    return RowLayout{self.size(dim), self.stride(dim),
                      self.numel() / std::max<int64_t>(self.size(dim), 1)};
}

// Decompose row index `r` into a base element offset over all dims
// except `dim`, using the same flat-index-over-sizes technique as
// gather/index_select in indexing_ops.cpp. NOTE: deliberately inlined
// into each lambda below rather than kept as a standalone free function
// -- a free function called from inside a launch_flat lambda isn't
// automatically device-compiled under -parasdevice cuda:sm_70 (it's a
// plain __host__ function unless annotated), which is exactly what
// indexing_ops.cpp avoids by writing this loop inline in every kernel
// there too. This struct is still useful as a value captured by the
// lambda (POD, cheap to copy); only the offset *computation* needs to
// live inside the device-compiled lambda body.
struct RowSpec {
    int ndim;
    int64_t sizes[16];
    int64_t strides[16];
    int64_t skip_dim;
};

RowSpec make_row_spec(const Tensor& t, int64_t dim) {
    RowSpec s{};
    s.ndim = static_cast<int>(t.dim());
    TORCH_CHECK(s.ndim <= 16, "paras argsort: rank exceeds kernel limit");
    for (int i = 0; i < s.ndim; ++i) {
        s.sizes[i] = t.size(i);
        s.strides[i] = t.stride(i);
    }
    s.skip_dim = dim;
    return s;
}

Tensor argsort_stable(const Tensor& self, bool stable, int64_t dim, bool descending) {
    PTSYCL_TRACE_OP("argsort.stable");
    TORCH_CHECK(self.dim() >= 1, "paras argsort: input must have >= 1 dim");
    const int64_t d = c10::maybe_wrap_dim(dim, self.dim());

    Tensor out = at::empty(self.sizes(), self.options().dtype(c10::kLong));
    const auto layout = make_row_layout(self, d);
    if (layout.num_rows == 0 || layout.dim_size == 0) return out;

    const auto self_spec = make_row_spec(self, d);
    const auto out_spec  = make_row_spec(out, d);
    auto& q = queue_for(self);
    const int64_t dim_size   = layout.dim_size;
    const int64_t dim_stride = layout.dim_stride;
    const int64_t out_dim_stride = out.stride(d);

#if defined(PTSYCL_BACKEND_CPU)
    AT_DISPATCH_ALL_TYPES_AND2(
        c10::kHalf, c10::kBFloat16, self.scalar_type(), "ptsycl_argsort", [&] {
            const scalar_t* pin = data_ptr<scalar_t>(self);
            int64_t*        pout = data_ptr<int64_t>(out);

            launch_flat(q, layout.num_rows, [=](std::size_t row_) {
                const int64_t row = static_cast<int64_t>(row_);

                auto base_offset = [](const RowSpec& s, int64_t r) {
                    int64_t rem = r;
                    int64_t off = 0;
                    for (int dd = s.ndim - 1; dd >= 0; --dd) {
                        if (dd == s.skip_dim) continue;
                        const int64_t c = rem % s.sizes[dd];
                        rem /= s.sizes[dd];
                        off += c * s.strides[dd];
                    }
                    return off;
                };
                const int64_t in_base  = base_offset(self_spec, row);
                const int64_t out_base = base_offset(out_spec, row);

                std::vector<int64_t> idx(dim_size);
                std::iota(idx.begin(), idx.end(), int64_t{0});
                const scalar_t* base = pin + in_base;
                auto cmp = [&](int64_t a, int64_t b) {
                    const float va = static_cast<float>(base[a * dim_stride]);
                    const float vb = static_cast<float>(base[b * dim_stride]);
                    return descending ? va > vb : va < vb;
                };
                if (stable) std::stable_sort(idx.begin(), idx.end(), cmp);
                else        std::sort(idx.begin(), idx.end(), cmp);

                for (int64_t e = 0; e < dim_size; ++e)
                    pout[out_base + e * out_dim_stride] = idx[e];
            });
        });
#else
    TORCH_CHECK(dim_size <= PTSYCL_ARGSORT_MAX_LEN,
                "paras argsort: dim size ", dim_size, " exceeds the CUDA/HIP "
                "reference kernel's fixed-buffer cap of ", PTSYCL_ARGSORT_MAX_LEN,
                " (see the note at the top of sort_ops.cpp -- this flavor needs "
                "a real parallel sort for rows this large)");

    AT_DISPATCH_ALL_TYPES_AND2(
        c10::kHalf, c10::kBFloat16, self.scalar_type(), "ptsycl_argsort", [&] {
            const scalar_t* pin = data_ptr<scalar_t>(self);
            int64_t*        pout = data_ptr<int64_t>(out);

            launch_flat(q, layout.num_rows, [=](std::size_t row_) {
                const int64_t row = static_cast<int64_t>(row_);

                auto base_offset = [](const RowSpec& s, int64_t r) {
                    int64_t rem = r;
                    int64_t off = 0;
                    for (int dd = s.ndim - 1; dd >= 0; --dd) {
                        if (dd == s.skip_dim) continue;
                        const int64_t c = rem % s.sizes[dd];
                        rem /= s.sizes[dd];
                        off += c * s.strides[dd];
                    }
                    return off;
                };
                const int64_t in_base  = base_offset(self_spec, row);
                const int64_t out_base = base_offset(out_spec, row);
                const scalar_t* base = pin + in_base;

                // Fixed-size, no-heap insertion sort -- O(dim_size^2),
                // correctness stopgap only (see file header). Comparisons
                // go through float rather than raw scalar_t: c10::Half
                // has an implicit conversion to CUDA's __half, and
                // __half also defines operator</operator> (cuda_fp16.hpp)
                // -- comparing two c10::Half values directly is genuinely
                // ambiguous to the compiler under nvcc/clang-cuda (two
                // equally-good overload resolution paths), not just a
                // style nit. Casting sidesteps the ambiguity and matches
                // the float-accumulation choice already made in
                // matmul_ops.cpp for the same underlying reason.
                int64_t local_idx[PTSYCL_ARGSORT_MAX_LEN];
                for (int64_t e = 0; e < dim_size; ++e) local_idx[e] = e;
                for (int64_t i = 1; i < dim_size; ++i) {
                    const int64_t key = local_idx[i];
                    const float key_val = static_cast<float>(base[key * dim_stride]);
                    int64_t j = i - 1;
                    while (j >= 0) {
                        const float vj = static_cast<float>(base[local_idx[j] * dim_stride]);
                        const bool should_shift =
                            descending ? vj < key_val : vj > key_val;
                        if (!should_shift) break;
                        local_idx[j + 1] = local_idx[j];
                        --j;
                    }
                    local_idx[j + 1] = key;
                }
                for (int64_t e = 0; e < dim_size; ++e)
                    pout[out_base + e * out_dim_stride] = local_idx[e];
            });
        });
#endif
    return out;
}

Tensor argsort(const Tensor& self, int64_t dim, bool descending) {
    PTSYCL_TRACE_OP("argsort");
    return argsort_stable(self, /*stable=*/false, dim, descending);
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("aten::argsort", &ptsycl::argsort);
    m.impl("aten::argsort.stable", &ptsycl::argsort_stable);
}

} // namespace ptsycl
