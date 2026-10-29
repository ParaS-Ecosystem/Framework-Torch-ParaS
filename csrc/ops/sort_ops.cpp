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
// So this file splits by flavor:
//   - PTSYCL_BACKEND_CPU: launch_flat runs one thread per *row* (not per
//     element), and each thread does a std::sort with a heap-allocated
//     buffer -- plain host C++ under the kem threadpool.
//   - PTSYCL_BACKEND_SYCL (GPU): a batched bitonic sort. Rows are padded to
//     a power of two and every compare-exchange stage is one launch_flat over
//     all padded elements, so there is no per-row length limit and no
//     thread-local storage. O(n log^2 n) work.
// Both flavours share one total order (exact keys, NaN largest, ties by
// position), so results are identical across devices and always stable.
// -----------------------------------------------------------------------------

#include "core/kernels.h"

#include <algorithm>
#include <numeric>
#include <type_traits>
#include <vector>


namespace ptsycl {
namespace {

using at::Tensor;

#if !defined(PTSYCL_BACKEND_CPU)
constexpr int64_t kRankSortMaxLen = 2048;
constexpr int64_t kBitonicLocalBlock = 1024;
template <typename T> class BitonicLocalStages;
#endif

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

#define PTSYCL_ARGSORT_BEFORE(KEY_OF)                                         \
    [=](int64_t a, int64_t b) {                                               \
        if (a < 0) return false;                                              \
        if (b < 0) return true;                                               \
        const auto ka = KEY_OF(a);                                            \
        const auto kb = KEY_OF(b);                                            \
        const bool a_nan = ka != ka;                                          \
        const bool b_nan = kb != kb;                                          \
        if (a_nan || b_nan) {                                                 \
            if (a_nan && b_nan) return a < b;                                 \
            return descending ? a_nan : b_nan;                                \
        }                                                                     \
        if (ka != kb) return descending ? ka > kb : ka < kb;                  \
        return a < b;                                                         \
    }

    AT_DISPATCH_ALL_TYPES_AND2(
        c10::kHalf, c10::kBFloat16, self.scalar_type(), "ptsycl_argsort", [&] {
            using key_t = std::conditional_t<std::is_integral_v<scalar_t>, int64_t, double>;
            const scalar_t* pin = data_ptr<scalar_t>(self);
            int64_t*        pout = data_ptr<int64_t>(out);

#if defined(PTSYCL_BACKEND_CPU)
            (void)stable;
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

                auto key_of = [=](int64_t e) {
                    if constexpr (std::is_same_v<scalar_t, double>)
                        return base[e * dim_stride];
                    else if constexpr (std::is_integral_v<scalar_t>)
                        return static_cast<key_t>(base[e * dim_stride]);
                    else
                        return static_cast<key_t>(static_cast<float>(base[e * dim_stride]));
                };
                auto before = PTSYCL_ARGSORT_BEFORE(key_of);

                std::vector<int64_t> idx(dim_size);
                std::iota(idx.begin(), idx.end(), int64_t{0});
                std::sort(idx.begin(), idx.end(), before);

                for (int64_t e = 0; e < dim_size; ++e)
                    pout[out_base + e * out_dim_stride] = idx[e];
            });
#else
            (void)stable;
            if (dim_size <= kRankSortMaxLen) {
                launch_flat(q, layout.num_rows * dim_size, [=](std::size_t t_) {
                    const int64_t t = static_cast<int64_t>(t_);
                    const int64_t row = t / dim_size;
                    const int64_t e = t % dim_size;
                    int64_t rem = row;
                    int64_t in_base = 0;
                    int64_t out_base = 0;
                    for (int dd = self_spec.ndim - 1; dd >= 0; --dd) {
                        if (dd == self_spec.skip_dim) continue;
                        const int64_t c = rem % self_spec.sizes[dd];
                        rem /= self_spec.sizes[dd];
                        in_base += c * self_spec.strides[dd];
                        out_base += c * out_spec.strides[dd];
                    }
                    const scalar_t* base = pin + in_base;
                    auto key_of = [=](int64_t i) {
                        if constexpr (std::is_same_v<scalar_t, double>)
                            return base[i * dim_stride];
                        else if constexpr (std::is_integral_v<scalar_t>)
                            return static_cast<key_t>(base[i * dim_stride]);
                        else
                            return static_cast<key_t>(static_cast<float>(base[i * dim_stride]));
                    };
                    auto before = PTSYCL_ARGSORT_BEFORE(key_of);
                    int64_t rank = 0;
                    for (int64_t j = 0; j < dim_size; ++j) rank += before(j, e) ? 1 : 0;
                    pout[out_base + rank * out_dim_stride] = e;
                });
            } else {
                int64_t P = 1;
                while (P < dim_size) P <<= 1;
                const int64_t rows = layout.num_rows;
                Tensor work = at::empty({rows, P}, self.options().dtype(c10::kLong));
                int64_t* pw = data_ptr<int64_t>(work);

                launch_flat(q, rows * P, [=](std::size_t t_) {
                    const int64_t t = static_cast<int64_t>(t_);
                    const int64_t e = t % P;
                    pw[t] = e < dim_size ? e : -1;
                });

                const bool local_ok = q.is_gpu();
                const int64_t BLK = local_ok ? std::min<int64_t>(P, kBitonicLocalBlock) : 1;
                auto local_stages = [&](int64_t kmin, int64_t kmax) {
                    const int64_t half = BLK / 2;
                    const int64_t groups = rows * (P / BLK);
                    q.sycl_queue().submit([&](sycl::handler& cgh) {
                        sycl::local_accessor<int64_t, 1> sh(sycl::range<1>(static_cast<std::size_t>(BLK)), cgh);
                        cgh.template parallel_for<BitonicLocalStages<scalar_t>>(
                            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(groups * half)),
                                              sycl::range<1>(static_cast<std::size_t>(half))),
                            [=](sycl::nd_item<1> it) {
                                const int64_t g = static_cast<int64_t>(it.get_group(0));
                                const int64_t row = g / (P / BLK);
                                const int64_t base_pos = (g % (P / BLK)) * BLK;
                                const int64_t tid = static_cast<int64_t>(it.get_local_id(0));
                                int64_t rem = row;
                                int64_t in_base = 0;
                                for (int dd = self_spec.ndim - 1; dd >= 0; --dd) {
                                    if (dd == self_spec.skip_dim) continue;
                                    const int64_t c = rem % self_spec.sizes[dd];
                                    rem /= self_spec.sizes[dd];
                                    in_base += c * self_spec.strides[dd];
                                }
                                const scalar_t* base = pin + in_base;
                                auto key_of = [=](int64_t e) {
                                    if constexpr (std::is_same_v<scalar_t, double>)
                                        return base[e * dim_stride];
                                    else if constexpr (std::is_integral_v<scalar_t>)
                                        return static_cast<key_t>(base[e * dim_stride]);
                                    else
                                        return static_cast<key_t>(static_cast<float>(base[e * dim_stride]));
                                };
                                auto before = PTSYCL_ARGSORT_BEFORE(key_of);
                                int64_t* w = pw + row * P + base_pos;
                                sh[tid] = w[tid];
                                sh[tid + half] = w[tid + half];
                                sycl::group_barrier(it.get_group());
                                for (int64_t k = kmin; k <= kmax; k <<= 1) {
                                    for (int64_t j = (k >> 1) < half ? (k >> 1) : half; j > 0; j >>= 1) {
                                        const int64_t i = 2 * j * (tid / j) + (tid % j);
                                        const int64_t l = i + j;
                                        const int64_t a = sh[i];
                                        const int64_t b = sh[l];
                                        const bool ascending_block = ((base_pos + i) & k) == 0;
                                        if (ascending_block ? before(b, a) : before(a, b)) {
                                            sh[i] = b;
                                            sh[l] = a;
                                        }
                                        sycl::group_barrier(it.get_group());
                                    }
                                }
                                w[tid] = sh[tid];
                                w[tid + half] = sh[tid + half];
                            });
                    });
                };
                if (BLK > 1) local_stages(2, BLK);
                for (int64_t k = BLK > 1 ? 2 * BLK : 2; k <= P; k <<= 1) {
                    for (int64_t j = k >> 1; j >= BLK && j > 0; j >>= 1) {
                        launch_flat(q, rows * P, [=](std::size_t t_) {
                            const int64_t t = static_cast<int64_t>(t_);
                            const int64_t i = t % P;
                            const int64_t l = i ^ j;
                            if (l <= i) return;
                            const int64_t row = t / P;

                            int64_t rem = row;
                            int64_t in_base = 0;
                            for (int dd = self_spec.ndim - 1; dd >= 0; --dd) {
                                if (dd == self_spec.skip_dim) continue;
                                const int64_t c = rem % self_spec.sizes[dd];
                                rem /= self_spec.sizes[dd];
                                in_base += c * self_spec.strides[dd];
                            }
                            const scalar_t* base = pin + in_base;
                            auto key_of = [=](int64_t e) {
                                if constexpr (std::is_same_v<scalar_t, double>)
                                    return base[e * dim_stride];
                                else if constexpr (std::is_integral_v<scalar_t>)
                                    return static_cast<key_t>(base[e * dim_stride]);
                                else
                                    return static_cast<key_t>(static_cast<float>(base[e * dim_stride]));
                            };
                            auto before = PTSYCL_ARGSORT_BEFORE(key_of);

                            int64_t* w = pw + row * P;
                            const int64_t a = w[i];
                            const int64_t b = w[l];
                            const bool ascending_block = (i & k) == 0;
                            if (ascending_block ? before(b, a) : before(a, b)) {
                                w[i] = b;
                                w[l] = a;
                            }
                        });
                    }
                    if (BLK > 1) local_stages(k, k);
                }

                launch_flat(q, rows * dim_size, [=](std::size_t t_) {
                    const int64_t t = static_cast<int64_t>(t_);
                    const int64_t row = t / dim_size;
                    const int64_t e = t % dim_size;
                    int64_t rem = row;
                    int64_t out_base = 0;
                    for (int dd = out_spec.ndim - 1; dd >= 0; --dd) {
                        if (dd == out_spec.skip_dim) continue;
                        const int64_t c = rem % out_spec.sizes[dd];
                        rem /= out_spec.sizes[dd];
                        out_base += c * out_spec.strides[dd];
                    }
                    pout[out_base + e * out_dim_stride] = pw[row * P + e];
                });
            }
#endif
        });
#undef PTSYCL_ARGSORT_BEFORE
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
