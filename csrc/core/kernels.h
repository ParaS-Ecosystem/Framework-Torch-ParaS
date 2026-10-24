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


#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <c10/util/Exception.h>

#if defined(PTSYCL_BACKEND_SYCL)
#include <sycl/sycl.hpp>
#endif

#include "core/common.h"

namespace ptsycl {

namespace compat {
template <typename F>
class ParasKernelKey;

template <typename F>
inline void parallel_for(Queue& queue, std::size_t n, F f) {
    if (n == 0) return;
#if defined(PTSYCL_BACKEND_SYCL)
    if (queue.is_gpu()) {
        queue.sycl_queue().template parallel_for<ParasKernelKey<F>>(
            sycl::range<1>(n),
            [f](sycl::id<1> i) { f(static_cast<std::size_t>(i[0])); });
        return;
    }
#endif
    host_parallel_for(n, f);
}
} // namespace compat

constexpr int kMaxDims = 8;

struct StridedSpec {
    int     ndim = 0;
    int64_t sizes[kMaxDims]{};
    int64_t strides[kMaxDims]{}; // in elements
    int64_t offset = 0;          // in elements
    bool    identity = false;
    bool    narrow = false;

    // Maps a flat logical index (row-major over sizes) to a storage offset.
    PTSYCL_HOST_DEVICE inline int64_t index(int64_t flat) const {
        if (identity) return offset + flat;
        int64_t off = offset;
        if (narrow) {
            uint32_t f = static_cast<uint32_t>(flat);
            for (int d = ndim - 1; d >= 0; --d) {
                const uint32_t sz = static_cast<uint32_t>(sizes[d]);
                off += static_cast<int64_t>(f % sz) * strides[d];
                f /= sz;
            }
            return off;
        }
        for (int d = ndim - 1; d >= 0; --d) {
            const int64_t c = flat % sizes[d];
            flat /= sizes[d];
            off += c * strides[d];
        }
        return off;
    }
};

inline StridedSpec flatten_spec(const StridedSpec& in) {
    StridedSpec s;
    s.offset = in.offset;
    int64_t numel = 1;
    for (int d = 0; d < in.ndim; ++d) {
        numel *= in.sizes[d];
        if (in.sizes[d] == 1) continue;
        if (s.ndim > 0 &&
            s.strides[s.ndim - 1] == in.strides[d] * in.sizes[d]) {
            s.sizes[s.ndim - 1] *= in.sizes[d];
            s.strides[s.ndim - 1] = in.strides[d];
        } else {
            s.sizes[s.ndim] = in.sizes[d];
            s.strides[s.ndim] = in.strides[d];
            ++s.ndim;
        }
    }
    s.identity = s.ndim == 0 || (s.ndim == 1 && s.strides[0] == 1);
    s.narrow = numel <= int64_t{0x7fffffff};
    return s;
}

inline bool spec_supported(const at::Tensor& t) { return t.dim() <= kMaxDims; }

inline StridedSpec make_spec(const at::Tensor& t) {
    TORCH_CHECK(t.dim() <= kMaxDims, "paras: tensor rank ", t.dim(),
                " exceeds kernel limit of ", kMaxDims);
    StridedSpec s;
    s.ndim   = static_cast<int>(t.dim());
    s.offset = 0; // data_ptr() already includes the storage offset
    for (int d = 0; d < s.ndim; ++d) {
        s.sizes[d]   = t.size(d);
        s.strides[d] = t.stride(d);
    }
    return s;
}

inline StridedSpec make_flat_spec(const at::Tensor& t) {
    return flatten_spec(make_spec(t));
}

// -----------------------------------------------------------------------------
// Elementwise helpers
// -----------------------------------------------------------------------------

template <typename T>
PTSYCL_HOST_DEVICE inline void atomic_add(T* address, T val) {
#if (defined(PTSYCL_BACKEND_CUDA) || defined(PTSYCL_BACKEND_HIP) || defined(PTSYCL_BACKEND_SYCL)) && (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__))
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double> ||
                  std::is_same_v<T, int> || std::is_same_v<T, unsigned int>) {
        atomicAdd(address, val);
    } else if constexpr (sizeof(T) == 8 && std::is_integral_v<T>) {
        atomicAdd(reinterpret_cast<unsigned long long*>(address), static_cast<unsigned long long>(val));
    } else if constexpr (std::is_same_v<T, bool>) {
        if (val) *address = true;
    } else {
        static_assert(!sizeof(T*), "atomic_add: unsupported dtype on device");
    }
#else
    if constexpr (std::is_same_v<T, bool>) {
        if (val) __atomic_store_n(address, true, __ATOMIC_RELAXED);
    } else if constexpr (std::is_integral_v<T>) {
        __atomic_fetch_add(address, val, __ATOMIC_RELAXED);
    } else if constexpr (std::is_floating_point_v<T>) {
        T expected;
        __atomic_load(address, &expected, __ATOMIC_RELAXED);
        T desired = expected + val;
        while (!__atomic_compare_exchange(address, &expected, &desired, true,
                                          __ATOMIC_RELAXED, __ATOMIC_RELAXED))
            desired = expected + val;
    } else {
        static_assert(!sizeof(T*), "atomic_add: unsupported dtype on host");
    }
#endif
}

PTSYCL_HOST_DEVICE inline float opmath_exp(float v) { return ::expf(v); }
PTSYCL_HOST_DEVICE inline double opmath_exp(double v) { return ::exp(v); }

template <typename F>
inline void launch_flat(compat::Queue& q, int64_t n, F fn) {
    if (n <= 0) return;
    compat::parallel_for(q, static_cast<std::size_t>(n), fn);
}

// Strided store: out[spec(i)] = fn(i).
template <typename T, typename F>
inline void launch_strided_store(compat::Queue& q, int64_t n, T* out,
                                 StridedSpec spec, F fn) {
    if (n <= 0) return;
    compat::parallel_for(q, static_cast<std::size_t>(n), [=](std::size_t i) {
        out[spec.index(static_cast<int64_t>(i))] = fn(static_cast<int64_t>(i));
    });
}


template <typename T>
inline T read_scalar(compat::Queue& q, const T* p) {
    T v;
    q.copy(&v, p, sizeof(T), true);
    return v;
}

template <typename T>
inline void write_scalar(compat::Queue& q, T* p, T v) {
    q.copy(p, &v, sizeof(T), true);
}

constexpr int64_t kReducePartials = 1024;

#if defined(PTSYCL_BACKEND_SYCL)
constexpr int kReduceWG = 256;
constexpr int64_t kReduceWGMax = 16384;

inline bool reduce_in_groups(compat::Queue& q, int64_t red_n) {
    return q.is_gpu() && red_n >= kReduceWG && red_n <= kReduceWGMax;
}

template <typename acc_t, typename Combine>
PTSYCL_HOST_DEVICE inline acc_t combine_in_group(const sycl::nd_item<1>& it, acc_t* lds,
                                                 acc_t acc, const Combine& combine) {
    const int l = static_cast<int>(it.get_local_id(0));
    if constexpr (std::is_arithmetic_v<acc_t>) {
        const auto sg = it.get_sub_group();
        acc = sycl::reduce_over_group(sg, acc, combine);
        const int groups = static_cast<int>(sg.get_group_linear_range());
        if (sg.get_local_linear_id() == 0) lds[sg.get_group_linear_id()] = acc;
        sycl::group_barrier(it.get_group());
        acc_t out = lds[0];
        for (int g = 1; g < groups; ++g) out = combine(out, lds[g]);
        return out;
    } else {
        lds[l] = acc;
        for (int s = kReduceWG / 2; s > 0; s >>= 1) {
            sycl::group_barrier(it.get_group());
            if (l < s) lds[l] = combine(lds[l], lds[l + s]);
        }
        sycl::group_barrier(it.get_group());
        return lds[0];
    }
}

template <typename acc_t, typename Map, typename Combine>
PTSYCL_HOST_DEVICE inline acc_t reduce_row_in_group(const sycl::nd_item<1>& it, acc_t* lds,
                                                    int64_t o, int64_t red_n, acc_t init,
                                                    const Map& map, const Combine& combine) {
    const int l = static_cast<int>(it.get_local_id(0));
    acc_t acc = init;
    constexpr int kAhead = 8;
    int64_t j = l;
    for (; j + (kAhead - 1) * kReduceWG < red_n; j += kAhead * kReduceWG) {
        acc_t v[kAhead];
#pragma unroll
        for (int t = 0; t < kAhead; ++t) v[t] = map(o, j + t * kReduceWG);
#pragma unroll
        for (int t = 0; t < kAhead; ++t) acc = combine(acc, v[t]);
    }
    for (; j < red_n; j += kReduceWG) acc = combine(acc, map(o, j));
    return combine_in_group(it, lds, acc, combine);
}

template <typename acc_t, typename Map, typename Combine, typename Store>
class ReduceRowsKernel;
#endif

template <typename acc_t, typename Map, typename Combine>
inline acc_t reduce_full(compat::Queue& q, int64_t n, acc_t init, Map map,
                         Combine combine) {
    if (n <= 0) return init;

    const int64_t stripes = n < kReducePartials ? n : kReducePartials;
    acc_t* partials = static_cast<acc_t*>(
        q.alloc(static_cast<std::size_t>(stripes) * sizeof(acc_t)));

    compat::parallel_for(q, static_cast<std::size_t>(stripes), [=](std::size_t s) {
        acc_t acc = init;
        for (int64_t i = static_cast<int64_t>(s); i < n; i += stripes)
            acc = combine(acc, map(i));
        partials[s] = acc;
    });
    std::vector<acc_t> host(static_cast<std::size_t>(stripes));
    q.copy(host.data(), partials, host.size() * sizeof(acc_t), true);
    q.dealloc(partials);

    acc_t result = init;
    for (const acc_t& p : host) result = combine(result, p);
    return result;
}

template <typename acc_t>
struct StripePartials {
    const acc_t* partials;
    int64_t stripes;
    PTSYCL_HOST_DEVICE acc_t operator()(int64_t o, int64_t s) const {
        return partials[o * stripes + s];
    }
};

template <typename acc_t, typename Map, typename Combine, typename Store>
inline void reduce_outer(compat::Queue& q, int64_t out_n, int64_t red_n,
                         acc_t init, Map map, Combine combine, Store store,
                         const at::Tensor& like, int64_t min_stripes = 1) {
    if (out_n <= 0) return;
#if defined(PTSYCL_BACKEND_SYCL)
    if (reduce_in_groups(q, red_n) ||
        (q.is_gpu() && red_n >= 32 && red_n < kReduceWG && out_n < (int64_t{1} << 16))) {
        q.sycl_queue().submit([&](sycl::handler& cgh) {
            sycl::local_accessor<acc_t, 1> lds(sycl::range<1>(kReduceWG), cgh);
            cgh.template parallel_for<ReduceRowsKernel<acc_t, Map, Combine, Store>>(
                sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(out_n * kReduceWG)),
                                  sycl::range<1>(kReduceWG)),
                [=](sycl::nd_item<1> it) {
                    const int64_t o = static_cast<int64_t>(it.get_group(0));
                    const acc_t acc = reduce_row_in_group<acc_t>(it, &lds[0], o, red_n, init,
                                                                 map, combine);
                    if (it.get_local_id(0) == 0) store(o, acc);
                });
        });
        return;
    }
#endif

    constexpr int64_t kTargetItems = int64_t{1} << 16;
    constexpr int64_t kMaxSerialStripes = 1024;
    int64_t stripes = 1;
    if (red_n >= 64 && out_n < kTargetItems) {
        stripes = std::min<int64_t>(red_n / 16, kTargetItems / out_n);
    }
    stripes = std::max<int64_t>(stripes, std::min<int64_t>(min_stripes, red_n / 16));
    if (stripes <= 1) {
        compat::parallel_for(q, static_cast<std::size_t>(out_n), [=](std::size_t o) {
            acc_t acc = init;
            for (int64_t j = 0; j < red_n; ++j)
                acc = combine(acc, map(static_cast<int64_t>(o), j));
            store(static_cast<int64_t>(o), acc);
        });
        return;
    }

    at::Tensor scratch = at::empty(
        {static_cast<int64_t>(out_n * stripes * static_cast<int64_t>(sizeof(acc_t)))},
        like.options().dtype(at::kByte));
    acc_t* partials = static_cast<acc_t*>(scratch.data_ptr());
    compat::parallel_for(q, static_cast<std::size_t>(out_n * stripes), [=](std::size_t t) {
        const int64_t o = static_cast<int64_t>(t) / stripes;
        const int64_t s = static_cast<int64_t>(t) % stripes;
        acc_t acc = init;
        for (int64_t j = s; j < red_n; j += stripes) acc = combine(acc, map(o, j));
        partials[t] = acc;
    });
    if (stripes > kMaxSerialStripes) {
        reduce_outer<acc_t>(q, out_n, stripes, init,
                            StripePartials<acc_t>{partials, stripes}, combine, store, like);
        return;
    }
    compat::parallel_for(q, static_cast<std::size_t>(out_n), [=](std::size_t o_) {
        const int64_t o = static_cast<int64_t>(o_);
        acc_t acc = init;
        for (int64_t s = 0; s < stripes; ++s) acc = combine(acc, partials[o * stripes + s]);
        store(o, acc);
    });
}

} // namespace ptsycl
