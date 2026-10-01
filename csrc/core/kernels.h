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


#include <cstdint>

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

    // Maps a flat logical index (row-major over sizes) to a storage offset.
    PTSYCL_HOST_DEVICE inline int64_t index(int64_t flat) const {
        int64_t off = offset;
        for (int d = ndim - 1; d >= 0; --d) {
            const int64_t c = flat % sizes[d];
            flat /= sizes[d];
            off += c * strides[d];
        }
        return off;
    }
};

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


constexpr int64_t kReducePartials = 1024;

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
    q.synchronize(); // partials are read on the host below

    acc_t result = init;
    for (int64_t s = 0; s < stripes; ++s) result = combine(result, partials[s]);
    q.dealloc(partials);
    return result;
}

template <typename acc_t, typename Map, typename Combine, typename Store>
inline void reduce_outer(compat::Queue& q, int64_t out_n, int64_t red_n,
                         acc_t init, Map map, Combine combine, Store store) {
    if (out_n <= 0) return;
    compat::parallel_for(q, static_cast<std::size_t>(out_n), [=](std::size_t o) {
        acc_t acc = init;
        for (int64_t j = 0; j < red_n; ++j)
            acc = combine(acc, map(static_cast<int64_t>(o), j));
        store(static_cast<int64_t>(o), acc);
    });
}

} // namespace ptsycl
