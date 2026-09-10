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

// NOTE: filename assumed -- adjust the #include below and the path this is
// copied to if your tree calls this something other than core/kernel_utils.h.

#pragma once

#include <cstdint>
#include <type_traits>

#include <c10/util/Exception.h>

#include "core/common.h"

#if defined(PTSYCL_BACKEND_SYCL)
#include <sycl/sycl.hpp>
#endif

namespace ptsycl {

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

// atomic_ref is one of the constructs the compiler-team audit confirmed
// working under parascc (unlike, e.g., sub_group::get_local_linear_id(),
// which always returns 0). It replaces the old CUDA/HIP atomicAdd()
// intrinsics entirely -- there is no more __CUDA_ARCH__/__HIP_DEVICE_COMPILE__
// branch here.
// template <typename T>
// PTSYCL_HOST_DEVICE inline void atomic_add(T* address, T val) {
//     if constexpr (std::is_same_v<T, bool>) {
//         if (val) *address = true;
//     } else {
// #if defined(PTSYCL_BACKEND_SYCL)
//         sycl::atomic_ref<T, sycl::memory_order::relaxed,
//                           sycl::memory_scope::device> ref(*address);
//         ref.fetch_add(val);
// #else
//         *address += val;
// #endif
//     }
// }

// core/kernel_utils.h — atomic_add
template <typename T>
PTSYCL_HOST_DEVICE inline void atomic_add(T* address, T val) {
    if constexpr (std::is_same_v<T, bool>) {
        if (val) *address = true;
    } else {
#if defined(PTSYCL_BACKEND_SYCL)
        // fetch_add on this parascc install passes through directly to a
        // native atomicAdd() overload with NO generic/CAS fallback beneath
        // it -- confirmed by int64_t (64-bit, same width as double) still
        // failing to compile. Only {int32_t, uint32_t, float, double} are
        // safe here today. Everything else needs a hand-rolled CAS loop,
        // which is unverified on this install -- see the open item with
        // Laxmikant re: atomic_ref::compare_exchange_strong before adding
        // support for int8/int16/int64/Half/BFloat16.
        static_assert(std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> ||
                      std::is_same_v<T, float> || std::is_same_v<T, double>,
                      "paras atomic_add: T not yet supported by this parascc "
                      "install's atomic_ref::fetch_add -- see kernel_utils.h");
        sycl::atomic_ref<T, sycl::memory_order::relaxed,
                          sycl::memory_scope::device> ref(*address);
        ref.fetch_add(val);
#else
        *address += val;
#endif
    }
}
template <typename F>
inline void launch_flat(compat::Queue& q, int64_t n, F fn) {
    if (n <= 0) return;
    q.parallel_for(static_cast<std::size_t>(n), fn);
}

// Strided store: out[spec(i)] = fn(i).
template <typename T, typename F>
inline void launch_strided_store(compat::Queue& q, int64_t n, T* out,
                                 StridedSpec spec, F fn) {
    if (n <= 0) return;
    q.parallel_for(static_cast<std::size_t>(n), [=](std::size_t i) {
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

    q.parallel_for(static_cast<std::size_t>(stripes), [=](std::size_t s) {
        acc_t acc = init;
        for (int64_t i = static_cast<int64_t>(s); i < n; i += stripes)
            acc = combine(acc, map(i));
        partials[s] = acc;
    });
    q.synchronize(); // partials are read on the host below (they're USM
                      // shared allocations, so this is a plain host read)

    acc_t result = init;
    for (int64_t s = 0; s < stripes; ++s) result = combine(result, partials[s]);
    q.dealloc(partials);
    return result;
}

template <typename acc_t, typename Map, typename Combine, typename Store>
inline void reduce_outer(compat::Queue& q, int64_t out_n, int64_t red_n,
                         acc_t init, Map map, Combine combine, Store store) {
    if (out_n <= 0) return;
    q.parallel_for(static_cast<std::size_t>(out_n), [=](std::size_t o) {
        acc_t acc = init;
        for (int64_t j = 0; j < red_n; ++j)
            acc = combine(acc, map(static_cast<int64_t>(o), j));
        store(static_cast<int64_t>(o), acc);
    });
}

} // namespace ptsycl