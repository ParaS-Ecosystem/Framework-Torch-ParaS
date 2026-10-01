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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#if !defined(PTSYCL_BACKEND_SYCL) && !defined(PTSYCL_BACKEND_CPU)
#error "Define exactly one of PTSYCL_BACKEND_SYCL / PTSYCL_BACKEND_CPU"
#endif

#if defined(PTSYCL_BACKEND_SYCL) && defined(PTSYCL_BACKEND_CPU)
#error "Define exactly one of PTSYCL_BACKEND_SYCL / PTSYCL_BACKEND_CPU"
#endif

#if defined(PTSYCL_BACKEND_SYCL)
#include <sycl/sycl.hpp>
#endif

#if (defined(PTSYCL_BACKEND_CUDA) || defined(PTSYCL_BACKEND_SYCL)) && \
    (defined(__CUDACC__) || defined(__CUDA__))
#define PTSYCL_HOST_DEVICE __host__ __device__
#elif defined(PTSYCL_BACKEND_HIP) && (defined(__HIPCC__) || defined(__HIP__))
#define PTSYCL_HOST_DEVICE __host__ __device__
#else
#define PTSYCL_HOST_DEVICE
#endif

namespace ptsycl {
namespace compat {

[[noreturn]] void fail(const std::string& what);

struct DeviceInfo {
    std::string name;
    bool        is_gpu        = false;
    int         native_id     = -1;
    int         compute_units = 0;
    std::size_t global_mem    = 0;
    bool        fp64          = true;
};

std::vector<DeviceInfo> enumerate_devices();

using host_chunk_fn = void (*)(void* ctx, std::size_t begin, std::size_t end);
void host_parallel_chunks(std::size_t n, host_chunk_fn body, void* ctx);
unsigned host_thread_count();

template <typename F>
inline void host_parallel_for(std::size_t n, F&& f) {
    using Fn = std::remove_reference_t<F>;
    struct Ctx { Fn* f; } ctx{std::addressof(f)};
    host_parallel_chunks(
        n,
        [](void* c, std::size_t b, std::size_t e) {
            Fn& fn = *static_cast<Ctx*>(c)->f;
            for (std::size_t i = b; i < e; ++i) fn(i);
        },
        &ctx);
}

class Queue {
public:
    Queue();
    ~Queue();

    Queue(const Queue&)            = delete;
    Queue& operator=(const Queue&) = delete;

    void init(const DeviceInfo& dev);

    bool initialized() const { return initialized_; }
    bool is_gpu()      const { return is_gpu_; }
    int  native_id()   const { return native_id_; }

    void* alloc(std::size_t nbytes);
    void  dealloc(void* ptr);
    void copy(void* dst, const void* src, std::size_t nbytes, bool blocking);
    void memset(void* ptr, int value, std::size_t nbytes);
    void synchronize();

#if defined(PTSYCL_BACKEND_SYCL)
    // Queue ownership and construction remain in paras_compat.cpp.
    sycl::queue& sycl_queue() const;
#endif

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    bool initialized_ = false;
    bool is_gpu_      = false;
    int  native_id_   = -1;
};

const char* backend_name();

} // namespace compat
} // namespace ptsycl
