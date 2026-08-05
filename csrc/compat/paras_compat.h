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
#include <string>
#include <vector>

#if !defined(PTSYCL_BACKEND_CUDA) && !defined(PTSYCL_BACKEND_CPU) && !defined(PTSYCL_BACKEND_HIP)
#error "Define exactly one of PTSYCL_BACKEND_CUDA / PTSYCL_BACKEND_HIP / PTSYCL_BACKEND_CPU"
#endif

#if (defined(PTSYCL_BACKEND_CUDA) + defined(PTSYCL_BACKEND_HIP) + defined(PTSYCL_BACKEND_CPU)) > 1
#error "Define exactly one of PTSYCL_BACKEND_CUDA / PTSYCL_BACKEND_HIP / PTSYCL_BACKEND_CPU"
#endif

#if defined(PTSYCL_BACKEND_CUDA)
#include <cuda_runtime.h>
#endif

#if defined(PTSYCL_BACKEND_HIP)
#include <hip/hip_runtime.h>
#endif


#if defined(PTSYCL_BACKEND_CUDA) && (defined(__CUDACC__) || defined(__CUDA__))
#define PTSYCL_HOST_DEVICE __host__ __device__
#elif defined(PTSYCL_BACKEND_HIP) && (defined(__HIPCC__) || defined(__HIP__))
#define PTSYCL_HOST_DEVICE __host__ __device__
#else
#define PTSYCL_HOST_DEVICE
#endif

// PARAS_GPU_BACKEND is the ParaS-side signal for which GPU backend is active.
// It is set independently of __HIP_DEVICE_COMPILE__ / __CUDA_ARCH__ because
// parascc invokes clang directly (bypassing hipcc/nvcc), so those compiler-
// driver-injected macros cannot be relied on to detect device-side compilation.
#if defined(PTSYCL_BACKEND_HIP)
#define PARAS_GPU_BACKEND 2
#elif defined(PTSYCL_BACKEND_CUDA)
#define PARAS_GPU_BACKEND 1
#else
#define PARAS_GPU_BACKEND 0
#endif

namespace ptsycl {
namespace compat {


[[noreturn]] void fail(const std::string& what);

// -----------------------------------------------------------------------------
// Device table
// -----------------------------------------------------------------------------
struct DeviceInfo {
    std::string name;
    bool        is_gpu        = false;
    int         native_id     = -1;   // CUDA/HIP device ordinal when is_gpu
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

#if defined(PTSYCL_BACKEND_CUDA)

namespace detail {

inline constexpr unsigned kBlockSize = 256;
inline constexpr unsigned kMaxBlocks = 4096;

template <typename F>
__global__ void flat_kernel(std::size_t n, F f) {
    std::size_t i      = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    std::size_t stride = gridDim.x * static_cast<std::size_t>(blockDim.x);
    for (; i < n; i += stride) f(i);
}

void throw_on_cuda_error(int err, const char* what);

} // namespace detail
#endif // PTSYCL_BACKEND_CUDA

#if defined(PTSYCL_BACKEND_HIP)

namespace detail {

inline constexpr unsigned kBlockSize = 256;
inline constexpr unsigned kMaxBlocks = 4096;

template <typename F>
__global__ void flat_kernel(std::size_t n, F f) {
    std::size_t i      = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    std::size_t stride = gridDim.x * static_cast<std::size_t>(blockDim.x);
    for (; i < n; i += stride) f(i);
}

void throw_on_hip_error(int err, const char* what);

} // namespace detail
#endif // PTSYCL_BACKEND_HIP


class Queue {
public:
    Queue() = default;
    ~Queue();

    Queue(const Queue&)            = delete;
    Queue& operator=(const Queue&) = delete;

    // Binds this queue to a device from enumerate_devices().
    void init(const DeviceInfo& dev);

    bool initialized() const { return initialized_; }
    bool is_gpu()      const { return is_gpu_; }
    int  native_id()   const { return native_id_; }

  
    void* alloc(std::size_t nbytes);
    void  dealloc(void* ptr);

  
    void copy(void* dst, const void* src, std::size_t nbytes, bool blocking);

  
    void memset(void* ptr, int value, std::size_t nbytes);

   
    void synchronize();

    
    template <typename F>
    void parallel_for(std::size_t n, F f) {
        if (n == 0) return;
#if defined(PTSYCL_BACKEND_CUDA)
        if (is_gpu_) {
            detail::throw_on_cuda_error(cudaSetDevice(native_id_),
                                        "cudaSetDevice(parallel_for)");
            unsigned blocks = static_cast<unsigned>(
                (n + detail::kBlockSize - 1) / detail::kBlockSize);
            if (blocks > detail::kMaxBlocks) blocks = detail::kMaxBlocks;
            detail::flat_kernel<<<blocks, detail::kBlockSize, 0, stream()>>>(n, f);
            detail::throw_on_cuda_error(cudaGetLastError(),
                                        "flat_kernel launch");
            return;
        }
#endif
#if defined(PTSYCL_BACKEND_HIP)
        if (is_gpu_) {
            detail::throw_on_hip_error(hipSetDevice(native_id_),
                                       "hipSetDevice(parallel_for)");
            unsigned blocks = static_cast<unsigned>(
                (n + detail::kBlockSize - 1) / detail::kBlockSize);
            if (blocks > detail::kMaxBlocks) blocks = detail::kMaxBlocks;
            detail::flat_kernel<<<blocks, detail::kBlockSize, 0, stream()>>>(n, f);
            detail::throw_on_hip_error(hipGetLastError(),
                                       "flat_kernel launch");
            return;
        }
#endif
        host_parallel_for(n, f);
    }

#if defined(PTSYCL_BACKEND_CUDA)
    cudaStream_t stream() const { return static_cast<cudaStream_t>(stream_); }
#endif
#if defined(PTSYCL_BACKEND_HIP)
    hipStream_t stream() const { return static_cast<hipStream_t>(stream_); }
#endif

private:
    bool initialized_ = false;
    bool is_gpu_      = false;
    int  native_id_   = -1;
    void* stream_     = nullptr; // cudaStream_t / hipStream_t depending on backend
};


const char* backend_name();

} // namespace compat
} // namespace ptsycl
