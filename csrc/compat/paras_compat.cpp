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


#include "compat/paras_compat.h"

#include <ATen/Parallel.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace ptsycl {
namespace compat {

void fail(const std::string& what) {
    throw std::runtime_error("ptsycl: " + what);
}

// -----------------------------------------------------------------------------
// Host CPU description
// -----------------------------------------------------------------------------
static std::string host_cpu_name() {
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    while (std::getline(f, line)) {
        auto pos = line.find("model name");
        if (pos != std::string::npos) {
            auto colon = line.find(':');
            if (colon != std::string::npos) {
                std::string name = line.substr(colon + 1);
                while (!name.empty() && name.front() == ' ') name.erase(name.begin());
                return name;
            }
        }
    }
    return "Host CPU";
}

static std::size_t host_mem_bytes() {
    std::ifstream f("/proc/meminfo");
    std::string key;
    std::size_t kb = 0;
    while (f >> key >> kb) {
        if (key == "MemTotal:") return kb * 1024;
        f.ignore(256, '\n');
    }
    return 0;
}

unsigned host_thread_count() {
    static unsigned n = [] {
        if (const char* env = std::getenv("PTSYCL_CPU_THREADS")) {
            int v = std::atoi(env);
            if (v > 0) return static_cast<unsigned>(v);
        }
        const int torch_threads = at::get_num_threads();
        if (torch_threads > 0) return static_cast<unsigned>(torch_threads);
        const unsigned hc = std::thread::hardware_concurrency();
        return hc == 0 ? 4u : hc;
    }();
    return n;
}

// -----------------------------------------------------------------------------
// Device enumeration
// -----------------------------------------------------------------------------
std::vector<DeviceInfo> enumerate_devices() {
    std::vector<DeviceInfo> out;

    DeviceInfo cpu;
    cpu.name          = host_cpu_name();
    cpu.is_gpu        = false;
    cpu.native_id     = -1;
    cpu.compute_units = static_cast<int>(host_thread_count());
    cpu.global_mem    = host_mem_bytes();
    cpu.fp64          = true;
    out.push_back(std::move(cpu));

#if defined(PTSYCL_BACKEND_CUDA)
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) count = 0; // no GPUs visible: CPU-only table
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, i) != cudaSuccess) continue;
        DeviceInfo gpu;
        gpu.name          = prop.name;
        gpu.is_gpu        = true;
        gpu.native_id     = i;
        gpu.compute_units = prop.multiProcessorCount;
        gpu.global_mem    = prop.totalGlobalMem;
        gpu.fp64          = true;
        out.push_back(std::move(gpu));
    }
#endif

#if defined(PTSYCL_BACKEND_HIP)
    int hip_count = 0;
    hipError_t herr = hipGetDeviceCount(&hip_count);
    if (herr != hipSuccess) hip_count = 0; // no GPUs visible: CPU-only table
    for (int i = 0; i < hip_count; ++i) {
        hipDeviceProp_t prop{};
        if (hipGetDeviceProperties(&prop, i) != hipSuccess) continue;
        DeviceInfo gpu;
        gpu.name          = prop.name;
        gpu.is_gpu        = true;
        gpu.native_id     = i;
        gpu.compute_units = prop.multiProcessorCount;
        gpu.global_mem    = prop.totalGlobalMem;
        gpu.fp64          = true;
        out.push_back(std::move(gpu));
    }
#endif
    return out;
}

// -----------------------------------------------------------------------------
// Host execution engine
// -----------------------------------------------------------------------------
namespace {

extern "C" void GOMP_parallel(void (*fn)(void*), void* data,
                              unsigned num_threads, unsigned flags);

std::size_t host_grain_size() {
    static const std::size_t value = [] {
        if (const char* env = std::getenv("PTSYCL_CPU_GRAIN_SIZE")) {
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(env, &end, 10);
            if (end != env && *end == '\0' && parsed > 0)
                return static_cast<std::size_t>(parsed);
        }
        return std::size_t{16384};
    }();
    return value;
}

struct HostParallelJob {
    std::size_t n;
    std::size_t chunks;
    std::size_t per;
    host_chunk_fn body;
    void* ctx;
    std::atomic<std::size_t> next{0};
};

void run_host_parallel_job(void* raw_job) {
    auto& job = *static_cast<HostParallelJob*>(raw_job);
    while (true) {
        const std::size_t chunk =
            job.next.fetch_add(1, std::memory_order_relaxed);
        if (chunk >= job.chunks) return;
        const std::size_t begin = chunk * job.per;
        const std::size_t end = std::min(begin + job.per, job.n);
        if (begin < end) job.body(job.ctx, begin, end);
    }
}

}

void host_parallel_chunks(std::size_t n, host_chunk_fn body, void* ctx) {
    if (n == 0) return;
    const std::size_t thread_count = host_thread_count();
    const std::size_t grain = host_grain_size();
    if (thread_count <= 1 || n <= grain || at::in_parallel_region()) {
        body(ctx, 0, n);
        return;
    }
    const std::size_t chunks =
        std::min(thread_count, (n + grain - 1) / grain);
    const std::size_t per = (n + chunks - 1) / chunks;
    HostParallelJob job{n, chunks, per, body, ctx};
    GOMP_parallel(run_host_parallel_job, &job,
                  static_cast<unsigned>(chunks), 0);
}

// -----------------------------------------------------------------------------
// CUDA helpers
// -----------------------------------------------------------------------------
#if defined(PTSYCL_BACKEND_CUDA)
namespace detail {
void throw_on_cuda_error(int err, const char* what) {
    if (err != cudaSuccess) {
        std::ostringstream os;
        os << what << ": " << cudaGetErrorString(static_cast<cudaError_t>(err));
        fail(os.str());
    }
}
} // namespace detail
#endif

// -----------------------------------------------------------------------------
// HIP/ROCm helpers
// -----------------------------------------------------------------------------
#if defined(PTSYCL_BACKEND_HIP)
namespace detail {
void throw_on_hip_error(int err, const char* what) {
    if (err != hipSuccess) {
        std::ostringstream os;
        os << what << ": " << hipGetErrorString(static_cast<hipError_t>(err));
        fail(os.str());
    }
}
} // namespace detail
#endif

// -----------------------------------------------------------------------------
// Queue
// -----------------------------------------------------------------------------
void Queue::init(const DeviceInfo& dev) {
    if (initialized_) fail("Queue::init called twice");
    is_gpu_    = dev.is_gpu;
    native_id_ = dev.native_id;
#if defined(PTSYCL_BACKEND_CUDA)
    if (is_gpu_) {
        detail::throw_on_cuda_error(cudaSetDevice(native_id_), "cudaSetDevice(init)");
        cudaStream_t s = nullptr;
        detail::throw_on_cuda_error(
            cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "cudaStreamCreate");
        stream_ = s;
    }
#elif defined(PTSYCL_BACKEND_HIP)
    if (is_gpu_) {
        detail::throw_on_hip_error(hipSetDevice(native_id_), "hipSetDevice(init)");
        hipStream_t s = nullptr;
        detail::throw_on_hip_error(
            hipStreamCreateWithFlags(&s, hipStreamNonBlocking), "hipStreamCreate");
        stream_ = s;
    }
#else
    if (is_gpu_) fail("GPU device requested in a CPU-only build");
#endif
    initialized_ = true;
}

Queue::~Queue() {
#if defined(PTSYCL_BACKEND_CUDA)
    if (stream_ != nullptr) {
        cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
        stream_ = nullptr;
    }
#elif defined(PTSYCL_BACKEND_HIP)
    if (stream_ != nullptr) {
        hipStreamDestroy(static_cast<hipStream_t>(stream_));
        stream_ = nullptr;
    }
#endif
}

void* Queue::alloc(std::size_t nbytes) {
    if (nbytes == 0) nbytes = 1;
#if defined(PTSYCL_BACKEND_CUDA)
    if (is_gpu_) {
        detail::throw_on_cuda_error(cudaSetDevice(native_id_), "cudaSetDevice(alloc)");
        void* p = nullptr;
        cudaError_t err = cudaMallocManaged(&p, nbytes);
        if (err != cudaSuccess) {
            std::ostringstream os;
            os << "cudaMallocManaged(" << nbytes << " bytes) on cuda:" << native_id_
               << ": " << cudaGetErrorString(err);
            fail(os.str());
        }
        return p;
    }
#endif
#if defined(PTSYCL_BACKEND_HIP)
    if (is_gpu_) {
        detail::throw_on_hip_error(hipSetDevice(native_id_), "hipSetDevice(alloc)");
        void* p = nullptr;
        hipError_t err = hipMallocManaged(&p, nbytes);
        if (err != hipSuccess) {
            std::ostringstream os;
            os << "hipMallocManaged(" << nbytes << " bytes) on hip:" << native_id_
               << ": " << hipGetErrorString(err);
            fail(os.str());
        }
        return p;
    }
#endif
    void* p = nullptr;
    if (posix_memalign(&p, 64, nbytes) != 0 || p == nullptr) {
        std::ostringstream os;
        os << "host allocation of " << nbytes << " bytes failed";
        fail(os.str());
    }
    return p;
}

void Queue::dealloc(void* ptr) {
    if (ptr == nullptr) return;
#if defined(PTSYCL_BACKEND_CUDA)
    if (is_gpu_) {
        cudaFree(ptr); 
        return;
    }
#endif
#if defined(PTSYCL_BACKEND_HIP)
    if (is_gpu_) {
        hipFree(ptr);
        return;
    }
#endif
    std::free(ptr);
}

void Queue::copy(void* dst, const void* src, std::size_t nbytes, bool blocking) {
    if (nbytes == 0) return;
#if defined(PTSYCL_BACKEND_CUDA)
    if (is_gpu_) {
        detail::throw_on_cuda_error(cudaSetDevice(native_id_), "cudaSetDevice(copy)");
        // cudaMemcpyDefault resolves host/managed/device pointers via UVA.
        detail::throw_on_cuda_error(
            cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDefault, stream()),
            "cudaMemcpyAsync");
        if (blocking) synchronize();
        return;
    }
#endif
#if defined(PTSYCL_BACKEND_HIP)
    if (is_gpu_) {
        detail::throw_on_hip_error(hipSetDevice(native_id_), "hipSetDevice(copy)");
        // hipMemcpyDefault resolves host/managed/device pointers via unified addressing.
        detail::throw_on_hip_error(
            hipMemcpyAsync(dst, src, nbytes, hipMemcpyDefault, stream()),
            "hipMemcpyAsync");
        if (blocking) synchronize();
        return;
    }
#endif
    std::memcpy(dst, src, nbytes);
    (void)blocking; 
}

void Queue::memset(void* ptr, int value, std::size_t nbytes) {
    if (nbytes == 0) return;
#if defined(PTSYCL_BACKEND_CUDA)
    if (is_gpu_) {
        detail::throw_on_cuda_error(cudaSetDevice(native_id_), "cudaSetDevice(memset)");
        detail::throw_on_cuda_error(
            cudaMemsetAsync(ptr, value, nbytes, stream()), "cudaMemsetAsync");
        return;
    }
#endif
#if defined(PTSYCL_BACKEND_HIP)
    if (is_gpu_) {
        detail::throw_on_hip_error(hipSetDevice(native_id_), "hipSetDevice(memset)");
        detail::throw_on_hip_error(
            hipMemsetAsync(ptr, value, nbytes, stream()), "hipMemsetAsync");
        return;
    }
#endif
    std::memset(ptr, value, nbytes);
}

void Queue::synchronize() {
#if defined(PTSYCL_BACKEND_CUDA)
    if (is_gpu_) {
        detail::throw_on_cuda_error(cudaSetDevice(native_id_), "cudaSetDevice(sync)");
        detail::throw_on_cuda_error(
            cudaStreamSynchronize(stream()), "cudaStreamSynchronize");
    }
#endif
#if defined(PTSYCL_BACKEND_HIP)
    if (is_gpu_) {
        detail::throw_on_hip_error(hipSetDevice(native_id_), "hipSetDevice(sync)");
        detail::throw_on_hip_error(
            hipStreamSynchronize(stream()), "hipStreamSynchronize");
    }
#endif
    
}

const char* backend_name() {
#if defined(PTSYCL_BACKEND_CUDA)
    return "paras-cuda";
#elif defined(PTSYCL_BACKEND_HIP)
    return "paras-hip";
#else
    return "paras-cpu";
#endif
}

} // namespace compat
} // namespace ptsycl
