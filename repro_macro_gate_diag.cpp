// repro_macro_gate_diag.cpp
//
// Isolates exactly where device-detection diverges between atomic_ref.hpp
// (confirmed correctly hitting its #if PARAS_GPU_BACKEND branch) and
// sub_group.hpp (confirmed NOT hitting it -- get_local_linear_id() always
// returned 0, matching its host-only #else branch).
//
// Dumps, per-thread, straight out of the kernel body:
//   raw_thread_idx     -- threadIdx.x itself, no wrapper, no macro gating at all
//   cuda_arch_defined  -- 1 if __CUDA_ARCH__ is defined AT THIS POINT IN THE
//                         KERNEL BODY, 0 otherwise (this is checked inside
//                         the lambda, not in main() -- main() would trivially
//                         always show 0 since host code never has __CUDA_ARCH__)
//   paras_gpu_backend  -- the actual value of PARAS_GPU_BACKEND as seen at
//                         this exact point (via sub_group.hpp's transitive
//                         include of gpu_utilities.hpp)
//   lane_via_subgroup  -- sub_group::get_local_linear_id()'s result, for
//                         direct side-by-side comparison against raw_thread_idx
//
// If raw_thread_idx varies correctly (0..31 within each warp) but
// cuda_arch_defined==0, that confirms the kernel genuinely runs per-thread
// on real hardware, but __CUDA_ARCH__ specifically is not visible at the
// point this header code is compiled -- meaning PARAS_KERNEL_D (which gates
// on __CUDA_ARCH__ directly, not on PARAS_GPU_BACKEND) picks the wrong
// qualifier for get_local_linear_id(), independent of whatever separately
// makes atomic_ref.hpp's PARAS_GPU_BACKEND-gated code work correctly.
//
// Build:
//   parascc \
//     -I$PARAS_HOME/include -I$CUDA/include \
//     -L$CUDA/lib64 -L"$STDCXX_DIR" \
//     -lcudart -lpthread \
//     repro_macro_gate_diag.cpp -o repro_macro_gate_diag
//
//   LD_LIBRARY_PATH="$STDCXX_DIR":$CUDA/lib64:$LD_LIBRARY_PATH \
//     ./repro_macro_gate_diag

#include <cstdio>
#include <cuda_runtime.h>
#include "sycl/sycl.hpp"
#include "sycl/sub_group.hpp"

constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = 2;
constexpr int N         = WARP_SIZE * NUM_WARPS;

struct MacroGateDiagKernel;

static void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(err));
        std::exit(1);
    }
}

int main() {
    auto gpu_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (gpu_devices.empty()) {
        std::fprintf(stderr, "No GPU devices found -- cannot run this test.\n");
        return 1;
    }
    sycl::device dev = gpu_devices.front();
    std::printf("Using device: native_id=%d\n", dev.get_native_id());
    sycl::queue q(dev);

    int* raw_thread_idx    = nullptr;
    int* cuda_arch_defined = nullptr;
    int* paras_gpu_backend = nullptr;
    int* lane_via_subgroup = nullptr;

    check_cuda(cudaMallocManaged(&raw_thread_idx, sizeof(int) * N), "alloc raw_thread_idx");
    check_cuda(cudaMallocManaged(&cuda_arch_defined, sizeof(int) * N), "alloc cuda_arch_defined");
    check_cuda(cudaMallocManaged(&paras_gpu_backend, sizeof(int) * N), "alloc paras_gpu_backend");
    check_cuda(cudaMallocManaged(&lane_via_subgroup, sizeof(int) * N), "alloc lane_via_subgroup");

    for (int i = 0; i < N; ++i) {
        raw_thread_idx[i] = -999;
        cuda_arch_defined[i] = -999;
        paras_gpu_backend[i] = -999;
        lane_via_subgroup[i] = -999;
    }

    {
        int* r_idx = raw_thread_idx;
        int* c_arch = cuda_arch_defined;
        int* p_backend = paras_gpu_backend;
        int* l_sg = lane_via_subgroup;

        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<MacroGateDiagKernel>(sycl::range<1>(N), [=](sycl::id<1> idx) {
                std::size_t gid = idx[0];

                // raw hardware register, no SYCL wrapper at all
                r_idx[gid] = static_cast<int>(threadIdx.x);

#if defined(__CUDA_ARCH__)
                c_arch[gid] = 1;
#else
                c_arch[gid] = 0;
#endif

#if defined(PARAS_GPU_BACKEND)
                p_backend[gid] = PARAS_GPU_BACKEND;
#else
                p_backend[gid] = -1; // sentinel: not even defined at this point
#endif

                sycl::sub_group sg;
                l_sg[gid] = static_cast<int>(sg.get_local_linear_id());
            });
        });
        q.wait();
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }

    std::printf("\n%-6s %-14s %-16s %-16s %-14s\n",
                "gid", "raw_threadIdx", "cuda_arch_def", "paras_backend", "lane_subgroup");
    for (int i = 0; i < N; ++i) {
        std::printf("%-6d %-14d %-16d %-16d %-14d\n",
                     i, raw_thread_idx[i], cuda_arch_defined[i], paras_gpu_backend[i], lane_via_subgroup[i]);
    }

    std::printf("\n=== INTERPRETATION ===\n");
    bool raw_varies = false;
    for (int i = 1; i < N; ++i) {
        if (raw_thread_idx[i] != raw_thread_idx[0]) { raw_varies = true; break; }
    }
    std::printf("raw_threadIdx varies across threads: %s\n", raw_varies ? "YES (kernel genuinely runs per-thread)" : "NO (something is fundamentally wrong with launch itself)");
    std::printf("__CUDA_ARCH__ defined inside kernel body: %s (value shown: 1=defined, 0=not defined)\n",
                 cuda_arch_defined[0] ? "YES" : "NO");
    std::printf("PARAS_GPU_BACKEND value inside kernel body: %d (-1 means undefined entirely)\n", paras_gpu_backend[0]);
    std::printf("sub_group::get_local_linear_id() result: %d (compare to raw_threadIdx %% 32 = %d)\n",
                 lane_via_subgroup[0], raw_thread_idx[0] % WARP_SIZE);

    cudaFree(raw_thread_idx);
    cudaFree(cuda_arch_defined);
    cudaFree(paras_gpu_backend);
    cudaFree(lane_via_subgroup);
    return 0;
}
