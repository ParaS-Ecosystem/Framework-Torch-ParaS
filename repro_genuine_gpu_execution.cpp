// repro_genuine_gpu_execution.cpp
//
// Settles a question the last two hours of testing have raised but not
// answered: when handler::parallel_for "passes" a data-correctness check,
// did it actually run on the GPU, or could it have silently executed on
// the host? A host-fallback that writes to host-visible memory and reads
// it back would pass a naive correctness check without ever touching the
// device -- this test can't be fooled that way.
//
// Method: allocate with plain cudaMalloc (DEVICE-ONLY memory -- NOT
// managed, NOT pinned, NOT host-accessible under any normal circumstance).
// Launch a kernel via the exact same handler::parallel_for path used
// throughout this investigation, writing known values into that buffer.
// Then attempt a RAW HOST memcpy (plain std::memcpy, not cudaMemcpy)
// directly against the device pointer.
//
//   - If the kernel genuinely ran on the GPU: the buffer is real device
//     memory. A raw host memcpy against it should either segfault
//     (most likely) or return garbage/unrelated memory -- NOT the values
//     the kernel wrote, because the host has no legitimate way to see them
//     without an explicit device-to-host copy.
//   - If the kernel silently executed on the host: the "device" pointer
//     from cudaMalloc would still be a real device address, so even host
//     fallback execution couldn't legitimately write through it either --
//     BUT if somehow the write happened on the host into address space
//     the host CAN see (e.g. if the compiler's kernel-launch substitution
//     mechanism doesn't actually respect cudaMalloc semantics), that would
//     show up here as a successful raw read where one should be impossible.
//
// This test is deliberately allowed to crash -- a segfault here is a
// GOOD, informative result (proves real device memory + real device
// execution), not a bug in the test.
//
// Build:
//   parascc -I$PARAS_HOME/include -I$CUDA/include -L$CUDA/lib64 -L"$STDCXX_DIR" \
//     -lcudart -lpthread repro_genuine_gpu_execution.cpp -o repro_genuine_gpu_execution
//   LD_LIBRARY_PATH="$STDCXX_DIR":$CUDA/lib64:$LD_LIBRARY_PATH \
//     ./repro_genuine_gpu_execution
//   echo "Exit code: $?"   # 139 (segfault) here is a PASS, not a crash to fix

#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include "sycl/sycl.hpp"

constexpr int N = 1000;

struct GenuineGpuKernel;

static void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(err));
        std::exit(1);
    }
}

int main() {
    auto gpu_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (gpu_devices.empty()) {
        std::fprintf(stderr, "No GPU devices found.\n"); return 1;
    }
    sycl::device dev = gpu_devices.front();
    sycl::queue q(dev);
    std::printf("Using device: native_id=%d\n", dev.get_native_id());

    // Plain cudaMalloc -- device-only memory, confirmed NOT host-accessible
    // under normal CUDA semantics (unlike cudaMallocManaged/cudaHostAlloc).
    int* device_only_buf = nullptr;
    check_cuda(cudaMalloc(&device_only_buf, N * sizeof(int)), "cudaMalloc (device-only)");
    std::printf("Allocated device-only buffer at %p (via cudaMalloc, NOT managed)\n",
                (void*)device_only_buf);

    {
        int* out = device_only_buf;
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<GenuineGpuKernel>(sycl::range<1>(N), [=](sycl::id<1> idx) {
                out[idx[0]] = static_cast<int>(idx[0]) * 11 + 7;
            });
        });
        q.wait();
        check_cuda(cudaDeviceSynchronize(), "sync after kernel");
    }

    std::printf("\nKernel launch completed without error. Now attempting a RAW HOST "
                "memcpy directly against the device pointer (bypassing cudaMemcpy "
                "entirely) -- this SHOULD crash or return garbage if the pointer is "
                "genuine device-only memory written by a genuine GPU kernel.\n");
    std::fflush(stdout); // flush before the likely crash so this message isn't lost

    int host_buf[N];
    std::memcpy(host_buf, device_only_buf, N * sizeof(int)); // <-- the decisive line

    // If we get here without crashing, check whether the values are correct.
    bool all_correct = true;
    for (int i = 0; i < N; ++i) {
        if (host_buf[i] != i * 11 + 7) { all_correct = false; break; }
    }

    std::printf("\n=== UNEXPECTED: raw host memcpy did NOT crash ===\n");
    std::printf("Values %s the kernel's expected output.\n",
                 all_correct ? "MATCH" : "DO NOT MATCH");
    if (all_correct) {
        std::printf("\n=== VERDICT ===\n");
        std::printf("SUSPICIOUS: raw host memcpy against device-only memory succeeded "
                     "AND returned exactly correct values. This should be impossible "
                     "under normal CUDA semantics for cudaMalloc'd memory. Possible "
                     "explanations: (1) the kernel actually executed on the host, not "
                     "the GPU, writing through some address the host can legitimately "
                     "see, or (2) this system's memory model (e.g. unified addressing "
                     "on certain platforms) makes this pointer host-readable in a way "
                     "that doesn't generalize. Needs follow-up before trusting ANY "
                     "prior 'confirmed working' GPU repro result at face value.\n");
    } else {
        std::printf("\n=== VERDICT ===\n");
        std::printf("Host memcpy succeeded but returned wrong/garbage data -- "
                     "consistent with reading unrelated host memory rather than "
                     "the device buffer's real contents either way.\n");
    }

    cudaFree(device_only_buf);
    return 0;
}
