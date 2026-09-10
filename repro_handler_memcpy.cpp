// repro_handler_memcpy.cpp
//
// Tests handler::memcpy specifically -- this is the exact construct
// compat::Queue::copy would be replaced with, which underlies
// ParasAllocator::copy_data (called with blocking=true always, per the
// real allocator.cpp source -- so the "always synchronous internally"
// behavior we found in cuda_threadpool::memcpy is not a regression for
// THIS specific call site, just worth confirming correctness first).
//
// Tests both directions since PyTorch's allocator copies data both ways:
//   A. host  -> device (simulates uploading a tensor)
//   B. device -> host   (simulates reading a tensor back, e.g. .cpu())
//
// Build:
//   parascc -I$PARAS_HOME/include -I$CUDA/include -L$CUDA/lib64 -L"$STDCXX_DIR" \
//     -lcudart -lpthread repro_handler_memcpy.cpp -o repro_handler_memcpy
//   LD_LIBRARY_PATH="$STDCXX_DIR":$CUDA/lib64:$LD_LIBRARY_PATH ./repro_handler_memcpy

#include <cstdio>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include "sycl/sycl.hpp"

constexpr int N = 1000;
constexpr std::size_t NBYTES = N * sizeof(int);

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

    // Plain cudaMalloc for the device-side buffer -- deliberately NOT using
    // sycl::malloc_device here, to keep this test isolated to handler::memcpy
    // alone, independent of repro_sycl_alloc.cpp's result.
    int* device_buf = nullptr;
    check_cuda(cudaMalloc(&device_buf, NBYTES), "cudaMalloc(device_buf)");

    std::vector<int> host_src(N), host_dst(N, -1);
    for (int i = 0; i < N; ++i) host_src[i] = i * 7 + 3;

    bool test_a_pass = false;
    bool test_b_pass = false;

    // --- Test A: host -> device, via handler::memcpy ---
    {
        q.submit([&](sycl::handler& cgh) {
            cgh.memcpy(device_buf, host_src.data(), NBYTES);
        });
        q.wait();
        check_cuda(cudaDeviceSynchronize(), "sync after H2D handler::memcpy");

        // verify via plain cudaMemcpy readback -- isolates whether the
        // WRITE (handler::memcpy) worked, independent of testing the
        // READ direction (Test B) with the same mechanism
        std::vector<int> verify(N);
        check_cuda(cudaMemcpy(verify.data(), device_buf, NBYTES, cudaMemcpyDeviceToHost),
                   "cudaMemcpy verify readback");
        test_a_pass = (verify == host_src);
        std::printf("TEST A (host -> device via handler::memcpy): %s\n",
                     test_a_pass ? "PASS -- data matches exactly" : "FAIL -- data mismatch");
    }

    // --- Test B: device -> host, via handler::memcpy ---
    {
        // seed device_buf with known values via plain cudaMemcpy first,
        // so Test B is isolated to the READ direction only
        check_cuda(cudaMemcpy(device_buf, host_src.data(), NBYTES, cudaMemcpyHostToDevice),
                   "cudaMemcpy seed for Test B");

        q.submit([&](sycl::handler& cgh) {
            cgh.memcpy(host_dst.data(), device_buf, NBYTES);
        });
        q.wait();
        check_cuda(cudaDeviceSynchronize(), "sync after D2H handler::memcpy");

        test_b_pass = (host_dst == host_src);
        std::printf("TEST B (device -> host via handler::memcpy): %s\n",
                     test_b_pass ? "PASS -- data matches exactly" : "FAIL -- data mismatch");
    }

    std::printf("\n=== VERDICT ===\n");
    std::printf("handler::memcpy host->device: %s\n", test_a_pass ? "SAFE TO USE" : "DO NOT USE -- broken");
    std::printf("handler::memcpy device->host: %s\n", test_b_pass ? "SAFE TO USE" : "DO NOT USE -- broken");

    cudaFree(device_buf);
    return (test_a_pass && test_b_pass) ? 0 : 1;
}
