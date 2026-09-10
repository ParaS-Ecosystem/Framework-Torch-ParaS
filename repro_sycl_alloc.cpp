// repro_sycl_alloc.cpp
//
// Tests whether sycl:: allocation functions actually work correctly, not
// just whether they compile/return non-null. A pointer that "succeeds" but
// points at the wrong memory, wrong device, or gets silently freed early
// would be a much worse bug than a clean crash -- so this checks real data
// round-trips, not just non-null returns.
//
// Two allocation styles tested, since MemoryPool::acquire/release would need
// ONE of these to replace compat::Queue::alloc/dealloc (currently raw
// cudaMallocManaged):
//   A. sycl::malloc_shared<T>  -- unified/managed memory, host+device visible
//   B. sycl::malloc_device<T>  -- device-only memory, needs explicit copy to read
//
// Build:
//   parascc -I$PARAS_HOME/include -I$CUDA/include -L$CUDA/lib64 -L"$STDCXX_DIR" \
//     -lcudart -lpthread repro_sycl_alloc.cpp -o repro_sycl_alloc
//   LD_LIBRARY_PATH="$STDCXX_DIR":$CUDA/lib64:$LD_LIBRARY_PATH ./repro_sycl_alloc

#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include "sycl/sycl.hpp"

constexpr int N = 1000;

struct WriteKernel;
struct ReadBackKernel;

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
    sycl::context ctx(dev);
    sycl::queue q(dev);
    std::printf("Using device: native_id=%d\n", dev.get_native_id());

    bool test_a_pass = false;
    bool test_b_pass = false;

    // --- Test A: malloc_shared ---
    {
        int* shared_buf = sycl::malloc_shared<int>(N, dev, ctx);
        if (shared_buf == nullptr) {
            std::printf("TEST A (malloc_shared): allocation returned nullptr.\n");
        } else {
            q.submit([&](sycl::handler& cgh) {
                cgh.parallel_for<WriteKernel>(sycl::range<1>(N), [=](sycl::id<1> idx) {
                    shared_buf[idx[0]] = static_cast<int>(idx[0]) * 2;
                });
            });
            q.wait();
            check_cuda(cudaDeviceSynchronize(), "sync after malloc_shared write");

            bool all_correct = true;
            for (int i = 0; i < N; ++i) {
                if (shared_buf[i] != i * 2) { all_correct = false; break; }
            }
            test_a_pass = all_correct;
            std::printf("TEST A (malloc_shared): %s\n", all_correct ? "PASS -- all values correct" : "FAIL -- data mismatch");
            sycl::free(shared_buf, ctx);
        }
    }

    // --- Test B: malloc_device (requires explicit host<->device copy to verify) ---
    {
        int* device_buf = sycl::malloc_device<int>(N, dev, ctx);
        if (device_buf == nullptr) {
            std::printf("TEST B (malloc_device): allocation returned nullptr.\n");
        } else {
            q.submit([&](sycl::handler& cgh) {
                cgh.parallel_for<ReadBackKernel>(sycl::range<1>(N), [=](sycl::id<1> idx) {
                    device_buf[idx[0]] = static_cast<int>(idx[0]) * 3;
                });
            });
            q.wait();
            check_cuda(cudaDeviceSynchronize(), "sync after malloc_device write");

            // Plain cudaMemcpy here deliberately -- this test is about whether
            // malloc_device gives back a real, valid CUDA device pointer, not
            // about testing handler::memcpy (that's repro_handler_memcpy.cpp).
            std::vector<int> host_buf(N);
            check_cuda(cudaMemcpy(host_buf.data(), device_buf, N * sizeof(int), cudaMemcpyDeviceToHost),
                       "cudaMemcpy readback from malloc_device buffer");

            bool all_correct = true;
            for (int i = 0; i < N; ++i) {
                if (host_buf[i] != i * 3) { all_correct = false; break; }
            }
            test_b_pass = all_correct;
            std::printf("TEST B (malloc_device): %s\n", all_correct ? "PASS -- all values correct" : "FAIL -- data mismatch");
            sycl::free(device_buf, ctx);
        }
    }

    std::printf("\n=== VERDICT ===\n");
    std::printf("malloc_shared: %s\n", test_a_pass ? "SAFE TO USE" : "DO NOT USE -- broken");
    std::printf("malloc_device: %s\n", test_b_pass ? "SAFE TO USE" : "DO NOT USE -- broken");
    return (test_a_pass && test_b_pass) ? 0 : 1;
}
