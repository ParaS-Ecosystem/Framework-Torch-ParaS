// repro_wait_and_device_targeting.cpp
//
// Two checks bundled together since both matter for Context::synchronize /
// Context()'s device enumeration loop:
//
//   CHECK A: does q.wait() ALONE (no accompanying cudaDeviceSynchronize)
//            correctly guarantee the kernel has finished before the host
//            reads results? Every prior repro called both together, so
//            this was never actually isolated -- if q.wait() alone is
//            insufficient, Context::synchronize (which would only call
//            q.wait()) would have a real race.
//
//   CHECK B: if more than one GPU is visible, does constructing a
//            sycl::queue from a device matched by get_native_id() == 1
//            (not 0) actually run kernels on device 1, not silently fall
//            back to device 0? Verified via cudaGetDevice() inside the
//            kernel-launching host code path AND via which device shows
//            memory allocated (nvidia-smi cross-check suggested in output).
//            If only one GPU is visible in this environment, Check B
//            reports that plainly rather than faking a result.
//
// Build:
//   parascc -I$PARAS_HOME/include -I$CUDA/include -L$CUDA/lib64 -L"$STDCXX_DIR" \
//     -lcudart -lpthread repro_wait_and_device_targeting.cpp -o repro_wait_and_device_targeting
//   LD_LIBRARY_PATH="$STDCXX_DIR":$CUDA/lib64:$LD_LIBRARY_PATH ./repro_wait_and_device_targeting

#include <cstdio>
#include <cuda_runtime.h>
#include "sycl/sycl.hpp"

constexpr int N = 500000; // deliberately largish -- gives a real race window
                          // if wait() returns before the kernel truly finishes

struct SlowishKernel;

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
    std::printf("Visible GPU count: %zu\n", gpu_devices.size());
    for (const auto& d : gpu_devices) {
        std::printf("  native_id=%d\n", d.get_native_id());
    }

    // --- CHECK A: q.wait() alone, no cudaDeviceSynchronize crutch ---
    {
        sycl::device dev = gpu_devices.front();
        sycl::queue q(dev);

        int* result = nullptr;
        check_cuda(cudaMallocManaged(&result, sizeof(int) * N), "alloc for Check A");
        for (int i = 0; i < N; ++i) result[i] = -1;

        int* out = result;
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<SlowishKernel>(sycl::range<1>(N), [=](sycl::id<1> idx) {
                // some real, non-trivial per-thread work so the kernel takes
                // measurable time -- a race would be more likely to show up
                // than with a single trivial write
                std::size_t i = idx[0];
                int acc = 0;
                for (int k = 0; k < 50; ++k) acc += static_cast<int>(i) + k;
                out[i] = acc;
            });
        });

        q.wait(); // <-- THE THING UNDER TEST: no cudaDeviceSynchronize after this

        bool all_written = true;
        int first_bad = -1;
        for (int i = 0; i < N; ++i) {
            if (result[i] == -1) { all_written = false; first_bad = i; break; }
        }

        std::printf("\n=== CHECK A: q.wait() alone ===\n");
        if (all_written) {
            std::printf("PASS -- all %d results written before q.wait() returned control.\n", N);
        } else {
            std::printf("FAIL -- result[%d] still holds sentinel -1 after q.wait() returned. "
                         "q.wait() does NOT reliably synchronize on its own; "
                         "any code relying on it without an extra sync (e.g. a rewritten "
                         "Context::synchronize) would have a real race.\n", first_bad);
        }
        cudaFree(result);
    }

    // --- CHECK B: device targeting by ordinal, if multiple GPUs visible ---
    std::printf("\n=== CHECK B: device targeting by ordinal ===\n");
    if (gpu_devices.size() < 2) {
        std::printf("Only %zu GPU(s) visible in this environment -- cannot test "
                     "non-default ordinal targeting here. This does NOT confirm "
                     "correctness for multi-GPU; re-run on a multi-GPU node before "
                     "trusting ordinal-based device selection in Context().\n",
                     gpu_devices.size());
    } else {
        int target_ordinal = 1;
        sycl::device* matched = nullptr;
        for (auto& d : gpu_devices) {
            if (d.get_native_id() == target_ordinal) { matched = &d; break; }
        }
        if (!matched) {
            std::printf("FAIL -- no device found matching native_id=%d despite %zu GPUs visible.\n",
                         target_ordinal, gpu_devices.size());
        } else {
            sycl::queue q(*matched);
            int* marker = nullptr;
            check_cuda(cudaMallocManaged(&marker, sizeof(int)), "alloc for Check B");
            *marker = -1;

            int* out = marker;
            q.submit([&](sycl::handler& cgh) {
                cgh.parallel_for<SlowishKernel>(sycl::range<1>(1), [=](sycl::id<1>) {
                    int dev_id = -1;
                    cudaGetDevice(&dev_id);
                    out[0] = dev_id;
                });
            });
            q.wait();
            check_cuda(cudaDeviceSynchronize(), "sync for Check B");

            std::printf("Requested native_id=%d, kernel actually ran on cudaGetDevice()=%d: %s\n",
                         target_ordinal, *marker,
                         (*marker == target_ordinal) ? "PASS -- correctly targeted" : "FAIL -- silently ran on wrong device");
            cudaFree(marker);
        }
    }

    return 0;
}
