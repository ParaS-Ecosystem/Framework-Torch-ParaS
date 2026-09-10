// repro_atomic_ref_fetch_add.cpp
//
// Direct test: is sycl::atomic_ref<T>::fetch_add actually atomic on real
// concurrent GPU hardware, or does it silently lose updates under a race?
//
// Two independent checks, because a lucky race can sometimes still produce
// the right final SUM even with lost updates (e.g. compensating errors are
// astronomically unlikely at this N, but the return-value check below rules
// it out structurally rather than statistically):
//
//   CHECK A (count): N threads each fetch_add(1) into one shared counter.
//                    If truly atomic, final value == N, no exceptions.
//                    If not atomic, some updates are lost -> final value < N.
//
//   CHECK B (linearizability): each thread records the value fetch_add
//                    RETURNS (the pre-increment value) into slot[global_id].
//                    If every increment is properly atomic and serialized,
//                    the N returned values must be exactly the set
//                    {0, 1, 2, ..., N-1} with no duplicates and no gaps --
//                    this is a much stronger property than the sum matching,
//                    since two threads could both see the same "old" value
//                    only if the increment wasn't truly atomic, even in
//                    cases where the final count happens to look right.
//
// Build (reuse the exact flags that worked for the queue::parallel_for repro):
//
//   parascc \
//     -I$PARAS_HOME/include -I$CUDA/include \
//     -L$CUDA/lib64 -L"$STDCXX_DIR" \
//     -lcudart -lpthread \
//     repro_atomic_ref_fetch_add.cpp -o repro_atomic_ref_fetch_add
//
//   LD_LIBRARY_PATH="$STDCXX_DIR":$CUDA/lib64:$LD_LIBRARY_PATH \
//     ./repro_atomic_ref_fetch_add

#include <algorithm>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include "sycl/sycl.hpp"
#include "sycl/atomic_ref.hpp"

constexpr int N = 200000; // large enough to make a real race likely to manifest

struct AtomicAddKernel;

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

    int* counter = nullptr;
    int* returned_old_values = nullptr;
    check_cuda(cudaMallocManaged(&counter, sizeof(int)), "cudaMallocManaged(counter)");
    check_cuda(cudaMallocManaged(&returned_old_values, sizeof(int) * N), "cudaMallocManaged(returned_old_values)");
    *counter = 0;
    for (int i = 0; i < N; ++i) returned_old_values[i] = -1; // sentinel: "never written"

    {
        int* ctr = counter;
        int* out = returned_old_values;
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<AtomicAddKernel>(sycl::range<1>(N), [=](sycl::id<1> idx) {
                sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(*ctr);
                int old_value = ref.fetch_add(1);
                out[idx[0]] = old_value;
            });
        });
        q.wait();
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }

    std::printf("\n=== CHECK A: final count ===\n");
    std::printf("counter = %d (expected %d)\n", *counter, N);
    bool check_a_pass = (*counter == N);

    std::printf("\n=== CHECK B: returned old-value uniqueness/coverage ===\n");
    std::vector<int> sorted_vals(returned_old_values, returned_old_values + N);
    std::sort(sorted_vals.begin(), sorted_vals.end());

    bool check_b_pass = true;
    int first_bad_index = -1;
    for (int i = 0; i < N; ++i) {
        if (sorted_vals[i] != i) {
            check_b_pass = false;
            first_bad_index = i;
            break;
        }
    }
    if (check_b_pass) {
        std::printf("All %d returned values form the exact set {0..%d} with no duplicates/gaps.\n", N, N - 1);
    } else {
        std::printf("MISMATCH at sorted position %d: expected %d, found %d "
                     "(duplicate or missing pre-increment value -- fetch_add is not "
                     "properly linearizable).\n", first_bad_index, first_bad_index, sorted_vals[first_bad_index]);
    }

    std::printf("\n=== VERDICT ===\n");
    if (check_a_pass && check_b_pass) {
        std::printf("CONFIRMED: sycl::atomic_ref::fetch_add is genuinely atomic under "
                     "real concurrency on this build -- both the final count and the "
                     "full set of returned pre-increment values are exactly correct.\n");
    } else if (!check_a_pass) {
        std::printf("REFUTED: fetch_add lost updates under real concurrency -- final "
                     "count %d != expected %d. This is NOT atomic on this build "
                     "regardless of what atomic_ref.hpp's source implies.\n", *counter, N);
    } else {
        std::printf("REFUTED: final count matched by coincidence, but the "
                     "linearizability check caught a real race (see CHECK B above). "
                     "Do not trust fetch_add for anything where correctness depends "
                     "on unique, ordered increments (e.g. work assignment via atomic "
                     "counters), even though naive count-based tests would have "
                     "looked fine.\n");
    }

    cudaFree(counter);
    cudaFree(returned_old_values);
    return (check_a_pass && check_b_pass) ? 0 : 1;
}
