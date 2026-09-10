// repro_queue_parallel_for_stub.cpp
//
// Direct test of a single, narrow claim: does sycl::queue::parallel_for
// (the queue-level shorthand — NOT queue.submit([](handler&){ ... }))
// actually invoke its kernel functor on this compiler build?
//
// Method: two counters in managed memory.
//   counter_via_queue    -- incremented by a kernel launched via q.parallel_for(...) directly
//   counter_via_handler  -- incremented by the SAME kernel logic launched via
//                           q.submit([](handler& cgh){ cgh.parallel_for(...); })
//
// counter_via_handler is the positive control: if IT also comes back 0,
// something is wrong with the test itself (wrong device, launch never
// reached, etc.), not with queue::parallel_for specifically. If
// counter_via_handler == N but counter_via_queue == 0, that's a clean,
// unambiguous confirmation that queue::parallel_for is a no-op stub on
// this build, with a same-run control ruling out test-harness error.
//
// Build (adjust include/lib paths and any extra flags to match however
// paras_compat.cpp is actually compiled in the real build -- check
// CMakeLists.txt / setup.py for the exact parascc invocation used there,
// since this repro needs to match that environment, not guess at it):
//
//   $PARAS_HOME/bin/parascc \
//     -I$PARAS_HOME/include \
//     -L$PARAS_HOME/lib \
//     -lcudart \
//     repro_queue_parallel_for_stub.cpp -o repro_queue_parallel_for_stub
//
//   LD_LIBRARY_PATH=$PARAS_HOME/lib:$LD_LIBRARY_PATH ./repro_queue_parallel_for_stub
//
// If parascc rejects any flag above, run it with -v (or check the real
// build's compile_commands.json / verbose build log) to get the exact
// flags this compiler expects, then substitute them here -- don't guess
// a second time, copy the working invocation directly.

#include <cstdio>
#include <cuda_runtime.h>
#include "sycl/sycl.hpp"
#include "sycl/atomic_ref.hpp"

constexpr int N = 1000;

// Kernel name tags -- queue::parallel_for and handler::parallel_for both
// require an explicit KernelName template argument (it can't be deduced),
// per the signatures seen in queue.hpp / handler.hpp.
struct QueuePathKernel;
struct HandlerPathKernel;

static void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(err));
        std::exit(1);
    }
}

int main() {
    // --- Device selection: first GPU device, mirroring the ordinal-matching
    // pattern from the real Queue::init patch (not gpu_selector_v directly,
    // to stay consistent, though for this single-GPU smoke test either works).
    auto gpu_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (gpu_devices.empty()) {
        std::fprintf(stderr, "No GPU devices found via sycl::device::get_devices — "
                              "cannot run this test.\n");
        return 1;
    }
    sycl::device dev = gpu_devices.front();
    std::printf("Using device: native_id=%d\n", dev.get_native_id());

    sycl::queue q(dev);
    std::printf("queue bound device is_gpu(): %s\n", q.get_device().is_gpu() ? "true" : "false");

    // --- Shared counters, plain cudaMallocManaged so allocation itself is
    // not something under test here -- keep this repro narrowly about
    // parallel_for, nothing else.
    int* counter_via_queue   = nullptr;
    int* counter_via_handler = nullptr;
    check_cuda(cudaMallocManaged(&counter_via_queue, sizeof(int)), "cudaMallocManaged(queue counter)");
    check_cuda(cudaMallocManaged(&counter_via_handler, sizeof(int)), "cudaMallocManaged(handler counter)");
    *counter_via_queue   = 0;
    *counter_via_handler = 0;
    std::printf("counter_via_queue address:   %p\n", (void*)counter_via_queue);
    std::printf("counter_via_handler address: %p\n", (void*)counter_via_handler);

    // --- Path A: queue::parallel_for directly (the claim under test)
    {
        int* ctr = counter_via_queue;
        q.parallel_for<QueuePathKernel>(sycl::range<1>(N), [=](sycl::id<1> idx) {
            (void)idx;
            // sycl::atomic_ref instead of raw atomicAdd: confirmed real in
            // atomic_ref.hpp (delegates to atomicAdd internally under
            // PARAS_GPU_BACKEND), and avoids needing --cuda-gpu-arch just to
            // make the raw CUDA builtin visible to clang's CUDA frontend.
            sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(*ctr);
            ref.fetch_add(1);
        });
        // queue::wait() IS confirmed real (delegates to gpu_pool_->wait()),
        // so call it regardless -- if queue::parallel_for never touched
        // gpu_pool_ (as its empty body suggests), this is a harmless no-op;
        // if it somehow did dispatch work, this ensures it's finished
        // before we read the counter, so either way the read below is fair.
        q.wait();
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after queue path");
    }

    // --- Path B: handler::parallel_for via submit() (positive control --
    // confirmed real via handler_impl.hpp in prior investigation)
    {
        int* ctr = counter_via_handler;
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<HandlerPathKernel>(sycl::range<1>(N), [=](sycl::id<1> idx) {
                (void)idx;
                sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(*ctr);
                ref.fetch_add(1);
            });
        });
        q.wait();
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after handler path");
    }

    std::printf("\n=== RESULTS (expected N=%d if the kernel actually ran) ===\n", N);
    std::printf("counter_via_queue   (queue.parallel_for directly) = %d\n", *counter_via_queue);
    std::printf("counter_via_handler (submit + handler.parallel_for) = %d\n", *counter_via_handler);

    bool queue_path_ran   = (*counter_via_queue == N);
    bool handler_path_ran = (*counter_via_handler == N);

    std::printf("\n=== VERDICT ===\n");
    if (!handler_path_ran) {
        std::printf("INCONCLUSIVE: the positive control (handler path) did NOT reach "
                     "the expected count either (%d != %d). Something about this test's "
                     "environment/device/launch is not working as assumed -- this does "
                     "NOT confirm or deny the queue::parallel_for claim. Check device "
                     "selection and build flags before drawing any conclusion.\n",
                     *counter_via_handler, N);
    } else if (!queue_path_ran) {
        std::printf("CONFIRMED: queue::parallel_for did NOT execute its kernel "
                     "(counter stayed at %d instead of reaching %d), while the "
                     "handler path in the SAME run correctly reached %d. This is a "
                     "silent no-op, not a crash -- the call compiled and returned "
                     "successfully while doing zero device work.\n",
                     *counter_via_queue, N, *counter_via_handler);
    } else {
        std::printf("queue::parallel_for DID execute its kernel correctly (%d == %d). "
                     "This would contradict the empty-body implementation read directly "
                     "from queue.hpp in prior investigation -- if you see this result, "
                     "the installed queue.hpp must differ from what was reviewed earlier "
                     "(check PARAS_HOME points to the same install, or the compiler team "
                     "shipped a fix).\n", *counter_via_queue, N);
    }

    cudaFree(counter_via_queue);
    cudaFree(counter_via_handler);
    return 0;
}
