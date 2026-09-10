// repro_sub_group_reduce.cpp
//
// Direct test: does sycl::reduce_over_group(sub_group, value, op) actually
// perform a real warp-level shuffle reduction, or does it silently return
// the input unchanged (the failure mode we already confirmed exists on the
// HIP path via paras_shfl_down's #else branch in gpu_utilities.hpp -- this
// test checks whether the SAME kind of silent pass-through happens on CUDA
// too, rather than assuming the CUDA branch is fine just because the source
// text looks complete).
//
// Design: launch N threads, N a multiple of 32 (warp size, confirmed in
// sub_group.hpp: `warp_size = 32`). Each thread's value = (lane_id + 1),
// i.e. within every warp the 32 threads hold values 1..32 in some order
// determined by lane id. The correct sum for EVERY warp is the closed-form
// 1+2+...+32 = 32*33/2 = 528, independent of which warp it is.
//
// If reduce_over_group works: every thread's output == 528.
// If it silently no-ops (returns input unchanged): output[i] == value[i]
//   (i.e. just the lane's own 1..32 value, never actually reduced) --
//   this exact pattern is checked for explicitly below, so a silent-passthrough
//   bug is distinguished from other kinds of wrongness, not just flagged "wrong".
//
// Build:
//   parascc \
//     -I$PARAS_HOME/include -I$CUDA/include \
//     -L$CUDA/lib64 -L"$STDCXX_DIR" \
//     -lcudart -lpthread \
//     repro_sub_group_reduce.cpp -o repro_sub_group_reduce
//
//   LD_LIBRARY_PATH="$STDCXX_DIR":$CUDA/lib64:$LD_LIBRARY_PATH \
//     ./repro_sub_group_reduce

#include <cstdio>
#include <cuda_runtime.h>
#include "sycl/sycl.hpp"
#include "sycl/sub_group.hpp"

constexpr int WARP_SIZE   = 32;
constexpr int NUM_WARPS   = 100;
constexpr int N           = WARP_SIZE * NUM_WARPS;
constexpr int EXPECTED_SUM_PER_WARP = (WARP_SIZE * (WARP_SIZE + 1)) / 2; // 1+2+...+32 = 528

struct SubGroupReduceKernel;

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

    int* input_values = nullptr;
    int* output_sums  = nullptr;
    check_cuda(cudaMallocManaged(&input_values, sizeof(int) * N), "cudaMallocManaged(input_values)");
    check_cuda(cudaMallocManaged(&output_sums, sizeof(int) * N), "cudaMallocManaged(output_sums)");
    for (int i = 0; i < N; ++i) output_sums[i] = -1; // sentinel

    {
        int* in  = input_values;
        int* out = output_sums;
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<SubGroupReduceKernel>(sycl::range<1>(N), [=](sycl::id<1> idx) {
                std::size_t gid = idx[0];
                sycl::sub_group sg;
                // lane id within this warp, independent of which warp -- this
                // relies on the confirmed mapping (from gpu_execute_1D_kernel:
                // i = blockIdx.x*blockDim.x + threadIdx.x, blockDim.x=256, a
                // multiple of 32) that consecutive global ids fall into
                // consecutive, warp-aligned groups of 32.
                int lane = static_cast<int>(sg.get_local_linear_id());
                int value = lane + 1; // 1..32 within every warp
                in[gid] = value;

                int sum = sycl::reduce_over_group(sg, value, [](int a, int b) { return a + b; });
                out[gid] = sum;
            });
        });
        q.wait();
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }

    int correct_count      = 0;
    int passthrough_count  = 0; // output == input exactly (the "silent no-op" signature)
    int other_wrong_count  = 0;
    int first_wrong_index  = -1;

    for (int i = 0; i < N; ++i) {
        if (output_sums[i] == EXPECTED_SUM_PER_WARP) {
            ++correct_count;
        } else if (output_sums[i] == input_values[i]) {
            ++passthrough_count;
            if (first_wrong_index < 0) first_wrong_index = i;
        } else {
            ++other_wrong_count;
            if (first_wrong_index < 0) first_wrong_index = i;
        }
    }

    std::printf("\n=== RESULTS (N=%d threads, %d warps, expected sum/warp = %d) ===\n",
                N, NUM_WARPS, EXPECTED_SUM_PER_WARP);
    std::printf("Correct (== %d):              %d / %d\n", EXPECTED_SUM_PER_WARP, correct_count, N);
    std::printf("Silent passthrough (== input): %d / %d\n", passthrough_count, N);
    std::printf("Other wrong values:            %d / %d\n", other_wrong_count, N);
    if (first_wrong_index >= 0) {
        std::printf("First wrong index %d: input=%d output=%d\n",
                     first_wrong_index, input_values[first_wrong_index], output_sums[first_wrong_index]);
    }

    std::printf("\n=== VERDICT ===\n");
    if (correct_count == N) {
        std::printf("CONFIRMED: sycl::reduce_over_group / sub_group shuffle reduction "
                     "is genuinely correct on this build -- every one of %d threads "
                     "across %d warps got the exact expected per-warp sum.\n", N, NUM_WARPS);
    } else if (passthrough_count == N) {
        std::printf("REFUTED, SILENT PASSTHROUGH: reduce_over_group returned the input "
                     "value UNCHANGED for every thread -- this is the exact 'no shuffle "
                     "happened' failure mode already confirmed on the HIP path "
                     "(paras_shfl_down's #else branch). This build's CUDA path is doing "
                     "the same silent no-op. Do NOT use this for any reduction where "
                     "correctness matters -- it would look like it compiled and ran fine "
                     "while computing nothing.\n");
    } else {
        std::printf("REFUTED, OTHER WRONGNESS: neither uniformly correct nor uniformly "
                     "passthrough -- %d correct, %d passthrough, %d other. This looks like "
                     "a genuine partial/racy implementation rather than a clean no-op. "
                     "Needs its own follow-up investigation before use.\n",
                     correct_count, passthrough_count, other_wrong_count);
    }

    cudaFree(input_values);
    cudaFree(output_sums);
    return (correct_count == N) ? 0 : 1;
}
