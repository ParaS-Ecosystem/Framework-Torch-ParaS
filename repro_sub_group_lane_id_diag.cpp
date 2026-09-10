// repro_sub_group_lane_id_diag.cpp
//
// Isolates ONE question, separate from reduce_over_group entirely:
// does sycl::sub_group::get_local_linear_id() return real, varying,
// per-thread lane IDs (0..31 within each warp), or is it also silently
// wrong (e.g. always 0)?
//
// This determines where the actual bug lives:
//   - If lane IDs come back correct (0..31, varying) -> the bug is
//     specifically in reduce_over_group's shuffle logic.
//   - If lane IDs come back wrong (e.g. all 0, or garbage) -> the bug is
//     earlier, in lane-id detection itself, and reduce_over_group failing
//     is a downstream symptom, not the root cause.
//
// Build:
//   parascc \
//     -I$PARAS_HOME/include -I$CUDA/include \
//     -L$CUDA/lib64 -L"$STDCXX_DIR" \
//     -lcudart -lpthread \
//     repro_sub_group_lane_id_diag.cpp -o repro_sub_group_lane_id_diag
//
//   LD_LIBRARY_PATH="$STDCXX_DIR":$CUDA/lib64:$LD_LIBRARY_PATH \
//     ./repro_sub_group_lane_id_diag

#include <cstdio>
#include <cuda_runtime.h>
#include "sycl/sycl.hpp"
#include "sycl/sub_group.hpp"

constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = 4;   // small on purpose -- we're going to print every value
constexpr int N         = WARP_SIZE * NUM_WARPS;

struct LaneIdKernel;

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

    int* lane_ids = nullptr;
    check_cuda(cudaMallocManaged(&lane_ids, sizeof(int) * N), "cudaMallocManaged(lane_ids)");
    for (int i = 0; i < N; ++i) lane_ids[i] = -999; // sentinel: "kernel never wrote here"

    {
        int* out = lane_ids;
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<LaneIdKernel>(sycl::range<1>(N), [=](sycl::id<1> idx) {
                std::size_t gid = idx[0];
                sycl::sub_group sg;
                out[gid] = static_cast<int>(sg.get_local_linear_id());
            });
        });
        q.wait();
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }

    std::printf("\n=== RAW LANE IDS (grouped by warp, expect 0..31 in each row) ===\n");
    for (int w = 0; w < NUM_WARPS; ++w) {
        std::printf("warp %d: ", w);
        for (int lane = 0; lane < WARP_SIZE; ++lane) {
            std::printf("%d ", lane_ids[w * WARP_SIZE + lane]);
        }
        std::printf("\n");
    }

    bool all_zero = true;
    bool all_correct = true;
    for (int w = 0; w < NUM_WARPS; ++w) {
        for (int lane = 0; lane < WARP_SIZE; ++lane) {
            int v = lane_ids[w * WARP_SIZE + lane];
            if (v != 0) all_zero = false;
            if (v != lane) all_correct = false;
        }
    }

    std::printf("\n=== VERDICT ===\n");
    if (all_correct) {
        std::printf("CONFIRMED: get_local_linear_id() returns correct, varying lane IDs "
                     "(0..31) in every warp. The bug is specifically in reduce_over_group's "
                     "shuffle logic, not in lane-id detection.\n");
    } else if (all_zero) {
        std::printf("REFUTED: every thread reports lane id 0, regardless of its real "
                     "position in the warp. Lane-id detection itself is broken -- "
                     "reduce_over_group failing is a downstream symptom of THIS bug, "
                     "not a separate issue in the shuffle logic.\n");
    } else {
        std::printf("REFUTED, OTHER PATTERN: lane IDs are neither correctly varying "
                     "0..31 nor uniformly 0 -- see the raw printout above for the actual "
                     "pattern. This needs manual inspection before concluding anything.\n");
    }

    cudaFree(lane_ids);
    return all_correct ? 0 : 1;
}
