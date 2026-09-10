// repro_ndrange.cpp
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstddef>

template <typename F>
class ParasKernelKey;

int main() {
    sycl::queue q(sycl::gpu_selector_v);

    constexpr size_t N = 256;
    constexpr size_t WG = 64;  // work-group size

    int* data = sycl::malloc_shared<int>(N, q);
    int* block_sums = sycl::malloc_shared<int>(N / WG, q);
    for (size_t i = 0; i < N; ++i) data[i] = 1;

    struct Kernel {};
    q.template submit([&](sycl::handler& h) {
        sycl::local_accessor<int, 1> local_mem(sycl::range<1>(WG), h);
        h.parallel_for<ParasKernelKey<Kernel>>(
            sycl::nd_range<1>(sycl::range<1>(N), sycl::range<1>(WG)),
            [=](sycl::nd_item<1> item) {
                size_t local_id = item.get_local_id(0);
                size_t global_id = item.get_global_id(0);
                size_t group_id = item.get_group(0);

                local_mem[local_id] = data[global_id];
                item.barrier(sycl::access::fence_space::local_space);

                // naive tree reduction within the work-group
                for (size_t stride = WG / 2; stride > 0; stride /= 2) {
                    if (local_id < stride)
                        local_mem[local_id] += local_mem[local_id + stride];
                    item.barrier(sycl::access::fence_space::local_space);
                }

                if (local_id == 0) block_sums[group_id] = local_mem[0];
            });
    });
    q.wait();

    int total = 0;
    for (size_t i = 0; i < N / WG; ++i) total += block_sums[i];
    std::printf("total = %d (expect %zu)\n", total, N);

    sycl::free(data, q);
    sycl::free(block_sums, q);
    return (total == static_cast<int>(N)) ? 0 : 1;
}
