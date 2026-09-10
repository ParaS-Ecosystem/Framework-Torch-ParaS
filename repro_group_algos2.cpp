// repro_group_algos2.cpp — isolate work-group vs sub-group reduce_over_group
#include <sycl/sycl.hpp>
#include <cstdio>

template <typename F>
class ParasKernelKey;

int main() {
    sycl::queue q(sycl::gpu_selector_v);
    constexpr size_t N = 32;

    int* wg_out = sycl::malloc_shared<int>(N, q);
    int* sg_out = sycl::malloc_shared<int>(N, q);

    struct KWg {};
    struct KSg {};

    // work-group level reduce_over_group(group<1>, ...)
    q.submit([&](sycl::handler& h) {
        h.parallel_for<ParasKernelKey<KWg>>(
            sycl::nd_range<1>(sycl::range<1>(N), sycl::range<1>(N)),
            [=](sycl::nd_item<1> item) {
                int v = static_cast<int>(item.get_local_id(0)) + 1;
                int total = sycl::reduce_over_group(item.get_group(), v, sycl::plus<int>());
                wg_out[item.get_global_id(0)] = total;
            });
    }).wait();

    // sub-group level reduce_over_group(sub_group, ...)
    q.submit([&](sycl::handler& h) {
        h.parallel_for<ParasKernelKey<KSg>>(
            sycl::nd_range<1>(sycl::range<1>(N), sycl::range<1>(N)),
            [=](sycl::nd_item<1> item) {
                int v = static_cast<int>(item.get_local_id(0)) + 1;
                sycl::sub_group sg = item.get_sub_group();
                int total = sycl::reduce_over_group(sg, v, sycl::plus<int>());
                sg_out[item.get_global_id(0)] = total;
            });
    }).wait();

    std::printf("work-group reduce_over_group: got %d, expect 528\n", wg_out[0]);
    std::printf("sub-group  reduce_over_group: got %d, expect 528\n", sg_out[0]);

    sycl::free(wg_out, q);
    sycl::free(sg_out, q);
    return 0;
}
