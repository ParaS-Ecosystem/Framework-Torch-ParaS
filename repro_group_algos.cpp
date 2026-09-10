// repro_group_algos.cpp
#include <sycl/sycl.hpp>
#include <cstdio>

template <typename F>
class ParasKernelKey;

int main() {
    sycl::queue q(sycl::gpu_selector_v);
    constexpr size_t N = 32;

    int* reduce_out = sycl::malloc_shared<int>(N, q);
    int* any_out    = sycl::malloc_shared<int>(N, q);
    int* shift_out  = sycl::malloc_shared<int>(N, q);

    struct KReduce {};
    struct KAny {};
    struct KShift {};

    // reduce_over_group: every work-item in the group should see the same total
    q.submit([&](sycl::handler& h) {
        h.parallel_for<ParasKernelKey<KReduce>>(
            sycl::nd_range<1>(sycl::range<1>(N), sycl::range<1>(N)),
            [=](sycl::nd_item<1> item) {
                int v = static_cast<int>(item.get_local_id(0)) + 1; // 1..N
                int total = sycl::reduce_over_group(item.get_group(), v, sycl::plus<int>());
                reduce_out[item.get_global_id(0)] = total;
            });
    }).wait();

    // any_of_group: true iff any work-item's local_id == N-1
    q.submit([&](sycl::handler& h) {
        h.parallel_for<ParasKernelKey<KAny>>(
            sycl::nd_range<1>(sycl::range<1>(N), sycl::range<1>(N)),
            [=](sycl::nd_item<1> item) {
                bool pred = (item.get_local_id(0) == N - 1);
                bool result = sycl::any_of_group(item.get_group(), pred);
                any_out[item.get_global_id(0)] = result ? 1 : 0;
            });
    }).wait();

    // shift_group_left within a sub-group (warp-level shuffle)
    q.submit([&](sycl::handler& h) {
        h.parallel_for<ParasKernelKey<KShift>>(
            sycl::nd_range<1>(sycl::range<1>(N), sycl::range<1>(32)),
            [=](sycl::nd_item<1> item) {
                sycl::sub_group sg = item.get_sub_group();
                int v = static_cast<int>(sg.get_local_id()[0]);
                int shifted = sycl::shift_group_left(sg, v, 1);
                shift_out[item.get_global_id(0)] = shifted;
            });
    }).wait();

    const int expected_sum = N * (N + 1) / 2;
    bool reduce_ok = true, any_ok = true;
    for (size_t i = 0; i < N; ++i) {
        if (reduce_out[i] != expected_sum) reduce_ok = false;
        if (any_out[i] != 1) any_ok = false;
    }
    std::printf("reduce_over_group: %s (got %d, expect %d)\n",
                reduce_ok ? "OK" : "WRONG", reduce_out[0], expected_sum);
    std::printf("any_of_group: %s\n", any_ok ? "OK" : "WRONG");
    for (size_t i = 0; i < 5; ++i)
        std::printf("shift_group_left[%zu] = %d\n", i, shift_out[i]);

    sycl::free(reduce_out, q);
    sycl::free(any_out, q);
    sycl::free(shift_out, q);
    return (reduce_ok && any_ok) ? 0 : 1;
}
