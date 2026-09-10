// repro_subgroup.cpp
#include <sycl/sycl.hpp>
#include <cstdio>

template <typename F>
class ParasKernelKey;

int main() {
    sycl::queue q(sycl::gpu_selector_v);
    constexpr size_t N = 64;
    int* out = sycl::malloc_shared<int>(N, q);

    struct Kernel {};
    q.submit([&](sycl::handler& h) {
        h.parallel_for<ParasKernelKey<Kernel>>(
            sycl::nd_range<1>(sycl::range<1>(N), sycl::range<1>(32)),
            [=](sycl::nd_item<1> item) {
                sycl::sub_group sg = item.get_sub_group();
                out[item.get_global_id(0)] = static_cast<int>(sg.get_local_linear_id());
            });
    });
    q.wait();

    bool all_zero = true;
    for (size_t i = 0; i < N; ++i) {
        std::printf("out[%zu] = %d\n", i, out[i]);
        if (out[i] != 0) all_zero = false;
    }
    std::printf(all_zero ? "BUG CONFIRMED: get_local_linear_id() always returns 0\n"
                          : "get_local_linear_id() returns distinct values (working)\n");
    sycl::free(out, q);
    return 0;
}
