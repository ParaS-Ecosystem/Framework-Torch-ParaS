// Standalone standard-SYCL probe for the installed ParaS compiler.
#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>
#include <vector>

class QueueProbeKernel;

int main(int argc, char** argv) {
#if defined(PROBE_GPU)
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
#else
    auto devices = sycl::device::get_devices(sycl::info::device_type::cpu);
#endif
    if (devices.empty()) return 77;
    const std::size_t index = argc > 1 ? static_cast<std::size_t>(std::atoi(argv[1])) : 0;
    sycl::queue q(devices.at(index), sycl::property::queue::in_order{});
    constexpr std::size_t n = 10;
    int* p = sycl::malloc_shared<int>(n, q);
    if (!p) return 2;
    q.memset(p, 0, n * sizeof(int));
    q.wait();
    for (std::size_t i = 0; i < n; ++i) if (p[i] != 0) return 3;
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<QueueProbeKernel>(sycl::range<1>(n), [=](sycl::id<1> i) {
            p[i[0]] = static_cast<int>(i[0]) + 7;
        });
    });
    q.wait();
    std::vector<int> out(n);
    q.memcpy(out.data(), p, n * sizeof(int));
    q.wait();
    for (std::size_t i = 0; i < n; ++i) if (out[i] != static_cast<int>(i) + 7) return 4;
    std::printf("PASS: allocation, memset, kernel, memcpy, free on device %d\n",
                q.get_device().get_native_id());
    sycl::free(p, q);
    return 0;
}
