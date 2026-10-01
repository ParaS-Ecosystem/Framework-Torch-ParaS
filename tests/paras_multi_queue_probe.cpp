// Standalone SYCL probe for queue operations across four NVIDIA devices.
#include <sycl/sycl.hpp>

#include <cstdio>
#include <exception>
#include <stdexcept>
#include <vector>

#if defined(__CUDACC__) || defined(__CUDA__)
#define PROBE_HOST_DEVICE __host__ __device__
#else
#define PROBE_HOST_DEVICE
#endif

class MultiQueueFirstKernel;
class MultiQueueSecondKernel;
class MultiQueueRestoreKernel;

template <typename F>
bool report_operation(const char* label, F operation) {
    try {
        operation();
        std::printf("PASS: %s\n", label);
        return true;
    } catch (const std::exception& error) {
        std::printf("ERROR: %s: %s\n", label, error.what());
        return false;
    } catch (...) {
        std::printf("ERROR: %s: unknown exception\n", label);
        return false;
    }
}

bool check_values(const char* label, const int* values, std::size_t n,
                  int first, bool increasing) {
    for (std::size_t i = 0; i < n; ++i) {
        const int expected = first + (increasing ? static_cast<int>(i) : 0);
        if (values[i] != expected) {
            std::printf("ERROR: %s: index %zu expected %d, got %d\n",
                        label, i, expected, values[i]);
            return false;
        }
    }
    std::printf("PASS: %s\n", label);
    return true;
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devices.size() != 4) {
        std::printf("SKIP: this probe requires exactly four GPU devices; found %zu\n",
                    devices.size());
        return 77;
    }

    sycl::queue q0(devices.at(0), sycl::property::queue::in_order{});
    sycl::queue q1(devices.at(1), sycl::property::queue::in_order{});
    sycl::queue q2(devices.at(2), sycl::property::queue::in_order{});
    sycl::queue q3(devices.at(3), sycl::property::queue::in_order{});
    std::printf("Constructed queues for native GPU devices %d, %d, %d, %d\n",
                q0.get_device().get_native_id(), q1.get_device().get_native_id(),
                q2.get_device().get_native_id(), q3.get_device().get_native_id());

    constexpr std::size_t n = 10;
    constexpr std::size_t bytes = n * sizeof(int);
    int* p0 = sycl::malloc_shared<int>(n, q0);
    int* p1 = sycl::malloc_shared<int>(n, q1);
    if (!p0 || !p1) {
        std::printf("ERROR: shared allocation returned null\n");
        if (p0) sycl::free(p0, q0);
        if (p1) sycl::free(p1, q1);
        return 2;
    }
    for (std::size_t i = 0; i < n; ++i) {
        p0[i] = -1;
        p1[i] = -1;
    }
    std::vector<int> host(n, -1);
    unsigned failures = 0;

    // Constructing q3 leaves its device current. Each earlier queue must
    // establish its own device before performing a memory operation or wait.
    if (!report_operation("q0.memset after constructing q3", [&] {
            q0.memset(p0, 0, bytes);
        })) ++failures;
    if (!report_operation("q0.memcpy after constructing q3", [&] {
            q0.memcpy(host.data(), p0, bytes);
        })) ++failures;
    if (!report_operation("q0.wait after constructing q3", [&] {
            q0.wait();
        })) ++failures;
    if (!check_values("q0 zero-fill result", host.data(), n, 0, false))
        ++failures;

    if (!report_operation("q0.submit nonzero range kernel", [&] {
            q0.submit([&](sycl::handler& cgh) {
                cgh.parallel_for<MultiQueueFirstKernel>(
                    sycl::range<1>(n),
                    [=] PROBE_HOST_DEVICE (sycl::id<1> i) {
                        p0[i[0]] = static_cast<int>(i[0]) + 7;
                    });
            });
            q0.wait();
        })) ++failures;
    if (!check_values("q0 kernel wrote all indices", p0, n, 7, true))
        ++failures;

    if (!report_operation("q1.memcpy managed pointers after q0 kernel", [&] {
            q1.memcpy(p1, p0, bytes);
        })) ++failures;
    if (!report_operation("q1.wait after q0 kernel", [&] {
            q1.wait();
        })) ++failures;
    if (!check_values("cross-device copy result", p1, n, 7, true))
        ++failures;

    if (!report_operation("q1.submit switches execution to second GPU", [&] {
            q1.submit([&](sycl::handler& cgh) {
                cgh.parallel_for<MultiQueueSecondKernel>(
                    sycl::range<1>(n),
                    [=] PROBE_HOST_DEVICE (sycl::id<1> i) {
                        p1[i[0]] = static_cast<int>(i[0]) + 17;
                    });
            });
            q1.wait();
        })) ++failures;
    if (!check_values("q1 kernel wrote all indices", p1, n, 17, true))
        ++failures;

    if (!report_operation("q0.memset after q1 kernel", [&] {
            q0.memset(p0, 0, bytes);
        })) ++failures;
    if (!report_operation("q0.memcpy after q1 kernel", [&] {
            q0.memcpy(host.data(), p0, bytes);
        })) ++failures;
    if (!report_operation("q0.wait after q1 kernel", [&] {
            q0.wait();
        })) ++failures;

    // A kernel explicitly restores execution to q0's device. Repeating the
    // memory checks distinguishes device-selection errors from queue lowering.
    if (!report_operation("q0.submit restores first GPU", [&] {
            q0.submit([&](sycl::handler& cgh) {
                cgh.parallel_for<MultiQueueRestoreKernel>(
                    sycl::range<1>(n),
                    [=] PROBE_HOST_DEVICE (sycl::id<1> i) {
                        p0[i[0]] = static_cast<int>(i[0]) + 27;
                    });
            });
            q0.wait();
        })) ++failures;
    if (!report_operation("q0.memcpy after its own kernel", [&] {
            q0.memcpy(host.data(), p0, bytes);
        })) ++failures;
    if (!check_values("q0 restored-device copy result", host.data(), n, 27, true))
        ++failures;
    if (!report_operation("q0.memset after its own kernel", [&] {
            q0.memset(p0, 0, bytes);
        })) ++failures;
    if (!report_operation("q0.wait after its own kernel", [&] {
            q0.wait();
        })) ++failures;
    if (!check_values("q0 restored-device zero-fill result", p0, n, 0, false))
        ++failures;

    // The explicit selector form is an optional compile-time architecture
    // check: enable it with -DPROBE_EXPLICIT_CPU_SELECTOR. The default device
    // form reaches the runtime mixed CPU/GPU check in a GPU translation unit.
    auto cpus = sycl::device::get_devices(sycl::info::device_type::cpu);
    if (!cpus.empty()) {
#if defined(PROBE_EXPLICIT_CPU_SELECTOR)
        sycl::queue q_cpu(sycl::cpu_selector_v,
                          sycl::property::queue::in_order{});
#else
        sycl::queue q_cpu(cpus.at(0), sycl::property::queue::in_order{});
#endif
        int* cpu_ptr = nullptr;
        const bool allocated = report_operation("mixed CPU/GPU shared allocation", [&] {
            cpu_ptr = sycl::malloc_shared<int>(n, q_cpu);
            if (!cpu_ptr) throw std::runtime_error("shared allocation returned null");
        });
        bool filled = false;
        if (allocated) {
            filled = report_operation("mixed CPU/GPU q_cpu.memset", [&] {
                q_cpu.memset(cpu_ptr, 0, bytes);
            });
        }
        const bool waited = report_operation("mixed CPU/GPU q_cpu.wait", [&] {
            q_cpu.wait();
        });
        std::printf("Mixed CPU/GPU queue capability: %s (reported separately from GPU checks)\n",
                    allocated && filled && waited ? "SUPPORTED" : "UNSUPPORTED");
        if (cpu_ptr) {
            report_operation("mixed CPU/GPU shared free", [&] {
                sycl::free(cpu_ptr, q_cpu);
            });
        }
    } else {
        std::printf("SKIP: mixed CPU/GPU queue check found no CPU device\n");
    }

    if (!report_operation("q0 shared free", [&] { sycl::free(p0, q0); }))
        ++failures;
    if (!report_operation("q1 shared free", [&] { sycl::free(p1, q1); }))
        ++failures;
    std::printf("GPU queue checks: %u failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
