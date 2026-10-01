#include <cassert>
#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>

// Host-only standalone test verifying atomic_add instantiation and correctness
// for all PTSYCL_INDEX_PUT_ATOMIC_SAFE_TYPES without requiring a GPU or PyTorch.

#define PTSYCL_HOST_DEVICE

namespace ptsycl {

template <typename T>
PTSYCL_HOST_DEVICE inline void atomic_add(T* address, T val) {
#if (defined(PTSYCL_BACKEND_CUDA) || defined(PTSYCL_BACKEND_HIP) || defined(PTSYCL_BACKEND_SYCL)) && (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__))
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double> ||
                  std::is_same_v<T, int> || std::is_same_v<T, unsigned int>) {
        atomicAdd(address, val);
    } else if constexpr (sizeof(T) == 8 && std::is_integral_v<T>) {
        atomicAdd(reinterpret_cast<unsigned long long*>(address), static_cast<unsigned long long>(val));
    } else if constexpr (std::is_same_v<T, bool>) {
        if (val) *address = true;
    } else {
        static_assert(!sizeof(T*), "atomic_add: unsupported dtype on device");
    }
#else
    if constexpr (std::is_same_v<T, bool>) {
        if (val) __atomic_store_n(address, true, __ATOMIC_RELAXED);
    } else if constexpr (std::is_integral_v<T>) {
        __atomic_fetch_add(address, val, __ATOMIC_RELAXED);
    } else if constexpr (std::is_floating_point_v<T>) {
        T expected;
        __atomic_load(address, &expected, __ATOMIC_RELAXED);
        T desired = expected + val;
        while (!__atomic_compare_exchange(address, &expected, &desired, true,
                                          __ATOMIC_RELAXED, __ATOMIC_RELAXED))
            desired = expected + val;
    } else {
        static_assert(!sizeof(T*), "atomic_add: unsupported dtype on host");
    }
#endif
}

} // namespace ptsycl

template <typename T>
void test_concurrent_atomic_add(T initial_val, T add_val, int num_threads, int iters_per_thread, T expected_val) {
    T target = initial_val;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&target, add_val, iters_per_thread]() {
            for (int i = 0; i < iters_per_thread; ++i) {
                ptsycl::atomic_add(&target, add_val);
            }
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    if constexpr (std::is_floating_point_v<T>) {
        assert(target == expected_val);
    } else if constexpr (std::is_same_v<T, bool>) {
        assert(target == true);
    } else {
        assert(target == expected_val);
    }
}

int main() {
    constexpr int kThreads = 16;
    constexpr int kIters = 1000;

    std::cout << "[test_atomic_add] Testing int32_t..." << std::endl;
    test_concurrent_atomic_add<int32_t>(0, 1, kThreads, kIters, kThreads * kIters);

    std::cout << "[test_atomic_add] Testing int64_t..." << std::endl;
    test_concurrent_atomic_add<int64_t>(0, 1, kThreads, kIters, static_cast<int64_t>(kThreads * kIters));

    std::cout << "[test_atomic_add] Testing float..." << std::endl;
    test_concurrent_atomic_add<float>(0.0f, 1.0f, kThreads, kIters, static_cast<float>(kThreads * kIters));

    std::cout << "[test_atomic_add] Testing double..." << std::endl;
    test_concurrent_atomic_add<double>(0.0, 1.0, kThreads, kIters, static_cast<double>(kThreads * kIters));

    std::cout << "[test_atomic_add] Testing bool..." << std::endl;
    test_concurrent_atomic_add<bool>(false, true, kThreads, kIters, true);

    std::cout << "All host atomic_add regression tests passed successfully!" << std::endl;
    return 0;
}
