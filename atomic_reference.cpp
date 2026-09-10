#include <sycl/sycl.hpp>

#include <cstdint>
#include <iostream>

template <typename T, typename KernelName>
void test_atomic_fetch_add(sycl::queue& q, const char* name) {
    T* data = sycl::malloc_shared<T>(1, q);

    if (!data) {
        std::cerr << name << ": allocation failed\n";
        return;
    }

    *data = T{0};

    constexpr int iterations = 1024;

    try {
        q.parallel_for<KernelName>(
            sycl::range<1>(iterations),
            [=](sycl::id<1>) {
                sycl::atomic_ref<
                    T,
                    sycl::memory_order::relaxed,
                    sycl::memory_scope::device,
                    sycl::access::address_space::generic_space>
                    atomic(*data);

                atomic.fetch_add(T{1});
            });

        q.wait_and_throw();

        std::cout << name
                  << ": actual=" << static_cast<long double>(*data)
                  << ", expected=" << iterations
                  << "\n";

    } catch (const sycl::exception& e) {
        std::cerr << name << ": SYCL error: "
                  << e.what() << "\n";
    }

    sycl::free(data, q);
}


// Unique kernel names for each datatype.
class AtomicInt8;
class AtomicUInt8;
class AtomicInt16;
class AtomicUInt16;
class AtomicInt32;
class AtomicUInt32;
class AtomicInt64;
class AtomicUInt64;
class AtomicFloat;
class AtomicDouble;


int main() {
    sycl::queue q{sycl::gpu_selector_v};

    std::cout << "Device: "
              << q.get_device().get_info<sycl::info::device::name>()
              << "\n\n";

    test_atomic_fetch_add<int8_t, AtomicInt8>(
        q, "int8");

    test_atomic_fetch_add<uint8_t, AtomicUInt8>(
        q, "uint8");

    test_atomic_fetch_add<int16_t, AtomicInt16>(
        q, "int16");

    test_atomic_fetch_add<uint16_t, AtomicUInt16>(
        q, "uint16");

    test_atomic_fetch_add<int32_t, AtomicInt32>(
        q, "int32");

    test_atomic_fetch_add<uint32_t, AtomicUInt32>(
        q, "uint32");

    test_atomic_fetch_add<int64_t, AtomicInt64>(
        q, "int64");

    test_atomic_fetch_add<uint64_t, AtomicUInt64>(
        q, "uint64");

    test_atomic_fetch_add<float, AtomicFloat>(
        q, "float");

    test_atomic_fetch_add<double, AtomicDouble>(
        q, "double");

    return 0;
}
