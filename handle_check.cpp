#include <iostream>
#include <vector>
#include <sycl/sycl.hpp>

constexpr int N = 1000;

int main() {
    try {
        // Get available GPU devices
        auto devices =
            sycl::device::get_devices(sycl::info::device_type::gpu);

        if (devices.empty()) {
            std::cerr << "No GPU device found.\n";
            return 1;
        }

        // Select the first GPU
        sycl::device dev = devices.front();
        sycl::queue q(dev);

        std::cout << "Using device: native_id="
                  << dev.get_native_id() << "\n";

        const std::size_t NBYTES = N * sizeof(int);

        // Allocate device memory using SYCL
        int* device_buf = sycl::malloc_device<int>(
            N, dev, q.get_context()
        );

        if (device_buf == nullptr) {
            std::cerr << "sycl::malloc_device failed.\n";
            return 1;
        }

        // Host source and destination buffers
        std::vector<int> host_src(N);
        std::vector<int> host_dst(N, -1);

        // Initialize source data
        for (int i = 0; i < N; ++i) {
            host_src[i] = i * 7 + 3;
        }

        bool test_a_pass = false;
        bool test_b_pass = false;

        // ============================================================
        // Test A: Host -> Device using handler::memcpy
        // ============================================================

        std::cout << "\n--- Test A: Host -> Device ---\n";

        q.submit([&](sycl::handler& cgh) {
            cgh.memcpy(
                device_buf,
                host_src.data(),
                NBYTES
            );
        });

        q.wait();

        // Read back using SYCL handler::memcpy
        std::vector<int> verify(N, -1);

        q.submit([&](sycl::handler& cgh) {
            cgh.memcpy(
                verify.data(),
                device_buf,
                NBYTES
            );
        });

        q.wait();

        test_a_pass = (verify == host_src);

        std::cout
            << "TEST A (host -> device): "
            << (test_a_pass
                    ? "PASS -- data matches exactly"
                    : "FAIL -- data mismatch")
            << "\n";


        // ============================================================
        // Test B: Device -> Host using handler::memcpy
        // ============================================================

        std::cout << "\n--- Test B: Device -> Host ---\n";

        // First copy known data to the device
        q.submit([&](sycl::handler& cgh) {
            cgh.memcpy(
                device_buf,
                host_src.data(),
                NBYTES
            );
        });

        q.wait();

        // Reset destination buffer
        std::fill(host_dst.begin(), host_dst.end(), -1);

        // Copy device -> host
        q.submit([&](sycl::handler& cgh) {
            cgh.memcpy(
                host_dst.data(),
                device_buf,
                NBYTES
            );
        });

        q.wait();

        test_b_pass = (host_dst == host_src);

        std::cout
            << "TEST B (device -> host): "
            << (test_b_pass
                    ? "PASS -- data matches exactly"
                    : "FAIL -- data mismatch")
            << "\n";


        // ============================================================
        // Final verdict
        // ============================================================

        std::cout << "\n=== VERDICT ===\n";

        std::cout
            << "handler::memcpy host->device: "
            << (test_a_pass
                    ? "SAFE TO USE"
                    : "DO NOT USE -- broken")
            << "\n";

        std::cout
            << "handler::memcpy device->host: "
            << (test_b_pass
                    ? "SAFE TO USE"
                    : "DO NOT USE -- broken")
            << "\n";

        // Free device memory
        sycl::free(device_buf, q.get_context());

        return (test_a_pass && test_b_pass) ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << "Exception: "
                  << e.what() << "\n";
        return 1;
    }
}
