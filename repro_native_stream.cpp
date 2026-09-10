// repro_native_stream.cpp
#include <sycl/sycl.hpp>
#include <cuda_runtime.h>
#include <cstdio>

int main() {
    sycl::queue q(sycl::gpu_selector_v);
    auto stream = q.get_native<sycl::backend::cuda>();
    std::printf("native stream handle: %p\n", (void*)stream);
    return 0;
}
