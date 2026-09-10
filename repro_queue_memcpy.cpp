// repro_queue_memcpy.cpp
#include <sycl/sycl.hpp>
int main() {
    sycl::queue q(sycl::gpu_selector_v);
    int* a = sycl::malloc_shared<int>(4, q);
    int* b = sycl::malloc_shared<int>(4, q);
    q.memcpy(b, a, sizeof(int) * 4);
    q.wait();
    sycl::free(a, q);
    sycl::free(b, q);
    return 0;
}
