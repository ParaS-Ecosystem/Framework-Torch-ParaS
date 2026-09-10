// repro_b2.cpp
#include "/storage/Goutam/Framework-Torch-ParaS/repro_a.h"

int main() {
    repro::QueueOpt q;
    q.init();
    void* a = static_cast<void*>(sycl::malloc_shared<std::byte>(sizeof(int) * 4, q.sycl_queue()));
    void* b = static_cast<void*>(sycl::malloc_shared<std::byte>(sizeof(int) * 4, q.sycl_queue()));
    q.copy(b, a, sizeof(int) * 4);
    return 0;
}