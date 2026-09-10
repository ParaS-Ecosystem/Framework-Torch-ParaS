#include "/storage/Goutam/Framework-Torch-ParaS/repro_c.h"
repro::GpuQueue* make_queue();
int main() {
    auto* q = make_queue();
    void* a = sycl::malloc_shared<char>(16, sycl::queue{});  // adjust as needed for your actual malloc pattern
    // simplified — main point is just exercising q->copy(...)
    return 0;
}
