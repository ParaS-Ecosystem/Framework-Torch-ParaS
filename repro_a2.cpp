// repro_a2.cpp
#include "/storage/Goutam/Framework-Torch-ParaS/repro_a.h"

namespace repro {

void QueueOpt::init() {
    sycl_queue_.emplace(sycl::gpu_selector_v);
}

void QueueOpt::copy(void* dst, const void* src, std::size_t nbytes) {
    sycl_queue().memcpy(dst, src, nbytes);
    sycl_queue().wait();
}

} // namespace repro
