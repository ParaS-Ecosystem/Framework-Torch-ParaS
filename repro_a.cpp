// repro_a.cpp
#include "/storage/Goutam/Framework-Torch-ParaS/repro_a.h"
#include <cstddef>

namespace repro {

void Queue::init() {
#if defined(PTSYCL_BACKEND_SYCL)
    sycl_queue_ = new sycl::queue(sycl::gpu_selector_v);
#endif
}

void* Queue::alloc(std::size_t nbytes) {
#if defined(PTSYCL_BACKEND_SYCL)
    return static_cast<void*>(sycl::malloc_shared<std::byte>(nbytes, sycl_queue()));
#else
    return nullptr;
#endif
}

void Queue::copy(void* dst, const void* src, std::size_t nbytes) {
#if defined(PTSYCL_BACKEND_SYCL)
    sycl_queue().memcpy(dst, src, nbytes);
    sycl_queue().wait();
#endif
}

} // namespace repro
