#include <sycl/sycl.hpp>

namespace repro {
class GpuQueue {
public:
    explicit GpuQueue(sycl::device& d) : queue_(d, sycl::property::queue::in_order{}) {}
    void copy(void* dst, const void* src, std::size_t nbytes) {
        queue_.memcpy(dst, src, nbytes);
        queue_.wait();
    }
private:
    sycl::queue queue_;
};
}