#include "/storage/Goutam/Framework-Torch-ParaS/repro_c.h"
repro::GpuQueue* make_queue() {
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    static sycl::device d = devs.empty() ? sycl::device(sycl::gpu_selector_v) : devs[0];
    return new repro::GpuQueue(d);
}
