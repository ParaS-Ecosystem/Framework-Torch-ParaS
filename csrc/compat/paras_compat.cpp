// -----------------------------------------------------------------------------
// Copyright (c) 2026 Centre for Development of Advanced Computing (C-DAC)
//
// This file is part of Torch_ParaS, a component of the ParaS Ecosystem
//
// This library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License (LGPL)
// version 3 as published by the Free Software Foundation.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this library. If not, see <https://www.gnu.org/licenses/>.
// -----------------------------------------------------------------------------

#if defined(PTSYCL_BACKEND_SYCL)
#include <cctype>
#include <sycl/sycl.hpp>
#endif
#include "compat/paras_compat.h"

#include <ATen/Parallel.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#if defined(PTSYCL_BACKEND_SYCL) && (PARAS_HIP_BACKEND || PARAS_CUDA_BACKEND)
#include <dlfcn.h>
#endif
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>



namespace ptsycl {
namespace compat {

#if defined(PTSYCL_BACKEND_SYCL) && defined(PARAS_EXT_ONEAPI_GRAPH_RECORD_REPLAY)
#define PTSYCL_HAVE_GRAPHS 1
namespace sycl_exp = sycl::ext::oneapi::experimental;
#endif

struct Queue::Impl {
#if defined(PTSYCL_BACKEND_SYCL)
    ~Impl() { delete queue; }
    sycl::queue* queue = nullptr;
#endif
#if defined(PTSYCL_HAVE_GRAPHS)
    std::optional<sycl_exp::command_graph<sycl_exp::graph_state::modifiable>> recording;
#endif
};

class Graph {
public:
#if defined(PTSYCL_HAVE_GRAPHS)
    explicit Graph(sycl_exp::command_graph<sycl_exp::graph_state::executable> exec)
        : exec_(std::move(exec)) {}
    sycl_exp::command_graph<sycl_exp::graph_state::executable> exec_;
#endif
};

Queue::Queue() = default;
Queue::~Queue() = default;

void fail(const std::string& what) {
    throw std::runtime_error("ptsycl: " + what);
}
static std::string host_cpu_name() {
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    while (std::getline(f, line)) {
        auto pos = line.find("model name");
        if (pos != std::string::npos) {
            auto colon = line.find(':');
            if (colon != std::string::npos) {
                std::string name = line.substr(colon + 1);
                while (!name.empty() && name.front() == ' ') name.erase(name.begin());
                return name;
            }
        }
    }
    return "Host CPU";
}

static std::size_t host_mem_bytes() {
    std::ifstream f("/proc/meminfo");
    std::string key;
    std::size_t kb = 0;
    while (f >> key >> kb) {
        if (key == "MemTotal:") return kb * 1024;
        f.ignore(256, '\n');
    }
    return 0;
}

unsigned host_thread_count() {
    static unsigned n = [] {
        if (const char* env = std::getenv("PTSYCL_CPU_THREADS")) {
            int v = std::atoi(env);
            if (v > 0) return static_cast<unsigned>(v);
        }
        const int torch_threads = at::get_num_threads();
        if (torch_threads > 0) return static_cast<unsigned>(torch_threads);
        const unsigned hc = std::thread::hardware_concurrency();
        return hc == 0 ? 4u : hc;
    }();
    return n;
}
#if defined(PTSYCL_BACKEND_SYCL)
namespace {

std::vector<sycl::device>& device_table() {
    static std::vector<sycl::device> table;
    return table;
}

int vendor_visible_gpu_count() {
#if PARAS_HIP_BACKEND || PARAS_CUDA_BACKEND
#if PARAS_HIP_BACKEND
    const char* libs[] = {"libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6"};
    const char* symbol = "hipGetDeviceCount";
#else
    const char* libs[] = {"libcudart.so", "libcudart.so.12", "libcudart.so.11.0"};
    const char* symbol = "cudaGetDeviceCount";
#endif
    void* handle = nullptr;
    for (const char* lib : libs)
        if (handle == nullptr) handle = dlopen(lib, RTLD_LAZY | RTLD_NOLOAD);
    for (const char* lib : libs)
        if (handle == nullptr) handle = dlopen(lib, RTLD_LAZY | RTLD_LOCAL);
    if (handle != nullptr) {
        using CountFn = int (*)(int*);
        auto count_fn = reinterpret_cast<CountFn>(dlsym(handle, symbol));
        int count = 0;
        if (count_fn != nullptr && count_fn(&count) == 0) return count;
        return -1;
    }
#endif
    return -1;
}
} // namespace

sycl::device& device_by_native_id(int native_id) {
    auto& table = device_table();
    if (native_id < 0 || static_cast<std::size_t>(native_id) >= table.size()) {
        std::ostringstream os;
        os << "device_by_native_id: invalid ordinal " << native_id;
        fail(os.str());
    }
    return table[static_cast<std::size_t>(native_id)];
}
#endif

void Queue::init(const DeviceInfo& dev) {
    if (initialized_) fail("Queue::init called twice");
    is_gpu_ = dev.is_gpu;
    native_id_ = dev.native_id;
    wave_size_ = dev.wave_size;
    mfma_ = dev.mfma;
#if defined(PTSYCL_BACKEND_SYCL)
    if (is_gpu_) {
        ::setenv("PARAS_ASYNC_KERNELS", "1", 0);
        sycl::device& device = device_by_native_id(native_id_);
        try {
            auto impl = std::make_unique<Impl>();
            impl->queue = new sycl::queue(
                device, sycl::property::queue::in_order{});
            impl_ = std::move(impl);
        } catch (const sycl::exception& e) {
            std::ostringstream os;
            os << "sycl::queue construction failed on device " << native_id_
               << ": " << e.what();
            fail(os.str());
        }
    }
#else
    if (is_gpu_) fail("GPU device requested in a CPU-only build");
#endif
    initialized_ = true;
}

#if defined(PTSYCL_BACKEND_SYCL)
sycl::queue& Queue::sycl_queue() const {
    if (!impl_ || !impl_->queue) fail("SYCL queue is not initialized");
    return *impl_->queue;
}
#endif

#if defined(PTSYCL_BACKEND_SYCL)
namespace {
unsigned gfx_number(const std::string& version) {
    const auto p = version.find("gfx");
    if (p == std::string::npos) return 0;
    unsigned v = 0;
    for (std::size_t i = p + 3; i < version.size() && std::isxdigit(static_cast<unsigned char>(version[i])); ++i)
        v = v * 16 + static_cast<unsigned>(std::stoi(std::string(1, version[i]), nullptr, 16));
    return v;
}

void gpu_kernel_caps(const sycl::device& d, DeviceInfo& info) {
    const auto sizes = d.get_info<sycl::info::device::sub_group_sizes>();
    info.wave_size = sizes.empty() ? 64 : static_cast<int>(sizes.front());
#if PARAS_HIP_BACKEND
    const unsigned gfx = gfx_number(d.get_info<sycl::info::device::version>());
    info.mfma = info.wave_size == 64 && (gfx == 0x90a || (gfx >= 0x940 && gfx <= 0x95f));
#else
    info.mfma = false;
#endif
    const char* generic = std::getenv("PTSYCL_GENERIC_KERNELS");
    if (generic != nullptr && std::string(generic) == "1") {
        info.wave_size = 32;
        info.mfma = false;
    }
}
}
#endif

std::vector<DeviceInfo> enumerate_devices() {

    std::vector<DeviceInfo> out;

#if defined(PTSYCL_BACKEND_SYCL)
    auto& table = device_table();
    table.clear();
    auto gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    const int visible = vendor_visible_gpu_count();
    if (visible >= 0 && static_cast<std::size_t>(visible) < gpus.size())
        gpus.resize(static_cast<std::size_t>(visible));
    for (std::size_t i = 0; i < gpus.size(); ++i) {
        DeviceInfo gpu;
        gpu.name          = gpus[i].get_info<sycl::info::device::name>();
        gpu.is_gpu        = true;
        gpu.native_id     = static_cast<int>(i);
        gpu.compute_units = static_cast<int>(gpus[i].get_info<sycl::info::device::max_compute_units>());
        gpu.global_mem    = gpus[i].get_info<sycl::info::device::global_mem_size>();
        gpu.fp64          = true;
        gpu_kernel_caps(gpus[i], gpu);

        table.push_back(gpus[i]);
        out.push_back(std::move(gpu));
    }
#endif

    DeviceInfo cpu;
    cpu.name          = host_cpu_name();
    cpu.is_gpu        = false;
    cpu.native_id     = -1;
    cpu.compute_units = static_cast<int>(host_thread_count());
    cpu.global_mem    = host_mem_bytes();
    cpu.fp64          = true;
    out.push_back(std::move(cpu));

    return out;
}

namespace {

extern "C" void GOMP_parallel(void (*fn)(void*), void* data,
                              unsigned num_threads, unsigned flags);

std::size_t host_grain_size() {
    static const std::size_t value = [] {
        if (const char* env = std::getenv("PTSYCL_CPU_GRAIN_SIZE")) {
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(env, &end, 10);
            if (end != env && *end == '\0' && parsed > 0)
                return static_cast<std::size_t>(parsed);
        }
        return std::size_t{16384};
    }();
    return value;
}

struct HostParallelJob {
    std::size_t n;
    std::size_t chunks;
    std::size_t per;
    host_chunk_fn body;
    void* ctx;
    std::atomic<std::size_t> next{0};
};

void run_host_parallel_job(void* raw_job) {
    auto& job = *static_cast<HostParallelJob*>(raw_job);
    while (true) {
        const std::size_t chunk =
            job.next.fetch_add(1, std::memory_order_relaxed);
        if (chunk >= job.chunks) return;
        const std::size_t begin = chunk * job.per;
        const std::size_t end = std::min(begin + job.per, job.n);
        if (begin < end) job.body(job.ctx, begin, end);
    }
}

}

void host_parallel_chunks(std::size_t n, host_chunk_fn body, void* ctx) {
    if (n == 0) return;
    const std::size_t thread_count = host_thread_count();
    const std::size_t grain = host_grain_size();
    if (thread_count <= 1 || n <= grain || at::in_parallel_region()) {
        body(ctx, 0, n);
        return;
    }
    const std::size_t chunks =
        std::min(thread_count, (n + grain - 1) / grain);
    const std::size_t per = (n + chunks - 1) / chunks;
    HostParallelJob job{n, chunks, per, body, ctx};
    GOMP_parallel(run_host_parallel_job, &job,
                  static_cast<unsigned>(chunks), 0);
}
void* Queue::alloc(std::size_t nbytes) {
    if (nbytes == 0) nbytes = 1;
#if defined(PTSYCL_BACKEND_SYCL)
    if (is_gpu_) {
        void* p = static_cast<void*>(
            sycl::malloc_device<std::byte>(nbytes, sycl_queue()));
        if (p == nullptr) {
            std::ostringstream os;
            os << "sycl::malloc_device(" << nbytes << " bytes) on device "
               << native_id_ << " failed";
            fail(os.str());
        }
        return p;
    }
#endif
    void* p = nullptr;
    if (posix_memalign(&p, 64, nbytes) != 0 || p == nullptr) {
        std::ostringstream os;
        os << "host allocation of " << nbytes << " bytes failed";
        fail(os.str());
    }
    return p;
}

void Queue::dealloc(void* ptr) {
    if (ptr == nullptr) return;
#if defined(PTSYCL_BACKEND_SYCL)
    if (is_gpu_) {
        sycl::free(ptr, sycl_queue());
        return;
    }
#endif
    std::free(ptr);
}

void Queue::copy(void* dst, const void* src, std::size_t nbytes, bool blocking) {
    if (nbytes == 0) return;
#if defined(PTSYCL_BACKEND_SYCL)
    if (is_gpu_) {
        sycl_queue().memcpy(dst, src, nbytes);
        if (blocking) synchronize();
        return;
    }
#endif
    std::memcpy(dst, src, nbytes);
    (void)blocking;
}

void Queue::memset(void* ptr, int value, std::size_t nbytes) {
    if (nbytes == 0) return;
#if defined(PTSYCL_BACKEND_SYCL)
    if (is_gpu_) {
        sycl_queue().memset(ptr, value, nbytes);
        return;
    }
#endif
    std::memset(ptr, value, nbytes);
}

void Queue::synchronize() {
#if defined(PTSYCL_BACKEND_SYCL)
    if (is_gpu_) {
        sycl_queue().wait();
    }
#endif
}

bool Queue::run_native(const std::function<void(void* native_stream)>& fn) {
#if defined(PTSYCL_BACKEND_SYCL) && (PARAS_HIP_BACKEND || PARAS_CUDA_BACKEND)
    if (!is_gpu_) return false;
    sycl_queue().parasSYCL_enqueue_custom_operation([&](sycl::interop_handle ih) {
#if PARAS_HIP_BACKEND
        fn(static_cast<void*>(ih.get_native_queue<sycl::backend::hip>()));
#else
        fn(static_cast<void*>(ih.get_native_queue<sycl::backend::cuda>()));
#endif
    });
    return true;
#else
    (void)fn;
    return false;
#endif
}

namespace {
struct EventApi {
    int (*create)(void** event, unsigned flags) = nullptr;
    int (*record)(void* event, void* stream) = nullptr;
    int (*wait)(void* stream, void* event, unsigned flags) = nullptr;
    int (*query)(void* event) = nullptr;
    int (*synchronize)(void* event) = nullptr;
    int (*elapsed)(float* ms, void* start, void* end) = nullptr;
    int (*destroy)(void* event) = nullptr;
    int (*stream_query)(void* stream) = nullptr;
};

const EventApi& event_api() {
    static const EventApi api = [] {
        EventApi a;
#if PARAS_HIP_BACKEND || PARAS_CUDA_BACKEND
#if PARAS_HIP_BACKEND
        const char* libs[] = {"libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6"};
        const char* names[] = {"hipEventCreateWithFlags", "hipEventRecord", "hipStreamWaitEvent", "hipEventQuery",
                               "hipEventSynchronize", "hipEventElapsedTime", "hipEventDestroy", "hipStreamQuery"};
#else
        const char* libs[] = {"libcudart.so", "libcudart.so.12", "libcudart.so.11.0"};
        const char* names[] = {"cudaEventCreateWithFlags", "cudaEventRecord", "cudaStreamWaitEvent", "cudaEventQuery",
                               "cudaEventSynchronize", "cudaEventElapsedTime", "cudaEventDestroy", "cudaStreamQuery"};
#endif
        void* handle = nullptr;
        for (const char* lib : libs)
            if (handle == nullptr) handle = dlopen(lib, RTLD_LAZY | RTLD_NOLOAD);
        for (const char* lib : libs)
            if (handle == nullptr) handle = dlopen(lib, RTLD_LAZY | RTLD_LOCAL);
        if (handle != nullptr) {
            a.create = reinterpret_cast<decltype(a.create)>(dlsym(handle, names[0]));
            a.record = reinterpret_cast<decltype(a.record)>(dlsym(handle, names[1]));
            a.wait = reinterpret_cast<decltype(a.wait)>(dlsym(handle, names[2]));
            a.query = reinterpret_cast<decltype(a.query)>(dlsym(handle, names[3]));
            a.synchronize = reinterpret_cast<decltype(a.synchronize)>(dlsym(handle, names[4]));
            a.elapsed = reinterpret_cast<decltype(a.elapsed)>(dlsym(handle, names[5]));
            a.destroy = reinterpret_cast<decltype(a.destroy)>(dlsym(handle, names[6]));
            a.stream_query = reinterpret_cast<decltype(a.stream_query)>(dlsym(handle, names[7]));
        }
#endif
        return a;
    }();
    return api;
}

const EventApi& require_events() {
    const EventApi& a = event_api();
    if (!a.create || !a.record || !a.wait || !a.query || !a.synchronize || !a.elapsed || !a.destroy ||
        !a.stream_query)
        fail("paras events: the HIP / CUDA runtime's event API was not found");
    return a;
}

void check_event(int rc, const char* what) {
    if (rc != 0) {
        std::ostringstream os;
        os << "paras events: " << what << " failed with error " << rc;
        fail(os.str());
    }
}
}

void Queue::record_event(void** event, bool timing) {
    if (!is_gpu_) return;
    const EventApi& api = require_events();
    const bool ran = run_native([&](void* stream) {
        if (*event == nullptr) check_event(api.create(event, timing ? 0u : 2u ), "event create");
        check_event(api.record(*event, stream), "event record");
    });
    if (!ran) fail("paras events: the queue has no native stream");
}

void Queue::wait_event(void* event) {
    if (!is_gpu_ || event == nullptr) return;
    const EventApi& api = require_events();
    run_native([&](void* stream) { check_event(api.wait(stream, event, 0), "stream wait"); });
}

bool Queue::query() {
    if (!is_gpu_) return true;
    const EventApi& api = require_events();
    int rc = 0;
    run_native([&](void* stream) { rc = api.stream_query(stream); });
    if (rc == 600) return false;
    check_event(rc, "stream query");
    return true;
}

bool Queue::event_query(void* event) {
    if (event == nullptr) return true;
    const int rc = require_events().query(event);
    if (rc == 600) return false;
    check_event(rc, "event query");
    return true;
}

void Queue::event_synchronize(void* event) {
    if (event != nullptr) check_event(require_events().synchronize(event), "event synchronize");
}

float Queue::event_elapsed_ms(void* start, void* end) {
    if (start == nullptr || end == nullptr) fail("paras events: elapsed time of an event that was never recorded");
    float ms = 0.f;
    check_event(require_events().elapsed(&ms, start, end), "event elapsed time (both events need timing)");
    return ms;
}

void Queue::event_destroy(void* event) noexcept {
    if (event == nullptr) return;
    const EventApi& api = event_api();
    if (api.destroy != nullptr) api.destroy(event);
}

bool Queue::graphs_supported() {
#if defined(PTSYCL_HAVE_GRAPHS)
    return true;
#else
    return false;
#endif
}

void Queue::begin_capture() {
#if defined(PTSYCL_HAVE_GRAPHS)
    if (!is_gpu_) fail("graph capture needs a GPU device");
    if (capturing_) fail("the device is already capturing a graph");
    auto& q = *impl_->queue;
    impl_->recording.emplace(q);
    try {
        impl_->recording->begin_recording(q);
    } catch (const sycl::exception& e) {
        impl_->recording.reset();
        fail(std::string("graph capture could not start: ") + e.what());
    }
    capturing_ = true;
#else
    fail("graph capture needs a ParaS build with record-and-replay command graphs");
#endif
}

std::shared_ptr<Graph> Queue::end_capture() {
#if defined(PTSYCL_HAVE_GRAPHS)
    if (!capturing_) fail("the device is not capturing a graph");
    capturing_ = false;
    auto& q = *impl_->queue;
    try {
        impl_->recording->end_recording(q);
        auto exec = impl_->recording->finalize();
        impl_->recording.reset();
        return std::make_shared<Graph>(std::move(exec));
    } catch (const sycl::exception& e) {
        impl_->recording.reset();
        fail(std::string("graph capture failed: ") + e.what());
    }
#else
    fail("graph capture needs a ParaS build with record-and-replay command graphs");
#endif
}

void Queue::replay(const Graph& graph) {
#if defined(PTSYCL_HAVE_GRAPHS)
    if (capturing_) fail("a graph cannot be replayed while capturing");
    impl_->queue->ext_oneapi_graph(graph.exec_);
#else
    (void)graph;
    fail("graph replay needs a ParaS build with record-and-replay command graphs");
#endif
}

const char* backend_name() {
#if defined(PTSYCL_BACKEND_SYCL)
    return "paras-sycl";
#else
    return "paras-cpu";
#endif
}

} // namespace compat
} // namespace ptsycl
