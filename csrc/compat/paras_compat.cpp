
#include "compat/paras_compat.h"

#include <ATen/Parallel.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace ptsycl {
namespace compat {

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

std::vector<DeviceInfo> enumerate_devices() {
   
    std::vector<DeviceInfo> out;

#if defined(PTSYCL_BACKEND_SYCL)
    auto& table = device_table();
    table.clear();
    try {
        sycl::device dev(sycl::gpu_selector_v);
        DeviceInfo gpu;
        gpu.name          = dev.get_info<sycl::info::device::name>();
        gpu.is_gpu        = true;
        gpu.native_id     = static_cast<int>(table.size()); // always 0 today
        gpu.compute_units = static_cast<int>(
            dev.get_info<sycl::info::device::max_compute_units>());
        gpu.global_mem    = dev.get_info<sycl::info::device::global_mem_size>();
        // aspect::fp64 is not in the supported construct matrix (Section 1's
        // `aspect` row lists only cpu, queue_profiling) and doesn't compile
        // on this parascc install. Hardcoded true -- these are V100s, which
        // support fp64 in hardware. Revisit if this backend ever targets
        // fp64-limited hardware; track alongside the enumeration gap above.
      //  gpu.fp64          = true;
      gpu.fp64 = dev.has(sycl::aspect::fp64);
        table.push_back(dev);
        out.push_back(std::move(gpu));
    } catch (const sycl::exception&) {
       
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
void Queue::init(const DeviceInfo& dev) {
    if (initialized_) fail("Queue::init called twice");
    is_gpu_    = dev.is_gpu;
    native_id_ = dev.native_id;
#if defined(PTSYCL_BACKEND_SYCL)
    if (is_gpu_) {
        sycl::device& d = device_by_native_id(native_id_);
        try {
            queue_ = new sycl::queue(d, sycl::property::queue::in_order{});
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

Queue::~Queue() {
#if defined(PTSYCL_BACKEND_SYCL)
    if (queue_ != nullptr) {
        delete static_cast<sycl::queue*>(queue_);
        queue_ = nullptr;
    }
#endif
}

void* Queue::alloc(std::size_t nbytes) {
    if (nbytes == 0) nbytes = 1;
#if defined(PTSYCL_BACKEND_SYCL)
    if (is_gpu_) {
        void* p = static_cast<void*>(
            sycl::malloc_shared<std::byte>(nbytes, sycl_queue()));
        if (p == nullptr) {
            std::ostringstream os;
            os << "sycl::malloc_shared(" << nbytes << " bytes) on device "
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
        sycl_queue().wait_and_throw();
    }
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


