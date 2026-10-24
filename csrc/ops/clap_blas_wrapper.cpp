#include "clap_blas_wrapper.h"

#include <clap/blas/openblas/openblas_backend.hpp>
#if defined(PTSYCL_CLAP_DEVICE_BACKEND_AMD)
#include <clap/blas/rocblas/rocblas_backend.hpp>
#include <dlfcn.h>
#elif defined(PTSYCL_CLAP_DEVICE_BACKEND_CUDA)
#include <clap/blas/cublas/cublas_backend.hpp>
#endif

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <unordered_map>

namespace {

thread_local std::unique_ptr<clap::IBlasBackend> cached_backend;

clap::IBlasBackend* get_backend()
{
    if (!cached_backend)
        cached_backend = std::make_unique<clap::OpenBlasBackend>();

    return cached_backend.get();
}

#if defined(PTSYCL_CLAP_DEVICE_BACKEND_AMD) || defined(PTSYCL_CLAP_DEVICE_BACKEND_CUDA)

struct DeviceBackend {
    std::unique_ptr<clap::IBlasBackend> backend;
    bool unavailable = false;
    bool bf16_unsupported = false;
    bool ex_unsupported = false;
};

thread_local std::unordered_map<int, DeviceBackend> device_backends;

DeviceBackend* get_device_backend(int device_index)
{
    DeviceBackend& entry = device_backends[device_index];
    if (!entry.backend && !entry.unavailable) {
        try {
#if defined(PTSYCL_CLAP_DEVICE_BACKEND_AMD)
            entry.backend = std::make_unique<clap::RocBlasBackend>();
#else
            entry.backend = std::make_unique<clap::CuBlasBackend>();
#endif
        } catch (...) {
            entry.unavailable = true;
        }
    }
    return entry.backend ? &entry : nullptr;
}

#if defined(PTSYCL_CLAP_DEVICE_BACKEND_AMD)
class StreamDeviceGuard {
public:
    explicit StreamDeviceGuard(void* stream)
    {
        using GetFn = int (*)(int*);
        using SetFn = int (*)(int);
        using StreamFn = int (*)(void*, int*);
        static void* hip = [] {
            for (const char* lib : {"libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6"})
                if (void* h = dlopen(lib, RTLD_LAZY | RTLD_NOLOAD)) return h;
            return static_cast<void*>(nullptr);
        }();
        static auto get = hip ? reinterpret_cast<GetFn>(dlsym(hip, "hipGetDevice")) : nullptr;
        static auto set = hip ? reinterpret_cast<SetFn>(dlsym(hip, "hipSetDevice")) : nullptr;
        static auto of_stream = hip ? reinterpret_cast<StreamFn>(
                                          dlsym(hip, "hipStreamGetDevice"))
                                    : nullptr;
        int target = -1;
        if (!get || !set || !of_stream || stream == nullptr ||
            of_stream(stream, &target) != 0 || get(&previous_) != 0 ||
            previous_ == target)
            return;
        if (set(target) == 0) set_ = set;
    }
    ~StreamDeviceGuard()
    {
        if (set_) set_(previous_);
    }
    StreamDeviceGuard(const StreamDeviceGuard&) = delete;
    StreamDeviceGuard& operator=(const StreamDeviceGuard&) = delete;

private:
    int previous_ = 0;
    int (*set_)(int) = nullptr;
};
#else
struct StreamDeviceGuard {
    explicit StreamDeviceGuard(void*) {}
};
#endif

bool run_device_gemm(DeviceBackend& entry,
                     clap::Transpose ta, clap::Transpose tb,
                     int m, int n, int k, double alpha,
                     const void* a, int lda, long long stride_a,
                     const void* b, int ldb, long long stride_b,
                     double beta, void* c, int ldc, long long stride_c,
                     int batch, int dtype, int c_dtype, void* stream)
{
    constexpr auto kRow = clap::Layout::RowMajor;
    clap::IBlasBackend* backend = entry.backend.get();
#if defined(PTSYCL_CLAP_HAS_GEMM_EX)
    const auto elem = [](int t) {
        return t == PTSYCL_CLAP_F16 ? clap::GemmElem::F16
             : t == PTSYCL_CLAP_BF16 ? clap::GemmElem::BF16 : clap::GemmElem::F32;
    };
    if ((dtype == PTSYCL_CLAP_F16 && (c_dtype == PTSYCL_CLAP_F16 || c_dtype == PTSYCL_CLAP_F32)) ||
        (dtype == PTSYCL_CLAP_BF16 && c_dtype == PTSYCL_CLAP_F32)) {
        if (entry.ex_unsupported) return false;
        try {
            backend->gemm_ex_strided_batched_device(
                kRow, ta, tb, m, n, k, static_cast<float>(alpha), a, lda, stride_a, b, ldb, stride_b,
                static_cast<float>(beta), c, ldc, stride_c, batch, elem(dtype), elem(c_dtype), stream);
        } catch (const std::invalid_argument&) {
            return false;
        } catch (const std::exception&) {
            entry.ex_unsupported = true;
            return false;
        }
        return true;
    }
#endif
    if (dtype != c_dtype)
        return false;
    switch (dtype) {
    case PTSYCL_CLAP_F32:
        if (batch == 1)
            backend->gemm_device(kRow, ta, tb, m, n, k, static_cast<float>(alpha),
                                 static_cast<const float*>(a), lda,
                                 static_cast<const float*>(b), ldb,
                                 static_cast<float>(beta), static_cast<float*>(c),
                                 ldc, stream);
        else
            backend->gemm_strided_batched_device(
                kRow, ta, tb, m, n, k, static_cast<float>(alpha), a, lda, stride_a,
                b, ldb, stride_b, static_cast<float>(beta), c, ldc, stride_c,
                batch, stream);
        return true;
    case PTSYCL_CLAP_F64:
        for (int i = 0; i < batch; ++i)
            backend->gemm_device(kRow, ta, tb, m, n, k, alpha,
                                 static_cast<const double*>(a) + i * stride_a, lda,
                                 static_cast<const double*>(b) + i * stride_b, ldb,
                                 beta, static_cast<double*>(c) + i * stride_c, ldc,
                                 stream);
        return true;
    case PTSYCL_CLAP_BF16: {
        const auto* pa = static_cast<const std::uint16_t*>(a);
        const auto* pb = static_cast<const std::uint16_t*>(b);
        auto* pc = static_cast<std::uint16_t*>(c);
        if (entry.bf16_unsupported)
            return false;
        int done = 0;
        auto fail = [&](bool unsupported) {
            if (done > 0 && beta != 0.0)
                throw std::runtime_error("bf16 device GEMM failed after " + std::to_string(done) + " of " +
                                         std::to_string(batch) + " batch entries");
            if (unsupported) entry.bf16_unsupported = true;
            return false;
        };
        try {
#if defined(PTSYCL_CLAP_HAS_BF16_BATCHED)
            if (batch > 1)
                backend->gemm_strided_batched_bf16_device(
                    kRow, ta, tb, m, n, k, static_cast<float>(alpha), a, lda, stride_a,
                    b, ldb, stride_b, static_cast<float>(beta), c, ldc, stride_c,
                    batch, stream);
            else
#endif
            for (int i = 0; i < batch; ++i, ++done)
                backend->gemm_bf16_device(kRow, ta, tb, m, n, k, static_cast<float>(alpha),
                                          pa + i * stride_a, lda, pb + i * stride_b, ldb,
                                          static_cast<float>(beta), pc + i * stride_c,
                                          ldc, stream);
        } catch (const std::invalid_argument&) {
            return fail(false);
        } catch (const std::exception&) {
            return fail(true);
        }
        return true;
    }
    default:
        return false;
    }
}

#endif

} // namespace

extern "C" bool ptsycl_clap_blas_gemm_raw(
    const void* a,
    const void* b,
    void* out,
    int m,
    int n,
    int k,
    bool transpose_b,
    bool is_double,
    double alpha) noexcept
{
    try {
        clap::IBlasBackend* backend = get_backend();

        if (backend == nullptr)
            return false;

        const clap::Transpose first_transpose =
            transpose_b
                ? clap::Transpose::Trans
                : clap::Transpose::NoTrans;

        const int lda = transpose_b ? k : n;
        const int ldb = k;
        const int ldc = n;

        if (is_double) {
            backend->gemm(
                clap::Layout::ColMajor,
                first_transpose,
                clap::Transpose::NoTrans,
                n, m, k,
                alpha,
                static_cast<const double*>(b), lda,
                static_cast<const double*>(a), ldb,
                0.0,
                static_cast<double*>(out), ldc);
        } else {
            backend->gemm(
                clap::Layout::ColMajor,
                first_transpose,
                clap::Transpose::NoTrans,
                n, m, k,
                static_cast<float>(alpha),
                static_cast<const float*>(b), lda,
                static_cast<const float*>(a), ldb,
                0.0f,
                static_cast<float*>(out), ldc);
        }

        return true;
    } catch (...) {
        cached_backend.reset();
        return false;
    }
}

extern "C" bool ptsycl_clap_blas_gemm_device_raw(
    int device_index,
    bool trans_a,
    bool trans_b,
    int m,
    int n,
    int k,
    double alpha,
    const void* a,
    int lda,
    long long stride_a,
    const void* b,
    int ldb,
    long long stride_b,
    double beta,
    void* c,
    int ldc,
    long long stride_c,
    int batch,
    int dtype,
    int c_dtype,
    void* stream) noexcept
{
#if defined(PTSYCL_CLAP_DEVICE_BACKEND_AMD) || defined(PTSYCL_CLAP_DEVICE_BACKEND_CUDA)
    try {
        const StreamDeviceGuard device_guard(stream);
        DeviceBackend* backend = get_device_backend(device_index);
        if (backend == nullptr)
            return false;

        const clap::Transpose ta =
            trans_a ? clap::Transpose::Trans : clap::Transpose::NoTrans;
        const clap::Transpose tb =
            trans_b ? clap::Transpose::Trans : clap::Transpose::NoTrans;
        return run_device_gemm(*backend, ta, tb, m, n, k, alpha, a, lda, stride_a,
                               b, ldb, stride_b, beta, c, ldc, stride_c, batch,
                               dtype, c_dtype, stream);
    } catch (...) {
        return false;
    }
#else
    (void)device_index; (void)trans_a; (void)trans_b; (void)m; (void)n;
    (void)k; (void)alpha; (void)a; (void)lda; (void)stride_a; (void)b;
    (void)ldb; (void)stride_b; (void)beta; (void)c; (void)ldc;
    (void)stride_c; (void)batch; (void)dtype; (void)c_dtype; (void)stream;
    return false;
#endif
}
