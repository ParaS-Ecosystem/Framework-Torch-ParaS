#include "clap_blas_wrapper.h"

#include <clap/blas_factory.hpp>

#include <memory>
#include <unordered_map>
#include <stdexcept>
#include <string>

#include <cstdint>
#include <dlfcn.h>

namespace {

#if defined(PTSYCL_CLAP_BLAS_CUDA) || defined(PTSYCL_BACKEND_CUDA)

extern "C" int cudaGetDevice(int*);
extern "C" int cudaSetDevice(int);

using CuPointerGetAttributeFn =
    int (*)(void*, int, std::uintptr_t);

CuPointerGetAttributeFn cu_pointer_get_attribute()
{
    static CuPointerGetAttributeFn fn = [] {
        void* lib = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
        if (!lib)
            throw std::runtime_error("Unable to load libcuda.so.1");

        void* sym = dlsym(lib, "cuPointerGetAttribute");
        if (!sym)
            throw std::runtime_error(
                "Unable to load cuPointerGetAttribute");

        return reinterpret_cast<CuPointerGetAttributeFn>(sym);
    }();

    return fn;
}

class CudaDeviceGuard {
public:
    explicit CudaDeviceGuard(const void* ptr)
    {
        int ptr_device = -1;

        // CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9
        const int driver_status =
            cu_pointer_get_attribute()(
                &ptr_device,
                9,
                reinterpret_cast<std::uintptr_t>(ptr));

        if (driver_status != 0) {
            throw std::runtime_error(
                "cuPointerGetAttribute DEVICE_ORDINAL failed: " +
                std::to_string(driver_status));
        }

        const int get_status = cudaGetDevice(&previous_device_);
        if (get_status != 0) {
            throw std::runtime_error(
                "cudaGetDevice failed: " +
                std::to_string(get_status));
        }

        if (previous_device_ != ptr_device) {
            const int set_status = cudaSetDevice(ptr_device);
            if (set_status != 0) {
                throw std::runtime_error(
                    "cudaSetDevice failed: " +
                    std::to_string(set_status));
            }

            restore_ = true;

        }
    }

    ~CudaDeviceGuard() noexcept
    {
        if (restore_)
            (void)cudaSetDevice(previous_device_);
    }

private:
    int previous_device_ = -1;
    bool restore_ = false;
};

#endif

thread_local std::unique_ptr<clap::IBlasBackend> cached_cpu_backend;

thread_local std::unordered_map<
    void*,
    std::unique_ptr<clap::IBlasBackend>
> cached_cuda_backends;

thread_local std::unordered_map<
    void*,
    std::unique_ptr<clap::IBlasBackend>
> cached_amd_backends;

clap::IBlasBackend* get_backend(
    clap::BackendType type,
    void* stream = nullptr)
{
    if (type == clap::BackendType::CPU) {
        if (!cached_cpu_backend)
            cached_cpu_backend = clap::BlasFactory::create(type);

        return cached_cpu_backend.get();
    }

    if (stream == nullptr)
        return nullptr;

    auto& cache =
        type == clap::BackendType::CUDA
            ? cached_cuda_backends
            : cached_amd_backends;

    auto& backend = cache[stream];

    if (!backend)
        backend = clap::BlasFactory::create(type);

    return backend.get();
}

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
        clap::IBlasBackend* backend = get_backend(clap::BackendType::CPU);

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
        cached_cpu_backend.reset();
        return false;
    }
}

extern "C" bool ptsycl_clap_blas_gemm_device_raw(
    const void* a,
    const void* b,
    void* out,
    int m,
    int n,
    int k,
    bool transpose_b,
    bool is_double,
    bool is_bf16,
    double alpha,
    double beta,
    void* stream,
    int backend_type) noexcept
{
    try {
#if defined(PTSYCL_CLAP_BLAS_CUDA) || defined(PTSYCL_BACKEND_CUDA)
        std::unique_ptr<CudaDeviceGuard> cuda_device_guard;
        if (backend_type == 1)
            cuda_device_guard = std::make_unique<CudaDeviceGuard>(out);
#endif

        const auto type =
            backend_type == 1
                ? clap::BackendType::CUDA
                : clap::BackendType::AMD;

        clap::IBlasBackend* backend = get_backend(type, stream);

        if (backend == nullptr)
            return false;

        const clap::Transpose first_transpose =
            transpose_b
                ? clap::Transpose::Trans
                : clap::Transpose::NoTrans;

        const int lda = transpose_b ? k : n;
        const int ldb = k;
        const int ldc = n;

        if (is_bf16) {
            backend->gemm_bf16_device(
                clap::Layout::ColMajor,
                first_transpose,
                clap::Transpose::NoTrans,
                n, m, k,
                static_cast<float>(alpha),
                b, lda,
                a, ldb,
                static_cast<float>(beta),
                out, ldc,
                stream);
        } else if (is_double) {
            backend->gemm_device(
                clap::Layout::ColMajor,
                first_transpose,
                clap::Transpose::NoTrans,
                n, m, k,
                alpha,
                static_cast<const double*>(b), lda,
                static_cast<const double*>(a), ldb,
                beta,
                static_cast<double*>(out), ldc,
                stream);
        } else {
            backend->gemm_device(
                clap::Layout::ColMajor,
                first_transpose,
                clap::Transpose::NoTrans,
                n, m, k,
                static_cast<float>(alpha),
                static_cast<const float*>(b), lda,
                static_cast<const float*>(a), ldb,
                static_cast<float>(beta),
                static_cast<float*>(out), ldc,
                stream);
        }

        return true;
    } catch (...) {
        if (backend_type == 1)
            cached_cuda_backends.erase(stream);
        else
            cached_amd_backends.erase(stream);

        return false;
    }
}
