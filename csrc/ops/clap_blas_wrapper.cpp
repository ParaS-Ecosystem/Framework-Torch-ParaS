#include "clap_blas_wrapper.h"

#include <clap/blas_factory.hpp>

#include <memory>
#include <unordered_map>

namespace {

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
