#include "clap_blas_wrapper.h"

#include <clap/blas_factory.hpp>

#include <memory>

namespace {

thread_local std::unique_ptr<clap::IBlasBackend> cached_backend;

clap::IBlasBackend* get_backend()
{
    if (!cached_backend)
        cached_backend = clap::BlasFactory::create();

    return cached_backend.get();
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
