#pragma once

#include <ATen/ATen.h>
#include <limits>

#ifdef PTSYCL_USE_CLAP_BLAS
// CLAP and PyTorch may expose overlapping CUDA identifiers.
#define cudaError_t clap_dummy_cudaError_t
#define cudaMemcpyKind clap_dummy_cudaMemcpyKind
#define cudaMemcpyHostToHost clap_dummy_cudaMemcpyHostToHost
#define cudaMemcpyHostToDevice clap_dummy_cudaMemcpyHostToDevice
#define cudaMemcpyDeviceToHost clap_dummy_cudaMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice clap_dummy_cudaMemcpyDeviceToDevice
#define cudaMemcpyDefault clap_dummy_cudaMemcpyDefault

#include <clap/blas_factory.hpp>

#undef cudaError_t
#undef cudaMemcpyKind
#undef cudaMemcpyHostToHost
#undef cudaMemcpyHostToDevice
#undef cudaMemcpyDeviceToHost
#undef cudaMemcpyDeviceToDevice
#undef cudaMemcpyDefault
#endif

namespace ptsycl {
namespace clap_blas {

inline bool supported_dtype(const at::Tensor& t)
{
    return t.scalar_type() == at::kFloat ||
           t.scalar_type() == at::kDouble;
}

inline bool gemm_cpu(
    const at::Tensor& a_in,
    const at::Tensor& b_in,
    at::Tensor& out,
    double alpha = 1.0)
{
#ifdef PTSYCL_USE_CLAP_BLAS
    if (!supported_dtype(a_in) ||
        a_in.scalar_type() != b_in.scalar_type() ||
        a_in.scalar_type() != out.scalar_type() ||
        a_in.dim() != 2 ||
        b_in.dim() != 2 ||
        out.dim() != 2 ||
        a_in.size(1) != b_in.size(0) ||
        out.size(0) != a_in.size(0) ||
        out.size(1) != b_in.size(1)) {
        return false;
    }

    at::Tensor a = a_in.contiguous();
    at::Tensor b = b_in.contiguous();

    if (!out.is_contiguous())
        return false;

    const int64_t m = a.size(0);
    const int64_t k = a.size(1);
    const int64_t n = b.size(1);

    if (m > std::numeric_limits<int>::max() ||
        n > std::numeric_limits<int>::max() ||
        k > std::numeric_limits<int>::max()) {
        return false;
    }

    if (m == 0 || n == 0 || k == 0) {
        out.zero_();
        return true;
    }

    auto backend = clap::BlasFactory::create();
    if (backend == nullptr)
        return false;

    // Row-major A(m,k) * B(k,n).
    // Viewed as column-major buffers:
    // B^T(n,k) * A^T(k,m) = C^T(n,m).
    if (a.scalar_type() == at::kFloat) {
        backend->gemm(
            clap::Layout::ColMajor,
            clap::Transpose::NoTrans,
            clap::Transpose::NoTrans,
            static_cast<int>(n),
            static_cast<int>(m),
            static_cast<int>(k),
            static_cast<float>(alpha),
            b.data_ptr<float>(),
            static_cast<int>(n),
            a.data_ptr<float>(),
            static_cast<int>(k),
            0.0f,
            out.data_ptr<float>(),
            static_cast<int>(n));
    } else {
        backend->gemm(
            clap::Layout::ColMajor,
            clap::Transpose::NoTrans,
            clap::Transpose::NoTrans,
            static_cast<int>(n),
            static_cast<int>(m),
            static_cast<int>(k),
            alpha,
            b.data_ptr<double>(),
            static_cast<int>(n),
            a.data_ptr<double>(),
            static_cast<int>(k),
            0.0,
            out.data_ptr<double>(),
            static_cast<int>(n));
    }

    return true;
#else
    (void)a_in;
    (void)b_in;
    (void)out;
    (void)alpha;
    return false;
#endif
}


} // namespace clap_blas
} // namespace ptsycl
