#pragma once

#include <ATen/ATen.h>

#include <limits>

#include "clap_blas_wrapper.h"

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
    double alpha = 1.0,
    bool transpose_b = false)
{
#ifdef PTSYCL_USE_CLAP_BLAS
    if (!supported_dtype(a_in) ||
        a_in.scalar_type() != b_in.scalar_type() ||
        a_in.scalar_type() != out.scalar_type() ||
        a_in.dim() != 2 ||
        b_in.dim() != 2 ||
        out.dim() != 2 ||
        !out.is_contiguous()) {
        return false;
    }

    const int64_t m = a_in.size(0);
    const int64_t k = a_in.size(1);

    const int64_t b_k =
        transpose_b ? b_in.size(1) : b_in.size(0);

    const int64_t n =
        transpose_b ? b_in.size(0) : b_in.size(1);

    if (k != b_k ||
        out.size(0) != m ||
        out.size(1) != n) {
        return false;
    }

    if (m > std::numeric_limits<int>::max() ||
        n > std::numeric_limits<int>::max() ||
        k > std::numeric_limits<int>::max()) {
        return false;
    }

    if (m == 0 || n == 0)
        return true;

    // A(m,0) * B(0,n) is an m x n zero matrix.
    if (k == 0) {
        out.zero_();
        return true;
    }

    at::Tensor a = a_in.contiguous();
    at::Tensor b = b_in.contiguous();

    return ptsycl_clap_blas_gemm_raw(
        a.data_ptr(),
        b.data_ptr(),
        out.data_ptr(),
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k),
        transpose_b,
        a.scalar_type() == at::kDouble,
        alpha);
#else
    (void)a_in;
    (void)b_in;
    (void)out;
    (void)alpha;
    (void)transpose_b;
    return false;
#endif
}

} // namespace clap_blas
} // namespace ptsycl
