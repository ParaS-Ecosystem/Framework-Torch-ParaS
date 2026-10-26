#pragma once

#include <ATen/ATen.h>

#include <limits>

#include "clap_blas_wrapper.h"
#include "core/common.h"

namespace ptsycl {
namespace clap_blas {

inline bool supported_dtype(const at::Tensor& t)
{
    return t.scalar_type() == at::kFloat ||
           t.scalar_type() == at::kDouble;
}

inline bool supported_device_dtype(const at::Tensor& t)
{
    return supported_dtype(t) ||
           t.scalar_type() == at::kBFloat16;
}

inline bool gemm_device(
    const at::Tensor& a,
    const at::Tensor& b,
    at::Tensor& out,
    double alpha = 1.0,
    bool transpose_b = false,
    double beta = 0.0)
{
#ifdef PTSYCL_USE_CLAP_BLAS
    if (!is_paras_tensor(a) ||
        !is_paras_tensor(b) ||
        !is_paras_tensor(out) ||
        a.device() != b.device() ||
        a.device() != out.device() ||
        !supported_device_dtype(a) ||
        a.scalar_type() != b.scalar_type() ||
        a.scalar_type() != out.scalar_type() ||
        a.dim() != 2 ||
        b.dim() != 2 ||
        out.dim() != 2 ||
        !a.is_contiguous() ||
        !b.is_contiguous() ||
        !out.is_contiguous()) {
        return false;
    }

    auto& q = queue_for(out);
    if (!q.is_gpu())
        return false;

    const int64_t m = a.size(0);
    const int64_t k = a.size(1);
    const int64_t b_k = transpose_b ? b.size(1) : b.size(0);
    const int64_t n = transpose_b ? b.size(0) : b.size(1);

    if (k != b_k ||
        out.size(0) != m ||
        out.size(1) != n ||
        m > std::numeric_limits<int>::max() ||
        n > std::numeric_limits<int>::max() ||
        k > std::numeric_limits<int>::max()) {
        return false;
    }

    if (m == 0 || n == 0)
        return true;

    if (k == 0) {
        if (beta == 0.0) {
            out.zero_();
        } else if (beta != 1.0) {
            out.mul_(beta);
        }
        return true;
    }

    const void* a_ptr = a.data_ptr();
    const void* b_ptr = b.data_ptr();
    void* out_ptr = out.data_ptr();
    const bool is_double = a.scalar_type() == at::kDouble;
    const bool is_bf16 = a.scalar_type() == at::kBFloat16;

#if defined(PTSYCL_CLAP_BLAS_CUDA)
    q.sycl_queue().parasSYCL_enqueue_custom_operation(
        [=](sycl::interop_handle ih) {
            auto stream = ih.get_native_queue<sycl::backend::cuda>();

            TORCH_CHECK(
                stream != nullptr,
                "Torch-ParaS CLAP-BLAS received a null CUDA stream");

            TORCH_CHECK(
                ptsycl_clap_blas_gemm_device_raw(
                    a_ptr,
                    b_ptr,
                    out_ptr,
                    static_cast<int>(m),
                    static_cast<int>(n),
                    static_cast<int>(k),
                    transpose_b,
                    is_double,
                    is_bf16,
                    alpha,
                    beta,
                    reinterpret_cast<void*>(stream),
                    1),
                "CLAP-BLAS CUDA device GEMM failed");
        });
    return true;
#elif defined(PTSYCL_CLAP_BLAS_ROCM)
    q.sycl_queue().parasSYCL_enqueue_custom_operation(
        [=](sycl::interop_handle ih) {
            auto stream = ih.get_native_queue<sycl::backend::hip>();

            TORCH_CHECK(
                stream != nullptr,
                "Torch-ParaS CLAP-BLAS received a null HIP stream");

            TORCH_CHECK(
                ptsycl_clap_blas_gemm_device_raw(
                    a_ptr,
                    b_ptr,
                    out_ptr,
                    static_cast<int>(m),
                    static_cast<int>(n),
                    static_cast<int>(k),
                    transpose_b,
                    is_double,
                    is_bf16,
                    alpha,
                    beta,
                    reinterpret_cast<void*>(stream),
                    2),
                "CLAP-BLAS ROCm device GEMM failed");
        });
    return true;
#else
    return false;
#endif
#else
    (void)a;
    (void)b;
    (void)out;
    (void)alpha;
    (void)transpose_b;
    (void)beta;
    return false;
#endif
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
