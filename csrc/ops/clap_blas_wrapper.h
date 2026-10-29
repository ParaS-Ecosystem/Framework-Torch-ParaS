#pragma once

extern "C" bool ptsycl_clap_blas_gemm_raw(
    const void* a,
    const void* b,
    void* out,
    int m,
    int n,
    int k,
    bool transpose_b,
    bool is_double,
    double alpha) noexcept;

enum {
    PTSYCL_CLAP_F32 = 0,
    PTSYCL_CLAP_F64 = 1,
    PTSYCL_CLAP_F16 = 2,
    PTSYCL_CLAP_BF16 = 3,
};

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
    void* stream) noexcept;
