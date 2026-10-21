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


#include "core/kernels.h"
#include "core/telemetry.h"
#include <ATen/ATen.h>
#include <torch/torch.h>
#include <ATen/ops/_native_multi_head_attention_cpu_dispatch.h>
#include <cmath>
#include <limits>



#include "clap_blas_utils.h"

namespace ptsycl {
namespace {

using namespace torch;
using torch::autograd::tensor_list;
using torch::autograd::AutogradContext;
using c10::Device;
using c10::DeviceType;

static Tensor to_cpu(const Tensor& t) {
    return t.contiguous().to(c10::Device(c10::kCPU));
}

static void upload(const Tensor& cpu_src, const Tensor& dst) {
    if (!dst.is_contiguous()) {
        const_cast<Tensor&>(dst).copy_(cpu_src);
        return;
    }
    Tensor c = cpu_src.contiguous();
    telemetry::record_transfer(telemetry::Transfer::HostToDevice,
                               static_cast<std::size_t>(c.nbytes()));
    queue_for(dst).copy(dst.data_ptr(), c.data_ptr(),
                        static_cast<std::size_t>(c.nbytes()), true);
}

static Tensor to_device(const Tensor& cpu_t, c10::Device dev) {
    Tensor result = at::empty(cpu_t.sizes(), at::device(dev).dtype(cpu_t.scalar_type()));
    upload(cpu_t, result);
    return result;
}

static c10::optional<Tensor> opt_to_cpu(const c10::optional<Tensor>& t) {
    if (!t || t->numel() == 0) return c10::nullopt;
    return to_cpu(*t);
}



Tensor convolution_overrideable(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool transposed,
    IntArrayRef output_padding,
    int64_t groups)
{
    PTSYCL_TRACE_OP("convolution_overrideable");
    c10::Device dev = input.device();

    Tensor cpu_in = to_cpu(input);
    Tensor cpu_w  = to_cpu(weight);
    auto   cpu_b  = opt_to_cpu(bias);

    Tensor cpu_out = at::convolution(
        cpu_in, cpu_w, cpu_b,
        stride, padding, dilation,
        transposed, output_padding, groups);

    Tensor result = to_device(cpu_out, dev);
    return result;
}

::std::tuple<Tensor, Tensor, Tensor> convolution_backward_overrideable(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool transposed,
    IntArrayRef output_padding,
    int64_t groups,
    ::std::array<bool, 3> output_mask)
{
    PTSYCL_TRACE_OP("convolution_backward_overrideable");
    c10::Device dev = input.device();

    Tensor cpu_dy = to_cpu(grad_output);
    Tensor cpu_x  = to_cpu(input);
    Tensor cpu_w  = to_cpu(weight);

    auto result = at::convolution_backward(
        cpu_dy, cpu_x, cpu_w,
        c10::nullopt,
        stride, padding, dilation,
        transposed, output_padding, groups,
        output_mask);

    auto cpu_dx = std::get<0>(result);
    auto cpu_dw = std::get<1>(result);
    auto cpu_db = std::get<2>(result);

    Tensor dx = cpu_dx.defined() ? to_device(cpu_dx, dev) : Tensor{};
    Tensor dw = cpu_dw.defined() ? to_device(cpu_dw, dev) : Tensor{};
    Tensor db = cpu_db.defined() ? to_device(cpu_db, dev) : Tensor{};

    return {dx, dw, db};
}

Tensor _adaptive_avg_pool2d(const Tensor& self, IntArrayRef output_size)
{
    PTSYCL_TRACE_OP("_adaptive_avg_pool2d");
    Tensor self_c = self.contiguous();
    int64_t N = self_c.size(0), C = self_c.size(1);
    int64_t H = self_c.size(2), W = self_c.size(3);
    int64_t oh = output_size[0], ow = output_size[1];

    TORCH_CHECK(
        (oh == 1 && ow == 1) || (oh == H && ow == W),
        "Only global pooling (1×1) or identity (same size) supported");

    c10::Device dev = self.device();
    auto& q = queue_for(self);

    if (oh == H && ow == W) {
        Tensor result = at::empty(self_c.sizes(), at::device(dev).dtype(self_c.scalar_type()));
        q.copy(result.data_ptr(), self_c.data_ptr(),
               static_cast<std::size_t>(self_c.nbytes()), false);
        return result;
    }

    Tensor result = at::empty({N, C, 1, 1}, at::device(dev).dtype(self_c.scalar_type()));

    AT_DISPATCH_FLOATING_TYPES(self_c.scalar_type(), "adaptive_avg_pool", [&] {
        const scalar_t* X = data_ptr<scalar_t>(self_c);
        scalar_t*       Y = data_ptr<scalar_t>(result);
        int64_t hw = H * W;
        float   inv_hw = 1.0f / static_cast<float>(hw);
        launch_flat(q, N * C, [=](std::size_t i) {
            int64_t nc   = static_cast<int64_t>(i);
            const scalar_t* src = X + nc * hw;
            float acc = 0.0f;
            for (int64_t j = 0; j < hw; j++)
                acc += static_cast<float>(src[j]);
            Y[nc] = static_cast<scalar_t>(acc * inv_hw);
        });
    });

    return result;
}

Tensor _adaptive_avg_pool2d_backward(
    const Tensor& grad_output, const Tensor& self)
{
    PTSYCL_TRACE_OP("_adaptive_avg_pool2d_backward");
    Tensor self_c      = self.contiguous();
    Tensor grad_out_c  = grad_output.contiguous();
    int64_t N = self_c.size(0), C = self_c.size(1);
    int64_t H = self_c.size(2), W = self_c.size(3);
    int64_t oh = grad_out_c.size(2), ow = grad_out_c.size(3);

    TORCH_CHECK(
        (oh == 1 && ow == 1) || (grad_out_c.sizes() == self_c.sizes()),
        "Only global or identity adaptive avg pool backward supported");

    c10::Device dev = self.device();
    auto& q = queue_for(self);
    Tensor result = at::empty(self_c.sizes(), at::device(dev).dtype(self_c.scalar_type()));

    if (oh == H && ow == W) {
        q.copy(result.data_ptr(), grad_out_c.data_ptr(),
               static_cast<std::size_t>(grad_out_c.nbytes()), false);
        return result;
    }

    AT_DISPATCH_FLOATING_TYPES(self_c.scalar_type(), "adaptive_avg_pool_bwd", [&] {
        const scalar_t* dY = data_ptr<scalar_t>(grad_out_c);
        scalar_t*       dX = data_ptr<scalar_t>(result);
        int64_t hw = H * W;
        float   inv_hw = 1.0f / static_cast<float>(hw);
        launch_flat(q, N * C * H * W, [=](std::size_t i) {
            int64_t idx  = static_cast<int64_t>(i);
            int64_t nc   = idx / hw;
            dX[idx] = static_cast<scalar_t>(static_cast<float>(dY[nc]) * inv_hw);
        });
    });

    return result;
}

Tensor& avg_pool2d_out(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override,
    Tensor& out)
{
    PTSYCL_TRACE_OP("avg_pool2d.out");
    TORCH_CHECK(!divisor_override, "Divisor override is not implemented");
    Tensor cpu_in = to_cpu(self);
    IntArrayRef eff_stride = stride.empty() ? kernel_size : stride;
    Tensor cpu_out = at::avg_pool2d(cpu_in, kernel_size, eff_stride,
                                     padding, ceil_mode, count_include_pad,
                                     divisor_override);
    upload(cpu_out, out);
    return out;
}

Tensor& avg_pool2d_backward_out(
    const Tensor& grad_output,
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override,
    Tensor& grad_input)
{
    PTSYCL_TRACE_OP("avg_pool2d_backward.grad_input");
    TORCH_CHECK(!divisor_override, "Divisor override is not implemented");
    Tensor cpu_dy = to_cpu(grad_output);
    Tensor cpu_x  = to_cpu(self);
    IntArrayRef eff_stride = stride.empty() ? kernel_size : stride;
    Tensor cpu_dx = at::avg_pool2d_backward(
        cpu_dy, cpu_x, kernel_size, eff_stride, padding,
        ceil_mode, count_include_pad, divisor_override);
    upload(cpu_dx, grad_input);
    return grad_input;
}

class MaxPool2dFunction : public torch::autograd::Function<MaxPool2dFunction> {
public:
    static Tensor forward(AutogradContext* ctx,
                          const Tensor& self,
                          IntArrayRef kernel_size,
                          IntArrayRef stride,
                          IntArrayRef padding,
                          IntArrayRef dilation,
                          bool ceil_mode)
    {
        at::AutoDispatchBelowADInplaceOrView g;
        c10::Device dev = self.device();

        Tensor cpu_in = to_cpu(self);
        Tensor cpu_out = at::max_pool2d(cpu_in, kernel_size, stride,
                                         padding, dilation, ceil_mode);
        Tensor result = to_device(cpu_out, dev);

        ctx->save_for_backward({self.contiguous()});
        ctx->saved_data["kernel_0"]  = kernel_size[0];
        ctx->saved_data["kernel_1"]  = kernel_size[1];
        ctx->saved_data["pad_0"]     = padding[0];
        ctx->saved_data["pad_1"]     = padding[1];
        ctx->saved_data["stride_0"]  = stride.empty() ? kernel_size[0] : stride[0];
        ctx->saved_data["stride_1"]  = stride.empty() ? kernel_size[1] : stride[1];
        ctx->saved_data["dil_0"]     = dilation[0];
        ctx->saved_data["dil_1"]     = dilation[1];
        ctx->saved_data["ceil_mode"] = ceil_mode;

        return result;
    }

    static tensor_list backward(AutogradContext* ctx,
                                 tensor_list grad_outputs)
    {
        Tensor input = ctx->get_saved_variables()[0];
        c10::Device dev = input.device();

        int64_t k0  = ctx->saved_data["kernel_0"].toInt();
        int64_t k1  = ctx->saved_data["kernel_1"].toInt();
        int64_t p0  = ctx->saved_data["pad_0"].toInt();
        int64_t p1  = ctx->saved_data["pad_1"].toInt();
        int64_t s0  = ctx->saved_data["stride_0"].toInt();
        int64_t s1  = ctx->saved_data["stride_1"].toInt();
        int64_t d0  = ctx->saved_data["dil_0"].toInt();
        int64_t d1  = ctx->saved_data["dil_1"].toInt();
        bool ceil_mode = ctx->saved_data["ceil_mode"].toBool();

        Tensor cpu_dy = to_cpu(grad_outputs[0]);
        Tensor cpu_x  = to_cpu(input);

        auto result_fwd = at::max_pool2d_with_indices(
            cpu_x, {k0, k1}, {s0, s1}, {p0, p1}, {d0, d1}, ceil_mode);
        auto cpu_idx = std::get<1>(result_fwd);

        Tensor cpu_dx = at::max_pool2d_with_indices_backward(
            cpu_dy, cpu_x, {k0, k1}, {s0, s1}, {p0, p1},
            {d0, d1}, ceil_mode, cpu_idx);

        Tensor dx = to_device(cpu_dx.contiguous(), dev);
        return {dx, Tensor{}, Tensor{}, Tensor{}, Tensor{}, Tensor{}};
    }
};

Tensor max_pool2d_autograd(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode)
{
    PTSYCL_TRACE_OP("max_pool2d");
    return MaxPool2dFunction::apply(self, kernel_size, stride,
                                    padding, dilation, ceil_mode);
}

struct GemmOperand {
    Tensor storage;
    bool trans = false;
    int64_t ld = 0;
    int64_t batch_stride = 0;
};

GemmOperand gemm_operand(const Tensor& t) {
    const int64_t rows = t.size(-2), cols = t.size(-1);
    const int64_t s_r = t.stride(-2), s_c = t.stride(-1);
    GemmOperand op;
    op.storage = t;
    if ((s_c == 1 || cols == 1) &&
        (rows == 1 || s_r >= std::max<int64_t>(1, cols))) {
        op.ld = rows == 1 ? std::max<int64_t>(1, cols) : s_r;
    } else if ((s_r == 1 || rows == 1) &&
               (cols == 1 || s_c >= std::max<int64_t>(1, rows))) {
        op.trans = true;
        op.ld = cols == 1 ? std::max<int64_t>(1, rows) : s_c;
    } else {
        op.storage = t.contiguous();
        op.ld = std::max<int64_t>(1, cols);
    }
    op.batch_stride = t.dim() == 3 ? op.storage.stride(0) : 0;
    return op;
}

int clap_dtype(c10::ScalarType st) {
    switch (st) {
    case c10::kFloat:    return PTSYCL_CLAP_F32;
    case c10::kDouble:   return PTSYCL_CLAP_F64;
    case c10::kHalf:     return PTSYCL_CLAP_F16;
    case c10::kBFloat16: return PTSYCL_CLAP_BF16;
    default:             return -1;
    }
}

bool fits_int(int64_t v) {
    return v >= 0 && v <= std::numeric_limits<int>::max();
}

bool device_gemm_eligible(const Tensor& a, const Tensor& b, const Tensor& out,
                          bool widen = false) {
#ifdef PTSYCL_USE_CLAP_BLAS
    if (!is_paras_tensor(a) || !is_paras_tensor(b) || !is_paras_tensor(out))
        return false;
    if (a.device() != out.device() || b.device() != out.device())
        return false;
    if (!queue_for(out).is_gpu())
        return false;
    if (clap_dtype(out.scalar_type()) < 0 || b.scalar_type() != a.scalar_type())
        return false;
    if (widen ? !((a.scalar_type() == c10::kHalf || a.scalar_type() == c10::kBFloat16) &&
                  out.scalar_type() == c10::kFloat)
              : a.scalar_type() != out.scalar_type())
        return false;
    const int64_t d = out.dim();
    if ((d != 2 && d != 3) || a.dim() != d || b.dim() != d)
        return false;
    if (d == 3 && (a.size(0) != out.size(0) || b.size(0) != out.size(0)))
        return false;
    return a.size(-2) == out.size(-2) && b.size(-1) == out.size(-1) &&
           a.size(-1) == b.size(-2) && fits_int(out.size(0)) &&
           fits_int(out.size(-2)) && fits_int(out.size(-1)) &&
           fits_int(a.size(-1));
#else
    (void)a; (void)b; (void)out; (void)widen;
    return false;
#endif
}

#if defined(PTSYCL_BACKEND_SYCL)
constexpr int64_t kGemvMaxRows = 8;
constexpr int kGemvCols = 4;
constexpr int kGemvThreads = 256;
constexpr int kGemvMaxSubGroups = kGemvThreads / 16;

template <typename T, int M> class GemvKernel;

template <typename T, int V>
struct alignas(sizeof(T) * V) GemvChunk {
    T v[V];
};

template <typename T, int M>
void small_m_gemm(compat::Queue& queue, const T* A, int64_t lda, const T* W, int64_t ldw,
                  T* C, int64_t ldc, int64_t n, int64_t k, float alpha, float beta) {
    constexpr int V = 16 / sizeof(T);
    using Ch = GemvChunk<T, V>;
    const int64_t chunks = k / V;
    const int64_t groups = (n + kGemvCols - 1) / kGemvCols;
    queue.sycl_queue().submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> part(sycl::range<1>(M * kGemvCols * kGemvMaxSubGroups), cgh);
      cgh.template parallel_for<GemvKernel<T, M>>(
        sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(groups * kGemvThreads)),
                          sycl::range<1>(kGemvThreads)),
        [=](sycl::nd_item<1> it) {
            const auto sg = it.get_sub_group();
            const int tid = static_cast<int>(it.get_local_id(0));
            const int sgid = static_cast<int>(sg.get_group_linear_id());
            const int sgs = static_cast<int>(sg.get_group_linear_range());
            const int64_t n0 = static_cast<int64_t>(it.get_group(0)) * kGemvCols;
            const T* rows[kGemvCols];
#pragma unroll
            for (int c = 0; c < kGemvCols; ++c)
                rows[c] = W + (n0 + c < n ? n0 + c : n - 1) * ldw;
            float acc[M][kGemvCols] = {};
            for (int64_t ch = tid; ch < chunks; ch += kGemvThreads) {
                Ch w[kGemvCols];
#pragma unroll
                for (int c = 0; c < kGemvCols; ++c)
                    w[c] = *reinterpret_cast<const Ch*>(rows[c] + ch * V);
#pragma unroll
                for (int m = 0; m < M; ++m) {
                    const Ch x = *reinterpret_cast<const Ch*>(A + m * lda + ch * V);
#pragma unroll
                    for (int e = 0; e < V; ++e) {
                        const float xv = static_cast<float>(x.v[e]);
#pragma unroll
                        for (int c = 0; c < kGemvCols; ++c)
                            acc[m][c] += xv * static_cast<float>(w[c].v[e]);
                    }
                }
            }
            float* pp = &part[0];
#pragma unroll
            for (int m = 0; m < M; ++m) {
#pragma unroll
                for (int c = 0; c < kGemvCols; ++c) {
                    const float v = sycl::reduce_over_group(sg, acc[m][c], sycl::plus<float>());
                    if (sg.get_local_linear_id() == 0) pp[(m * kGemvCols + c) * kGemvMaxSubGroups + sgid] = v;
                }
            }
            sycl::group_barrier(it.get_group());
            if (tid < M * kGemvCols) {
                const int m = tid / kGemvCols, c = tid % kGemvCols;
                if (n0 + c < n) {
                    float v = 0.f;
                    for (int g = 0; g < sgs; ++g) v += pp[tid * kGemvMaxSubGroups + g];
                    T* o = C + m * ldc + n0 + c;
                    const float prev = beta != 0.f ? static_cast<float>(*o) : 0.f;
                    *o = static_cast<T>(alpha * v + beta * prev);
                }
            }
        });
    });
}

bool try_small_m_gemm(const Tensor& a, const Tensor& b, Tensor& out, double alpha, double beta) {
    if (out.dim() != 2 || a.dim() != 2 || b.dim() != 2) return false;
    const int64_t m = out.size(0), n = out.size(1), k = a.size(1);
    if (m < 1 || m > kGemvMaxRows || n < 1 || k < 1) return false;
    const auto st = a.scalar_type();
    if (!(st == c10::kBFloat16 || st == c10::kHalf || st == c10::kFloat) ||
        b.scalar_type() != st || out.scalar_type() != st)
        return false;
    const int64_t V = 16 / static_cast<int64_t>(a.element_size());
    const int64_t lda = m == 1 ? k : a.stride(0), ldw = b.stride(1), ldc = m == 1 ? n : out.stride(0);
    if (a.stride(1) != 1 || b.stride(0) != 1 || out.stride(1) != 1) return false;
    if (k % V || lda % V || ldw % V || ldw < k || lda < k || ldc < n) return false;
    if (reinterpret_cast<std::uintptr_t>(a.data_ptr()) % 16 ||
        reinterpret_cast<std::uintptr_t>(b.data_ptr()) % 16)
        return false;
    auto& q = queue_for(out);
    if (!q.is_gpu()) return false;
    AT_DISPATCH_FLOATING_TYPES_AND2(c10::kHalf, c10::kBFloat16, st, "ptsycl_small_m_gemm", [&] {
        if constexpr (!std::is_same_v<scalar_t, double>) {
            const scalar_t* pa = data_ptr<scalar_t>(a);
            const scalar_t* pw = data_ptr<scalar_t>(b);
            scalar_t* pc = data_ptr<scalar_t>(out);
            const float al = static_cast<float>(alpha), be = static_cast<float>(beta);
            switch (m) {
                case 1: small_m_gemm<scalar_t, 1>(q, pa, lda, pw, ldw, pc, ldc, n, k, al, be); break;
                case 2: small_m_gemm<scalar_t, 2>(q, pa, lda, pw, ldw, pc, ldc, n, k, al, be); break;
                case 3: small_m_gemm<scalar_t, 3>(q, pa, lda, pw, ldw, pc, ldc, n, k, al, be); break;
                case 4: small_m_gemm<scalar_t, 4>(q, pa, lda, pw, ldw, pc, ldc, n, k, al, be); break;
                case 5: small_m_gemm<scalar_t, 5>(q, pa, lda, pw, ldw, pc, ldc, n, k, al, be); break;
                case 6: small_m_gemm<scalar_t, 6>(q, pa, lda, pw, ldw, pc, ldc, n, k, al, be); break;
                case 7: small_m_gemm<scalar_t, 7>(q, pa, lda, pw, ldw, pc, ldc, n, k, al, be); break;
                case 8: small_m_gemm<scalar_t, 8>(q, pa, lda, pw, ldw, pc, ldc, n, k, al, be); break;
            }
        }
    });
    PTSYCL_TRACE_OP("gemm.small_m");
    return true;
}
#endif

bool device_gemm(const Tensor& a, const Tensor& b, Tensor& out,
                 double alpha, double beta) {
#ifdef PTSYCL_USE_CLAP_BLAS
    const bool batched = out.dim() == 3;
    const int64_t batch = batched ? out.size(0) : 1;
    const int64_t m = out.size(-2), n = out.size(-1), k = a.size(-1);
    if (batch == 0 || m == 0 || n == 0) return true;
    if (k == 0) {
        if (beta == 0.0) out.zero_();
        else out.mul_(beta);
        return true;
    }

#if defined(PTSYCL_BACKEND_SYCL)
    if (!batched && try_small_m_gemm(a, b, out, alpha, beta)) return true;
#endif
    const GemmOperand A = gemm_operand(a);
    const GemmOperand B = gemm_operand(b);
    if (!fits_int(A.ld) || !fits_int(B.ld)) return false;
    const bool direct = out.stride(-1) == 1 && out.is_non_overlapping_and_dense() &&
                        out.stride(-2) >= n && fits_int(out.stride(-2));
    Tensor c = direct ? out : out.contiguous();
    const int64_t ldc = direct ? out.stride(-2) : n;
    const int64_t stride_c = batched ? (direct ? out.stride(0) : m * n) : 0;

    bool ok = false;
    const bool ran = queue_for(out).run_native([&](void* stream) {
        ok = ptsycl_clap_blas_gemm_device_raw(
            static_cast<int>(out.device().index()), A.trans, B.trans,
            static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
            alpha, A.storage.data_ptr(), static_cast<int>(A.ld),
            A.batch_stride, B.storage.data_ptr(), static_cast<int>(B.ld),
            B.batch_stride, beta, c.data_ptr(), static_cast<int>(ldc),
            stride_c, static_cast<int>(batch),
            clap_dtype(a.scalar_type()), clap_dtype(out.scalar_type()), stream);
    });
    if (!ran) return false;
    if (!ok) {
        if (a.scalar_type() == c10::kFloat || a.scalar_type() == c10::kDouble)
            return false;
        Tensor out32 = beta == 0.0 ? at::empty(out.sizes(), out.options().dtype(c10::kFloat))
                                   : out.to(c10::kFloat);
        if (!device_gemm(a.to(c10::kFloat), b.to(c10::kFloat), out32, alpha, beta))
            return false;
        out.copy_(out32);
        PTSYCL_TRACE_OP("gemm.clap_device_fp32");
        return true;
    }
    if (!c.is_same(out)) out.copy_(c);
    PTSYCL_TRACE_OP("gemm.clap_device");
    return true;
#else
    (void)a; (void)b; (void)out; (void)alpha; (void)beta;
    return false;
#endif
}

Tensor& mm_out(const Tensor& self, const Tensor& mat2, Tensor& out)
{
    PTSYCL_TRACE_OP("mm.out");

    if (device_gemm_eligible(self, mat2, out) &&
        device_gemm(self, mat2, out, 1.0, 0.0))
        return out;

    Tensor cpu_a = to_cpu(self).contiguous();
    Tensor cpu_b = to_cpu(mat2).contiguous();

    // Validate the destination before any raw device copy.
    if (cpu_a.dim() == 2 &&
        cpu_b.dim() == 2 &&
        cpu_a.size(1) == cpu_b.size(0)) {

        TORCH_CHECK(
            out.dim() == 2 &&
            out.size(0) == cpu_a.size(0) &&
            out.size(1) == cpu_b.size(1),
            "mm.out: output shape mismatch");

        TORCH_CHECK(
            out.scalar_type() == cpu_a.scalar_type(),
            "mm.out: output dtype mismatch");

        TORCH_CHECK(
            out.is_contiguous(),
            "mm.out: output must be contiguous");
    }

    if (cpu_a.dim() == 2 && cpu_b.dim() == 2 &&
        cpu_a.size(1) == cpu_b.size(0)) {

        Tensor cpu_out = at::empty(
            {cpu_a.size(0), cpu_b.size(1)},
            at::TensorOptions()
                .dtype(cpu_a.scalar_type())
                .device(c10::kCPU));

        if (clap_blas::gemm_cpu(cpu_a, cpu_b, cpu_out)) {
            PTSYCL_TRACE_OP("mm.out.clap_blas");
            upload(cpu_out, out);
            return out;
        }
    }

    PTSYCL_TRACE_OP("mm.out.fallback");
    Tensor cpu_out = at::mm(cpu_a, cpu_b);
    upload(cpu_out, out);
    return out;
}

Tensor& bmm_out(const Tensor& self, const Tensor& mat2, Tensor& out)
{
    PTSYCL_TRACE_OP("bmm.out");

    if (device_gemm_eligible(self, mat2, out) &&
        device_gemm(self, mat2, out, 1.0, 0.0))
        return out;
    Tensor cpu_out = at::bmm(to_cpu(self), to_cpu(mat2));
    upload(cpu_out, out);
    return out;
}

Tensor& addmm_out(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out)
{
    PTSYCL_TRACE_OP("addmm.out");

    if (device_gemm_eligible(mat1, mat2, out) &&
        self.scalar_type() == out.scalar_type() &&
        is_paras_tensor(self) && self.device() == out.device() &&
        at::is_expandable_to(self.sizes(), out.sizes())) {
        const double b = beta.toDouble();
        if (b != 0.0 && !self.is_same(out))
            out.copy_(self.expand(out.sizes()));
        if (device_gemm(mat1, mat2, out, alpha.toDouble(), b))
            return out;
    }

    Tensor cpu_self = to_cpu(self).contiguous();
    Tensor cpu_a = to_cpu(mat1).contiguous();
    Tensor cpu_b = to_cpu(mat2).contiguous();

    // Validate the destination before any raw device copy.
    if (cpu_a.dim() == 2 &&
        cpu_b.dim() == 2 &&
        cpu_a.size(1) == cpu_b.size(0)) {

        TORCH_CHECK(
            out.dim() == 2 &&
            out.size(0) == cpu_a.size(0) &&
            out.size(1) == cpu_b.size(1),
            "addmm.out: output shape mismatch");

        TORCH_CHECK(
            out.scalar_type() == cpu_a.scalar_type(),
            "addmm.out: output dtype mismatch");

        TORCH_CHECK(
            out.is_contiguous(),
            "addmm.out: output must be contiguous");
    }

    const int64_t out_rows =
        cpu_a.dim() == 2 ? cpu_a.size(0) : 0;
    const int64_t out_cols =
        cpu_b.dim() == 2 ? cpu_b.size(1) : 0;

    const bool self_broadcastable =
        cpu_self.dim() == 0 ||
        (cpu_self.dim() == 1 &&
         (cpu_self.size(0) == 1 ||
          cpu_self.size(0) == out_cols)) ||
        (cpu_self.dim() == 2 &&
         (cpu_self.size(0) == 1 ||
          cpu_self.size(0) == out_rows) &&
         (cpu_self.size(1) == 1 ||
          cpu_self.size(1) == out_cols));

    if (cpu_a.dim() == 2 &&
        cpu_b.dim() == 2 &&
        cpu_a.size(1) == cpu_b.size(0) &&
        cpu_a.scalar_type() == cpu_b.scalar_type() &&
        cpu_self.scalar_type() == cpu_a.scalar_type() &&
        out.scalar_type() == cpu_a.scalar_type() &&
        out.dim() == 2 &&
        out.size(0) == cpu_a.size(0) &&
        out.size(1) == cpu_b.size(1) &&
        self_broadcastable &&
        clap_blas::supported_dtype(cpu_a)) {

        Tensor cpu_out = at::empty(
            {cpu_a.size(0), cpu_b.size(1)},
            at::TensorOptions()
                .dtype(cpu_a.scalar_type())
                .device(c10::kCPU));

        if (clap_blas::gemm_cpu(
                cpu_a,
                cpu_b,
                cpu_out,
                alpha.toDouble())) {

            PTSYCL_TRACE_OP("addmm.out.clap_blas");

            if (beta.toDouble() != 0.0)
                cpu_out.add_(cpu_self, beta);
            upload(cpu_out, out);
            return out;
        }
    }

    PTSYCL_TRACE_OP("addmm.out.fallback");
    Tensor cpu_out =
        at::addmm(cpu_self, cpu_a, cpu_b, beta, alpha);
    upload(cpu_out, out);
    return out;
}

::std::tuple<Tensor&, Tensor&> _native_multi_head_attention_out(
    const Tensor& query, const Tensor& key, const Tensor& value,
    int64_t embed_dim, int64_t num_head,
    const Tensor& qkv_weight, const Tensor& qkv_bias,
    const Tensor& proj_weight, const Tensor& proj_bias,
    const ::std::optional<Tensor>& mask,
    bool need_weights, bool average_attn_weights,
    ::std::optional<int64_t> mask_type,
    Tensor& out0, Tensor& out1)
{
    PTSYCL_TRACE_OP("_native_multi_head_attention.out");
    auto r = at::cpu::_native_multi_head_attention(
        query, key, value, embed_dim, num_head,
        qkv_weight, qkv_bias, proj_weight, proj_bias,
        mask, need_weights, average_attn_weights, mask_type);
    out0.copy_(std::get<0>(r));
    out1.copy_(std::get<1>(r));
    return {out0, out1};
}

::std::tuple<Tensor, Tensor> _native_multi_head_attention(
    const Tensor& query, const Tensor& key, const Tensor& value,
    int64_t embed_dim, int64_t num_head,
    const Tensor& qkv_weight, const Tensor& qkv_bias,
    const Tensor& proj_weight, const Tensor& proj_bias,
    const ::std::optional<Tensor>& mask,
    bool need_weights, bool average_attn_weights,
    ::std::optional<int64_t> mask_type)
{
    PTSYCL_TRACE_OP("_native_multi_head_attention");
    return at::cpu::_native_multi_head_attention(
        query, key, value, embed_dim, num_head,
        qkv_weight, qkv_bias, proj_weight, proj_bias,
        mask, need_weights, average_attn_weights, mask_type);
}

std::tuple<Tensor, Tensor, Tensor> transform_bias_rescale_qkv(
    const Tensor& qkv, const Tensor& qkv_bias, int64_t num_head)
{
    PTSYCL_TRACE_OP("_transform_bias_rescale_qkv");
    auto qkv_ = qkv.is_nested()
        ? c10::MaybeOwned<Tensor>::owned(qkv.to_padded_tensor(0))
        : c10::MaybeOwned<Tensor>::borrowed(qkv);

    int64_t B        = qkv_->size(0);
    int64_t T        = qkv_->size(1);
    int64_t _3D      = qkv_->size(2);
    int64_t D        = _3D / 3;
    TORCH_CHECK(D % num_head == 0, "D must be divisible by num_head");
    TORCH_CHECK(_3D % 3 == 0,      "last dim must be divisible by 3");
    int64_t dim_per_head = D / num_head;

    const auto qkv_contig      = qkv_->expect_contiguous();
    const auto qkv_bias_contig = qkv_bias.expect_contiguous();

    Tensor q_k_v_same_order = at::empty(
        {B, T, 3, num_head, dim_per_head}, qkv_->options());

    auto& q = queue_for(*qkv_);

    int64_t BT  = B * T;
    double  scale = 1.0 / std::sqrt(static_cast<double>(dim_per_head));
    float   fscale = static_cast<float>(scale);
    int64_t fD  = D;

    AT_DISPATCH_FLOATING_TYPES(qkv_contig->scalar_type(), "transform_qkv", [&] {
        const scalar_t* QKV  = data_ptr<scalar_t>(*qkv_contig);
        const scalar_t* BIAS = data_ptr<scalar_t>(*qkv_bias_contig);
        scalar_t*       OUT  = data_ptr<scalar_t>(q_k_v_same_order);
        int64_t ld = _3D;

        launch_flat(q, BT * _3D, [=](std::size_t i) {
            int64_t bt  = static_cast<int64_t>(i) / ld;
            int64_t d1  = static_cast<int64_t>(i) % ld;
            float   s   = (d1 < fD) ? fscale : 1.0f;
            scalar_t val = (QKV[bt * ld + d1] + BIAS[d1])
                           * static_cast<scalar_t>(s);
            OUT[bt * ld + d1] = val;
        });
    });

    Tensor q_k_v = at::empty({3, B, num_head, T, dim_per_head},
                                  qkv_->options());
    Tensor tmp = q_k_v_same_order;
    tmp = torch::transpose(tmp, 1, 2);
    tmp = torch::transpose(tmp, 2, 3);
    tmp = torch::transpose(tmp, 0, 1);
    q_k_v.copy_(tmp);

    auto q_k_v_s = at::native::split(
        q_k_v.view({3 * B, num_head, T, dim_per_head}), B, 0);

    return {q_k_v_s[0], q_k_v_s[1], q_k_v_s[2]};
}

static Tensor& upsample_fwd_cpu(
    const Tensor& self,
    c10::SymIntArrayRef /*output_size*/,
    Tensor& out,
    std::function<Tensor(const Tensor&)> cpu_fn)
{
    Tensor cpu_in  = to_cpu(self);
    Tensor cpu_out = cpu_fn(cpu_in);
    upload(cpu_out, out);
    return out;
}

static Tensor& upsample_bwd_cpu(
    const Tensor& grad_output,
    Tensor& grad_input,
    std::function<Tensor(const Tensor&)> cpu_fn)
{
    Tensor cpu_dy  = to_cpu(grad_output);
    Tensor cpu_dx  = cpu_fn(cpu_dy);
    upload(cpu_dx, grad_input);
    return grad_input;
}

Tensor& upsample_nearest2d_out(
    const Tensor& self,
    c10::SymIntArrayRef output_size,
    ::std::optional<double> scales_h,
    ::std::optional<double> scales_w,
    Tensor& out)
{
    PTSYCL_TRACE_OP("upsample_nearest2d.out");
    std::vector<int64_t> os;
    for (auto v : output_size) os.push_back(v.expect_int());
    return upsample_fwd_cpu(self, output_size, out, [&](const Tensor& x) {
        return at::upsample_nearest2d(x, c10::IntArrayRef(os), scales_h, scales_w);
    });
}

Tensor& upsample_nearest2d_backward_out(
    const Tensor& grad_output,
    c10::SymIntArrayRef output_size,
    c10::SymIntArrayRef input_size,
    ::std::optional<double> scales_h,
    ::std::optional<double> scales_w,
    Tensor& grad_input)
{
    PTSYCL_TRACE_OP("upsample_nearest2d_backward.grad_input");
    std::vector<int64_t> os, is;
    for (auto v : output_size) os.push_back(v.expect_int());
    for (auto v : input_size)  is.push_back(v.expect_int());
    return upsample_bwd_cpu(grad_output, grad_input, [&](const Tensor& dy) {
        return at::upsample_nearest2d_backward(dy,
            c10::IntArrayRef(os), c10::IntArrayRef(is), scales_h, scales_w);
    });
}

Tensor& _upsample_nearest_exact2d_out(
    const Tensor& self,
    c10::SymIntArrayRef output_size,
    ::std::optional<double> scales_h,
    ::std::optional<double> scales_w,
    Tensor& out)
{
    PTSYCL_TRACE_OP("_upsample_nearest_exact2d.out");
    std::vector<int64_t> os;
    for (auto v : output_size) os.push_back(v.expect_int());
    return upsample_fwd_cpu(self, output_size, out, [&](const Tensor& x) {
        return at::_upsample_nearest_exact2d(x, c10::IntArrayRef(os),
                                              scales_h, scales_w);
    });
}

Tensor& _upsample_nearest_exact2d_backward_out(
    const Tensor& grad_output,
    c10::SymIntArrayRef output_size,
    c10::SymIntArrayRef input_size,
    ::std::optional<double> scales_h,
    ::std::optional<double> scales_w,
    Tensor& grad_input)
{
    PTSYCL_TRACE_OP("_upsample_nearest_exact2d_backward.grad_input");
    std::vector<int64_t> os, is;
    for (auto v : output_size) os.push_back(v.expect_int());
    for (auto v : input_size)  is.push_back(v.expect_int());
    return upsample_bwd_cpu(grad_output, grad_input, [&](const Tensor& dy) {
        return at::_upsample_nearest_exact2d_backward(dy,
            c10::IntArrayRef(os), c10::IntArrayRef(is), scales_h, scales_w);
    });
}

Tensor& upsample_bilinear2d_out(
    const Tensor& self,
    c10::SymIntArrayRef output_size,
    bool align_corners,
    ::std::optional<double> scales_h,
    ::std::optional<double> scales_w,
    Tensor& out)
{
    PTSYCL_TRACE_OP("upsample_bilinear2d.out");
    std::vector<int64_t> os;
    for (auto v : output_size) os.push_back(v.expect_int());
    return upsample_fwd_cpu(self, output_size, out, [&](const Tensor& x) {
        return at::upsample_bilinear2d(x, c10::IntArrayRef(os),
                                        align_corners, scales_h, scales_w);
    });
}

Tensor& upsample_bilinear2d_backward_out(
    const Tensor& grad_output,
    c10::SymIntArrayRef output_size,
    c10::SymIntArrayRef input_size,
    bool align_corners,
    ::std::optional<double> scales_h,
    ::std::optional<double> scales_w,
    Tensor& grad_input)
{
    PTSYCL_TRACE_OP("upsample_bilinear2d_backward.grad_input");
    std::vector<int64_t> os, is;
    for (auto v : output_size) os.push_back(v.expect_int());
    for (auto v : input_size)  is.push_back(v.expect_int());
    return upsample_bwd_cpu(grad_output, grad_input, [&](const Tensor& dy) {
        return at::upsample_bilinear2d_backward(dy,
            c10::IntArrayRef(os), c10::IntArrayRef(is),
            align_corners, scales_h, scales_w);
    });
}

} // namespace

bool device_bmm_widened(const Tensor& a, const Tensor& b, Tensor& out) {
    return device_gemm_eligible(a, b, out, true) &&
           device_gemm(a, b, out, 1.0, 0.0);
}

} // namespace ptsycl

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("aten::convolution_overrideable",
           &ptsycl::convolution_overrideable);
    m.impl("aten::convolution_backward_overrideable",
           &ptsycl::convolution_backward_overrideable);
    m.impl("aten::_adaptive_avg_pool2d",
           &ptsycl::_adaptive_avg_pool2d);
    m.impl("aten::_adaptive_avg_pool2d_backward",
           &ptsycl::_adaptive_avg_pool2d_backward);
    m.impl("aten::avg_pool2d.out",
           &ptsycl::avg_pool2d_out);
    m.impl("aten::avg_pool2d_backward.grad_input",
           &ptsycl::avg_pool2d_backward_out);
    m.impl("aten::mm.out",
           &ptsycl::mm_out);
    m.impl("aten::bmm.out",
           &ptsycl::bmm_out);
    m.impl("aten::addmm.out",
           &ptsycl::addmm_out);
    m.impl("aten::_native_multi_head_attention.out",
           &ptsycl::_native_multi_head_attention_out);
    m.impl("aten::_native_multi_head_attention",
           &ptsycl::_native_multi_head_attention);
    m.impl("aten::_transform_bias_rescale_qkv",
           &ptsycl::transform_bias_rescale_qkv);
    m.impl("aten::upsample_nearest2d.out",
           &ptsycl::upsample_nearest2d_out);
    m.impl("aten::upsample_nearest2d_backward.grad_input",
           &ptsycl::upsample_nearest2d_backward_out);
    m.impl("aten::_upsample_nearest_exact2d.out",
           &ptsycl::_upsample_nearest_exact2d_out);
    m.impl("aten::_upsample_nearest_exact2d_backward.grad_input",
           &ptsycl::_upsample_nearest_exact2d_backward_out);
    m.impl("aten::upsample_bilinear2d.out",
           &ptsycl::upsample_bilinear2d_out);
    m.impl("aten::upsample_bilinear2d_backward.grad_input",
           &ptsycl::upsample_bilinear2d_backward_out);
}

TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
    m.impl("aten::max_pool2d", &ptsycl::max_pool2d_autograd);
}
