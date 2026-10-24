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
#include <optional>

#include <ATen/ATen.h>
#include <torch/torch.h>

#include "core/kernels.h"
#include "core/telemetry.h"
#include "ops/clap_blas_wrapper.h"

namespace ptsycl {
namespace {

using at::Tensor;

struct Operand {
    Tensor t;
    bool trans;
    int64_t ld;
};

Operand operand(const Tensor& x) {
    if (x.stride(1) == 1 && x.stride(0) >= x.size(1)) return {x, false, x.stride(0)};
    if (x.stride(0) == 1 && x.stride(1) >= x.size(0)) return {x, true, x.stride(1)};
    Tensor c = x.contiguous();
    return {c, false, c.size(1)};
}

bool fits_int(int64_t v) { return v >= 0 && v <= std::numeric_limits<int>::max(); }

#ifdef PTSYCL_USE_CLAP_BLAS
int clap_fp8(c10::ScalarType t) {
    switch (t) {
    case c10::kFloat8_e4m3fn: return PTSYCL_CLAP_F8E4M3;
    case c10::kFloat8_e5m2: return PTSYCL_CLAP_F8E5M2;
    case c10::kFloat8_e4m3fnuz: return PTSYCL_CLAP_F8E4M3FNUZ;
    case c10::kFloat8_e5m2fnuz: return PTSYCL_CLAP_F8E5M2FNUZ;
    case c10::kFloat: return PTSYCL_CLAP_F32;
    case c10::kHalf: return PTSYCL_CLAP_F16;
    case c10::kBFloat16: return PTSYCL_CLAP_BF16;
    default: return -1;
    }
}
#endif

void check_2d(const Tensor& a, const Tensor& b, const char* name) {
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, name, ": expected 2-D tensors");
    TORCH_CHECK(a.size(1) == b.size(0), name, ": a and b shapes cannot be multiplied (", a.sizes(), " and ",
                b.sizes(), ")");
    TORCH_CHECK(a.device() == b.device(), name, ": a and b on different devices");
}

Tensor& int_mm_out(const Tensor& self, const Tensor& mat2, Tensor& out) {
    PTSYCL_TRACE_OP("_int_mm.out");
    check_2d(self, mat2, "_int_mm");
    TORCH_CHECK(self.scalar_type() == c10::kChar && mat2.scalar_type() == c10::kChar,
                "_int_mm: expected int8 operands");
    TORCH_CHECK(out.scalar_type() == c10::kInt, "_int_mm: expected an int32 result");
    const int64_t m = self.size(0), n = mat2.size(1), k = self.size(1);
    out.resize_({m, n});
    if (m == 0 || n == 0) return out;
    if (k == 0) return out.zero_();
#ifdef PTSYCL_USE_CLAP_BLAS
    const Operand A = operand(self), B = operand(mat2);
    if (fits_int(m) && fits_int(n) && fits_int(k) && fits_int(A.ld) && fits_int(B.ld)) {
        Tensor c = out.is_contiguous() ? out : at::empty({m, n}, out.options());
        bool ok = false;
        const bool ran = queue_for(out).run_native([&](void* stream) {
            ok = ptsycl_clap_blas_gemm_device_raw(
                static_cast<int>(out.device().index()), A.trans, B.trans, static_cast<int>(m),
                static_cast<int>(n), static_cast<int>(k), 1.0, A.t.data_ptr(), static_cast<int>(A.ld), 0,
                B.t.data_ptr(), static_cast<int>(B.ld), 0, 0.0, c.data_ptr(), static_cast<int>(n), 0, 1,
                PTSYCL_CLAP_I8, PTSYCL_CLAP_I32, stream);
        });
        if (ran && ok) {
            if (!c.is_same(out)) out.copy_(c);
            return out;
        }
    }
#endif
    return out.copy_(at::mm(self.to(c10::kDouble), mat2.to(c10::kDouble)));
}

Tensor int_mm(const Tensor& self, const Tensor& mat2) {
    PTSYCL_TRACE_OP("_int_mm");
    Tensor out = at::empty({0}, self.options().dtype(c10::kInt));
    return int_mm_out(self, mat2, out);
}

bool is_fp8(c10::ScalarType t) {
    return t == c10::kFloat8_e4m3fn || t == c10::kFloat8_e5m2 || t == c10::kFloat8_e4m3fnuz ||
           t == c10::kFloat8_e5m2fnuz;
}

Tensor& scaled_mm_out(const Tensor& self, const Tensor& mat2, const Tensor& scale_a, const Tensor& scale_b,
                      const std::optional<Tensor>& bias, const std::optional<Tensor>& scale_result,
                      std::optional<c10::ScalarType> out_dtype, bool , Tensor& out) {
    PTSYCL_TRACE_OP("_scaled_mm.out");
    check_2d(self, mat2, "_scaled_mm");
    TORCH_CHECK(is_fp8(self.scalar_type()) && is_fp8(mat2.scalar_type()), "_scaled_mm: expected fp8 operands");
    TORCH_CHECK(scale_a.scalar_type() == c10::kFloat && scale_b.scalar_type() == c10::kFloat,
                "_scaled_mm: scales must be float32");
    const int64_t m = self.size(0), n = mat2.size(1), k = self.size(1);
    out.resize_({m, n});
    if (m == 0 || n == 0) return out;
    const auto ot = out.scalar_type();
#ifdef PTSYCL_USE_CLAP_BLAS
    const bool per_tensor = scale_a.numel() == 1 && scale_b.numel() == 1;
    if (per_tensor && k > 0 && clap_fp8(ot) >= 0 && !is_fp8(ot) && fits_int(m) && fits_int(n) && fits_int(k)) {
        const Operand A = operand(self), B = operand(mat2);
        const Tensor sa = scale_a.contiguous(), sb = scale_b.contiguous();
        Tensor c = out.is_contiguous() ? out : at::empty({m, n}, out.options());
        bool ok = false;
        if (fits_int(A.ld) && fits_int(B.ld)) {
            const bool ran = queue_for(out).run_native([&](void* stream) {
                ok = ptsycl_clap_blas_gemm_scaled_raw(
                    static_cast<int>(out.device().index()), A.trans, B.trans, static_cast<int>(m),
                    static_cast<int>(n), static_cast<int>(k), A.t.data_ptr(), static_cast<int>(A.ld),
                    B.t.data_ptr(), static_cast<int>(B.ld), c.data_ptr(), static_cast<int>(n),
                    clap_fp8(self.scalar_type()), clap_fp8(mat2.scalar_type()), clap_fp8(ot), sa.data_ptr(),
                    sb.data_ptr(), stream);
            });
            ok = ok && ran;
        }
        if (ok) {
            if (bias && bias->defined()) c.add_(bias->to(ot));
            if (!c.is_same(out)) out.copy_(c);
            return out;
        }
    }
#endif
    Tensor r = at::mm(self.to(c10::kFloat), mat2.to(c10::kFloat)).mul_(scale_a).mul_(scale_b);
    if (bias && bias->defined()) r.add_(bias->to(c10::kFloat));
    if (scale_result && scale_result->defined() && is_fp8(ot)) r.mul_(*scale_result);
    return out.copy_(r);
}

Tensor scaled_mm(const Tensor& self, const Tensor& mat2, const Tensor& scale_a, const Tensor& scale_b,
                 const std::optional<Tensor>& bias, const std::optional<Tensor>& scale_result,
                 std::optional<c10::ScalarType> out_dtype, bool use_fast_accum) {
    PTSYCL_TRACE_OP("_scaled_mm");
    Tensor out = at::empty({0}, self.options().dtype(out_dtype.value_or(self.scalar_type())));
    return scaled_mm_out(self, mat2, scale_a, scale_b, bias, scale_result, out_dtype, use_fast_accum, out);
}

}
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("aten::_int_mm", &ptsycl::int_mm);
    m.impl("aten::_int_mm.out", &ptsycl::int_mm_out);
    m.impl("aten::_scaled_mm", &ptsycl::scaled_mm);
    m.impl("aten::_scaled_mm.out", &ptsycl::scaled_mm_out);
}
