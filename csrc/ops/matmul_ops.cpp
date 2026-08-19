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
//
// !!! DO NOT ADD THIS FILE TO CMakeLists.txt WITHOUT CHECKING FIRST !!!
//
// `aten::matmul` is CompositeImplicitAutograd upstream: it inspects
// self/other's dims and decomposes into aten::dot / aten::mv / aten::mm /
// aten::bmm. If your backend has PrivateUse1 registrations for those --
// and the README's "multi-head attention already implemented" strongly
// implies it does, since MHA needs batched GEMM -- then:
//   grep -n "aten::mm\|aten::bmm\|aten::matmul\|aten::addmm" csrc/ops/*.cpp
// A PrivateUse1-specific override of a composite op (registering
// aten::matmul directly, as this file also does at the bottom) takes
// priority over the composite decomposition, so it's *not* a hard load
// -time abort the way double-registering aten::copy_ was -- but it does
// mean your existing (presumably tuned/vendor-GEMM-backed) mm/bmm would
// go completely unused in favor of the naive O(n^3) kernels below.
// Registering aten::mm/aten::bmm directly IS a hard collision (duplicate
// dispatch key registration -> abort at .so load) if they already exist.
//
// The kernels below are a correctness-first triple-loop GEMM (one output
// element per thread, serial reduction over k) -- fine for tests and
// small shapes, not competitive with a real tuned kernel or vendor BLAS
// for anything performance-sensitive. Treat this as a stopgap only if
// mm/bmm genuinely don't exist yet.
// -----------------------------------------------------------------------------

#include "core/kernels.h"

namespace ptsycl {
namespace {

using at::Tensor;

// -----------------------------------------------------------------------------
// mm: (n, k) @ (k, p) -> (n, p), both 2-D, no broadcasting.
// -----------------------------------------------------------------------------
Tensor& mm_out(const Tensor& self, const Tensor& mat2, Tensor& out) {
    PTSYCL_TRACE_OP("mm.out");
    TORCH_CHECK(self.dim() == 2 && mat2.dim() == 2,
                "paras mm: both inputs must be 2-D");
    TORCH_CHECK(self.size(1) == mat2.size(0),
                "paras mm: shape mismatch (", self.sizes(), " vs ", mat2.sizes(), ")");

    auto& q = queue_for(self);
    const int64_t n = self.size(0), kdim = self.size(1), p = mat2.size(1);
    const int64_t total = n * p;
    if (total == 0) return out;

    Tensor a = self.contiguous();
    Tensor b = mat2.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        c10::kHalf, c10::kBFloat16, self.scalar_type(), "ptsycl_mm", [&] {
            const scalar_t* pa = data_ptr<scalar_t>(a);
            const scalar_t* pb = data_ptr<scalar_t>(b);
            scalar_t*       po = data_ptr<scalar_t>(out);

            launch_flat(q, total, [=](std::size_t flat_) {
                const int64_t flat = static_cast<int64_t>(flat_);
                const int64_t i = flat / p;
                const int64_t j = flat % p;
                // accumulate in float for half/bfloat16 to avoid the
                // obvious precision blowup a k-length serial sum in
                // native low-precision arithmetic would otherwise have.
                float acc = 0.f;
                for (int64_t kk = 0; kk < kdim; ++kk)
                    acc += static_cast<float>(pa[i * kdim + kk]) *
                           static_cast<float>(pb[kk * p + j]);
                po[flat] = static_cast<scalar_t>(acc);
            });
        });
    return out;
}

Tensor mm(const Tensor& self, const Tensor& mat2) {
    PTSYCL_TRACE_OP("mm");
    Tensor out = at::empty({self.size(0), mat2.size(1)}, self.options());
    return ptsycl::mm_out(self, mat2, out);
}

// -----------------------------------------------------------------------------
// bmm: (B, n, k) @ (B, k, p) -> (B, n, p). Batch dims must already match
// exactly (no broadcasting -- that's matmul's job, below).
// -----------------------------------------------------------------------------
Tensor& bmm_out(const Tensor& self, const Tensor& mat2, Tensor& out) {
    PTSYCL_TRACE_OP("bmm.out");
    TORCH_CHECK(self.dim() == 3 && mat2.dim() == 3,
                "paras bmm: both inputs must be 3-D (B, n, k) / (B, k, p)");
    TORCH_CHECK(self.size(0) == mat2.size(0) && self.size(2) == mat2.size(1),
                "paras bmm: shape mismatch (", self.sizes(), " vs ", mat2.sizes(), ")");

    auto& q = queue_for(self);
    const int64_t B = self.size(0), n = self.size(1), kdim = self.size(2), p = mat2.size(2);
    const int64_t total = B * n * p;
    if (total == 0) return out;

    Tensor a = self.contiguous();
    Tensor b = mat2.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        c10::kHalf, c10::kBFloat16, self.scalar_type(), "ptsycl_bmm", [&] {
            const scalar_t* pa = data_ptr<scalar_t>(a);
            const scalar_t* pb = data_ptr<scalar_t>(b);
            scalar_t*       po = data_ptr<scalar_t>(out);

            launch_flat(q, total, [=](std::size_t flat_) {
                const int64_t flat = static_cast<int64_t>(flat_);
                const int64_t bi = flat / (n * p);
                const int64_t rem = flat % (n * p);
                const int64_t i = rem / p;
                const int64_t j = rem % p;
                const scalar_t* pa_b = pa + bi * n * kdim;
                const scalar_t* pb_b = pb + bi * kdim * p;

                float acc = 0.f;
                for (int64_t kk = 0; kk < kdim; ++kk)
                    acc += static_cast<float>(pa_b[i * kdim + kk]) *
                           static_cast<float>(pb_b[kk * p + j]);
                po[flat] = static_cast<scalar_t>(acc);
            });
        });
    return out;
}

Tensor bmm(const Tensor& self, const Tensor& mat2) {
    PTSYCL_TRACE_OP("bmm");
    Tensor out = at::empty({self.size(0), self.size(1), mat2.size(2)}, self.options());
    return ptsycl::bmm_out(self, mat2, out);
}

// -----------------------------------------------------------------------------
// matmul: dimension-dispatching wrapper, mirroring upstream's own
// CompositeImplicitAutograd decomposition (aten/src/ATen/native/
// LinearAlgebra.cpp) closely enough for the common cases. Built entirely
// on mm/bmm above (or whatever's already registered for them) -- no new
// low-level loop here.
// -----------------------------------------------------------------------------
Tensor matmul(const Tensor& self, const Tensor& other) {
    PTSYCL_TRACE_OP("matmul");
    const int64_t d1 = self.dim(), d2 = other.dim();
    TORCH_CHECK(d1 >= 1 && d2 >= 1, "paras matmul: both inputs must have >= 1 dim");

    if (d1 == 1 && d2 == 1) {
        // dot product -> 0-D scalar
        return ptsycl::mm(self.unsqueeze(0), other.unsqueeze(1)).reshape({});
    }
    if (d1 == 2 && d2 == 2) {
        return ptsycl::mm(self, other);
    }
    if (d1 == 1 && d2 == 2) {
        return ptsycl::mm(self.unsqueeze(0), other).squeeze(0);
    }
    if (d1 == 2 && d2 == 1) {
        return ptsycl::mm(self, other.unsqueeze(1)).squeeze(1);
    }

    // Batched case: at least one side has dim > 2. Fold to 3-D (batch, n, k)
    // after broadcasting leading batch dims, run bmm, then reshape/squeeze
    // back. Vector operands get a temporary size-1 dim added on the side
    // that needs it and removed from the result, same trick upstream uses.
    Tensor a = self, b = other;
    const bool a_is_vec = (a.dim() == 1);
    const bool b_is_vec = (b.dim() == 1);
    if (a_is_vec) a = a.unsqueeze(0);
    if (b_is_vec) b = b.unsqueeze(-1);

    const int64_t n = a.size(-2), k = a.size(-1), p = b.size(-1);
    TORCH_CHECK(b.size(-2) == k,
                "paras matmul: shape mismatch (", self.sizes(), " vs ", other.sizes(), ")");

    auto a_batch = a.sizes().slice(0, a.dim() - 2);
    auto b_batch = b.sizes().slice(0, b.dim() - 2);
    std::vector<int64_t> batch_shape = at::infer_size(a_batch, b_batch);

    std::vector<int64_t> a_expand = batch_shape; a_expand.push_back(n); a_expand.push_back(k);
    std::vector<int64_t> b_expand = batch_shape; b_expand.push_back(k); b_expand.push_back(p);
    Tensor ae = a.expand(a_expand).contiguous();
    Tensor be = b.expand(b_expand).contiguous();

    int64_t batch_n = 1;
    for (int64_t s : batch_shape) batch_n *= s;

    std::vector<int64_t> flat3{batch_n, n, k};
    std::vector<int64_t> flat3b{batch_n, k, p};
    Tensor out3 = ptsycl::bmm(ae.reshape(flat3), be.reshape(flat3b));

    std::vector<int64_t> out_shape = batch_shape;
    out_shape.push_back(n);
    out_shape.push_back(p);
    Tensor out = out3.reshape(out_shape);

    if (a_is_vec) out = out.squeeze(static_cast<int64_t>(batch_shape.size()));
    if (b_is_vec) out = out.squeeze(-1);
    return out;
}

Tensor dot(const Tensor& self, const Tensor& other) {
    PTSYCL_TRACE_OP("dot");
    TORCH_CHECK(self.dim() == 1 && other.dim() == 1, "paras dot: both inputs must be 1-D");
    return ptsycl::mm(self.unsqueeze(0), other.unsqueeze(1)).reshape({});
}

Tensor mv(const Tensor& self, const Tensor& vec) {
    PTSYCL_TRACE_OP("mv");
    TORCH_CHECK(self.dim() == 2 && vec.dim() == 1, "paras mv: self must be 2-D, vec 1-D");
    return ptsycl::mm(self, vec.unsqueeze(1)).squeeze(1);
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("aten::mm", &ptsycl::mm);
    m.impl("aten::mm.out", &ptsycl::mm_out);
    m.impl("aten::bmm", &ptsycl::bmm);
    m.impl("aten::bmm.out", &ptsycl::bmm_out);
    m.impl("aten::matmul", &ptsycl::matmul);
    m.impl("aten::dot", &ptsycl::dot);
    m.impl("aten::mv", &ptsycl::mv);
}

} // namespace ptsycl
