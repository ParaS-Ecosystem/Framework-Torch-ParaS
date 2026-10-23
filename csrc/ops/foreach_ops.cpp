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

#include <cmath>
#include <cstring>

#include "core/kernels.h"

#include <ATen/ops/_foreach_add_native.h>
#include <ATen/ops/_foreach_addcdiv_native.h>
#include <ATen/ops/_foreach_addcmul_native.h>
#include <ATen/ops/_foreach_div_native.h>
#include <ATen/ops/_foreach_lerp_native.h>
#include <ATen/ops/_foreach_mul_native.h>
#include <ATen/ops/_foreach_sqrt_native.h>

namespace ptsycl {
namespace {

using at::Tensor;
using at::TensorList;

#if defined(PTSYCL_BACKEND_SYCL)
constexpr int64_t kApplyChunk = int64_t{1} << 16;
constexpr int64_t kApplyGroup = 256;
struct alignas(16) Word16 { uint32_t v[4]; };

struct ApplyTable {
    const int64_t* out; const int64_t* a; const int64_t* b; const int64_t* c;
    const int64_t* numel; const int64_t* first; const int64_t* aligned;
    const double* scalar;
    int64_t n;
};

template <typename F>
bool foreach_apply(TensorList out, TensorList a, TensorList b, TensorList c,
                   const std::vector<double>& scalars, F f) {
    const int64_t n = static_cast<int64_t>(out.size());
    if (n == 0) return true;
    const auto dt = out[0].scalar_type();
    const auto dev = out[0].device();
    if (!c10::isFloatingType(dt) || dev.type() != c10::DeviceType::PrivateUse1) return false;
    if (!queue_for(out[0]).is_gpu()) return false;
    auto ok = [&](TensorList l) {
        if (l.empty()) return true;
        if (static_cast<int64_t>(l.size()) != n) return false;
        for (int64_t i = 0; i < n; ++i)
            if (l[i].scalar_type() != dt || l[i].device() != dev || !l[i].is_contiguous() ||
                l[i].numel() != out[i].numel())
                return false;
        return true;
    };
    if (!ok(out) || !ok(a) || !ok(b) || !ok(c)) return false;

    std::vector<int64_t> meta(static_cast<std::size_t>(7 * n + 1));
    int64_t* m_out = meta.data();
    int64_t* m_a = m_out + n;
    int64_t* m_b = m_a + n;
    int64_t* m_c = m_b + n;
    int64_t* m_numel = m_c + n;
    int64_t* m_first = m_numel + n;
    int64_t* m_aligned = m_first + n + 1;
    int64_t chunks = 0;
    for (int64_t i = 0; i < n; ++i) {
        const Tensor& ai = a.empty() ? out[i] : a[i];
        const Tensor& bi = b.empty() ? ai : b[i];
        const Tensor& ci = c.empty() ? ai : c[i];
        m_out[i] = reinterpret_cast<int64_t>(out[i].data_ptr());
        m_a[i] = reinterpret_cast<int64_t>(ai.data_ptr());
        m_b[i] = reinterpret_cast<int64_t>(bi.data_ptr());
        m_c[i] = reinterpret_cast<int64_t>(ci.data_ptr());
        m_numel[i] = out[i].numel();
        m_first[i] = chunks;
        chunks += (m_numel[i] + kApplyChunk - 1) / kApplyChunk;
        m_aligned[i] = ((m_out[i] | m_a[i] | m_b[i] | m_c[i]) % 16) == 0;
    }
    m_first[n] = chunks;
    if (chunks == 0) return true;
    std::vector<double> sc(static_cast<std::size_t>(n), 0.0);
    for (int64_t i = 0; i < n && i < static_cast<int64_t>(scalars.size()); ++i) sc[i] = scalars[i];
    Tensor dmeta = at::from_blob(meta.data(), {static_cast<int64_t>(meta.size())}, at::kLong)
                       .to(out[0].options().dtype(at::kLong));
    Tensor dsc = at::from_blob(sc.data(), {n}, at::kDouble).to(out[0].options().dtype(at::kDouble));
    const int64_t* pm = data_ptr<int64_t>(dmeta);
    const ApplyTable tab{pm, pm + n, pm + 2 * n, pm + 3 * n, pm + 4 * n, pm + 5 * n, pm + 6 * n + 1,
                         dsc.data_ptr<double>(), n};

    auto& q = queue_for(out[0]);
    AT_DISPATCH_FLOATING_TYPES_AND2(c10::kHalf, c10::kBFloat16, dt, "ptsycl_foreach_apply", [&] {
        using T = scalar_t;
        q.sycl_queue().parallel_for(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(chunks * kApplyGroup)),
                              sycl::range<1>(static_cast<std::size_t>(kApplyGroup))),
            [=](sycl::nd_item<1> it) {
                const int64_t k = static_cast<int64_t>(it.get_group(0));
                int64_t lo = 0, hi = tab.n - 1;
                while (lo < hi) {
                    const int64_t mid = (lo + hi + 1) / 2;
                    if (tab.first[mid] <= k) lo = mid; else hi = mid - 1;
                }
                const int64_t t = lo;
                const int64_t s0 = (k - tab.first[t]) * kApplyChunk;
                const int64_t s1 = s0 + kApplyChunk < tab.numel[t] ? s0 + kApplyChunk : tab.numel[t];
                T* po = reinterpret_cast<T*>(tab.out[t]);
                const T* pa = reinterpret_cast<const T*>(tab.a[t]);
                const T* pb = reinterpret_cast<const T*>(tab.b[t]);
                const T* pc = reinterpret_cast<const T*>(tab.c[t]);
                const double s = tab.scalar[t];
                const T* tag = nullptr;
                const int64_t lid = static_cast<int64_t>(it.get_local_id(0));
                constexpr int64_t W = 16 / static_cast<int64_t>(sizeof(T));
                int64_t j0 = s0;
                if (tab.aligned[t]) {
                    for (int64_t j = s0 + lid * W; j + W <= s1; j += kApplyGroup * W) {
                        Word16 wa = *reinterpret_cast<const Word16*>(pa + j);
                        Word16 wb = *reinterpret_cast<const Word16*>(pb + j);
                        Word16 wc = *reinterpret_cast<const Word16*>(pc + j);
                        T va[W], vb[W], vc[W], vo[W];
                        std::memcpy(va, &wa, 16); std::memcpy(vb, &wb, 16); std::memcpy(vc, &wc, 16);
                        for (int64_t e = 0; e < W; ++e)
                            vo[e] = static_cast<T>(f(static_cast<double>(va[e]), static_cast<double>(vb[e]),
                                                     static_cast<double>(vc[e]), s, tag));
                        Word16 wo;
                        std::memcpy(&wo, vo, 16);
                        *reinterpret_cast<Word16*>(po + j) = wo;
                    }
                    j0 = s0 + (s1 - s0) / W * W;
                }
                for (int64_t j = j0 + lid; j < s1; j += kApplyGroup)
                    po[j] = static_cast<T>(f(static_cast<double>(pa[j]), static_cast<double>(pb[j]),
                                             static_cast<double>(pc[j]), s, tag));
            });
    });
    return true;
}

template <typename T> double round_to(double v, const T*) { return static_cast<double>(static_cast<T>(v)); }
double scalar_as(const Tensor& like, double v) {
    return like.scalar_type() == c10::kDouble ? v : static_cast<double>(static_cast<float>(v));
}
std::vector<double> scalars_as(TensorList like, at::ArrayRef<c10::Scalar> s) {
    std::vector<double> out(s.size());
    for (std::size_t i = 0; i < s.size(); ++i) out[i] = scalar_as(like[i], s[i].toDouble());
    return out;
}
std::vector<double> repeat(TensorList l, double v) { return std::vector<double>(l.size(), v); }
#endif

void foreach_mul_scalar_(TensorList self, const c10::Scalar& w) {
    PTSYCL_TRACE_OP("_foreach_mul_.Scalar");
#if defined(PTSYCL_BACKEND_SYCL)
    if (!self.empty() &&
        foreach_apply(self, {}, {}, {}, repeat(self, scalar_as(self[0], w.toDouble())),
                      [](double x, double, double, double s, auto) { return x * s; }))
        return;
#endif
    at::native::foreach_tensor_mul_scalar_kernel_slow_(self, w);
}

void foreach_mul_tensor_(TensorList self, const Tensor& other) {
    PTSYCL_TRACE_OP("_foreach_mul_.Tensor");
#if defined(PTSYCL_BACKEND_SYCL)
    if (other.dim() == 0 && !self.empty() && other.device() == self[0].device()) {
        const Tensor coef = other.to(self[0].scalar_type()).to(at::kDouble);
        const double* pc = coef.data_ptr<double>();
        if (foreach_apply(self, {}, {}, {}, {}, [pc](double x, double, double, double, auto) { return x * pc[0]; }))
            return;
    }
#endif
    at::native::foreach_tensor_mul_tensor_kernel_slow_(self, other);
}

void foreach_add_list_(TensorList self, TensorList other, const c10::Scalar& alpha) {
    PTSYCL_TRACE_OP("_foreach_add_.List");
#if defined(PTSYCL_BACKEND_SYCL)
    if (foreach_apply(self, {}, other, {}, repeat(self, alpha.toDouble()),
                      [](double x, double y, double, double s, auto) { return x + s * y; }))
        return;
#endif
    at::native::foreach_tensor_add_list_kernel_slow_(self, other, alpha);
}

void foreach_add_scalar_(TensorList self, const c10::Scalar& v) {
    PTSYCL_TRACE_OP("_foreach_add_.Scalar");
#if defined(PTSYCL_BACKEND_SYCL)
    if (!self.empty() &&
        foreach_apply(self, {}, {}, {}, repeat(self, scalar_as(self[0], v.toDouble())),
                      [](double x, double, double, double s, auto) { return x + 1.0 * s; }))
        return;
#endif
    at::native::foreach_tensor_add_scalar_kernel_slow_(self, v);
}

void foreach_lerp_scalar_(TensorList self, TensorList end, const c10::Scalar& w) {
    PTSYCL_TRACE_OP("_foreach_lerp_.Scalar");
#if defined(PTSYCL_BACKEND_SYCL)
    if (foreach_apply(self, {}, end, {}, repeat(self, w.toDouble()),
                      [](double a, double b, double, double s, auto) { return a + s * (b - a); }))
        return;
#endif
    at::native::foreach_tensor_lerp_list_kernel_slow_(self, end, w);
}

void foreach_addcmul_scalar_(TensorList self, TensorList t1, TensorList t2, const c10::Scalar& v) {
    PTSYCL_TRACE_OP("_foreach_addcmul_.Scalar");
#if defined(PTSYCL_BACKEND_SYCL)
    if (foreach_apply(self, {}, t1, t2, repeat(self, v.toDouble()),
                      [](double x, double a, double b, double s, auto tag) { return x + round_to(s * a * b, tag); }))
        return;
#endif
    at::native::foreach_tensor_addcmul_scalar_slow_(self, t1, t2, v);
}

void foreach_addcdiv_scalarlist_(TensorList self, TensorList t1, TensorList t2, at::ArrayRef<c10::Scalar> v) {
    PTSYCL_TRACE_OP("_foreach_addcdiv_.ScalarList");
#if defined(PTSYCL_BACKEND_SYCL)
    std::vector<double> s(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) s[i] = v[i].toDouble();
    if (v.size() == self.size() &&
        foreach_apply(self, {}, t1, t2, s,
                      [](double x, double a, double b, double w, auto tag) { return x + round_to(w * a / b, tag); }))
        return;
#endif
    at::native::foreach_tensor_addcdiv_scalarlist_slow_(self, t1, t2, v);
}

void foreach_div_scalarlist_(TensorList self, at::ArrayRef<c10::Scalar> v) {
    PTSYCL_TRACE_OP("_foreach_div_.ScalarList");
#if defined(PTSYCL_BACKEND_SYCL)
    if (v.size() == self.size() &&
        foreach_apply(self, {}, {}, {}, scalars_as(self, v),
                      [](double x, double, double, double s, auto) { return x / s; }))
        return;
#endif
    at::native::foreach_tensor_div_scalarlist_kernel_slow_(self, v);
}

std::vector<Tensor> foreach_sqrt(TensorList self) {
    PTSYCL_TRACE_OP("_foreach_sqrt");
#if defined(PTSYCL_BACKEND_SYCL)
    std::vector<Tensor> out;
    out.reserve(self.size());
    for (const auto& t : self) out.push_back(at::empty_like(t, at::MemoryFormat::Contiguous));
    if (foreach_apply(out, self, {}, {}, {}, [](double x, double, double, double, auto) { return ::sqrt(x); }))
        return out;
#endif
    return at::native::foreach_tensor_sqrt_slow(self);
}

}
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("aten::_foreach_mul_.Scalar", &ptsycl::foreach_mul_scalar_);
    m.impl("aten::_foreach_mul_.Tensor", &ptsycl::foreach_mul_tensor_);
    m.impl("aten::_foreach_add_.List", &ptsycl::foreach_add_list_);
    m.impl("aten::_foreach_add_.Scalar", &ptsycl::foreach_add_scalar_);
    m.impl("aten::_foreach_lerp_.Scalar", &ptsycl::foreach_lerp_scalar_);
    m.impl("aten::_foreach_addcmul_.Scalar", &ptsycl::foreach_addcmul_scalar_);
    m.impl("aten::_foreach_addcdiv_.ScalarList", &ptsycl::foreach_addcdiv_scalarlist_);
    m.impl("aten::_foreach_div_.ScalarList", &ptsycl::foreach_div_scalarlist_);
    m.impl("aten::_foreach_sqrt", &ptsycl::foreach_sqrt);
}
