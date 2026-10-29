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

#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <ATen/Context.h>
#include <ATen/SDPBackend.h>

#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

#include "core/kernels.h"
#include "ops/device_gemm.h"

namespace ptsycl {
namespace {

using at::Tensor;

constexpr int64_t kFusedMaxQuery = 16;

#if defined(PTSYCL_BACKEND_SYCL) && PARAS_HIP_BACKEND
#define PTSYCL_SDPA_GEMM_PATH 1
constexpr int64_t kRowLanes = 64;
constexpr int64_t kRowPerLane = 64;
#else
#define PTSYCL_SDPA_GEMM_PATH 0
#endif

bool needs_grad(const Tensor& t) { return t.defined() && t.requires_grad(); }

bool fused_supported(const Tensor& q, const Tensor& k, const Tensor& v,
                     const std::optional<Tensor>& mask, double dropout_p,
                     bool enable_gqa) {
    if (!at::globalContext().userEnabledOverrideableSDP()) return false;
    if (dropout_p != 0.0) return false;
    if (q.dim() != 4 || k.dim() != 4 || v.dim() != 4) return false;
    const auto dt = q.scalar_type();
    if (dt != c10::kFloat && dt != c10::kHalf && dt != c10::kBFloat16) return false;
    if (k.scalar_type() != dt || v.scalar_type() != dt) return false;
    if (q.sym_size(0) != k.sym_size(0) || k.sym_size(0) != v.sym_size(0)) return false;
    if (q.sym_size(1) != k.sym_size(1) || k.sym_size(1) != v.sym_size(1)) return false;
    (void)enable_gqa;
    if (q.sym_size(3) != k.sym_size(3) || k.sym_size(2) != v.sym_size(2)) return false;
    if (k.sym_size(2) == 0) return false;
    if (q.sym_size(2) > kFusedMaxQuery) {
#if PTSYCL_SDPA_GEMM_PATH
        if (!queue_for(q).is_gpu()) return false;
        const bool flash = (dt == c10::kBFloat16 || dt == c10::kHalf) && v.sym_size(3) == q.sym_size(3) &&
                           (q.sym_size(3) == 64 || q.sym_size(3) == 128);
        if (!flash && k.sym_size(2) > kRowLanes * kRowPerLane) return false;
#else
        return false;
#endif
    }
    if (at::GradMode::is_enabled() &&
        (needs_grad(q) || needs_grad(k) || needs_grad(v) ||
         (mask.has_value() && needs_grad(*mask))))
        return false;
    if (mask.has_value() && mask->defined()) {
        const Tensor& m = *mask;
        if (m.scalar_type() != c10::kBool && m.scalar_type() != dt) return false;
        if (m.dim() > 4) return false;
        const c10::SymInt want[4] = {q.sym_size(0), q.sym_size(1), q.sym_size(2), k.sym_size(2)};
        for (int64_t i = 0; i < m.dim(); ++i) {
            const c10::SymInt s = m.sym_size(m.dim() - 1 - i);
            if (s != 1 && s != want[3 - i]) return false;
        }
    }
    return true;
}

int64_t fused_sdp_choice(const Tensor& q, const Tensor& k, const Tensor& v,
                         const std::optional<Tensor>& mask, double dropout_p,
                         bool , std::optional<double> ,
                         bool enable_gqa) {
    const bool fused = fused_supported(q, k, v, mask, dropout_p, enable_gqa);
    return static_cast<int64_t>(fused ? sdp::SDPBackend::overrideable
                                      : sdp::SDPBackend::math);
}

struct Strides4 {
    int64_t s[4];
};

Strides4 strides4(const Tensor& t) {
    Strides4 r{};
    for (int i = 0; i < 4; ++i) r.s[i] = t.size(i) == 1 ? 0 : t.stride(i);
    return r;
}

struct MaxSum {
    float m;
    float s;
};

#if PTSYCL_SDPA_GEMM_PATH
template <typename T, int Vec>
class SoftmaxRowsKernel;

template <typename T, int Vec>
struct alignas(sizeof(T) * Vec) Chunk {
    T v[Vec];
};

template <typename scalar_t, int Vec>
void softmax_rows(compat::Queue& queue, const float* S, scalar_t* P, float* L,
                  int64_t rows, int64_t Lq, int64_t H, int64_t Lk, float sc,
                  bool is_causal, const bool* pmb, const scalar_t* pmf,
                  Strides4 ms) {
    constexpr float kNegInf = -std::numeric_limits<float>::infinity();
    constexpr int kIters = kRowPerLane / Vec;
    using C = Chunk<float, Vec>;
    using PC = Chunk<scalar_t, Vec>;
    using MC = Chunk<scalar_t, Vec>;
    using BC = Chunk<bool, Vec>;
    queue.sycl_queue().template parallel_for<SoftmaxRowsKernel<scalar_t, Vec>>(
        sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(rows * kRowLanes)),
                          sycl::range<1>(static_cast<std::size_t>(kRowLanes))),
        [=](sycl::nd_item<1> it) {
            const auto sg = it.get_sub_group();
            const int64_t r = static_cast<int64_t>(it.get_group(0));
            const int64_t lane = static_cast<int64_t>(it.get_local_id(0));
            const int64_t i = r % Lq, h = (r / Lq) % H, b = r / (Lq * H);
            const float* srow = S + r * Lk;
            const int64_t mrow = b * ms.s[0] + h * ms.s[1] + i * ms.s[2];
            float x[kRowPerLane];
            float m = kNegInf;
#pragma unroll
            for (int t = 0; t < kIters; ++t) {
                const int64_t j0 = (t * kRowLanes + lane) * Vec;
                if (j0 < Lk) {
                    const C sv = *reinterpret_cast<const C*>(srow + j0);
                    const MC* mv = pmf != nullptr
                        ? reinterpret_cast<const MC*>(pmf + mrow + j0 * ms.s[3]) : nullptr;
                    const BC* bv = pmb != nullptr
                        ? reinterpret_cast<const BC*>(pmb + mrow + j0 * ms.s[3]) : nullptr;
#pragma unroll
                    for (int e = 0; e < Vec; ++e) {
                        const int64_t j = j0 + e;
                        float v = sv.v[e] * sc;
                        if (mv != nullptr) v += static_cast<float>(mv->v[e]);
                        if ((bv != nullptr && !bv->v[e]) || (is_causal && j > i)) v = kNegInf;
                        x[t * Vec + e] = v;
                        m = v > m ? v : m;
                    }
                } else {
#pragma unroll
                    for (int e = 0; e < Vec; ++e) x[t * Vec + e] = kNegInf;
                }
            }
            m = sycl::reduce_over_group(sg, m, sycl::maximum<float>());
            float sum = 0.f;
#pragma unroll
            for (int t = 0; t < kRowPerLane; ++t) {
                x[t] = x[t] == kNegInf ? 0.f : ::expf(x[t] - m);
                sum += x[t];
            }
            sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>());
            const bool empty = m == kNegInf;
            const float inv = empty ? 0.f : 1.f / sum;
            scalar_t* prow = P + r * Lk;
#pragma unroll
            for (int t = 0; t < kIters; ++t) {
                const int64_t j0 = (t * kRowLanes + lane) * Vec;
                if (j0 < Lk) {
                    PC pv;
#pragma unroll
                    for (int e = 0; e < Vec; ++e) pv.v[e] = static_cast<scalar_t>(x[t * Vec + e] * inv);
                    *reinterpret_cast<PC*>(prow + j0) = pv;
                }
            }
            if (lane == 0) L[r] = empty ? kNegInf : m + ::logf(sum);
        });
}
#endif

#if PTSYCL_SDPA_GEMM_PATH
constexpr int kFlashTiles = 1;
constexpr int kFlashRowsPerWave = 16 * kFlashTiles;
constexpr int kFlashWaves = 4;
constexpr int kFlashBlockQ = kFlashRowsPerWave * kFlashWaves;
constexpr int kFlashBlockK = 64;
constexpr int kFlashLanes = 64;
constexpr int kFlashThreads = kFlashWaves * kFlashLanes;
constexpr int kFlashPad = 4;
constexpr int kFlashPLD = kFlashBlockK + kFlashPad;

template <int D>
constexpr int flash_kp_elems() {
    return kFlashBlockK * (D + kFlashPad) > kFlashWaves * kFlashRowsPerWave * kFlashPLD
               ? kFlashBlockK * (D + kFlashPad)
               : kFlashWaves * kFlashRowsPerWave * kFlashPLD;
}
template <int D>
constexpr int flash_lds_elems() {
    return flash_kp_elems<D>() + D * (kFlashBlockK + kFlashPad);
}

template <typename T>
struct FlashArgs {
    const T* q; const T* k; const T* v; T* o; float* lse;
    const T* mf; const bool* mb;
    Strides4 qs, ks, vs, os, ms;
    int64_t H, Lq, Lk, q_blocks;
    float scale;
    bool causal;
    const signed char* mcls; const unsigned char* mqok;
    int64_t k_blocks, mb_stride, mh_stride;
};

constexpr signed char kMaskDead = 0;
constexpr signed char kMaskZero = 1;
constexpr signed char kMaskMixed = 2;

template <typename T, int D> class FlashKernel;

template <typename T>
PTSYCL_HOST_DEVICE inline short flash_bits(T x) {
    return static_cast<short>(x.x);
}

#if defined(__HIP_DEVICE_COMPILE__)
typedef unsigned short flash_v8u __attribute__((ext_vector_type(8)));

template <typename T>
__attribute__((always_inline)) PTSYCL_HOST_DEVICE inline unsigned short flash_p_bits(float p) {
    if constexpr (std::is_same_v<T, c10::BFloat16>) {
        const unsigned int u = __builtin_bit_cast(unsigned int, p);
        return static_cast<unsigned short>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
    } else {
        return static_cast<unsigned short>(flash_bits(static_cast<T>(p)));
    }
}

template <typename T, int D, int kPer>
__attribute__((always_inline)) PTSYCL_HOST_DEVICE inline void flash_load_kv(
    const T* kp, const T* vp, const FlashArgs<T>& a, int64_t j0, int tid, bool kvec, bool vvec,
    flash_v8u (&kreg)[kPer], flash_v8u (&vreg)[kPer]) {
#pragma unroll
    for (int t = 0; t < kPer; ++t) {
        const int c = tid + t * kFlashThreads;
        const int r = c / (D / 8), c8 = (c % (D / 8)) * 8;
        const int64_t key = j0 + r;
        if (key < a.Lk && kvec) {
            kreg[t] = *reinterpret_cast<const flash_v8u*>(kp + key * a.ks.s[2] + c8);
        } else {
#pragma unroll
            for (int e = 0; e < 8; ++e)
                kreg[t][e] = key < a.Lk ? static_cast<unsigned short>(flash_bits(kp[key * a.ks.s[2] + (c8 + e) * a.ks.s[3]])) : 0;
        }
        if (key < a.Lk && vvec) {
            vreg[t] = *reinterpret_cast<const flash_v8u*>(vp + key * a.vs.s[2] + c8);
        } else {
#pragma unroll
            for (int e = 0; e < 8; ++e)
                vreg[t][e] = key < a.Lk ? static_cast<unsigned short>(flash_bits(vp[key * a.vs.s[2] + (c8 + e) * a.vs.s[3]])) : 0;
        }
    }
}

template <int D, int kPer>
__attribute__((always_inline)) PTSYCL_HOST_DEVICE inline void flash_store_kv(
    unsigned short* klds, unsigned short* vlds, int tid,
    const flash_v8u (&kreg)[kPer], const flash_v8u (&vreg)[kPer]) {
    constexpr int KLD = D + kFlashPad;
    constexpr int VLD = kFlashBlockK + kFlashPad;
#pragma unroll
    for (int t = 0; t < kPer; ++t) {
        const int c = tid + t * kFlashThreads;
        const int r = c / (D / 8), c8 = (c % (D / 8)) * 8;
        if constexpr (KLD % 8 == 0) {
            *reinterpret_cast<flash_v8u*>(klds + r * KLD + c8) = kreg[t];
        } else {
            typedef unsigned short flash_v4u __attribute__((ext_vector_type(4)));
            *reinterpret_cast<flash_v4u*>(klds + r * KLD + c8) = kreg[t].lo;
            *reinterpret_cast<flash_v4u*>(klds + r * KLD + c8 + 4) = kreg[t].hi;
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) vlds[(c8 + e) * VLD + r] = vreg[t][e];
    }
}
#endif

template <typename T, int D, typename Item>
PTSYCL_HOST_DEVICE inline void flash_block(const FlashArgs<T>& a, const Item& it,
                                           unsigned short* lds) {
#if defined(__HIP_DEVICE_COMPILE__)
    typedef short v4s __attribute__((ext_vector_type(4)));
    typedef _Float16 v4h __attribute__((ext_vector_type(4)));
    typedef float v4f __attribute__((ext_vector_type(4)));
    constexpr bool kBf16 = std::is_same_v<T, c10::BFloat16>;
    constexpr int KS = D / 16;
    constexpr int MT = kFlashTiles;
    constexpr float kNegInf = -std::numeric_limits<float>::infinity();
    constexpr int KLD = D + kFlashPad;
    constexpr int VLD = kFlashBlockK + kFlashPad;
    constexpr int kChunks = kFlashBlockK * D / 8;
    constexpr int kPer = kChunks / kFlashThreads;
    auto mfma = [](v4s x, v4s y, v4f c) -> v4f {
        if constexpr (kBf16) {
            return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(x, y, c, 0, 0, 0);
        } else {
            v4h hx, hy;
            __builtin_memcpy(&hx, &x, sizeof(hx));
            __builtin_memcpy(&hy, &y, sizeof(hy));
            return __builtin_amdgcn_mfma_f32_16x16x16f16(hx, hy, c, 0, 0, 0);
        }
    };

    const auto sg = it.get_sub_group();
    const int tid = static_cast<int>(it.get_local_id(0));
    const int lane = tid % kFlashLanes, wave = tid / kFlashLanes;
    const int lr = lane % 16, lg = lane / 16;
    const int64_t g = static_cast<int64_t>(it.get_group(0));
    const int64_t bh = g / a.q_blocks, qb = g % a.q_blocks;
    const int64_t b = bh / a.H, h = bh % a.H;
    const int64_t q0 = qb * kFlashBlockQ;
    const int64_t row0 = q0 + wave * kFlashRowsPerWave;
    const T* qp = a.q + b * a.qs.s[0] + h * a.qs.s[1];
    const T* kp = a.k + b * a.ks.s[0] + h * a.ks.s[1];
    const T* vp = a.v + b * a.vs.s[0] + h * a.vs.s[1];
    unsigned short* klds = lds;
    unsigned short* plds = lds + wave * (kFlashRowsPerWave * kFlashPLD);
    unsigned short* vlds = lds + flash_kp_elems<D>();
    const bool kvec = a.ks.s[3] == 1 && a.ks.s[2] % 8 == 0 &&
                      reinterpret_cast<std::uintptr_t>(kp) % 16 == 0;
    const bool vvec = a.vs.s[3] == 1 && a.vs.s[2] % 8 == 0 &&
                      reinterpret_cast<std::uintptr_t>(vp) % 16 == 0;

    flash_v8u kreg[kPer], vreg[kPer];

    v4s qa[MT][KS];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        const int64_t r = row0 + 16 * mt + lr;
#pragma unroll
        for (int ks = 0; ks < KS; ++ks)
#pragma unroll
            for (int i = 0; i < 4; ++i)
                qa[mt][ks][i] = r < a.Lq ? flash_bits(qp[r * a.qs.s[2] + (ks * 16 + 4 * lg + i) * a.qs.s[3]])
                                         : static_cast<short>(0);
    }
    v4f oacc[MT][KS];
    float m_i[MT][4], l_i[MT][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
        for (int n = 0; n < KS; ++n) oacc[mt][n] = v4f{0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int i = 0; i < 4; ++i) { m_i[mt][i] = kNegInf; l_i[mt][i] = 0.f; }
    }

    const int64_t k_end = a.causal ? (q0 + kFlashBlockQ < a.Lk ? q0 + kFlashBlockQ : a.Lk) : a.Lk;
    const bool has_mask = a.mf != nullptr || a.mb != nullptr;
    const int64_t msum = a.mcls == nullptr ? 0
        : ((a.mb_stride ? b : 0) * a.mb_stride + (a.mh_stride ? h : 0)) * a.q_blocks + qb;
    const bool may_skip = a.mcls != nullptr && a.mqok[msum] != 0;
    for (int64_t j0 = 0; j0 < k_end; j0 += kFlashBlockK) {
        const signed char tile = a.mcls == nullptr
                                     ? (has_mask ? kMaskMixed : kMaskZero)
                                     : a.mcls[msum * a.k_blocks + j0 / kFlashBlockK];
        if (tile == kMaskDead && may_skip) continue;
        const bool masked = tile != kMaskZero;
        flash_load_kv<T, D, kPer>(kp, vp, a, j0, tid, kvec, vvec, kreg, vreg);
        flash_store_kv<D, kPer>(klds, vlds, tid, kreg, vreg);
        sycl::group_barrier(it.get_group());

        v4f sacc[MT][4];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int n = 0; n < 4; ++n) sacc[mt][n] = v4f{0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int n = 0; n < 4; ++n)
#pragma unroll
            for (int ks = 0; ks < KS; ++ks) {
                const v4s kb = *reinterpret_cast<const v4s*>(klds + (n * 16 + lr) * KLD + ks * 16 + 4 * lg);
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) sacc[mt][n] = mfma(qa[mt][ks], kb, sacc[mt][n]);
            }
        sycl::group_barrier(it.get_group());

#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            float x[4][4], mx[4];
            const int64_t tile_row0 = row0 + 16 * mt;
            const bool edge = j0 + kFlashBlockK > a.Lk || tile_row0 + 16 > a.Lq ||
                              (a.causal && j0 + kFlashBlockK - 1 > tile_row0);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int row_l = 16 * mt + 4 * lg + i;
                mx[i] = kNegInf;
#pragma unroll
                for (int n = 0; n < 4; ++n) {
                    float sv = sacc[mt][n][i] * a.scale;
                    if (edge || masked) {
                        const int64_t row = row0 + row_l;
                        const int64_t key = j0 + n * 16 + lr;
                        if (key >= a.Lk || row >= a.Lq || (a.causal && key > row)) {
                            sv = kNegInf;
                        } else if (masked) {
                            const int64_t mo = b * a.ms.s[0] + h * a.ms.s[1] + row * a.ms.s[2] + key * a.ms.s[3];
                            if (a.mf != nullptr) sv += static_cast<float>(a.mf[mo]);
                            if (a.mb != nullptr && !a.mb[mo]) sv = kNegInf;
                        }
                    }
                    x[i][n] = sv;
                    mx[i] = sv > mx[i] ? sv : mx[i];
                }
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int off = 1; off < 16; off <<= 1) {
                    const float o = sycl::permute_group_by_xor(sg, mx[i], off);
                    mx[i] = o > mx[i] ? o : mx[i];
                }
                const float m_new = mx[i] > m_i[mt][i] ? mx[i] : m_i[mt][i];
                const float alpha = m_new == kNegInf ? 1.f : __expf(m_i[mt][i] - m_new);
                float rs = 0.f;
#pragma unroll
                for (int n = 0; n < 4; ++n) {
                    const float pv = x[i][n] == kNegInf ? 0.f : __expf(x[i][n] - m_new);
                    x[i][n] = pv;
                    rs += pv;
                }
#pragma unroll
                for (int off = 1; off < 16; off <<= 1) rs += sycl::permute_group_by_xor(sg, rs, off);
                l_i[mt][i] = l_i[mt][i] * alpha + rs;
                m_i[mt][i] = m_new;
#pragma unroll
                for (int nd = 0; nd < KS; ++nd) oacc[mt][nd][i] *= alpha;
#pragma unroll
                for (int n = 0; n < 4; ++n)
                    plds[(16 * mt + 4 * lg + i) * kFlashPLD + n * 16 + lr] =
                        flash_p_bits<T>(x[i][n]);
            }
        }
        sycl::group_barrier(it.get_group());
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            v4s pa[MT];
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
                pa[mt] = *reinterpret_cast<const v4s*>(plds + (16 * mt + lr) * kFlashPLD + kk * 16 + 4 * lg);
#pragma unroll
            for (int nd = 0; nd < KS; ++nd) {
                const v4s vb = *reinterpret_cast<const v4s*>(vlds + (nd * 16 + lr) * VLD + kk * 16 + 4 * lg);
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) oacc[mt][nd] = mfma(pa[mt], vb, oacc[mt][nd]);
            }
        }
        sycl::group_barrier(it.get_group());
    }
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int64_t row = row0 + 16 * mt + 4 * lg + i;
            if (row >= a.Lq) continue;
            const float inv = l_i[mt][i] > 0.f ? 1.f / l_i[mt][i] : 0.f;
#pragma unroll
            for (int nd = 0; nd < KS; ++nd)
                a.o[b * a.os.s[0] + h * a.os.s[1] + row * a.os.s[2] + (nd * 16 + lr) * a.os.s[3]] =
                    static_cast<T>(oacc[mt][nd][i] * inv);
            if (lr == 0)
                a.lse[(b * a.H + h) * a.Lq + row] =
                    l_i[mt][i] > 0.f ? m_i[mt][i] + ::logf(l_i[mt][i]) : kNegInf;
        }
#else
    (void)a; (void)it; (void)lds;
#endif
}

template <typename T> class MaskSummaryKernel;

template <typename T, typename Item>
PTSYCL_HOST_DEVICE inline void mask_summary_block(
    const T* mf, const bool* mb, Strides4 ms, int64_t Hm, int64_t Lq, int64_t Lk,
    int64_t q_blocks, int64_t k_blocks, float dead_below, signed char* cls,
    unsigned char* qok, const Item& it, int* flags, int* rowlive) {
    const int tid = static_cast<int>(it.get_local_id(0));
    const int64_t g = static_cast<int64_t>(it.get_group(0));
    const int64_t qb = g % q_blocks, bh = g / q_blocks;
    const int64_t bm = bh / Hm, hm = bh % Hm;
    if (tid < kFlashBlockQ) rowlive[tid] = 0;
    for (int64_t kb = 0; kb < k_blocks; ++kb) {
        if (tid < 2) flags[tid] = 0;
        sycl::group_barrier(it.get_group());
        for (int e = tid; e < kFlashBlockQ * kFlashBlockK; e += kFlashThreads) {
            const int r = e / kFlashBlockK;
            const int64_t row = qb * kFlashBlockQ + r, key = kb * kFlashBlockK + e % kFlashBlockK;
            if (row >= Lq || key >= Lk) continue;
            const int64_t o = bm * ms.s[0] + hm * ms.s[1] + row * ms.s[2] + key * ms.s[3];
            bool dead, zero;
            if (mb != nullptr) {
                dead = !mb[o];
                zero = mb[o];
            } else {
                const float v = static_cast<float>(mf[o]);
                dead = v <= dead_below;
                zero = v == 0.f;
            }
            if (!dead) { flags[0] = 1; rowlive[r] = 1; }
            if (!zero) flags[1] = 1;
        }
        sycl::group_barrier(it.get_group());
        if (tid == 0)
            cls[(bh * q_blocks + qb) * k_blocks + kb] =
                !flags[0] ? kMaskDead : (!flags[1] ? kMaskZero : kMaskMixed);
        sycl::group_barrier(it.get_group());
    }
    if (tid == 0) {
        unsigned char ok = 1;
        for (int r = 0; r < kFlashBlockQ; ++r)
            if (qb * kFlashBlockQ + r < Lq && !rowlive[r]) ok = 0;
        qok[bh * q_blocks + qb] = ok;
    }
}

struct MaskSummary {
    Tensor cls, qok;
    int64_t k_blocks = 0, mb_stride = 0, mh_stride = 0;
};

template <typename T>
MaskSummary mask_block_summary(compat::Queue& queue, const Tensor& attn_bias,
                               const Tensor& mask, int64_t B, int64_t H, int64_t Lq,
                               int64_t Lk) {
    struct Cached {
        c10::weak_intrusive_ptr<c10::StorageImpl> storage{c10::intrusive_ptr<c10::StorageImpl>()};
        int64_t offset = -1, version = -1, Lq = -1, Lk = -1, B = -1, H = -1;
        std::vector<int64_t> sizes, strides;
        c10::ScalarType dtype = c10::ScalarType::Undefined;
        MaskSummary summary;
    };
    thread_local Cached& cached = *new Cached;
    const bool cacheable = !attn_bias.is_inference();
    if (cacheable) {
        auto live = cached.storage.lock();
        if (live && live.get() == attn_bias.storage().unsafeGetStorageImpl() &&
            cached.offset == attn_bias.storage_offset() &&
            cached.version == attn_bias._version() && cached.dtype == attn_bias.scalar_type() &&
            cached.sizes == attn_bias.sizes().vec() && cached.strides == attn_bias.strides().vec() &&
            cached.B == B && cached.H == H && cached.Lq == Lq && cached.Lk == Lk &&
            cached.summary.cls.device() == mask.device())
            return cached.summary;
    }
    const Strides4 ms = strides4(mask);
    const int64_t Bm = ms.s[0] ? B : 1, Hm = ms.s[1] ? H : 1;
    const int64_t q_blocks = (Lq + kFlashBlockQ - 1) / kFlashBlockQ;
    const int64_t k_blocks = (Lk + kFlashBlockK - 1) / kFlashBlockK;
    MaskSummary out;
    out.cls = at::empty({Bm * Hm * q_blocks * k_blocks}, mask.options().dtype(c10::kChar));
    out.qok = at::empty({Bm * Hm * q_blocks}, mask.options().dtype(c10::kByte));
    out.k_blocks = k_blocks;
    out.mb_stride = ms.s[0] ? Hm : 0;
    out.mh_stride = ms.s[1] ? 1 : 0;
    const bool is_bool = mask.scalar_type() == c10::kBool;
    const T* mf = is_bool ? nullptr : data_ptr<T>(mask);
    const bool* mb = is_bool ? mask.data_ptr<bool>() : nullptr;
    const float dead_below = 0.5f * static_cast<float>(std::numeric_limits<T>::lowest());
    signed char* cls = out.cls.data_ptr<signed char>();
    unsigned char* qok = out.qok.data_ptr<unsigned char>();
    queue.sycl_queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int, 1> flags(sycl::range<1>(2), cgh);
        sycl::local_accessor<int, 1> rowlive(sycl::range<1>(kFlashBlockQ), cgh);
        cgh.template parallel_for<MaskSummaryKernel<T>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(Bm * Hm * q_blocks * kFlashThreads)),
                              sycl::range<1>(static_cast<std::size_t>(kFlashThreads))),
            [=](sycl::nd_item<1> it) {
                mask_summary_block<T>(mf, mb, ms, Hm, Lq, Lk, q_blocks, k_blocks, dead_below,
                                      cls, qok, it, &flags[0], &rowlive[0]);
            });
    });
    if (cacheable) {
        cached.storage = c10::weak_intrusive_ptr<c10::StorageImpl>(
            attn_bias.storage().getWeakStorageImpl());
        cached.offset = attn_bias.storage_offset();
        cached.version = attn_bias._version();
        cached.dtype = attn_bias.scalar_type();
        cached.sizes = attn_bias.sizes().vec();
        cached.strides = attn_bias.strides().vec();
        cached.B = B; cached.H = H; cached.Lq = Lq; cached.Lk = Lk;
        cached.summary = out;
    }
    return out;
}

template <typename T, int D>
void flash_attention(compat::Queue& queue, const FlashArgs<T>& args, int64_t groups) {
    queue.sycl_queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<unsigned short, 1> lds(sycl::range<1>(flash_lds_elems<D>()), cgh);
        const FlashArgs<T> a = args;
        cgh.template parallel_for<FlashKernel<T, D>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(groups * kFlashThreads)),
                              sycl::range<1>(static_cast<std::size_t>(kFlashThreads))),
            [=](sycl::nd_item<1> it) { flash_block<T, D>(a, it, &lds[0]); });
    });
}
#endif

std::tuple<Tensor, Tensor, Tensor, Tensor, c10::SymInt, c10::SymInt, Tensor,
           Tensor, Tensor>
fused_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                const std::optional<Tensor>& attn_bias, double dropout_p,
                bool is_causal, bool return_debug_mask,
                std::optional<double> scale) {
    PTSYCL_TRACE_OP("_scaled_dot_product_fused_attention_overrideable");
    TORCH_CHECK(dropout_p == 0.0, "paras fused attention: dropout is not supported");
    TORCH_CHECK(!return_debug_mask, "paras fused attention: no debug mask");
    auto& queue = queue_for(q);

    const int64_t B = q.size(0), H = q.size(1), Lq = q.size(2), D = q.size(3);
    const int64_t Lk = k.size(2), Dv = v.size(3);
    const int64_t rows = B * H * Lq;
    const float sc = static_cast<float>(
        scale.has_value() ? *scale : 1.0 / std::sqrt(static_cast<double>(D)));

    Tensor out = (Dv == D && q.is_non_overlapping_and_dense())
                     ? at::empty_like(q)
                     : at::empty({B, H, Lq, Dv}, q.options());
    Tensor lse = at::empty({B, H, Lq}, q.options().dtype(c10::kFloat));

    const bool has_mask = attn_bias.has_value() && attn_bias->defined();
    Tensor mask = has_mask ? attn_bias->expand({B, H, Lq, Lk}) : Tensor();
    const Strides4 ms = has_mask ? strides4(mask) : Strides4{};
    const bool bool_mask = has_mask && mask.scalar_type() == c10::kBool;
    const auto lopts = q.options().dtype(c10::kLong);

#if PTSYCL_SDPA_GEMM_PATH
    if (Lq > kFusedMaxQuery && Dv == D && (D == 64 || D == 128) &&
        (q.scalar_type() == c10::kBFloat16 || q.scalar_type() == c10::kHalf)) {
        AT_DISPATCH_REDUCED_FLOATING_TYPES(q.scalar_type(), "ptsycl_flash_attention", [&] {
            FlashArgs<scalar_t> fa{};
            fa.q = data_ptr<scalar_t>(q); fa.k = data_ptr<scalar_t>(k); fa.v = data_ptr<scalar_t>(v);
            fa.o = data_ptr<scalar_t>(out); fa.lse = lse.data_ptr<float>();
            fa.mf = has_mask && !bool_mask ? data_ptr<scalar_t>(mask) : nullptr;
            fa.mb = bool_mask ? mask.data_ptr<bool>() : nullptr;
            fa.qs = strides4(q); fa.ks = strides4(k); fa.vs = strides4(v); fa.os = strides4(out); fa.ms = ms;
            fa.H = H; fa.Lq = Lq; fa.Lk = Lk;
            fa.q_blocks = (Lq + kFlashBlockQ - 1) / kFlashBlockQ;
            fa.scale = sc; fa.causal = is_causal;
            MaskSummary summary;
            if (has_mask) {
                summary = mask_block_summary<scalar_t>(queue, *attn_bias, mask, B, H, Lq, Lk);
                fa.mcls = summary.cls.data_ptr<signed char>();
                fa.mqok = summary.qok.data_ptr<unsigned char>();
                fa.k_blocks = summary.k_blocks;
                fa.mb_stride = summary.mb_stride;
                fa.mh_stride = summary.mh_stride;
            }
            const int64_t groups = B * H * fa.q_blocks;
            if (D == 64) flash_attention<scalar_t, 64>(queue, fa, groups);
            else flash_attention<scalar_t, 128>(queue, fa, groups);
        });
        const Tensor none = Tensor();
        return {out, lse, none, none, c10::SymInt(Lq), c10::SymInt(Lk),
                at::empty({}, lopts), at::empty({}, lopts), none};
    }
    if (Lq > kFusedMaxQuery) {
        Tensor S = at::empty({B, H, Lq, Lk}, q.options().dtype(c10::kFloat));
        Tensor P = at::empty({B, H, Lq, Lk}, q.options());
        for (int64_t b = 0; b < B; ++b) {
            at::Tensor sb = S[b];
            const at::Tensor kt = k[b].transpose(-2, -1);
            const bool widened = q.scalar_type() == c10::kFloat
                                     ? (at::bmm_out(sb, q[b], kt), true)
                                     : device_bmm_widened(q[b], kt, sb);
            if (!widened)
                at::bmm_out(sb, q[b].to(c10::kFloat), kt.to(c10::kFloat));
        }
        const bool vec4 = Lk % 4 == 0 && (!has_mask || (ms.s[3] == 1 &&
                          ms.s[0] % 4 == 0 && ms.s[1] % 4 == 0 && ms.s[2] % 4 == 0 &&
                          reinterpret_cast<std::uintptr_t>(mask.data_ptr()) %
                                  (4 * mask.element_size()) == 0));
        AT_DISPATCH_FLOATING_TYPES_AND2(
            c10::kHalf, c10::kBFloat16, q.scalar_type(), "ptsycl_attention_rows", [&] {
                const bool* pmb = bool_mask ? mask.data_ptr<bool>() : nullptr;
                const scalar_t* pmf = has_mask && !bool_mask ? data_ptr<scalar_t>(mask) : nullptr;
                auto run = [&](auto vec) {
                    softmax_rows<scalar_t, decltype(vec)::value>(
                        queue, S.data_ptr<float>(), data_ptr<scalar_t>(P),
                        lse.data_ptr<float>(), rows, Lq, H, Lk, sc, is_causal, pmb, pmf, ms);
                };
                if (vec4 && sizeof(scalar_t) <= 4) run(std::integral_constant<int, 4>{});
                else run(std::integral_constant<int, 1>{});
            });
        for (int64_t b = 0; b < B; ++b) {
            at::Tensor ob = out[b];
            at::bmm_out(ob, P[b], v[b]);
        }
        const Tensor none = Tensor();
        return {out, lse, none, none, c10::SymInt(Lq), c10::SymInt(Lk),
                at::empty({}, lopts), at::empty({}, lopts), none};
    }
#endif

    Tensor scores = at::empty({rows, Lk}, q.options().dtype(c10::kFloat));
    const Strides4 qs = strides4(q), ks = strides4(k), vs = strides4(v),
                   os = strides4(out);

    float* S = scores.data_ptr<float>();
    float* L = lse.data_ptr<float>();
    constexpr float kNegInf = -std::numeric_limits<float>::infinity();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        c10::kHalf, c10::kBFloat16, q.scalar_type(), "ptsycl_fused_attention", [&] {
            const scalar_t* pq = data_ptr<scalar_t>(q);
            const scalar_t* pk = data_ptr<scalar_t>(k);
            const scalar_t* pv = data_ptr<scalar_t>(v);
            scalar_t* po = data_ptr<scalar_t>(out);
            const bool* pmb = bool_mask ? mask.data_ptr<bool>() : nullptr;
            const scalar_t* pmf =
                has_mask && !bool_mask ? data_ptr<scalar_t>(mask) : nullptr;

            compat::parallel_for(queue, static_cast<std::size_t>(rows * Lk), [=](std::size_t t) {
                const int64_t r = static_cast<int64_t>(t) / Lk;
                const int64_t j = static_cast<int64_t>(t) % Lk;
                const int64_t i = r % Lq, h = (r / Lq) % H, b = r / (Lq * H);
                float s;
                if (is_causal && j > i) {
                    s = kNegInf;
                } else {
                    const scalar_t* qr = pq + b * qs.s[0] + h * qs.s[1] + i * qs.s[2];
                    const scalar_t* kj = pk + b * ks.s[0] + h * ks.s[1] + j * ks.s[2];
                    float acc = 0.f;
                    for (int64_t d = 0; d < D; ++d)
                        acc += static_cast<float>(qr[d * qs.s[3]]) *
                               static_cast<float>(kj[d * ks.s[3]]);
                    s = acc * sc;
                    const int64_t mo = b * ms.s[0] + h * ms.s[1] + i * ms.s[2] + j * ms.s[3];
                    if (pmb != nullptr && !pmb[mo]) s = kNegInf;
                    if (pmf != nullptr) s += static_cast<float>(pmf[mo]);
                }
                S[t] = s;
            });

            reduce_outer<MaxSum>(
                queue, rows, Lk, MaxSum{kNegInf, 0.f},
                [=](int64_t r, int64_t j) { return MaxSum{S[r * Lk + j], 1.f}; },
                [](MaxSum a, MaxSum b) {
                    if (a.m == kNegInf) return b;
                    if (b.m == kNegInf) return a;
                    const float m = a.m > b.m ? a.m : b.m;
                    return MaxSum{m, a.s * ::expf(a.m - m) + b.s * ::expf(b.m - m)};
                },
                [=](int64_t r, MaxSum acc) {
                    L[r] = acc.m == kNegInf ? kNegInf : acc.m + ::logf(acc.s);
                },
                q);

            compat::parallel_for(queue, static_cast<std::size_t>(rows * Dv), [=](std::size_t t) {
                const int64_t r = static_cast<int64_t>(t) / Dv;
                const int64_t d = static_cast<int64_t>(t) % Dv;
                const int64_t i = r % Lq, h = (r / Lq) % H, b = r / (Lq * H);
                const float l = L[r];
                float acc = 0.f;
                if (l != kNegInf) {
                    const scalar_t* vd = pv + b * vs.s[0] + h * vs.s[1] + d * vs.s[3];
                    for (int64_t j = 0; j < Lk; ++j) {
                        const float s = S[r * Lk + j];
                        if (s != kNegInf)
                            acc += ::expf(s - l) * static_cast<float>(vd[j * vs.s[2]]);
                    }
                }
                po[b * os.s[0] + h * os.s[1] + i * os.s[2] + d * os.s[3]] =
                    static_cast<scalar_t>(acc);
            });
        });

    const Tensor none = Tensor();
    return {out, lse, none, none, c10::SymInt(Lq), c10::SymInt(Lk),
            at::empty({}, lopts), at::empty({}, lopts), none};
}

}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("aten::_fused_sdp_choice", &fused_sdp_choice);
    m.impl("aten::_scaled_dot_product_fused_attention_overrideable",
           &fused_attention);
}

}

namespace at::native {
REGISTER_PRIVATEUSE1_DISPATCH(_fused_sdp_choice_stub, &ptsycl::fused_sdp_choice);
}
