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

#include <ATen/ATen.h>
#include <torch/torch.h>

#include "core/kernels.h"
#include "core/telemetry.h"

namespace ptsycl {
namespace {

using at::Tensor;

constexpr int64_t kMoeDeviceRoutes = 768;
constexpr int64_t kMoeDeviceMaxRoutes = 4096;
constexpr int64_t kMoeMaxExperts = 256;

Tensor moe_experts_composite(const Tensor& x, const Tensor& ids, const Tensor& w,
                             const Tensor& w_gate, const Tensor& w_up, const Tensor& w_down) {
    const int64_t T = x.size(0), K = ids.size(1), H = x.size(1), E = w_gate.size(0);
    const Tensor valid = ids.ge(0).logical_and_(ids.lt(E));
    const Tensor routed_ids = ids.masked_fill(valid.logical_not(), 0);
    const Tensor weights = w.masked_fill(valid.logical_not(), 0);
    const Tensor flat = routed_ids.reshape({-1});
    const Tensor counts = at::zeros({T, E}, ids.options()).scatter_(1, routed_ids, 1).sum(0).to(c10::kCPU);
    const Tensor idxs = flat.argsort();
    const Tensor sorted = x.index_select(0, at::floor_divide(idxs, K));
    const int64_t* cnt = counts.data_ptr<int64_t>();
    std::vector<Tensor> outs;
    int64_t start = 0;
    for (int64_t e = 0; e < E; ++e) {
        const int64_t n = cnt[e];
        if (n == 0) continue;
        const Tensor xs = sorted.slice(0, start, start + n);
        const Tensor h = at::silu(at::linear(xs, w_gate[e])) * at::linear(xs, w_up[e]);
        outs.push_back(at::linear(h, w_down[e]));
        start += n;
    }
    if (outs.empty()) return at::zeros({T, H}, x.options());
    const Tensor cat = at::cat(outs, 0);
    Tensor routed = at::empty_like(cat);
    routed.index_put_({idxs}, cat);
    return routed.view({T, K, H}).to(w.scalar_type()).mul_(weights.unsqueeze(-1)).sum(1).to(x.scalar_type());
}

#if defined(PTSYCL_BACKEND_SYCL) && (PARAS_HIP_BACKEND || PARAS_CUDA_BACKEND)
constexpr int kMoeThreads = 256;
constexpr int kMoeCols = 4;
constexpr int kMoeRows = 4;

template <typename T, int V>
struct alignas(sizeof(T) * V) MoeChunk {
    T v[V];
};

class MoeRoutesKernel;
template <typename T, bool GateUp, int Lanes> class MoeExpertKernel;
template <typename T, typename WT> class MoeCombineKernel;

void moe_routes(compat::Queue& q, const int64_t* ids, int64_t R, int64_t E, int64_t S, int32_t* meta) {
    q.sycl_queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int32_t, 1> cnt(sycl::range<1>(kMoeMaxExperts), cgh);
        sycl::local_accessor<int32_t, 1> off(sycl::range<1>(kMoeMaxExperts), cgh);
        cgh.parallel_for<MoeRoutesKernel>(
            sycl::nd_range<1>(sycl::range<1>(kMoeThreads), sycl::range<1>(kMoeThreads)),
            [=](sycl::nd_item<1> it) {
                const int tid = static_cast<int>(it.get_local_id(0));
                int32_t* slot_expert = meta;
                int32_t* slot_off = meta + S;
                int32_t* slot_cnt = meta + 2 * S;
                int32_t* routes = meta + 3 * S;
                for (int64_t e = tid; e < E; e += kMoeThreads) {
                    int32_t c = 0;
                    for (int64_t r = 0; r < R; ++r) c += ids[r] == e;
                    cnt[e] = c;
                }
                sycl::group_barrier(it.get_group());
                if (tid == 0) {
                    int32_t o = 0;
                    int64_t s = 0;
                    for (int64_t e = 0; e < E; ++e) {
                        off[e] = o;
                        if (cnt[e] != 0) {
                            slot_expert[s] = static_cast<int32_t>(e);
                            slot_off[s] = o;
                            slot_cnt[s] = cnt[e];
                            ++s;
                            o += cnt[e];
                        }
                    }
                    for (; s < S; ++s) {
                        slot_expert[s] = -1;
                        slot_off[s] = 0;
                        slot_cnt[s] = 0;
                    }
                }
                sycl::group_barrier(it.get_group());
                for (int64_t e = tid; e < E; e += kMoeThreads) {
                    if (cnt[e] == 0) continue;
                    int32_t p = off[e];
                    for (int64_t r = 0; r < R; ++r)
                        if (ids[r] == e) routes[p++] = static_cast<int32_t>(r);
                }
            });
    });
}

template <typename T, bool GateUp, int Lanes>
void moe_expert_gemv(compat::Queue& q, const int32_t* meta, int64_t S, const T* X, int64_t ldx,
                     int64_t div, const T* W0, int64_t es0, const T* W1, int64_t es1, int64_t N,
                     int64_t Kd, T* out) {
    constexpr int kMoeLanes = Lanes, kMoeWaves = kMoeThreads / Lanes;
    static_assert(kMoeRows * kMoeCols <= Lanes, "a lane per output of a pass");
    constexpr int V = 16 / sizeof(T);
    constexpr int M = GateUp ? 2 : 1;
    using Ch = MoeChunk<T, V>;
    const int64_t chunks = Kd / V;
    const int64_t tiles = (N + kMoeWaves * kMoeCols - 1) / (kMoeWaves * kMoeCols);
    q.sycl_queue().submit([&](sycl::handler& cgh) {
        cgh.template parallel_for<MoeExpertKernel<T, GateUp, Lanes>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(S * tiles * kMoeThreads)),
                              sycl::range<1>(kMoeThreads)),
            [=](sycl::nd_item<1> it) {
                const int64_t g = static_cast<int64_t>(it.get_group(0));
                const int64_t s = g / tiles, tile = g % tiles;
                const int32_t e = meta[s];
                if (e < 0) return;
                const int32_t first = meta[S + s], count = meta[2 * S + s];
                const int32_t* routes = meta + 3 * S + first;
                const auto sg = it.get_sub_group();
                const int tid = static_cast<int>(it.get_local_id(0));
                const int lane = tid % kMoeLanes, wave = tid / kMoeLanes;
                const int64_t n0 = (tile * kMoeWaves + wave) * kMoeCols;
                const T* rows[M][kMoeCols];
#pragma unroll
                for (int c = 0; c < kMoeCols; ++c) {
                    const int64_t n = n0 + c < N ? n0 + c : N - 1;
                    rows[0][c] = W0 + e * es0 + n * Kd;
                    if constexpr (GateUp) rows[1][c] = W1 + e * es1 + n * Kd;
                }
                for (int32_t m0 = 0; m0 < count; m0 += kMoeRows) {
                    const int32_t mr = count - m0 < kMoeRows ? count - m0 : kMoeRows;
                    const T* src[kMoeRows];
#pragma unroll
                    for (int m = 0; m < kMoeRows; ++m)
                        src[m] = X + (routes[m0 + (m < mr ? m : 0)] / div) * ldx;
                    float acc[kMoeRows][M][kMoeCols] = {};
                    for (int64_t ch = lane; ch < chunks; ch += kMoeLanes) {
                        Ch wv[M][kMoeCols];
#pragma unroll
                        for (int j = 0; j < M; ++j)
#pragma unroll
                            for (int c = 0; c < kMoeCols; ++c)
                                wv[j][c] = *reinterpret_cast<const Ch*>(rows[j][c] + ch * V);
#pragma unroll
                        for (int m = 0; m < kMoeRows; ++m) {
                            const Ch xv = *reinterpret_cast<const Ch*>(src[m] + ch * V);
#pragma unroll
                            for (int i = 0; i < V; ++i) {
                                const float xf = static_cast<float>(xv.v[i]);
#pragma unroll
                                for (int j = 0; j < M; ++j)
#pragma unroll
                                    for (int c = 0; c < kMoeCols; ++c)
                                        acc[m][j][c] += xf * static_cast<float>(wv[j][c].v[i]);
                            }
                        }
                    }
                    float mine[M] = {};
#pragma unroll
                    for (int m = 0; m < kMoeRows; ++m)
#pragma unroll
                        for (int j = 0; j < M; ++j)
#pragma unroll
                            for (int c = 0; c < kMoeCols; ++c) {
                                const float v = sycl::reduce_over_group(sg, acc[m][j][c], sycl::plus<float>());
                                if (lane == m * kMoeCols + c) mine[j] = v;
                            }
                    if (lane < kMoeRows * kMoeCols) {
                        const int m = lane / kMoeCols, c = lane % kMoeCols;
                        if (m < mr && n0 + c < N) {
                            T res;
                            if constexpr (GateUp) {
                                const float gt = static_cast<float>(static_cast<T>(mine[0]));
                                const T act = static_cast<T>(gt / (1.f + ::expf(-gt)));
                                const T up = static_cast<T>(mine[1]);
                                res = static_cast<T>(static_cast<double>(act) * static_cast<double>(up));
                            } else {
                                res = static_cast<T>(mine[0]);
                            }
                            out[static_cast<int64_t>(routes[m0 + m]) * N + n0 + c] = res;
                        }
                    }
                }
            });
    });
}

constexpr int kGmThreads = 256;
constexpr int kGmRows = 128;
constexpr int kGmK = 32;
constexpr int kGmLd = kGmK + 8;

template <bool GateUp>
constexpr int gm_cols() { return GateUp ? kGmRows / 2 : kGmRows; }

template <typename T>
struct MoeGemmArgs {
    const int32_t* meta;
    int64_t S;
    const T* X;
    int64_t ldx, div;
    const T* W0;
    int64_t es0;
    const T* W1;
    int64_t es1;
    int64_t N, Kd;
    T* out;
    int64_t col_tiles;
};

template <typename T, bool GateUp, typename Item>
PTSYCL_HOST_DEVICE inline void moe_gemm_block(const MoeGemmArgs<T>& a, const Item& it, unsigned short* lds) {
#if PTSYCL_MFMA_DEVICE
    typedef short v4s __attribute__((ext_vector_type(4)));
    typedef _Float16 v4h __attribute__((ext_vector_type(4)));
    typedef float v4f __attribute__((ext_vector_type(4)));
    typedef unsigned short v8u __attribute__((ext_vector_type(8)));
    constexpr bool kBf16 = std::is_same_v<T, c10::BFloat16>;
    constexpr int kCols = gm_cols<GateUp>();
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
    const int64_t g = static_cast<int64_t>(it.get_group(0));
    const int64_t ct = g % a.col_tiles;
    int64_t rt = g / a.col_tiles;
    int64_t s = 0;
    for (; s < a.S; ++s) {
        const int32_t cnt = a.meta[2 * a.S + s];
        const int64_t tiles = (cnt + kGmRows - 1) / kGmRows;
        if (rt < tiles) break;
        rt -= tiles;
    }
    if (s >= a.S || a.meta[s] < 0) return;
    const int32_t e = a.meta[s];
    const int32_t count = a.meta[2 * a.S + s];
    const int32_t* routes = a.meta + 3 * a.S + a.meta[a.S + s];
    const int64_t m0 = rt * kGmRows, n0 = ct * kCols;

    const int tid = static_cast<int>(it.get_local_id(0));
    const int lane = tid % 64, wave = tid / 64;
    const int lr = lane % 16, lg = lane / 16;
    const T* asrc[2];
    const T* bsrc[2];
    int arow[2], acol[2];
#pragma unroll
    for (int t = 0; t < 2; ++t) {
        const int c = tid + t * kGmThreads;
        arow[t] = c / (kGmK / 8);
        acol[t] = (c % (kGmK / 8)) * 8;
        const int64_t m = m0 + arow[t] < count ? m0 + arow[t] : m0;
        asrc[t] = a.X + (static_cast<int64_t>(routes[m]) / a.div) * a.ldx + acol[t];
        const int j = arow[t];
        if constexpr (GateUp) {
            const int64_t n = n0 + (j % kCols);
            bsrc[t] = (j < kCols ? a.W0 + e * a.es0 : a.W1 + e * a.es1) + n * a.Kd + acol[t];
        } else {
            bsrc[t] = a.W0 + e * a.es0 + (n0 + j) * a.Kd + acol[t];
        }
    }
    unsigned short* alds = lds;
    unsigned short* blds = lds + kGmRows * kGmLd;
    v8u areg[2], breg[2];
#pragma unroll
    for (int t = 0; t < 2; ++t) {
        areg[t] = *reinterpret_cast<const v8u*>(asrc[t]);
        breg[t] = *reinterpret_cast<const v8u*>(bsrc[t]);
    }
    v4f acc[2][8];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] = v4f{0.f, 0.f, 0.f, 0.f};

    for (int64_t k0 = 0; k0 < a.Kd; k0 += kGmK) {
        sycl::group_barrier(it.get_group());
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            *reinterpret_cast<v8u*>(alds + arow[t] * kGmLd + acol[t]) = areg[t];
            *reinterpret_cast<v8u*>(blds + arow[t] * kGmLd + acol[t]) = breg[t];
        }
        sycl::group_barrier(it.get_group());
        if (k0 + kGmK < a.Kd) {
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                areg[t] = *reinterpret_cast<const v8u*>(asrc[t] + k0 + kGmK);
                breg[t] = *reinterpret_cast<const v8u*>(bsrc[t] + k0 + kGmK);
            }
        }
#pragma unroll
        for (int ks = 0; ks < kGmK / 16; ++ks) {
            v4s av[2];
#pragma unroll
            for (int i = 0; i < 2; ++i)
                av[i] = *reinterpret_cast<const v4s*>(alds + (wave * 32 + i * 16 + lr) * kGmLd + ks * 16 + lg * 4);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const v4s bv = *reinterpret_cast<const v4s*>(blds + (j * 16 + lr) * kGmLd + ks * 16 + lg * 4);
#pragma unroll
                for (int i = 0; i < 2; ++i) acc[i][j] = mfma(av[i], bv, acc[i][j]);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            const int64_t m = m0 + wave * 32 + i * 16 + lg * 4 + v;
            if (m >= count) continue;
            T* orow = a.out + static_cast<int64_t>(routes[m]) * a.N + n0;
            if constexpr (GateUp) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float gt = static_cast<float>(static_cast<T>(acc[i][j][v]));
                    const T act = static_cast<T>(gt / (1.f + ::expf(-gt)));
                    const T up = static_cast<T>(acc[i][j + 4][v]);
                    orow[j * 16 + lr] = static_cast<T>(static_cast<double>(act) * static_cast<double>(up));
                }
            } else {
#pragma unroll
                for (int j = 0; j < 8; ++j) orow[j * 16 + lr] = static_cast<T>(acc[i][j][v]);
            }
        }
#endif
}

template <typename T, bool GateUp> class MoeGemmKernel;

struct MoeGemmOccupancy {
    static constexpr int paras_max_work_group_size = kGmThreads;
};

template <typename T, bool GateUp>
struct MoeGemmFunctor : MoeGemmOccupancy {
    MoeGemmArgs<T> a;
    sycl::local_accessor<unsigned short, 1> lds;
    PTSYCL_HOST_DEVICE void operator()(sycl::nd_item<1> it) const { moe_gemm_block<T, GateUp>(a, it, &lds[0]); }
};

template <typename T, bool GateUp>
void moe_expert_gemm(compat::Queue& q, const int32_t* meta, int64_t S, int64_t R, const T* X, int64_t ldx,
                     int64_t div, const T* W0, int64_t es0, const T* W1, int64_t es1, int64_t N, int64_t Kd,
                     T* out) {
    const int64_t col_tiles = N / gm_cols<GateUp>();
    const int64_t row_tiles = (R + kGmRows - 1) / kGmRows + S;
    const MoeGemmArgs<T> args{meta, S, X, ldx, div, W0, es0, W1, es1, N, Kd, out, col_tiles};
    q.sycl_queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<unsigned short, 1> lds(sycl::range<1>(2 * kGmRows * kGmLd), cgh);
        const MoeGemmFunctor<T, GateUp> f{{}, args, lds};
        cgh.template parallel_for<MoeGemmKernel<T, GateUp>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(row_tiles * col_tiles * kGmThreads)),
                              sycl::range<1>(kGmThreads)),
            f);
    });
}

constexpr int64_t kMoeGroupedRoutes = 8;

bool moe_grouped_ok(const Tensor& x, int64_t H, int64_t I) {
    const auto st = x.scalar_type();
    return queue_for(x).mfma() && (st == c10::kBFloat16 || st == c10::kHalf) && H % kGmK == 0 && I % kGmK == 0 &&
           I % gm_cols<true>() == 0 && H % gm_cols<false>() == 0;
}

template <typename T, typename WT>
void moe_combine(compat::Queue& q, const T* y, const int64_t* ids, const WT* w, int64_t Tn,
                 int64_t K, int64_t H, int64_t E, T* out) {
    const int64_t n = Tn * H;
    q.sycl_queue().submit([&](sycl::handler& cgh) {
        cgh.template parallel_for<MoeCombineKernel<T, WT>>(
            sycl::range<1>(static_cast<std::size_t>(n)), [=](sycl::id<1> i) {
                const int64_t idx = static_cast<int64_t>(i[0]);
                const int64_t t = idx / H, h = idx % H;
                float acc = 0.f;
                for (int64_t k = 0; k < K; ++k) {
                    const int64_t r = t * K + k;
                    if (ids[r] < 0 || ids[r] >= E) continue;
                    acc += static_cast<float>(static_cast<WT>(
                        static_cast<float>(y[r * H + h]) * static_cast<float>(w[r])));
                }
                out[idx] = static_cast<T>(static_cast<float>(static_cast<WT>(acc)));
            });
    });
}

bool aligned16(const Tensor& t) {
    return reinterpret_cast<std::uintptr_t>(t.data_ptr()) % 16 == 0;
}

bool device_path(const Tensor& x, const Tensor& ids, const Tensor& w, const Tensor& w_gate,
                 const Tensor& w_up, const Tensor& w_down) {
    if (x.device().type() != c10::DeviceType::PrivateUse1) return false;
    auto& q = queue_for(x);
    if (!q.wave64() && q.wave_size() != 32) return false;
    const int64_t R = ids.numel(), E = w_gate.size(0);
    if (R == 0) return false;
    const bool grouped = moe_grouped_ok(x, w_gate.size(2), w_gate.size(1)) && R >= kMoeGroupedRoutes;
    if (!grouped && (R > kMoeDeviceMaxRoutes || (R > kMoeDeviceRoutes && !q.capturing()))) return false;
    const auto st = x.scalar_type();
    if (!(st == c10::kBFloat16 || st == c10::kHalf || st == c10::kFloat)) return false;
    if (w.scalar_type() != st && w.scalar_type() != c10::kFloat) return false;
    if (E > kMoeMaxExperts) return false;
    const int64_t V = 16 / static_cast<int64_t>(x.element_size());
    if (x.size(1) % V || w_gate.size(1) % V) return false;
    for (const Tensor* t : {&x, &ids, &w})
        if (!t->is_contiguous()) return false;
    for (const Tensor* t : {&w_gate, &w_up, &w_down})
        if (t->stride(2) != 1 || t->stride(1) != t->size(2) || t->stride(0) % V) return false;
    for (const Tensor* t : {&x, &w_gate, &w_up, &w_down})
        if (!aligned16(*t)) return false;
    return true;
}

Tensor moe_experts_device(const Tensor& x, const Tensor& ids, const Tensor& w, const Tensor& w_gate,
                          const Tensor& w_up, const Tensor& w_down) {
    auto& q = queue_for(x);
    const int64_t T = x.size(0), H = x.size(1), K = ids.size(1);
    const int64_t E = w_gate.size(0), I = w_gate.size(1), R = T * K;
    const int64_t S = std::min(E, R);
    Tensor meta = at::empty({3 * S + R}, x.options().dtype(c10::kInt));
    Tensor h = at::empty({R, I}, x.options());
    Tensor y = at::empty({R, H}, x.options());
    Tensor out = at::empty({T, H}, x.options());
    int32_t* pm = data_ptr<int32_t>(meta);
    const int64_t* pid = data_ptr<int64_t>(ids);
    moe_routes(q, pid, R, E, S, pm);
    const bool grouped = moe_grouped_ok(x, H, I) && R >= kMoeGroupedRoutes;
    AT_DISPATCH_FLOATING_TYPES_AND2(c10::kHalf, c10::kBFloat16, x.scalar_type(), "paras_moe_experts", [&] {
        if constexpr (!std::is_same_v<scalar_t, double>) {
            constexpr bool k16 = !std::is_same_v<scalar_t, float>;
            if (k16 && grouped) {
                if constexpr (k16) {
                    moe_expert_gemm<scalar_t, true>(q, pm, S, R, data_ptr<scalar_t>(x), H, K,
                                                    data_ptr<scalar_t>(w_gate), w_gate.stride(0),
                                                    data_ptr<scalar_t>(w_up), w_up.stride(0), I, H,
                                                    data_ptr<scalar_t>(h));
                    moe_expert_gemm<scalar_t, false>(q, pm, S, R, data_ptr<scalar_t>(h), I, 1,
                                                     data_ptr<scalar_t>(w_down), w_down.stride(0), nullptr, 0,
                                                     H, I, data_ptr<scalar_t>(y));
                }
            } else {
                auto gemvs = [&](auto lanes) {
                    constexpr int L = decltype(lanes)::value;
                    moe_expert_gemv<scalar_t, true, L>(q, pm, S, data_ptr<scalar_t>(x), H, K,
                                                       data_ptr<scalar_t>(w_gate), w_gate.stride(0),
                                                       data_ptr<scalar_t>(w_up), w_up.stride(0), I, H,
                                                       data_ptr<scalar_t>(h));
                    moe_expert_gemv<scalar_t, false, L>(q, pm, S, data_ptr<scalar_t>(h), I, 1,
                                                        data_ptr<scalar_t>(w_down), w_down.stride(0), nullptr, 0,
                                                        H, I, data_ptr<scalar_t>(y));
                };
                if (q.wave64()) gemvs(std::integral_constant<int, 64>{});
                else gemvs(std::integral_constant<int, 32>{});
            }
            if (w.scalar_type() == c10::kFloat)
                moe_combine<scalar_t, float>(q, data_ptr<scalar_t>(y), pid, data_ptr<float>(w), T, K, H, E,
                                             data_ptr<scalar_t>(out));
            else
                moe_combine<scalar_t, scalar_t>(q, data_ptr<scalar_t>(y), pid, data_ptr<scalar_t>(w), T, K,
                                                H, E, data_ptr<scalar_t>(out));
        }
    });
    return out;
}
#endif

void check_args(const Tensor& x, const Tensor& ids, const Tensor& w, const Tensor& w_gate,
                const Tensor& w_up, const Tensor& w_down) {
    TORCH_CHECK(x.dim() == 2, "paras::moe_experts: x must be [tokens, hidden], got ", x.sizes());
    TORCH_CHECK(ids.dim() == 2 && ids.size(0) == x.size(0),
                "paras::moe_experts: topk_ids must be [tokens, k], got ", ids.sizes());
    TORCH_CHECK(w.sizes() == ids.sizes(), "paras::moe_experts: topk_weight must have the shape of topk_ids");
    TORCH_CHECK(ids.scalar_type() == c10::kLong, "paras::moe_experts: topk_ids must be int64");
    TORCH_CHECK(w_gate.dim() == 3 && w_gate.size(2) == x.size(1) && w_up.sizes() == w_gate.sizes(),
                "paras::moe_experts: w_gate and w_up must be [experts, intermediate, hidden]");
    TORCH_CHECK(w_down.dim() == 3 && w_down.size(0) == w_gate.size(0) &&
                    w_down.size(1) == x.size(1) && w_down.size(2) == w_gate.size(1),
                "paras::moe_experts: w_down must be [experts, hidden, intermediate]");
    for (const Tensor* t : {&ids, &w, &w_gate, &w_up, &w_down})
        TORCH_CHECK(t->device() == x.device(), "paras::moe_experts: all tensors must be on x's device");
    for (const Tensor* t : {&w_gate, &w_up, &w_down})
        TORCH_CHECK(t->scalar_type() == x.scalar_type(), "paras::moe_experts: expert weights must have x's dtype");
    TORCH_CHECK(!(at::GradMode::is_enabled() &&
                  (x.requires_grad() || w.requires_grad() || w_gate.requires_grad() ||
                   w_up.requires_grad() || w_down.requires_grad())),
                "paras::moe_experts is for inference: call it under torch.no_grad()");
}

Tensor moe_experts(const Tensor& x, const Tensor& topk_ids, const Tensor& topk_weight,
                   const Tensor& w_gate, const Tensor& w_up, const Tensor& w_down) {
    PTSYCL_TRACE_OP("paras::moe_experts");
    check_args(x, topk_ids, topk_weight, w_gate, w_up, w_down);
    if (x.size(0) == 0) return at::empty({0, x.size(1)}, x.options());
#if defined(PTSYCL_BACKEND_SYCL) && (PARAS_HIP_BACKEND || PARAS_CUDA_BACKEND)
    if (device_path(x, topk_ids, topk_weight, w_gate, w_up, w_down))
        return moe_experts_device(x, topk_ids, topk_weight, w_gate, w_up, w_down);
#endif
    return moe_experts_composite(x, topk_ids, topk_weight, w_gate, w_up, w_down);
}

}
}

TORCH_LIBRARY_FRAGMENT(paras, m) {
    m.def("moe_experts(Tensor x, Tensor topk_ids, Tensor topk_weight, Tensor w_gate, "
          "Tensor w_up, Tensor w_down) -> Tensor");
}

TORCH_LIBRARY_IMPL(paras, PrivateUse1, m) {
    m.impl("moe_experts", &ptsycl::moe_experts);
}

TORCH_LIBRARY_IMPL(paras, CPU, m) {
    m.impl("moe_experts", &ptsycl::moe_experts);
}
