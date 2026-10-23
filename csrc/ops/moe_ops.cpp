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

#if defined(PTSYCL_BACKEND_SYCL) && PARAS_HIP_BACKEND
constexpr int kMoeThreads = 256;
constexpr int kMoeLanes = 64;
constexpr int kMoeWaves = kMoeThreads / kMoeLanes;
constexpr int kMoeCols = 4;
constexpr int kMoeRows = 4;

template <typename T, int V>
struct alignas(sizeof(T) * V) MoeChunk {
    T v[V];
};

class MoeRoutesKernel;
template <typename T, bool GateUp> class MoeExpertKernel;
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

template <typename T, bool GateUp>
void moe_expert_gemv(compat::Queue& q, const int32_t* meta, int64_t S, const T* X, int64_t ldx,
                     int64_t div, const T* W0, int64_t es0, const T* W1, int64_t es1, int64_t N,
                     int64_t Kd, T* out) {
    constexpr int V = 16 / sizeof(T);
    constexpr int M = GateUp ? 2 : 1;
    using Ch = MoeChunk<T, V>;
    const int64_t chunks = Kd / V;
    const int64_t tiles = (N + kMoeWaves * kMoeCols - 1) / (kMoeWaves * kMoeCols);
    q.sycl_queue().submit([&](sycl::handler& cgh) {
        cgh.template parallel_for<MoeExpertKernel<T, GateUp>>(
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
                                const double gt = static_cast<double>(static_cast<T>(mine[0]));
                                const T act = static_cast<T>(gt / (1.0 + ::exp(-gt)));
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
    if (!q.is_gpu()) return false;
    const int64_t R = ids.numel(), E = w_gate.size(0);
    if (R == 0 || R > kMoeDeviceMaxRoutes || (R > kMoeDeviceRoutes && !q.capturing())) return false;
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
    AT_DISPATCH_FLOATING_TYPES_AND2(c10::kHalf, c10::kBFloat16, x.scalar_type(), "paras_moe_experts", [&] {
        if constexpr (!std::is_same_v<scalar_t, double>) {
            moe_expert_gemv<scalar_t, true>(q, pm, S, data_ptr<scalar_t>(x), H, K,
                                            data_ptr<scalar_t>(w_gate), w_gate.stride(0),
                                            data_ptr<scalar_t>(w_up), w_up.stride(0), I, H,
                                            data_ptr<scalar_t>(h));
            moe_expert_gemv<scalar_t, false>(q, pm, S, data_ptr<scalar_t>(h), I, 1,
                                             data_ptr<scalar_t>(w_down), w_down.stride(0), nullptr, 0,
                                             H, I, data_ptr<scalar_t>(y));
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
#if defined(PTSYCL_BACKEND_SYCL) && PARAS_HIP_BACKEND
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
