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

namespace ptsycl {
namespace {

using at::Tensor;

struct Philox2x32 {
    static PTSYCL_HOST_DEVICE inline void round(uint32_t& c0, uint32_t& c1, uint32_t key) {
        constexpr uint32_t M = 0xD256D193u;
        const uint64_t prod = static_cast<uint64_t>(M) * c0;
        c0 = static_cast<uint32_t>(prod >> 32) ^ key ^ c1;
        c1 = static_cast<uint32_t>(prod);
    }

    static PTSYCL_HOST_DEVICE inline void generate(uint64_t seed, uint64_t sequence, uint64_t idx,
                                uint32_t& r0, uint32_t& r1) {
        uint32_t c0 = static_cast<uint32_t>(idx);
        uint32_t c1 = static_cast<uint32_t>(sequence + (idx >> 32));
        uint32_t k0 = static_cast<uint32_t>(seed);
        for (int i = 0; i < 10; ++i) {
            round(c0, c1, k0);
            k0 += 0x9E3779B9u;
        }
        r0 = c0;
        r1 = c1;
    }

    static PTSYCL_HOST_DEVICE inline float to_float(uint32_t v) {
        return (v >> 8) * (1.0f / (1u << 24)); // [0, 1)
    }

    static PTSYCL_HOST_DEVICE inline float box_muller(uint32_t u1, uint32_t u2) {
        float f1 = to_float(u1);
        const float f2 = to_float(u2);
        if (f1 < 1e-7f) f1 = 1e-7f;
        return ::sqrtf(-2.0f * ::logf(f1)) *
               ::cosf(2.0f * 3.14159265358979323846f * f2);
    }
};

std::pair<uint64_t, uint64_t> draw_state(const Tensor& t, int64_t items) {
    return Context::instance().rng_draw(
        static_cast<int>(t.device().index()), items);
}

// Fills a (possibly strided) tensor through gen(i) -> float.
template <typename Gen>
void fill_random(Tensor& self, Gen gen) {
    auto& q = queue_for(self);
    const int64_t n = self.numel();
    if (n == 0) return;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        c10::kHalf, c10::kBFloat16, self.scalar_type(), "ptsycl_random", [&] {
            scalar_t* d = data_ptr<scalar_t>(self);
            if (self.is_contiguous()) {
                launch_flat(q, n, [=](std::size_t i) {
                    d[i] = static_cast<scalar_t>(gen(static_cast<int64_t>(i)));
                });
            } else {
                const auto spec = make_spec(self);
                launch_flat(q, n, [=](std::size_t i) {
                    d[spec.index(static_cast<int64_t>(i))] =
                        static_cast<scalar_t>(gen(static_cast<int64_t>(i)));
                });
            }
        });
}

Tensor& uniform_(Tensor& self, double from, double to,
                 std::optional<at::Generator> /*generator*/) {
    PTSYCL_TRACE_OP("uniform_");
    const auto state = draw_state(self, self.numel());
    const uint64_t seed = state.first;
    const uint64_t seq = state.second;
    const float lo = static_cast<float>(from);
    const float hi = static_cast<float>(to);
    fill_random(self, [=](int64_t i) {
        uint32_t r0, r1;
        Philox2x32::generate(seed, seq, static_cast<uint64_t>(i), r0, r1);
        return lo + Philox2x32::to_float(r0) * (hi - lo);
    });
    return self;
}

Tensor& normal_(Tensor& self, double mean, double std,
                std::optional<at::Generator> /*generator*/) {
    PTSYCL_TRACE_OP("normal_");
    const auto state = draw_state(self, self.numel());
    const uint64_t seed = state.first;
    const uint64_t seq = state.second;
    const float m = static_cast<float>(mean);
    const float s = static_cast<float>(std);
    fill_random(self, [=](int64_t i) {
        uint32_t r0, r1;
        const uint64_t pair = static_cast<uint64_t>(i) / 2;
        Philox2x32::generate(seed, seq, pair, r0, r1);
        const float v = (i % 2 == 0) ? Philox2x32::box_muller(r0, r1)
                                     : Philox2x32::box_muller(r1, r0);
        return m + s * v;
    });
    return self;
}

Tensor& bernoulli_(Tensor& self, double p,
                   std::optional<at::Generator> /*generator*/) {
    PTSYCL_TRACE_OP("bernoulli_");
    const auto state = draw_state(self, self.numel());
    const uint64_t seed = state.first;
    const uint64_t seq = state.second;
    const float fp = static_cast<float>(p);
    fill_random(self, [=](int64_t i) {
        uint32_t r0, r1;
        Philox2x32::generate(seed, seq, static_cast<uint64_t>(i), r0, r1);
        return Philox2x32::to_float(r0) < fp ? 1.0f : 0.0f;
    });
    return self;
}

std::tuple<Tensor, Tensor> native_dropout(const Tensor& input, double p,
                                          std::optional<bool> train) {
    PTSYCL_TRACE_OP("native_dropout");
    if (!train.value_or(true) || p == 0.0) {
        Tensor mask = at::ones_like(
            input, input.options().dtype(c10::kBool));
        return {input.clone(), mask};
    }
    TORCH_CHECK(p < 1.0, "native_dropout: p must be < 1, got ", p);

    auto& q = queue_for(input);
    Tensor in_c  = input.contiguous();
    Tensor out   = at::empty_like(in_c);
    Tensor mask  = at::empty(in_c.sizes(),
                             in_c.options().dtype(c10::kBool));
    const int64_t n = in_c.numel();
    const auto state = draw_state(input, n);
    const uint64_t seed = state.first;
    const uint64_t seq = state.second;
    const float keep  = static_cast<float>(1.0 - p);
    const float scale = 1.0f / keep;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        c10::kHalf, c10::kBFloat16, in_c.scalar_type(), "ptsycl_dropout", [&] {
            const scalar_t* x = data_ptr<scalar_t>(in_c);
            scalar_t* y = data_ptr<scalar_t>(out);
            bool* mk = data_ptr<bool>(mask);
            launch_flat(q, n, [=](std::size_t i) {
                uint32_t r0, r1;
                Philox2x32::generate(seed, seq, static_cast<uint64_t>(i), r0, r1);
                const bool k = Philox2x32::to_float(r0) < keep;
                mk[i] = k;
                y[i] = k ? static_cast<scalar_t>(
                               static_cast<float>(x[i]) * scale)
                         : static_cast<scalar_t>(0);
            });
        });
    return {out, mask};
}

Tensor native_dropout_backward(const Tensor& grad_output, const Tensor& mask,
                               double scale) {
    PTSYCL_TRACE_OP("native_dropout_backward");
    auto& q = queue_for(grad_output);
    Tensor g_c = grad_output.contiguous();
    Tensor m_c = mask.contiguous();
    Tensor out = at::empty_like(g_c);
    const int64_t n = g_c.numel();
    const float s = static_cast<float>(scale);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        c10::kHalf, c10::kBFloat16, g_c.scalar_type(), "ptsycl_dropout_bwd",
        [&] {
            const scalar_t* g = data_ptr<scalar_t>(g_c);
            const bool* mk = data_ptr<bool>(m_c);
            scalar_t* y = data_ptr<scalar_t>(out);
            launch_flat(q, n, [=](std::size_t i) {
                y[i] = mk[i] ? static_cast<scalar_t>(
                                   static_cast<float>(g[i]) * s)
                             : static_cast<scalar_t>(0);
            });
        });
    return out;
}

} // namespace
} // namespace ptsycl

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("aten::uniform_", &ptsycl::uniform_);
    m.impl("aten::normal_", &ptsycl::normal_);
    m.impl("aten::bernoulli_.float", &ptsycl::bernoulli_);
    m.impl("aten::native_dropout", &ptsycl::native_dropout);
    m.impl("aten::native_dropout_backward", &ptsycl::native_dropout_backward);
}
