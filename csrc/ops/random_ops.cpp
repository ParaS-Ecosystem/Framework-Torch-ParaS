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


#include <limits>

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

Tensor& multinomial_out(
    const Tensor& self, int64_t num_samples,
    bool replacement, std::optional<at::Generator> /*generator*/,
    Tensor& out) {
    PTSYCL_TRACE_OP("multinomial.out");
    TORCH_CHECK(self.dim() == 1 || self.dim() == 2,
                "multinomial: input must be 1D or 2D");
    TORCH_CHECK(num_samples > 0, "multinomial: num_samples must be > 0");

    const bool batched = (self.dim() == 2);
    const Tensor input = (batched ? self : self.unsqueeze(0)).contiguous();
    const int64_t B = input.size(0);
    const int64_t K = input.size(1);

    if (!replacement) {
        TORCH_CHECK(num_samples <= K,
            "multinomial without replacement: num_samples > categories");
    }

    out.resize_(batched ? c10::IntArrayRef({B, num_samples})
                        : c10::IntArrayRef({num_samples}));

    if (replacement) {
        Tensor cdf = input.cumsum(/*dim=*/-1);
        auto& q = queue_for(self);
        const auto [seed, seq] = draw_state(input, B * num_samples);

        AT_DISPATCH_FLOATING_TYPES_AND2(
            c10::kHalf, c10::kBFloat16, input.scalar_type(),
            "ptsycl_multinomial_repl", [&] {
                const scalar_t* pcdf = data_ptr<scalar_t>(cdf);
                int64_t* pout = data_ptr<int64_t>(out);

                launch_flat(q, B * num_samples, [=](std::size_t idx) {
                    const int64_t b = static_cast<int64_t>(idx) / num_samples;
                    const int64_t s = static_cast<int64_t>(idx) % num_samples;

                    uint32_t r0, r1;
                    Philox2x32::generate(seed, seq,
                        static_cast<uint64_t>(idx), r0, r1);
                    const float u = Philox2x32::to_float(r0);

                    const scalar_t* row_cdf = pcdf + b * K;
                    const double total = static_cast<double>(row_cdf[K - 1]);
                    const double target = static_cast<double>(u) * total;

                    int64_t lo = 0, hi = K;
                    while (lo < hi) {
                        const int64_t mid = (lo + hi) / 2;
                        if (static_cast<double>(row_cdf[mid]) <= target)
                            lo = mid + 1;
                        else
                            hi = mid;
                    }
                    if (lo >= K) lo = K - 1;

                    pout[b * num_samples + s] = lo;
                });
            });
    } else {
        auto& q = queue_for(self);
        Tensor keys = at::empty_like(input);
        const auto [seed, seq] = draw_state(input, B * K);

        AT_DISPATCH_FLOATING_TYPES_AND2(
            c10::kHalf, c10::kBFloat16, input.scalar_type(),
            "ptsycl_multinomial_norepl_keys", [&] {
                const scalar_t* pw = data_ptr<scalar_t>(input);
                scalar_t* pk = data_ptr<scalar_t>(keys);

                launch_flat(q, B * K, [=](std::size_t idx) {
                    const int64_t b = static_cast<int64_t>(idx) / K;
                    const int64_t j = static_cast<int64_t>(idx) % K;

                    uint32_t r0, r1;
                    Philox2x32::generate(seed, seq,
                        static_cast<uint64_t>(idx), r0, r1);
                    float u = Philox2x32::to_float(r0);
                    if (u < 1e-7f) u = 1e-7f;

                    const double w = static_cast<double>(pw[b * K + j]);
                    const double key = (w > 0.0)
                        ? (-::log(static_cast<double>(u)) / w)
                        : 1e38;
                    pk[b * K + j] = static_cast<scalar_t>(key);
                });
            });

        auto [topk_vals, topk_idxs] = keys.topk(
            num_samples, /*dim=*/-1, /*largest=*/false, /*sorted=*/false);
        Tensor out_view = out.view({B, num_samples});
        out_view.copy_(topk_idxs);
    }

    if (!batched) out = out.squeeze(0);
    return out;
}

Tensor& random_range_(Tensor& self, int64_t from, std::optional<int64_t> to_opt) {
    const int64_t n = self.numel();
    if (n == 0) return self;
    int64_t to = 0;
    if (to_opt.has_value()) {
        to = *to_opt;
    } else {
        switch (self.scalar_type()) {
            case c10::kHalf: to = (int64_t{1} << 11) + 1; break;
            case c10::kBFloat16: to = (int64_t{1} << 8) + 1; break;
            case c10::kFloat: to = (int64_t{1} << 24) + 1; break;
            case c10::kDouble: to = (int64_t{1} << 53) + 1; break;
            case c10::kBool: to = 2; break;
            default:
                AT_DISPATCH_INTEGRAL_TYPES(self.scalar_type(), "ptsycl_random_max", [&] {
                    to = static_cast<int64_t>(std::numeric_limits<scalar_t>::max());
                    to = to == std::numeric_limits<int64_t>::max() ? to : to + 1;
                });
        }
    }
    TORCH_CHECK(from < to, "random_ expects 'from' to be less than 'to', but got from=", from, " >= to=", to);
    const uint64_t range = static_cast<uint64_t>(to) - static_cast<uint64_t>(from);
    const auto state = draw_state(self, n);
    const uint64_t seed = state.first, seq = state.second;
    auto& q = queue_for(self);
    Tensor dst = self.is_contiguous() ? self : at::empty(self.sizes(), self.options());
    AT_DISPATCH_ALL_TYPES_AND3(c10::kBool, c10::kHalf, c10::kBFloat16, self.scalar_type(), "ptsycl_random_", [&] {
        scalar_t* d = data_ptr<scalar_t>(dst);
        launch_flat(q, n, [=](std::size_t i) {
            uint32_t r0, r1;
            Philox2x32::generate(seed, seq, static_cast<uint64_t>(i), r0, r1);
            const uint64_t r = range > 0xffffffffull ? ((static_cast<uint64_t>(r1) << 32) | r0) : r0;
            d[i] = static_cast<scalar_t>(static_cast<int64_t>(from + static_cast<int64_t>(r % range)));
        });
    });
    if (!dst.is_same(self)) self.copy_(dst);
    return self;
}

Tensor& random_from_(Tensor& self, int64_t from, std::optional<int64_t> to, std::optional<at::Generator>) {
    PTSYCL_TRACE_OP("random_.from");
    return random_range_(self, from, to);
}
Tensor& random_to_(Tensor& self, int64_t to, std::optional<at::Generator>) {
    PTSYCL_TRACE_OP("random_.to");
    return random_range_(self, 0, to);
}
Tensor& random_default_(Tensor& self, std::optional<at::Generator>) {
    PTSYCL_TRACE_OP("random_");
    return random_range_(self, 0, std::nullopt);
}

template <typename F>
Tensor& fill_from_uniform(Tensor& self, F inverse_cdf) {
    const auto state = draw_state(self, self.numel());
    const uint64_t seed = state.first, seq = state.second;
    fill_random(self, [=](int64_t i) {
        uint32_t r0, r1;
        Philox2x32::generate(seed, seq, static_cast<uint64_t>(i), r0, r1);
        return inverse_cdf(Philox2x32::to_float(r0));
    });
    return self;
}

Tensor& exponential_(Tensor& self, double lambd, std::optional<at::Generator> ) {
    PTSYCL_TRACE_OP("exponential_");
    TORCH_CHECK(lambd > 0, "exponential_ expects lambda > 0.0");
    const float l = static_cast<float>(lambd);
    return fill_from_uniform(self, [=](float u) { return -::log1pf(-u) / l; });
}

Tensor& cauchy_(Tensor& self, double median, double sigma, std::optional<at::Generator> ) {
    PTSYCL_TRACE_OP("cauchy_");
    const float m = static_cast<float>(median), s = static_cast<float>(sigma);
    return fill_from_uniform(self, [=](float u) { return m + s * ::tanf(3.14159265358979323846f * (u - 0.5f)); });
}

Tensor& geometric_(Tensor& self, double p, std::optional<at::Generator> ) {
    PTSYCL_TRACE_OP("geometric_");
    TORCH_CHECK(p > 0 && p < 1, "geometric_ expects p in (0, 1)");
    const float lq = ::log1pf(static_cast<float>(-p));
    return fill_from_uniform(self, [=](float u) {
        const float k = ::ceilf(::log1pf(-u) / lq);
        return k < 1.0f ? 1.0f : k;
    });
}

Tensor& log_normal_(Tensor& self, double mean, double std, std::optional<at::Generator> generator) {
    PTSYCL_TRACE_OP("log_normal_");
    TORCH_CHECK(std > 0, "log_normal_ expects std > 0.0");
    ptsycl::normal_(self, mean, std, generator);
    return self.exp_();
}

Tensor multinomial(
    const Tensor& self, int64_t num_samples,
    bool replacement, std::optional<at::Generator> generator) {
    PTSYCL_TRACE_OP("multinomial");
    Tensor out = at::empty({0}, self.options().dtype(c10::kLong));
    return ptsycl::multinomial_out(self, num_samples, replacement,
                                   generator, out);
}

} // namespace
} // namespace ptsycl

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("aten::uniform_", &ptsycl::uniform_);
    m.impl("aten::normal_", &ptsycl::normal_);
    m.impl("aten::bernoulli_.float", &ptsycl::bernoulli_);
    m.impl("aten::native_dropout", &ptsycl::native_dropout);
    m.impl("aten::native_dropout_backward", &ptsycl::native_dropout_backward);
    m.impl("aten::multinomial", &ptsycl::multinomial);
    m.impl("aten::exponential_", &ptsycl::exponential_);
    m.impl("aten::random_.from", &ptsycl::random_from_);
    m.impl("aten::random_.to", &ptsycl::random_to_);
    m.impl("aten::random_", &ptsycl::random_default_);
    m.impl("aten::cauchy_", &ptsycl::cauchy_);
    m.impl("aten::geometric_", &ptsycl::geometric_);
    m.impl("aten::log_normal_", &ptsycl::log_normal_);
    m.impl("aten::multinomial.out", &ptsycl::multinomial_out);
}
