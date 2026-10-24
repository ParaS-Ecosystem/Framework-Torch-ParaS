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
#include <limits>

#include <ATen/ATen.h>
#include <torch/torch.h>

#include "core/kernels.h"
#include "core/telemetry.h"
#include "ops/conv_pool_ops.h"

namespace ptsycl {

using at::Tensor;

namespace {

struct Conv2dGeom {
    int64_t N, C, H, W, O, kh, kw, sh, sw, ph, pw, dh, dw, Ho, Wo, groups;
};

template <typename T>
void im2col_kernel(compat::Queue& q, const T* x, T* cols, const Conv2dGeom g) {
    const int64_t L = g.Ho * g.Wo, K = g.C * g.kh * g.kw, NL = g.N * L;
    launch_flat(q, K * NL, [=](std::size_t idx) {
        const int64_t i = static_cast<int64_t>(idx);
        const int64_t k = i / NL, col = i % NL;
        const int64_t n = col / L, l = col % L, oh = l / g.Wo, ow = l % g.Wo;
        const int64_t c = k / (g.kh * g.kw), kk = k % (g.kh * g.kw), ki = kk / g.kw, kj = kk % g.kw;
        const int64_t h = oh * g.sh - g.ph + ki * g.dh, w = ow * g.sw - g.pw + kj * g.dw;
        cols[i] = (h >= 0 && h < g.H && w >= 0 && w < g.W)
                      ? x[((n * g.C + c) * g.H + h) * g.W + w]
                      : static_cast<T>(0);
    });
}

template <typename T>
void col2im_kernel(compat::Queue& q, const T* cols, T* x, const Conv2dGeom g) {
    const int64_t L = g.Ho * g.Wo, NL = g.N * L;
    using acc_t = std::conditional_t<std::is_same_v<T, double>, double, float>;
    launch_flat(q, g.N * g.C * g.H * g.W, [=](std::size_t idx) {
        const int64_t i = static_cast<int64_t>(idx);
        const int64_t w = i % g.W, h = (i / g.W) % g.H, c = (i / (g.W * g.H)) % g.C, n = i / (g.W * g.H * g.C);
        acc_t acc = 0;
        for (int64_t ki = 0; ki < g.kh; ++ki) {
            const int64_t th = h + g.ph - ki * g.dh;
            if (th < 0 || th % g.sh) continue;
            const int64_t oh = th / g.sh;
            if (oh >= g.Ho) continue;
            for (int64_t kj = 0; kj < g.kw; ++kj) {
                const int64_t tw = w + g.pw - kj * g.dw;
                if (tw < 0 || tw % g.sw) continue;
                const int64_t ow = tw / g.sw;
                if (ow >= g.Wo) continue;
                const int64_t k = (c * g.kh + ki) * g.kw + kj;
                acc += static_cast<acc_t>(cols[k * NL + n * L + oh * g.Wo + ow]);
            }
        }
        x[i] = static_cast<T>(acc);
    });
}

Tensor im2col(const Tensor& x, const Conv2dGeom& g) {
    Tensor cols = at::empty({g.C * g.kh * g.kw, g.N * g.Ho * g.Wo}, x.options());
    auto& q = queue_for(x);
    AT_DISPATCH_FLOATING_TYPES_AND2(c10::kHalf, c10::kBFloat16, x.scalar_type(), "ptsycl_im2col", [&] {
        im2col_kernel<scalar_t>(q, data_ptr<scalar_t>(x), data_ptr<scalar_t>(cols), g);
    });
    return cols;
}

Tensor col2im(const Tensor& cols, const Conv2dGeom& g) {
    Tensor x = at::empty({g.N, g.C, g.H, g.W}, cols.options());
    auto& q = queue_for(cols);
    AT_DISPATCH_FLOATING_TYPES_AND2(c10::kHalf, c10::kBFloat16, cols.scalar_type(), "ptsycl_col2im", [&] {
        col2im_kernel<scalar_t>(q, data_ptr<scalar_t>(cols), data_ptr<scalar_t>(x), g);
    });
    return x;
}

bool conv_geometry(const Tensor& input, const Tensor& weight, at::IntArrayRef stride,
                   at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, Conv2dGeom& g) {
    const int64_t d = input.dim();
    if ((d != 3 && d != 4) || weight.dim() != d) return false;
    const bool one = d == 3;
    auto at = [&](at::IntArrayRef v, int64_t i, int64_t dflt) {
        if (v.empty()) return dflt;
        return v.size() == 1 ? v[0] : v[i];
    };
    g.N = input.size(0);
    g.C = input.size(1);
    g.H = one ? 1 : input.size(2);
    g.W = input.size(d - 1);
    g.O = weight.size(0);
    g.kh = one ? 1 : weight.size(2);
    g.kw = weight.size(d - 1);
    g.sh = one ? 1 : at(stride, 0, 1);
    g.sw = at(stride, one ? 0 : 1, 1);
    g.ph = one ? 0 : at(padding, 0, 0);
    g.pw = at(padding, one ? 0 : 1, 0);
    g.dh = one ? 1 : at(dilation, 0, 1);
    g.dw = at(dilation, one ? 0 : 1, 1);
    g.groups = groups;
    if (groups < 1 || g.C % groups || g.O % groups || weight.size(1) * groups != g.C) return false;
    g.Ho = (g.H + 2 * g.ph - g.dh * (g.kh - 1) - 1) / g.sh + 1;
    g.Wo = (g.W + 2 * g.pw - g.dw * (g.kw - 1) - 1) / g.sw + 1;
    return g.Ho > 0 && g.Wo > 0 && g.sh > 0 && g.sw > 0;
}

Tensor shaped_output(const Tensor& y_ol, const Conv2dGeom& g, int64_t dim) {
    Tensor y = y_ol.view({g.O, g.N, g.Ho * g.Wo}).permute({1, 0, 2}).contiguous();
    return dim == 3 ? y.view({g.N, g.O, g.Wo}) : y.view({g.N, g.O, g.Ho, g.Wo});
}

}

bool conv_device_supported(const Tensor& input, const Tensor& weight, bool transposed) {
    if (transposed || !(input.dim() == 3 || input.dim() == 4)) return false;
    const auto st = input.scalar_type();
    return (st == c10::kFloat || st == c10::kDouble || st == c10::kHalf || st == c10::kBFloat16) &&
           weight.scalar_type() == st && input.device() == weight.device();
}

Tensor conv_forward_device(const Tensor& input, const Tensor& weight, const c10::optional<Tensor>& bias,
                           at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
                           int64_t groups) {
    Conv2dGeom g{};
    TORCH_CHECK(conv_geometry(input, weight, stride, padding, dilation, groups, g),
                "paras convolution: unsupported geometry");
    const Tensor x = input.contiguous();
    const Tensor cols = im2col(x, g);
    const int64_t Kg = (g.C / groups) * g.kh * g.kw, Og = g.O / groups;
    const Tensor w2 = weight.contiguous().view({g.O, Kg});
    Tensor y;
    if (groups == 1) {
        y = at::mm(w2, cols);
    } else {
        y = at::empty({g.O, g.N * g.Ho * g.Wo}, x.options());
        for (int64_t gi = 0; gi < groups; ++gi) {
            Tensor yg = y.narrow(0, gi * Og, Og);
            at::mm_out(yg, w2.narrow(0, gi * Og, Og), cols.narrow(0, gi * Kg, Kg));
        }
    }
    if (bias && bias->defined()) y.add_(bias->to(y.scalar_type()).view({g.O, 1}));
    return shaped_output(y, g, input.dim());
}

std::tuple<Tensor, Tensor, Tensor> conv_backward_device(
    const Tensor& grad_output, const Tensor& input, const Tensor& weight, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, std::array<bool, 3> mask) {
    Conv2dGeom g{};
    TORCH_CHECK(conv_geometry(input, weight, stride, padding, dilation, groups, g),
                "paras convolution backward: unsupported geometry");
    const int64_t Kg = (g.C / groups) * g.kh * g.kw, Og = g.O / groups, L = g.Ho * g.Wo;
    const Tensor dy = grad_output.contiguous().view({g.N, g.O, L}).permute({1, 0, 2}).contiguous().view({g.O, g.N * L});
    const Tensor w2 = weight.contiguous().view({g.O, Kg});
    Tensor dx, dw, db;
    if (mask[0]) {
        Tensor dcols = at::empty({g.C * g.kh * g.kw, g.N * L}, dy.options());
        for (int64_t gi = 0; gi < groups; ++gi) {
            Tensor dg = dcols.narrow(0, gi * Kg, Kg);
            at::mm_out(dg, w2.narrow(0, gi * Og, Og).t(), dy.narrow(0, gi * Og, Og));
        }
        dx = col2im(dcols, g);
        if (input.dim() == 3) dx = dx.view(input.sizes());
    }
    if (mask[1]) {
        const Tensor cols = im2col(input.contiguous(), g);
        Tensor dw2 = at::empty({g.O, Kg}, dy.options());
        for (int64_t gi = 0; gi < groups; ++gi) {
            Tensor dg = dw2.narrow(0, gi * Og, Og);
            at::mm_out(dg, dy.narrow(0, gi * Og, Og), cols.narrow(0, gi * Kg, Kg).t());
        }
        dw = dw2.view(weight.sizes());
    }
    if (mask[2]) db = dy.sum(1);
    return {dx, dw, db};
}

namespace {

struct Pool2dGeom {
    int64_t NC, H, W, Ho, Wo, kh, kw, sh, sw, ph, pw, dh, dw;
};

Pool2dGeom pool_geom(const Tensor& in, const Tensor& out, at::IntArrayRef k, at::IntArrayRef s,
                     at::IntArrayRef p, at::IntArrayRef d) {
    auto v = [](at::IntArrayRef a, int i, int64_t dflt) { return a.empty() ? dflt : (a.size() == 1 ? a[0] : a[i]); };
    Pool2dGeom g{};
    g.H = in.size(-2);
    g.W = in.size(-1);
    g.NC = in.numel() / (g.H * g.W);
    g.Ho = out.size(-2);
    g.Wo = out.size(-1);
    g.kh = v(k, 0, 1);
    g.kw = v(k, 1, 1);
    g.sh = s.empty() ? g.kh : v(s, 0, 1);
    g.sw = s.empty() ? g.kw : v(s, 1, 1);
    g.ph = v(p, 0, 0);
    g.pw = v(p, 1, 0);
    g.dh = v(d, 0, 1);
    g.dw = v(d, 1, 1);
    return g;
}

template <typename T>
using acc_of = std::conditional_t<std::is_same_v<T, double>, double, float>;

void max_pool2d_fwd(const Tensor& self, const Pool2dGeom g, Tensor& out, Tensor& indices) {
    const Tensor x = self.contiguous();
    auto& q = queue_for(x);
    AT_DISPATCH_FLOATING_TYPES_AND2(c10::kHalf, c10::kBFloat16, x.scalar_type(), "ptsycl_max_pool2d", [&] {
        const scalar_t* px = data_ptr<scalar_t>(x);
        scalar_t* po = data_ptr<scalar_t>(out);
        int64_t* pi = data_ptr<int64_t>(indices);
        launch_flat(q, g.NC * g.Ho * g.Wo, [=](std::size_t idx) {
            const int64_t i = static_cast<int64_t>(idx);
            const int64_t ow = i % g.Wo, oh = (i / g.Wo) % g.Ho, nc = i / (g.Wo * g.Ho);
            const scalar_t* plane = px + nc * g.H * g.W;
            acc_of<scalar_t> best = -std::numeric_limits<acc_of<scalar_t>>::infinity();
            int64_t arg = -1;
            for (int64_t ki = 0; ki < g.kh; ++ki) {
                const int64_t h = oh * g.sh - g.ph + ki * g.dh;
                if (h < 0 || h >= g.H) continue;
                for (int64_t kj = 0; kj < g.kw; ++kj) {
                    const int64_t w = ow * g.sw - g.pw + kj * g.dw;
                    if (w < 0 || w >= g.W) continue;
                    const acc_of<scalar_t> v = static_cast<acc_of<scalar_t>>(plane[h * g.W + w]);
                    if (arg < 0 || v > best || v != v) {
                        best = v;
                        arg = h * g.W + w;
                        if (v != v) { ki = g.kh; break; }
                    }
                }
            }
            po[i] = static_cast<scalar_t>(best);
            pi[i] = arg;
        });
    });
}

void max_pool2d_bwd(const Tensor& grad_output, const Tensor& indices, const Pool2dGeom g, Tensor& grad_input) {
    const Tensor dy = grad_output.contiguous();
    const Tensor ix = indices.contiguous();
    auto& q = queue_for(dy);
    AT_DISPATCH_FLOATING_TYPES_AND2(c10::kHalf, c10::kBFloat16, dy.scalar_type(), "ptsycl_max_pool2d_bwd", [&] {
        const scalar_t* pdy = data_ptr<scalar_t>(dy);
        const int64_t* pi = data_ptr<int64_t>(ix);
        scalar_t* pdx = data_ptr<scalar_t>(grad_input);
        launch_flat(q, g.NC * g.H * g.W, [=](std::size_t idx) {
            const int64_t i = static_cast<int64_t>(idx);
            const int64_t w = i % g.W, h = (i / g.W) % g.H, nc = i / (g.W * g.H);
            const int64_t me = h * g.W + w;
            acc_of<scalar_t> acc = 0;
            for (int64_t ki = 0; ki < g.kh; ++ki) {
                const int64_t th = h + g.ph - ki * g.dh;
                if (th < 0 || th % g.sh) continue;
                const int64_t oh = th / g.sh;
                if (oh >= g.Ho) continue;
                for (int64_t kj = 0; kj < g.kw; ++kj) {
                    const int64_t tw = w + g.pw - kj * g.dw;
                    if (tw < 0 || tw % g.sw) continue;
                    const int64_t ow = tw / g.sw;
                    if (ow >= g.Wo) continue;
                    const int64_t o = (nc * g.Ho + oh) * g.Wo + ow;
                    if (pi[o] == me) acc += static_cast<acc_of<scalar_t>>(pdy[o]);
                }
            }
            pdx[i] = static_cast<scalar_t>(acc);
        });
    });
}

struct AvgWin {
    int64_t h0, h1, w0, w1, div;
};

PTSYCL_HOST_DEVICE inline AvgWin avg_window(const Pool2dGeom& g, int64_t oh, int64_t ow, bool count_include_pad,
                                             int64_t divisor) {
    int64_t h0 = oh * g.sh - g.ph, w0 = ow * g.sw - g.pw;
    int64_t h1 = h0 + g.kh < g.H + g.ph ? h0 + g.kh : g.H + g.ph;
    int64_t w1 = w0 + g.kw < g.W + g.pw ? w0 + g.kw : g.W + g.pw;
    const int64_t pool = (h1 - h0) * (w1 - w0);
    h0 = h0 > 0 ? h0 : 0;
    w0 = w0 > 0 ? w0 : 0;
    h1 = h1 < g.H ? h1 : g.H;
    w1 = w1 < g.W ? w1 : g.W;
    const int64_t div = divisor != 0 ? divisor : (count_include_pad ? pool : (h1 - h0) * (w1 - w0));
    return {h0, h1, w0, w1, div};
}

void avg_pool2d_fwd(const Tensor& self, const Pool2dGeom g, bool cip, int64_t divisor, Tensor& out) {
    const Tensor x = self.contiguous();
    auto& q = queue_for(x);
    AT_DISPATCH_FLOATING_TYPES_AND2(c10::kHalf, c10::kBFloat16, x.scalar_type(), "ptsycl_avg_pool2d", [&] {
        const scalar_t* px = data_ptr<scalar_t>(x);
        scalar_t* po = data_ptr<scalar_t>(out);
        launch_flat(q, g.NC * g.Ho * g.Wo, [=](std::size_t idx) {
            const int64_t i = static_cast<int64_t>(idx);
            const int64_t ow = i % g.Wo, oh = (i / g.Wo) % g.Ho, nc = i / (g.Wo * g.Ho);
            const AvgWin wv = avg_window(g, oh, ow, cip, divisor);
            const scalar_t* plane = px + nc * g.H * g.W;
            acc_of<scalar_t> acc = 0;
            for (int64_t h = wv.h0; h < wv.h1; ++h)
                for (int64_t w = wv.w0; w < wv.w1; ++w) acc += static_cast<acc_of<scalar_t>>(plane[h * g.W + w]);
            po[i] = static_cast<scalar_t>(wv.div > 0 ? acc / static_cast<acc_of<scalar_t>>(wv.div) : acc);
        });
    });
}

void avg_pool2d_bwd(const Tensor& grad_output, const Pool2dGeom g, bool cip, int64_t divisor, Tensor& grad_input) {
    const Tensor dy = grad_output.contiguous();
    auto& q = queue_for(dy);
    AT_DISPATCH_FLOATING_TYPES_AND2(c10::kHalf, c10::kBFloat16, dy.scalar_type(), "ptsycl_avg_pool2d_bwd", [&] {
        const scalar_t* pdy = data_ptr<scalar_t>(dy);
        scalar_t* pdx = data_ptr<scalar_t>(grad_input);
        launch_flat(q, g.NC * g.H * g.W, [=](std::size_t idx) {
            const int64_t i = static_cast<int64_t>(idx);
            const int64_t w = i % g.W, h = (i / g.W) % g.H, nc = i / (g.W * g.H);
            acc_of<scalar_t> acc = 0;
            for (int64_t ki = 0; ki < g.kh; ++ki) {
                const int64_t th = h + g.ph - ki;
                if (th < 0 || th % g.sh) continue;
                const int64_t oh = th / g.sh;
                if (oh >= g.Ho) continue;
                for (int64_t kj = 0; kj < g.kw; ++kj) {
                    const int64_t tw = w + g.pw - kj;
                    if (tw < 0 || tw % g.sw) continue;
                    const int64_t ow = tw / g.sw;
                    if (ow >= g.Wo) continue;
                    const AvgWin wv = avg_window(g, oh, ow, cip, divisor);
                    if (h < wv.h0 || h >= wv.h1 || w < wv.w0 || w >= wv.w1 || wv.div <= 0) continue;
                    acc += static_cast<acc_of<scalar_t>>(pdy[(nc * g.Ho + oh) * g.Wo + ow]) /
                           static_cast<acc_of<scalar_t>>(wv.div);
                }
            }
            pdx[i] = static_cast<scalar_t>(acc);
        });
    });
}

PTSYCL_HOST_DEVICE inline int64_t ada_start(int64_t o, int64_t in, int64_t out) { return (o * in) / out; }
PTSYCL_HOST_DEVICE inline int64_t ada_end(int64_t o, int64_t in, int64_t out) { return ((o + 1) * in + out - 1) / out; }

}

Tensor adaptive_avg_pool2d_device(const Tensor& self, at::IntArrayRef output_size) {
    TORCH_CHECK(self.dim() == 3 || self.dim() == 4, "adaptive_avg_pool2d: 3-D or 4-D input expected");
    TORCH_CHECK(output_size.size() == 2, "adaptive_avg_pool2d: output_size must have 2 elements");
    const Tensor x = self.contiguous();
    const int64_t H = x.size(-2), W = x.size(-1), Ho = output_size[0], Wo = output_size[1];
    std::vector<int64_t> os(x.sizes().begin(), x.sizes().end() - 2);
    os.push_back(Ho);
    os.push_back(Wo);
    Tensor out = at::empty(os, x.options());
    const int64_t NC = H * W > 0 ? x.numel() / (H * W) : 0;
    if (out.numel() == 0) return out;
    auto& q = queue_for(x);
    AT_DISPATCH_FLOATING_TYPES_AND2(c10::kHalf, c10::kBFloat16, x.scalar_type(), "ptsycl_adaptive_avg_pool2d", [&] {
        const scalar_t* px = data_ptr<scalar_t>(x);
        scalar_t* po = data_ptr<scalar_t>(out);
        launch_flat(q, NC * Ho * Wo, [=](std::size_t idx) {
            const int64_t i = static_cast<int64_t>(idx);
            const int64_t ow = i % Wo, oh = (i / Wo) % Ho, nc = i / (Wo * Ho);
            const int64_t h0 = ada_start(oh, H, Ho), h1 = ada_end(oh, H, Ho);
            const int64_t w0 = ada_start(ow, W, Wo), w1 = ada_end(ow, W, Wo);
            const scalar_t* plane = px + nc * H * W;
            acc_of<scalar_t> acc = 0;
            for (int64_t h = h0; h < h1; ++h)
                for (int64_t w = w0; w < w1; ++w) acc += static_cast<acc_of<scalar_t>>(plane[h * W + w]);
            po[i] = static_cast<scalar_t>(acc / static_cast<acc_of<scalar_t>>((h1 - h0) * (w1 - w0)));
        });
    });
    return out;
}

Tensor adaptive_avg_pool2d_backward_device(const Tensor& grad_output, const Tensor& self) {
    const Tensor dy = grad_output.contiguous();
    const int64_t H = self.size(-2), W = self.size(-1), Ho = dy.size(-2), Wo = dy.size(-1);
    Tensor dx = at::empty(self.sizes(), dy.options());
    const int64_t NC = H * W > 0 ? self.numel() / (H * W) : 0;
    if (dx.numel() == 0) return dx;
    auto& q = queue_for(dy);
    AT_DISPATCH_FLOATING_TYPES_AND2(c10::kHalf, c10::kBFloat16, dy.scalar_type(), "ptsycl_adaptive_avg_pool2d_bwd", [&] {
        const scalar_t* pdy = data_ptr<scalar_t>(dy);
        scalar_t* pdx = data_ptr<scalar_t>(dx);
        launch_flat(q, NC * H * W, [=](std::size_t idx) {
            const int64_t i = static_cast<int64_t>(idx);
            const int64_t w = i % W, h = (i / W) % H, nc = i / (W * H);
            const int64_t oh0 = (h * Ho) / H, ow0 = (w * Wo) / W;
            acc_of<scalar_t> acc = 0;
            for (int64_t oh = oh0; oh < Ho && ada_start(oh, H, Ho) <= h; ++oh) {
                const int64_t h0 = ada_start(oh, H, Ho), h1 = ada_end(oh, H, Ho);
                if (h < h0 || h >= h1) continue;
                for (int64_t ow = ow0; ow < Wo && ada_start(ow, W, Wo) <= w; ++ow) {
                    const int64_t w0 = ada_start(ow, W, Wo), w1 = ada_end(ow, W, Wo);
                    if (w < w0 || w >= w1) continue;
                    acc += static_cast<acc_of<scalar_t>>(pdy[(nc * Ho + oh) * Wo + ow]) /
                           static_cast<acc_of<scalar_t>>((h1 - h0) * (w1 - w0));
                }
            }
            pdx[i] = static_cast<scalar_t>(acc);
        });
    });
    return dx;
}

namespace {

double area_scale(int64_t in, int64_t out, bool align_corners, const c10::optional<double>& scale) {
    if (align_corners) return out > 1 ? static_cast<double>(in - 1) / static_cast<double>(out - 1) : 0.0;
    if (scale && *scale > 0) return 1.0 / *scale;
    return static_cast<double>(in) / static_cast<double>(out);
}

double nearest_scale(int64_t in, int64_t out, const c10::optional<double>& scale) {
    if (scale && *scale > 0) return 1.0 / *scale;
    return static_cast<double>(in) / static_cast<double>(out);
}

PTSYCL_HOST_DEVICE inline int64_t nearest_src(int64_t dst, double scale, int64_t in, bool exact) {
    const float sc = static_cast<float>(scale);
    const float f = exact ? (static_cast<float>(dst) + 0.5f) * sc : static_cast<float>(dst) * sc;
    const int64_t s = static_cast<int64_t>(::floorf(f));
    return s < in - 1 ? s : in - 1;
}

struct Up2dGeom {
    int64_t NC, H, W, Ho, Wo;
    double sh, sw;
};

template <typename T>
void nearest_fwd(const Tensor& x, Tensor& out, const Up2dGeom g, bool exact) {
    auto& q = queue_for(x);
    const T* px = data_ptr<T>(x);
    T* po = data_ptr<T>(out);
    launch_flat(q, g.NC * g.Ho * g.Wo, [=](std::size_t idx) {
        const int64_t i = static_cast<int64_t>(idx);
        const int64_t ow = i % g.Wo, oh = (i / g.Wo) % g.Ho, nc = i / (g.Wo * g.Ho);
        po[i] = px[(nc * g.H + nearest_src(oh, g.sh, g.H, exact)) * g.W + nearest_src(ow, g.sw, g.W, exact)];
    });
}

template <typename T>
void nearest_bwd(const Tensor& dy, Tensor& acc, const Up2dGeom g, bool exact) {
    auto& q = queue_for(dy);
    const T* pdy = data_ptr<T>(dy);
    using A = acc_of<T>;
    A* pa = data_ptr<A>(acc);
    launch_flat(q, g.NC * g.Ho * g.Wo, [=](std::size_t idx) {
        const int64_t i = static_cast<int64_t>(idx);
        const int64_t ow = i % g.Wo, oh = (i / g.Wo) % g.Ho, nc = i / (g.Wo * g.Ho);
        atomic_add(pa + (nc * g.H + nearest_src(oh, g.sh, g.H, exact)) * g.W + nearest_src(ow, g.sw, g.W, exact),
                   static_cast<A>(pdy[i]));
    });
}

PTSYCL_HOST_DEVICE inline void linear_src(int64_t dst, double scale, int64_t in, bool align_corners,
                                          int64_t& i0, int64_t& i1, double& l1) {
    double s = align_corners ? static_cast<double>(dst) * scale
                             : (static_cast<double>(dst) + 0.5) * scale - 0.5;
    if (s < 0) s = 0;
    i0 = static_cast<int64_t>(s);
    if (i0 > in - 1) i0 = in - 1;
    i1 = i0 < in - 1 ? i0 + 1 : i0;
    l1 = s - static_cast<double>(i0);
}

template <typename T>
void bilinear_fwd(const Tensor& x, Tensor& out, const Up2dGeom g, bool ac) {
    auto& q = queue_for(x);
    const T* px = data_ptr<T>(x);
    T* po = data_ptr<T>(out);
    using A = acc_of<T>;
    launch_flat(q, g.NC * g.Ho * g.Wo, [=](std::size_t idx) {
        const int64_t i = static_cast<int64_t>(idx);
        const int64_t ow = i % g.Wo, oh = (i / g.Wo) % g.Ho, nc = i / (g.Wo * g.Ho);
        int64_t h0, h1, w0, w1;
        double lh, lw;
        linear_src(oh, g.sh, g.H, ac, h0, h1, lh);
        linear_src(ow, g.sw, g.W, ac, w0, w1, lw);
        const T* p = px + nc * g.H * g.W;
        const A a = static_cast<A>(1.0 - lh), b = static_cast<A>(lh), c = static_cast<A>(1.0 - lw), d = static_cast<A>(lw);
        po[i] = static_cast<T>(a * (c * static_cast<A>(p[h0 * g.W + w0]) + d * static_cast<A>(p[h0 * g.W + w1])) +
                               b * (c * static_cast<A>(p[h1 * g.W + w0]) + d * static_cast<A>(p[h1 * g.W + w1])));
    });
}

template <typename T>
void bilinear_bwd(const Tensor& dy, Tensor& acc, const Up2dGeom g, bool ac) {
    auto& q = queue_for(dy);
    const T* pdy = data_ptr<T>(dy);
    using A = acc_of<T>;
    A* pa = data_ptr<A>(acc);
    launch_flat(q, g.NC * g.Ho * g.Wo, [=](std::size_t idx) {
        const int64_t i = static_cast<int64_t>(idx);
        const int64_t ow = i % g.Wo, oh = (i / g.Wo) % g.Ho, nc = i / (g.Wo * g.Ho);
        int64_t h0, h1, w0, w1;
        double lh, lw;
        linear_src(oh, g.sh, g.H, ac, h0, h1, lh);
        linear_src(ow, g.sw, g.W, ac, w0, w1, lw);
        A* p = pa + nc * g.H * g.W;
        const A v = static_cast<A>(pdy[i]);
        atomic_add(p + h0 * g.W + w0, static_cast<A>((1.0 - lh) * (1.0 - lw)) * v);
        atomic_add(p + h0 * g.W + w1, static_cast<A>((1.0 - lh) * lw) * v);
        atomic_add(p + h1 * g.W + w0, static_cast<A>(lh * (1.0 - lw)) * v);
        atomic_add(p + h1 * g.W + w1, static_cast<A>(lh * lw) * v);
    });
}

Up2dGeom up_geom(const Tensor& in_like, int64_t Ho, int64_t Wo) {
    Up2dGeom g{};
    g.H = in_like.size(-2);
    g.W = in_like.size(-1);
    g.NC = g.H * g.W > 0 ? in_like.numel() / (g.H * g.W) : 0;
    g.Ho = Ho;
    g.Wo = Wo;
    return g;
}

}

std::tuple<Tensor&, Tensor&> max_pool2d_with_indices_out_device(
    const Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding,
    at::IntArrayRef dilation, bool ceil_mode, Tensor& out, Tensor& indices) {
    PTSYCL_TRACE_OP("max_pool2d_with_indices.out");
    (void)ceil_mode;
    if (out.numel() > 0) max_pool2d_fwd(self, pool_geom(self, out, kernel_size, stride, padding, dilation), out, indices);
    return {out, indices};
}

Tensor& max_pool2d_with_indices_backward_out_device(
    const Tensor& grad_output, const Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const Tensor& indices, Tensor& grad_input) {
    PTSYCL_TRACE_OP("max_pool2d_with_indices_backward.grad_input");
    (void)ceil_mode;
    if (grad_input.numel() > 0)
        max_pool2d_bwd(grad_output, indices, pool_geom(self, grad_output, kernel_size, stride, padding, dilation),
                       grad_input);
    return grad_input;
}

Tensor& avg_pool2d_out_device(const Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
                              at::IntArrayRef padding, bool ceil_mode, bool count_include_pad,
                              c10::optional<int64_t> divisor_override, Tensor& out) {
    PTSYCL_TRACE_OP("avg_pool2d.out");
    (void)ceil_mode;
    if (out.numel() > 0)
        avg_pool2d_fwd(self, pool_geom(self, out, kernel_size, stride, padding, {}), count_include_pad,
                       divisor_override.value_or(0), out);
    return out;
}

Tensor& avg_pool2d_backward_out_device(const Tensor& grad_output, const Tensor& self, at::IntArrayRef kernel_size,
                                       at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode,
                                       bool count_include_pad, c10::optional<int64_t> divisor_override,
                                       Tensor& grad_input) {
    PTSYCL_TRACE_OP("avg_pool2d_backward.grad_input");
    (void)ceil_mode;
    if (grad_input.numel() > 0)
        avg_pool2d_bwd(grad_output, pool_geom(self, grad_output, kernel_size, stride, padding, {}),
                       count_include_pad, divisor_override.value_or(0), grad_input);
    return grad_input;
}

Tensor& upsample_nearest2d_device(const Tensor& self, int64_t Ho, int64_t Wo, c10::optional<double> scales_h,
                                  c10::optional<double> scales_w, bool exact, Tensor& out) {
    const Tensor x = self.contiguous();
    Up2dGeom g = up_geom(x, Ho, Wo);
    g.sh = nearest_scale(g.H, Ho, scales_h);
    g.sw = nearest_scale(g.W, Wo, scales_w);
    if (out.numel() == 0) return out;
    Tensor o = out.is_contiguous() ? out : at::empty(out.sizes(), out.options());
    AT_DISPATCH_FLOATING_TYPES_AND3(c10::kHalf, c10::kBFloat16, c10::kByte, x.scalar_type(), "ptsycl_upsample_nearest2d", [&] {
        nearest_fwd<scalar_t>(x, o, g, exact);
    });
    if (!o.is_same(out)) out.copy_(o);
    return out;
}

Tensor& upsample_nearest2d_backward_device(const Tensor& grad_output, at::IntArrayRef input_size,
                                           c10::optional<double> scales_h, c10::optional<double> scales_w,
                                           bool exact, Tensor& grad_input) {
    const Tensor dy = grad_output.contiguous();
    const int64_t H = input_size[input_size.size() - 2], W = input_size[input_size.size() - 1];
    Up2dGeom g = up_geom(grad_input, dy.size(-2), dy.size(-1));
    g.sh = nearest_scale(H, g.Ho, scales_h);
    g.sw = nearest_scale(W, g.Wo, scales_w);
    Tensor acc = at::zeros(grad_input.sizes(), dy.options().dtype(dy.scalar_type() == c10::kDouble ? c10::kDouble : c10::kFloat));
    if (dy.numel() > 0) {
        AT_DISPATCH_FLOATING_TYPES_AND2(c10::kHalf, c10::kBFloat16, dy.scalar_type(), "ptsycl_upsample_nearest2d_bwd", [&] {
            nearest_bwd<scalar_t>(dy, acc, g, exact);
        });
    }
    grad_input.copy_(acc);
    return grad_input;
}

Tensor& upsample_bilinear2d_device(const Tensor& self, int64_t Ho, int64_t Wo, bool align_corners,
                                   c10::optional<double> scales_h, c10::optional<double> scales_w, Tensor& out) {
    const Tensor x = self.contiguous();
    Up2dGeom g = up_geom(x, Ho, Wo);
    g.sh = area_scale(g.H, Ho, align_corners, scales_h);
    g.sw = area_scale(g.W, Wo, align_corners, scales_w);
    if (out.numel() == 0) return out;
    Tensor o = out.is_contiguous() ? out : at::empty(out.sizes(), out.options());
    AT_DISPATCH_FLOATING_TYPES_AND2(c10::kHalf, c10::kBFloat16, x.scalar_type(), "ptsycl_upsample_bilinear2d", [&] {
        bilinear_fwd<scalar_t>(x, o, g, align_corners);
    });
    if (!o.is_same(out)) out.copy_(o);
    return out;
}

Tensor& upsample_bilinear2d_backward_device(const Tensor& grad_output, at::IntArrayRef input_size,
                                            bool align_corners, c10::optional<double> scales_h,
                                            c10::optional<double> scales_w, Tensor& grad_input) {
    const Tensor dy = grad_output.contiguous();
    const int64_t H = input_size[input_size.size() - 2], W = input_size[input_size.size() - 1];
    Up2dGeom g = up_geom(grad_input, dy.size(-2), dy.size(-1));
    g.sh = area_scale(H, g.Ho, align_corners, scales_h);
    g.sw = area_scale(W, g.Wo, align_corners, scales_w);
    Tensor acc = at::zeros(grad_input.sizes(), dy.options().dtype(dy.scalar_type() == c10::kDouble ? c10::kDouble : c10::kFloat));
    if (dy.numel() > 0) {
        AT_DISPATCH_FLOATING_TYPES_AND2(c10::kHalf, c10::kBFloat16, dy.scalar_type(), "ptsycl_upsample_bilinear2d_bwd", [&] {
            bilinear_bwd<scalar_t>(dy, acc, g, align_corners);
        });
    }
    grad_input.copy_(acc);
    return grad_input;
}

}
