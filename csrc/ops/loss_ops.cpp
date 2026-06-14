#include <ATen/core/Reduction.h>

#include "core/kernels.h"

namespace ptsycl {
namespace {

using at::Tensor;

double mean_scale(int64_t reduction, int64_t n) {
    return (reduction == at::Reduction::Mean && n > 0)
               ? 1.0 / static_cast<double>(n)
               : 1.0;
}

// -----------------------------------------------------------------------------
// mse_loss
// -----------------------------------------------------------------------------
Tensor mse_loss(const Tensor& self, const Tensor& target, int64_t reduction) {
    PTSYCL_TRACE_OP("mse_loss");
    auto& q = queue_for(self);
    Tensor x = self.contiguous();
    Tensor y = target.expand_as(self).contiguous();
    const int64_t n = x.numel();

    if (reduction == at::Reduction::None) {
        Tensor out = at::empty_like(x);
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "mse_none", [&] {
            const scalar_t* px = data_ptr<scalar_t>(x);
            const scalar_t* py = data_ptr<scalar_t>(y);
            scalar_t* po = data_ptr<scalar_t>(out);
            launch_flat(q, n, [=](std::size_t i) {
                const double d = static_cast<double>(px[i]) -
                                 static_cast<double>(py[i]);
                po[i] = static_cast<scalar_t>(d * d);
            });
        });
        return out;
    }

    Tensor out = at::empty({}, x.options());
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "mse_reduce", [&] {
        const scalar_t* px = data_ptr<scalar_t>(x);
        const scalar_t* py = data_ptr<scalar_t>(y);
        const double r = reduce_full<double>(
            q, n, 0.0,
            [=](int64_t i) {
                const double d = static_cast<double>(px[i]) -
                                 static_cast<double>(py[i]);
                return d * d;
            },
            [](double a, double b) { return a + b; });
        q.synchronize();
        *data_ptr<scalar_t>(out) =
            static_cast<scalar_t>(r * mean_scale(reduction, n));
    });
    return out;
}

Tensor mse_loss_backward(const Tensor& grad_output, const Tensor& self,
                         const Tensor& target, int64_t reduction) {
    PTSYCL_TRACE_OP("mse_loss_backward");
    auto& q = queue_for(self);
    Tensor x = self.contiguous();
    Tensor y = target.expand_as(self).contiguous();
    Tensor g = grad_output.expand_as(self).contiguous();
    Tensor out = at::empty_like(x);
    const int64_t n = x.numel();
    const double scale = 2.0 * mean_scale(reduction, n);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "mse_bwd", [&] {
        const scalar_t* px = data_ptr<scalar_t>(x);
        const scalar_t* py = data_ptr<scalar_t>(y);
        const scalar_t* pg = data_ptr<scalar_t>(g);
        scalar_t* po = data_ptr<scalar_t>(out);
        launch_flat(q, n, [=](std::size_t i) {
            po[i] = static_cast<scalar_t>(
                scale *
                (static_cast<double>(px[i]) - static_cast<double>(py[i])) *
                static_cast<double>(pg[i]));
        });
    });
    return out;
}

// -----------------------------------------------------------------------------
// binary_cross_entropy
// -----------------------------------------------------------------------------
constexpr double kBceLogFloor = -100.0; // matches ATen clamp on log values

Tensor binary_cross_entropy(const Tensor& self, const Tensor& target,
                            const std::optional<Tensor>& weight,
                            int64_t reduction) {
    PTSYCL_TRACE_OP("binary_cross_entropy");
    auto& q = queue_for(self);
    Tensor x = self.contiguous();
    Tensor y = target.expand_as(self).contiguous();
    const bool has_w = weight.has_value() && weight->defined();
    Tensor w = has_w ? weight->expand_as(self).contiguous() : Tensor();
    const int64_t n = x.numel();

    if (reduction == at::Reduction::None) {
        Tensor out = at::empty_like(x);
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "bce_none", [&] {
            const scalar_t* px = data_ptr<scalar_t>(x);
            const scalar_t* py = data_ptr<scalar_t>(y);
            const scalar_t* pw = has_w ? data_ptr<scalar_t>(w) : nullptr;
            scalar_t* po = data_ptr<scalar_t>(out);
            launch_flat(q, n, [=](std::size_t i) {
                const double xi = static_cast<double>(px[i]);
                const double yi = static_cast<double>(py[i]);
                double lx  = ::log(xi);
                double l1x = ::log(1.0 - xi);
                if (lx < kBceLogFloor) lx = kBceLogFloor;
                if (l1x < kBceLogFloor) l1x = kBceLogFloor;
                double v = -(yi * lx + (1.0 - yi) * l1x);
                if (pw) v *= static_cast<double>(pw[i]);
                po[i] = static_cast<scalar_t>(v);
            });
        });
        return out;
    }

    Tensor out = at::empty({}, x.options());
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "bce_reduce", [&] {
        const scalar_t* px = data_ptr<scalar_t>(x);
        const scalar_t* py = data_ptr<scalar_t>(y);
        const scalar_t* pw = has_w ? data_ptr<scalar_t>(w) : nullptr;
        const double r = reduce_full<double>(
            q, n, 0.0,
            [=](int64_t i) {
                const double xi = static_cast<double>(px[i]);
                const double yi = static_cast<double>(py[i]);
                double lx  = ::log(xi);
                double l1x = ::log(1.0 - xi);
                if (lx < kBceLogFloor) lx = kBceLogFloor;
                if (l1x < kBceLogFloor) l1x = kBceLogFloor;
                double v = -(yi * lx + (1.0 - yi) * l1x);
                if (pw) v *= static_cast<double>(pw[i]);
                return v;
            },
            [](double a, double b) { return a + b; });
        q.synchronize();
        *data_ptr<scalar_t>(out) =
            static_cast<scalar_t>(r * mean_scale(reduction, n));
    });
    return out;
}

Tensor bce_backward_impl(const Tensor& grad_output, const Tensor& self,
                         const Tensor& target,
                         const std::optional<Tensor>& weight,
                         int64_t reduction) {
    auto& q = queue_for(self);
    Tensor x = self.contiguous();
    Tensor y = target.expand_as(self).contiguous();
    Tensor g = grad_output.expand_as(self).contiguous();
    const bool has_w = weight.has_value() && weight->defined();
    Tensor w = has_w ? weight->expand_as(self).contiguous() : Tensor();
    Tensor out = at::empty_like(x);
    const int64_t n = x.numel();
    const double scale = mean_scale(reduction, n);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "bce_bwd", [&] {
        const scalar_t* px = data_ptr<scalar_t>(x);
        const scalar_t* py = data_ptr<scalar_t>(y);
        const scalar_t* pg = data_ptr<scalar_t>(g);
        const scalar_t* pw = has_w ? data_ptr<scalar_t>(w) : nullptr;
        scalar_t* po = data_ptr<scalar_t>(out);
        launch_flat(q, n, [=](std::size_t i) {
            const double xi = static_cast<double>(px[i]);
            const double yi = static_cast<double>(py[i]);
            double denom = xi * (1.0 - xi);
            if (denom < 1e-12) denom = 1e-12;
            double v = (xi - yi) / denom * static_cast<double>(pg[i]) * scale;
            if (pw) v *= static_cast<double>(pw[i]);
            po[i] = static_cast<scalar_t>(v);
        });
    });
    return out;
}

Tensor binary_cross_entropy_backward(const Tensor& grad_output,
                                     const Tensor& self, const Tensor& target,
                                     const std::optional<Tensor>& weight,
                                     int64_t reduction) {
    PTSYCL_TRACE_OP("binary_cross_entropy_backward");
    return bce_backward_impl(grad_output, self, target, weight, reduction);
}

Tensor& binary_cross_entropy_backward_out(const Tensor& grad_output,
                                          const Tensor& self,
                                          const Tensor& target,
                                          const std::optional<Tensor>& weight,
                                          int64_t reduction,
                                          Tensor& grad_input) {
    Tensor r = bce_backward_impl(grad_output, self, target, weight, reduction);
    grad_input.copy_(r);
    return grad_input;
}

// -----------------------------------------------------------------------------
// nll_loss (2-D input [N, C], 1-D target [N])
// -----------------------------------------------------------------------------
std::tuple<Tensor&, Tensor&> nll_loss_forward_out(
    const Tensor& self, const Tensor& target,
    const std::optional<Tensor>& weight, int64_t reduction,
    int64_t ignore_index, Tensor& output, Tensor& total_weight) {
    PTSYCL_TRACE_OP("nll_loss_forward.output");
    TORCH_CHECK(self.dim() == 2, "paras nll_loss: expected 2-D input");
    TORCH_CHECK(target.dim() == 1 && target.size(0) == self.size(0),
                "paras nll_loss: bad target shape");

    auto& q = queue_for(self);
    Tensor x = self.contiguous();
    Tensor t = target.contiguous();
    const bool has_w = weight.has_value() && weight->defined();
    Tensor w = has_w ? weight->contiguous() : Tensor();
    const int64_t N = x.size(0);
    const int64_t C = x.size(1);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "nll_fwd", [&] {
        const scalar_t* px = data_ptr<scalar_t>(x);
        const int64_t* pt = data_ptr<int64_t>(t);
        const scalar_t* pw = has_w ? data_ptr<scalar_t>(w) : nullptr;

        if (reduction == at::Reduction::None) {
            if (output.numel() != N) output.resize_({N});
            scalar_t* po = data_ptr<scalar_t>(output);
            launch_flat(q, N, [=](std::size_t i) {
                const int64_t ti = pt[i];
                if (ti == ignore_index) {
                    po[i] = static_cast<scalar_t>(0);
                    return;
                }
                const double wv = pw ? static_cast<double>(pw[ti]) : 1.0;
                po[i] = static_cast<scalar_t>(
                    -wv * static_cast<double>(px[i * C + ti]));
            });
            q.synchronize();
            *data_ptr<scalar_t>(total_weight) = static_cast<scalar_t>(0);
            return;
        }

        // Reduced: loss = sum(-w[t_i] * x[i, t_i]) and total weight.
        const double loss_sum = reduce_full<double>(
            q, N, 0.0,
            [=](int64_t i) {
                const int64_t ti = pt[i];
                if (ti == ignore_index) return 0.0;
                const double wv = pw ? static_cast<double>(pw[ti]) : 1.0;
                return -wv * static_cast<double>(px[i * C + ti]);
            },
            [](double a, double b) { return a + b; });
        const double weight_sum = reduce_full<double>(
            q, N, 0.0,
            [=](int64_t i) {
                const int64_t ti = pt[i];
                if (ti == ignore_index) return 0.0;
                return pw ? static_cast<double>(pw[ti]) : 1.0;
            },
            [](double a, double b) { return a + b; });
        q.synchronize();

        double result = loss_sum;
        if (reduction == at::Reduction::Mean)
            result = weight_sum > 0 ? loss_sum / weight_sum : 0.0;
        *data_ptr<scalar_t>(output) = static_cast<scalar_t>(result);
        *data_ptr<scalar_t>(total_weight) = static_cast<scalar_t>(weight_sum);
    });
    return {output, total_weight};
}

Tensor& nll_loss_backward_out(const Tensor& grad_output, const Tensor& self,
                              const Tensor& target,
                              const std::optional<Tensor>& weight,
                              int64_t reduction, int64_t ignore_index,
                              const Tensor& total_weight, Tensor& grad_input) {
    PTSYCL_TRACE_OP("nll_loss_backward.grad_input");
    auto& q = queue_for(self);
    Tensor t = target.contiguous();
    const bool has_w = weight.has_value() && weight->defined();
    Tensor w = has_w ? weight->contiguous() : Tensor();
    const int64_t N = self.size(0);
    const int64_t C = self.size(1);

    grad_input.zero_();

    AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "nll_bwd", [&] {
        const int64_t* pt = data_ptr<int64_t>(t);
        const scalar_t* pw = has_w ? data_ptr<scalar_t>(w) : nullptr;
        scalar_t* pg = data_ptr<scalar_t>(grad_input);

        q.synchronize(); // host reads of small scalars below
        double tw = static_cast<double>(*data_ptr<scalar_t>(total_weight));
        double g0 = 0.0;
        Tensor go = grad_output.contiguous();
        const scalar_t* pgo = data_ptr<scalar_t>(go);
        const bool per_element = (reduction == at::Reduction::None);
        if (!per_element) g0 = static_cast<double>(pgo[0]);
        const bool mean_red = (reduction == at::Reduction::Mean);

        launch_flat(q, N, [=](std::size_t i) {
            const int64_t ti = pt[i];
            if (ti == ignore_index) return;
            const double wv = pw ? static_cast<double>(pw[ti]) : 1.0;
            double g = per_element ? static_cast<double>(pgo[i]) : g0;
            if (mean_red && tw > 0) g /= tw;
            pg[i * C + ti] = static_cast<scalar_t>(-wv * g);
        });
    });
    return grad_input;
}

} // namespace
} // namespace ptsycl

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("aten::mse_loss", &ptsycl::mse_loss);
    m.impl("aten::mse_loss_backward", &ptsycl::mse_loss_backward);
    m.impl("aten::binary_cross_entropy", &ptsycl::binary_cross_entropy);
    m.impl("aten::binary_cross_entropy_backward",
           &ptsycl::binary_cross_entropy_backward);
    m.impl("aten::binary_cross_entropy_backward.grad_input",
           &ptsycl::binary_cross_entropy_backward_out);
    m.impl("aten::nll_loss_forward.output", &ptsycl::nll_loss_forward_out);
    m.impl("aten::nll_loss_backward.grad_input",
           &ptsycl::nll_loss_backward_out);
}
