"""
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
"""

"""Fake kernels and autograd formulas for the fused ``paras::`` operators.

The kernels live in csrc/ops/fused_ops.cpp. Registering their shape/dtype
rules (fake kernels) lets FakeTensor tracing see through them, and
registering their backward formulas lets AOTAutograd build training graphs in
which the backward also runs on the ParaS backward kernels.
"""

import torch

from .. import _C  # noqa: F401  (registers the paras:: operators)

_registered = False


def register() -> None:
    """Idempotently registers fake kernels and autograd for paras:: ops."""
    global _registered
    if _registered:
        return

    @torch.library.register_fake("paras::rms_norm")
    def _rms_norm_fake(input, weight, eps):
        rows = input.numel() // input.shape[-1] if input.shape[-1] != 0 else 0
        out = torch.empty_like(input, memory_format=torch.contiguous_format)
        return out, input.new_empty((rows,), dtype=torch.float32)

    @torch.library.register_fake("paras::rms_norm_backward")
    def _rms_norm_backward_fake(grad_out, input, weight, rstd):
        return (torch.empty_like(input, memory_format=torch.contiguous_format),
                weight.new_empty((input.shape[-1],)))

    @torch.library.register_fake("paras::swiglu")
    def _swiglu_fake(gate, up):
        return torch.empty_like(gate)

    @torch.library.register_fake("paras::swiglu_backward")
    def _swiglu_backward_fake(grad_out, gate, up):
        return torch.empty_like(gate), torch.empty_like(up)

    @torch.library.register_fake("paras::rotate_half")
    def _rotate_half_fake(x):
        return torch.empty_like(x)

    @torch.library.register_fake("paras::rotate_half_backward")
    def _rotate_half_backward_fake(grad_out):
        return torch.empty_like(grad_out)

    def _rms_norm_setup(ctx, inputs, output):
        input, weight, _eps = inputs
        _out, rstd = output
        ctx.mark_non_differentiable(rstd)
        ctx.save_for_backward(input, weight, rstd)

    def _rms_norm_backward(ctx, grad_out, _grad_rstd):
        input, weight, rstd = ctx.saved_tensors
        grad_input, grad_weight = torch.ops.paras.rms_norm_backward(
            grad_out.contiguous(), input, weight, rstd)
        return grad_input, grad_weight, None

    torch.library.register_autograd(
        "paras::rms_norm", _rms_norm_backward, setup_context=_rms_norm_setup)

    def _swiglu_setup(ctx, inputs, output):
        ctx.save_for_backward(*inputs)

    def _swiglu_backward(ctx, grad_out):
        gate, up = ctx.saved_tensors
        return torch.ops.paras.swiglu_backward(grad_out.contiguous(), gate, up)

    torch.library.register_autograd(
        "paras::swiglu", _swiglu_backward, setup_context=_swiglu_setup)

    def _rotate_half_setup(ctx, inputs, output):
        pass

    def _rotate_half_backward(ctx, grad_out):
        return torch.ops.paras.rotate_half_backward(grad_out.contiguous())

    torch.library.register_autograd(
        "paras::rotate_half", _rotate_half_backward,
        setup_context=_rotate_half_setup)
    _registered = True
