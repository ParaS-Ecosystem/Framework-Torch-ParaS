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

"""Fake kernels for the fused ``paras::`` operators.

The kernels and their autograd (backward = the ParaS backward kernel) live in
csrc/ops/fused_ops.cpp. Registering their shape/dtype rules (fake kernels)
lets FakeTensor tracing -- Dynamo and AOTAutograd -- see through them, so
training graphs also run their backward on the ParaS backward kernels.
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

    @torch.library.register_fake("paras::add_rms_norm")
    def _add_rms_norm_fake(a, b, weight, eps):
        rows = a.numel() // a.shape[-1] if a.shape[-1] != 0 else 0
        sum_ = torch.empty_like(a, memory_format=torch.contiguous_format)
        return sum_, torch.empty_like(sum_), a.new_empty((rows,), dtype=torch.float32)

    @torch.library.register_fake("paras::rope")
    def _rope_fake(x, cos, sin):
        return torch.empty_like(x)

    @torch.library.register_fake("paras::sdpa")
    def _sdpa_fake(q, k, v, mask, scale, is_causal):
        if v.shape[-1] == q.shape[-1]:
            return torch.empty_like(q)
        return q.new_empty(q.shape[:-1] + (v.shape[-1],))

    _registered = True
