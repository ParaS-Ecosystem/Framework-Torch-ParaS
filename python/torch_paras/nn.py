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


"""Fused kernels exposed as ordinary differentiable functions.

These call the ``paras::`` custom ops (csrc/ops/fused_ops.cpp), whose
autograd is registered there: the backward runs the ParaS backward kernel.
"""

import torch

from . import _C  # noqa: F401  (ensures the paras:: ops are registered)

__all__ = ["rms_norm", "swiglu", "rotate_half"]


def rms_norm(input: torch.Tensor, weight: torch.Tensor,
             eps: float = 1e-6) -> torch.Tensor:
    """Fused RMSNorm over the last dimension of ``input``."""
    return torch.ops.paras.rms_norm(input, weight, eps)[0]


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused ``silu(gate) * up``."""
    return torch.ops.paras.swiglu(gate, up)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Fused RoPE helper: ``cat(-x2, x1, dim=-1)`` where
    ``x1, x2 = x.chunk(2, dim=-1)``."""
    return torch.ops.paras.rotate_half(x)
