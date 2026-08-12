"""Fused kernels exposed as ordinary differentiable functions.

These wrap the ``paras::`` custom ops (registered in
csrc/ops/fused_ops.cpp) in torch.autograd.Function subclasses, since a
freshly defined custom-op namespace has no autograd formula of its own.
"""

import torch

from . import _C  # noqa: F401  (ensures the paras:: ops are registered)

__all__ = ["rms_norm", "swiglu", "rotate_half"]


class _RMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, eps):
        out, rstd = torch.ops.paras.rms_norm(input, weight, eps)
        ctx.save_for_backward(input, weight, rstd)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        input, weight, rstd = ctx.saved_tensors
        grad_input, grad_weight = torch.ops.paras.rms_norm_backward(
            grad_out.contiguous(), input, weight, rstd)
        return grad_input, grad_weight, None


class _SwiGLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        ctx.save_for_backward(gate, up)
        return torch.ops.paras.swiglu(gate, up)

    @staticmethod
    def backward(ctx, grad_out):
        gate, up = ctx.saved_tensors
        grad_gate, grad_up = torch.ops.paras.swiglu_backward(
            grad_out.contiguous(), gate, up)
        return grad_gate, grad_up


class _RotateHalfFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.ops.paras.rotate_half(x)

    @staticmethod
    def backward(ctx, grad_out):
        return torch.ops.paras.rotate_half_backward(grad_out.contiguous())


def rms_norm(input: torch.Tensor, weight: torch.Tensor,
             eps: float = 1e-6) -> torch.Tensor:
    """Fused RMSNorm over the last dimension of ``input``."""
    return _RMSNormFn.apply(input, weight, eps)


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused ``silu(gate) * up``."""
    return _SwiGLUFn.apply(gate, up)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Fused RoPE helper: ``cat(-x2, x1, dim=-1)`` where
    ``x1, x2 = x.chunk(2, dim=-1)``."""
    return _RotateHalfFn.apply(x)
