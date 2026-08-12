import torch
import torch.nn.functional as F

from common import assert_close
from torch_paras import nn as paras_nn


def _rms_norm_ref(x, weight, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def test_rms_norm_forward(device):
    x = torch.randn(4, 8, requires_grad=False)
    w = torch.randn(8)
    out_ref = _rms_norm_ref(x, w, 1e-6)
    out_dev = paras_nn.rms_norm(x.to(device), w.to(device), 1e-6)
    assert_close(out_dev, out_ref, "rms_norm")


def test_rms_norm_backward(device):
    x = torch.randn(4, 8, requires_grad=True)
    w = torch.randn(8, requires_grad=True)
    _rms_norm_ref(x, w, 1e-6).sum().backward()

    xd = x.detach().to(device).requires_grad_()
    wd = w.detach().to(device).requires_grad_()
    paras_nn.rms_norm(xd, wd, 1e-6).sum().backward()

    assert_close(xd.grad, x.grad, "rms_norm.grad_input")
    assert_close(wd.grad, w.grad, "rms_norm.grad_weight")


def test_swiglu_forward(device):
    gate = torch.randn(4, 8)
    up = torch.randn(4, 8)
    out_ref = F.silu(gate) * up
    out_dev = paras_nn.swiglu(gate.to(device), up.to(device))
    assert_close(out_dev, out_ref, "swiglu")


def test_swiglu_backward(device):
    gate = torch.randn(4, 8, requires_grad=True)
    up = torch.randn(4, 8, requires_grad=True)
    (F.silu(gate) * up).sum().backward()

    gd = gate.detach().to(device).requires_grad_()
    ud = up.detach().to(device).requires_grad_()
    paras_nn.swiglu(gd, ud).sum().backward()

    assert_close(gd.grad, gate.grad, "swiglu.grad_gate")
    assert_close(ud.grad, up.grad, "swiglu.grad_up")


def test_rotate_half_forward(device):
    x = torch.randn(4, 8)
    x1, x2 = x.chunk(2, dim=-1)
    out_ref = torch.cat((-x2, x1), dim=-1)
    out_dev = paras_nn.rotate_half(x.to(device))
    assert_close(out_dev, out_ref, "rotate_half")


def test_rotate_half_backward(device):
    x = torch.randn(4, 8, requires_grad=True)
    x1, x2 = x.chunk(2, dim=-1)
    torch.cat((-x2, x1), dim=-1).sum().backward()

    xd = x.detach().to(device).requires_grad_()
    paras_nn.rotate_half(xd).sum().backward()

    assert_close(xd.grad, x.grad, "rotate_half.grad")
