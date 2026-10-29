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


def test_index_copy_out_of_range_index_does_not_write_past_the_tensor(device):
    import torch_paras
    torch.manual_seed(0)
    store = torch.zeros(2, 8, 16, device=device)
    cache = store[0]
    src = torch.randn(1, 16)
    cache.index_copy_(0, torch.tensor([8], device=device), src.to(device))
    torch_paras.synchronize()
    assert torch.equal(store.cpu(), torch.zeros(2, 8, 16)), "index_copy_ wrote outside the cache"
    cache.index_copy_(0, torch.tensor([3], device=device), src.to(device))
    assert torch.equal(cache.cpu()[3], src[0])
    x = torch.randn(1, 2, 1, 16)
    cos, sin = torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16)
    kv = torch.zeros(1, 2, 8, 16, device=device)
    guard = torch.zeros(1, 2, 8, 16, device=device)
    torch.ops.paras.rope_index_copy_(kv, 2, torch.tensor([8], device=device), x.to(device), cos.to(device), sin.to(device))
    torch_paras.synchronize()
    assert torch.equal(kv.cpu(), torch.zeros(1, 2, 8, 16)) and torch.equal(guard.cpu(), torch.zeros(1, 2, 8, 16))
