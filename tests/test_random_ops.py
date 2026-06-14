import torch


def test_uniform_(device):
    x = torch.empty((100_000,), device=device, dtype=torch.float32)
    x.uniform_(2.0, 5.0)
    cpu = x.cpu()
    assert torch.all(cpu >= 2.0)
    assert torch.all(cpu < 5.0)
    assert abs(float(cpu.mean()) - 3.5) < 0.05


def test_bernoulli_(device):
    p = 0.30
    x = torch.empty((100_000,), device=device, dtype=torch.float32)
    x.bernoulli_(p)
    assert abs(float(x.cpu().mean()) - p) < 0.03


def test_dropout_train(device):
    p = 0.4
    inp = torch.ones((100_000,), device=device, dtype=torch.float32)
    out, mask = torch.native_dropout(inp, p, True)
    scale = 1.0 / (1.0 - p)
    cpu_out = out.cpu()
    valid = (cpu_out == 0.0) | torch.isclose(
        cpu_out, torch.tensor(scale), atol=1e-5)
    assert valid.all()
    keep_ratio = float(mask.float().mean().cpu())
    assert abs(keep_ratio - (1.0 - p)) < 0.03


def test_dropout_eval(device):
    inp = torch.randn((1000,), device=device)
    out, mask = torch.native_dropout(inp, 0.5, False)
    assert torch.allclose(out.cpu(), inp.cpu())
    assert torch.all(mask.cpu() == 1)


def test_dropout_backward(device):
    grad = torch.randn((10_000,), device=device)
    mask = torch.rand((10_000,), device=device) < 0.7
    scale = 1.0 / 0.7
    dx = torch.ops.aten.native_dropout_backward(grad, mask, scale)
    expected = grad.cpu() * mask.cpu() * scale
    assert torch.allclose(dx.cpu(), expected, atol=1e-4, rtol=1e-4)


def test_rng_advances(device):
    a = torch.empty((10,), device=device).uniform_(0, 1)
    b = torch.empty((10,), device=device).uniform_(0, 1)
    assert not torch.equal(a.cpu(), b.cpu())
