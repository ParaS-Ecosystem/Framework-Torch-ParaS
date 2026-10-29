"""Mixed precision on paras devices: autocast's dtype policy, GradScaler
(unscale / inf check and scale update on the device, as on the CPU), and
fp16 / mixed-precision GEMMs."""

import torch
import torch_paras



def test_autocast_policy(device):
    x = torch.randn(8, 32, device=device)
    w = torch.randn(16, 32, device=device)
    with torch.autocast(device_type="paras", dtype=torch.bfloat16):
        y = torch.nn.functional.linear(x, w)
        z = x @ w.t()
        s = torch.softmax(y, -1)
    assert y.dtype == torch.bfloat16 and z.dtype == torch.bfloat16, (y.dtype, z.dtype)
    assert s.dtype == torch.float32, s.dtype
    assert (y.float().cpu() - (x.cpu() @ w.cpu().t())).abs().max() < 0.5


def test_grad_scaler_matches_cpu(device):
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 4))
    ref = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 4))
    ref.load_state_dict(model.state_dict())
    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    opt_ref = torch.optim.SGD(ref.parameters(), lr=0.1)
    kw = dict(init_scale=2.0 ** 8, growth_factor=2.0, backoff_factor=0.5, growth_interval=2)
    scaler = torch.amp.GradScaler("paras", **kw)
    scaler_ref = torch.amp.GradScaler("cpu", **kw)
    for step in range(6):
        x = torch.randn(8, 16)
        if step == 3:
            x[0, 0] = float("inf")
        for m, o, sc, xx in ((model, opt, scaler, x.to(device)), (ref, opt_ref, scaler_ref, x)):
            o.zero_grad()
            sc.scale(m(xx).float().pow(2).mean()).backward()
            sc.step(o)
            sc.update()
        assert scaler.get_scale() == scaler_ref.get_scale(), (step, scaler.get_scale(), scaler_ref.get_scale())
    for p, q in zip(model.parameters(), ref.parameters()):
        assert torch.allclose(p.detach().cpu(), q.detach(), atol=1e-5, rtol=1e-4, equal_nan=True)


def test_fp16_and_mixed_gemm(device):
    torch.manual_seed(0)
    a = torch.randn(64, 96).half()
    b = torch.randn(96, 48).half()
    got = (a.to(device) @ b.to(device)).cpu()
    ref = (a.float() @ b.float())
    assert got.dtype == torch.float16
    assert (got.float() - ref).abs().max() / ref.abs().max() < 2e-3
    for dt in (torch.float16, torch.bfloat16):
        x = torch.randn(3, 33, 40).to(dt)
        y = torch.randn(3, 40, 17).to(dt)
        got = torch.bmm(x.to(device), y.to(device)).cpu().float()
        ref = torch.bmm(x.float(), y.float())
        assert (got - ref).abs().max() / ref.abs().max() < 1e-2, dt


def test_amp_unscale_fp16_gradients_at_large_scales(device):
    g = (torch.randn(1000) * 2.0 ** 25).half()
    inv = torch.tensor(2.0 ** -25)
    found = torch.zeros(1)
    gd = g.to(device)
    torch._amp_foreach_non_finite_check_and_unscale_([gd], found.to(device), inv.to(device))
    ref = (g.float() * inv).half()
    assert torch.equal(gd.cpu(), ref) and float(gd.cpu().abs().max()) > 0
