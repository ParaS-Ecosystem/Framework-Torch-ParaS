"""Normalisation, convolution, pooling and upsampling on the device: forward
and backward against PyTorch on the CPU, and no copies to the host."""

import torch
import torch.nn.functional as F
import torch_paras


def _close(got, ref, what, tol=1e-4):
    got = got.cpu().to(ref.dtype)
    assert got.shape == ref.shape, (what, got.shape, ref.shape)
    assert torch.allclose(got, ref, rtol=tol, atol=tol), (what, (got - ref).abs().max().item())


def _check(device, fn, *inputs, tol=1e-4, what=""):
    """fn on CPU leaves and on device copies: outputs and input gradients."""
    ref_in = [x.detach().clone().requires_grad_(x.is_floating_point()) for x in inputs]
    dev_in = [x.detach().to(device).requires_grad_(x.is_floating_point()) for x in inputs]
    ref = fn(*ref_in)
    torch_paras.reset_runtime_stats()
    got = fn(*dev_in)
    copies = torch_paras.runtime_stats().get("device_to_host_copies", 0)
    assert copies == 0, (what, "forward copied to the host", copies)
    _close(got, ref.detach(), what + " forward", tol)
    g = torch.randn_like(ref)
    ref.backward(g)
    got.backward(g.to(device))
    for i, (r, d) in enumerate(zip(ref_in, dev_in)):
        if r.grad is not None:
            _close(d.grad, r.grad, f"{what} grad[{i}]", tol)


def test_layer_norm(device):
    torch.manual_seed(0)
    for shape, norm in (((4, 7, 256), (256,)), ((3, 1000), (1000,)), ((2, 5, 8, 16), (8, 16)),
                        ((2, 20000), (20000,)), ((6, 100), (100,))):
        x = torch.randn(shape) * 3 + 1
        w = torch.randn(norm)
        b = torch.randn(norm)
        _check(device, lambda x, w, b: F.layer_norm(x, norm, w, b, 1e-5), x, w, b, what=f"layer_norm {shape}")
        _check(device, lambda x: F.layer_norm(x, norm), x, what=f"layer_norm no affine {shape}")


def test_layer_norm_bf16(device):
    torch.manual_seed(0)
    x = torch.randn(8, 512, dtype=torch.bfloat16)
    w = torch.randn(512, dtype=torch.bfloat16)
    ref = F.layer_norm(x.float(), (512,), w.float()).bfloat16()
    got = F.layer_norm(x.to(device), (512,), w.to(device))
    assert (got.cpu().float() - ref.float()).abs().max().item() <= 0.04


def test_batch_norm(device):
    torch.manual_seed(0)
    for shape in ((4, 3, 5, 6), (8, 16), (2, 4, 10)):
        x = torch.randn(shape) * 2 + 0.5
        C = shape[1]
        w, b = torch.randn(C), torch.randn(C)
        rm, rv = torch.zeros(C), torch.ones(C)
        rmd, rvd = rm.clone().to(device), rv.clone().to(device)
        _check(device, lambda x, w, b: F.batch_norm(x, rm if x.device.type == "cpu" else rmd,
                                                    rv if x.device.type == "cpu" else rvd, w, b,
                                                    training=True, momentum=0.1), x, w, b,
               what=f"batch_norm train {shape}")
        _close(rmd, rm, "running_mean")
        _close(rvd, rv, "running_var")
        _check(device, lambda x, w, b: F.batch_norm(x, rm.to(x.device), rv.to(x.device), w, b, training=False),
               x, w, b, what=f"batch_norm eval {shape}")
        _check(device, lambda x: F.batch_norm(x, None, None, training=True), x, what=f"batch_norm no stats {shape}")


def test_conv2d(device):
    torch.manual_seed(0)
    cases = [dict(C=3, O=8, k=3, stride=1, padding=1, dilation=1, groups=1),
             dict(C=4, O=6, k=3, stride=2, padding=0, dilation=1, groups=2),
             dict(C=6, O=6, k=3, stride=1, padding=2, dilation=2, groups=6),
             dict(C=5, O=7, k=1, stride=1, padding=0, dilation=1, groups=1),
             dict(C=3, O=4, k=(3, 5), stride=(2, 1), padding=(1, 2), dilation=1, groups=1)]
    for c in cases:
        k = c["k"] if isinstance(c["k"], tuple) else (c["k"], c["k"])
        x = torch.randn(2, c["C"], 11, 13)
        w = torch.randn(c["O"], c["C"] // c["groups"], *k) * 0.3
        b = torch.randn(c["O"])
        _check(device, lambda x, w, b: F.conv2d(x, w, b, c["stride"], c["padding"], c["dilation"], c["groups"]),
               x, w, b, tol=2e-4, what=f"conv2d {c}")


def test_conv1d(device):
    torch.manual_seed(0)
    x = torch.randn(3, 4, 50)
    w = torch.randn(6, 2, 5) * 0.3
    _check(device, lambda x, w: F.conv1d(x, w, None, 2, 2, 1, 2), x, w, tol=2e-4, what="conv1d")


def test_pooling(device):
    torch.manual_seed(0)
    x = torch.randn(2, 3, 13, 11)
    for kw in (dict(kernel_size=3, stride=2), dict(kernel_size=3, stride=2, padding=1, ceil_mode=True),
               dict(kernel_size=(2, 3), stride=1, dilation=2)):
        _check(device, lambda x: F.max_pool2d(x, **kw), x, what=f"max_pool2d {kw}")
    for kw in (dict(kernel_size=3, stride=2), dict(kernel_size=3, stride=2, padding=1, ceil_mode=True),
               dict(kernel_size=3, stride=1, padding=1, count_include_pad=False),
               dict(kernel_size=2, divisor_override=3)):
        _check(device, lambda x: F.avg_pool2d(x, **kw), x, what=f"avg_pool2d {kw}")
    for size in ((1, 1), (5, 7), (13, 11), (4, 3)):
        _check(device, lambda x: F.adaptive_avg_pool2d(x, size), x, what=f"adaptive_avg_pool2d {size}")


def test_upsample(device):
    torch.manual_seed(0)
    x = torch.randn(2, 3, 7, 5)
    for kw in (dict(scale_factor=2, mode="nearest"), dict(size=(10, 9), mode="nearest"),
               dict(size=(10, 9), mode="nearest-exact"), dict(scale_factor=1.7, mode="nearest"),
               dict(size=(13, 11), mode="bilinear", align_corners=False),
               dict(size=(13, 11), mode="bilinear", align_corners=True),
               dict(scale_factor=0.6, mode="bilinear", align_corners=False)):
        _check(device, lambda x: F.interpolate(x, **kw), x, what=f"interpolate {kw}")


def test_resnet_block_trains_on_device(device):
    import copy
    torch.manual_seed(0)
    ref = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.BatchNorm2d(8), torch.nn.ReLU(),
                              torch.nn.MaxPool2d(2), torch.nn.Conv2d(8, 8, 3, padding=1, groups=2),
                              torch.nn.AdaptiveAvgPool2d((2, 2)), torch.nn.Flatten(), torch.nn.Linear(32, 4))
    dev = copy.deepcopy(ref).to(device)
    x = torch.randn(4, 3, 16, 16)
    out_ref = ref(x)
    out_ref.sum().backward()
    torch_paras.reset_runtime_stats()
    out = dev(x.to(device))
    out.sum().backward()
    assert torch_paras.runtime_stats().get("device_to_host_copies", 0) == 0
    _close(out.detach(), out_ref.detach(), "resnet block", tol=2e-4)
    for (n, p), (_, q) in zip(ref.named_parameters(), dev.named_parameters()):
        _close(q.grad, p.grad, f"resnet block grad {n}", tol=5e-4)


def test_conv3d_and_transposed(device):
    torch.manual_seed(0)
    x = torch.randn(2, 4, 6, 7, 5)
    w = torch.randn(6, 2, 3, 2, 3) * 0.3
    b = torch.randn(6)
    _check(device, lambda x, w, b: F.conv3d(x, w, b, (1, 2, 1), (1, 0, 1), (1, 1, 2), 2), x, w, b,
           tol=2e-4, what="conv3d")
    for dim, (x, w, kw) in {
            1: (torch.randn(2, 4, 9), torch.randn(4, 3, 4) * 0.3, dict(stride=2, padding=1, output_padding=1)),
            2: (torch.randn(2, 4, 5, 6), torch.randn(4, 3, 3, 2) * 0.3,
                dict(stride=(2, 1), padding=(1, 0), output_padding=(1, 0), groups=2, dilation=(1, 2))),
            3: (torch.randn(1, 2, 3, 4, 3), torch.randn(2, 2, 2, 3, 2) * 0.3, dict(stride=2))}.items():
        fn = {1: F.conv_transpose1d, 2: F.conv_transpose2d, 3: F.conv_transpose3d}[dim]
        b = torch.randn(w.size(1) * kw.get("groups", 1))
        _check(device, lambda x, w, b: fn(x, w, b, **kw), x, w, b, tol=2e-4, what=f"conv_transpose{dim}d")
