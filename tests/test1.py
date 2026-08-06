
import argparse
import importlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import numpy as np
import torch
import torch_paras  # importing this registers the "paras" device with torch

# Device 0 is always the host CPU engine; CUDA GPUs start at paras:1.
# See python/torch_paras/__init__.py for the mapping.
device = "paras:1" if torch_paras.device_count() > 1 else "paras:0"
print(f"Using device={device!r} ({torch_paras.device_count()} paras device(s) visible)")

RUN_AUTOGRAD_SMOKE = True
RUN_BROADCAST_SMOKE = True
RUN_IMAGE_PIPELINE_SMOKE = True

# -----------------------------------------------------------------------------
# 1. Basic elementwise op + autograd round-trip
# -----------------------------------------------------------------------------
if RUN_AUTOGRAD_SMOKE:
    x = torch.randn(10, requires_grad=True)
    xd = x.detach().clone().to(device).requires_grad_(True)

    y = x.relu().mul(2.0)
    yd = xd.relu().mul(2.0)

    dy = torch.randn(10)
    y.backward(dy)
    yd.backward(dy.to(device))

    print("=== autograd smoke ===")
    print("x         =", x)
    print("y (cpu)   =", y)
    print("y (paras) =", yd.cpu())
    print("dx (cpu)   =", x.grad)
    print("dx (paras) =", xd.grad.cpu())
    print("fwd match:", torch.allclose(y, yd.cpu()))
    print("bwd match:", torch.allclose(x.grad, xd.grad.cpu()))
    print()

# -----------------------------------------------------------------------------
# 2. Broadcasting add
# -----------------------------------------------------------------------------
if RUN_BROADCAST_SMOKE:
    t1 = torch.ones((20, 10), requires_grad=True)
    t2 = torch.randn(1, 10)

    t1d = t1.detach().clone().to(device)
    t2d = t2.to(device)

    with torch.no_grad():
        print("=== broadcast smoke ===")
        print("t1.shape =", t1.shape, " t1d.shape =", t1d.shape)
        print("t2.shape =", t2.shape, " t2d.shape =", t2d.shape)

        t3 = t1 + t2
        t3d = t1d + t2d
        print("t3d (cpu) =", t3d.cpu())
        print("broadcast add match:", torch.allclose(t3, t3d.cpu()))
        print()

# -----------------------------------------------------------------------------
# 3. Image-normalize-style pipeline: mul/add_/clamp_/permute/to(uint8)/numpy
# -----------------------------------------------------------------------------
if RUN_IMAGE_PIPELINE_SMOKE:
    grid_src = torch.randn(2, 3, 4)
    grid_dev = grid_src.detach().clone().to(device)

    ref = (
        grid_src.mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .to("cpu", torch.uint8)
        .numpy()
    )
    out = (
        grid_dev.mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .to("cpu", torch.uint8)
        .numpy()
    )

    print("=== image pipeline smoke ===")
    print("REF")
    print(ref)
    print("DEV")
    print(out)
    print("max abs diff:", np.max(np.abs(ref.astype(np.float32) - out.astype(np.float32))))
