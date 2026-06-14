# Torch-ParaS

A PyTorch out-of-tree backend for the ParaS compiler. It registers a
`paras` device with PyTorch, so existing models run on the ParaS runtime
without code changes:

```python
import torch
import torch_paras

x = torch.randn(64, 64, device="paras")     # device 0 = host CPU engine
y = torch.randn(64, 64, device="paras:1")   # devices 1.. = NVIDIA GPUs
z = (x @ x).relu().cpu()
```

Device 0 always exists and runs on the host CPU through the ParaS
threadpool engine. In CUDA builds, devices 1..N map to the visible NVIDIA
GPUs. Tested on Intel CPUs and NVIDIA GPUs.

## What is implemented

- Tensor lifecycle: allocation, strided layouts, views, copies in every
  direction (host/device, cross-device, dtype conversion), resize.
- Around 130 aten kernels: elementwise math, activations, comparisons,
  reductions, matrix multiply (mm/bmm/addmm), convolution, pooling,
  upsampling, batch/layer norm, losses, RNG (Philox counter-based),
  dropout, multi-head attention.
- Anything not implemented natively falls back to CPU through the boxed
  fallback, so models keep working while coverage grows.
- A binned memory pool per device on top of CUDA unified memory (GPU) or
  aligned host memory (CPU).

## Repository layout

```
csrc/compat/      compatibility layer, the only code that knows about the
                  ParaS compiler and CUDA 
csrc/core/        device context, allocator, kernel launch helpers, registration
csrc/ops/         the aten kernels, plain C++ lambdas, compiler agnostic
python/torch_paras/  the python package
scripts/          env.sh (toolchain paths) and build.sh
tests/            test suite, run with tests/run_all.py
docs/             install guide, testing guide
```

## Quick start

```bash
source scripts/env.sh          # adjust paths for your machine first
scripts/build.sh cpu           # CPU-only build, or:
scripts/build.sh cuda          # CPU + NVIDIA GPU build

python tests/run_all.py --all-devices
```

See docs/INSTALL.md for the full toolchain setup, docs/TESTING.md




