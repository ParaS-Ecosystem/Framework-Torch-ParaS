# Torch-ParaS

A PyTorch out-of-tree backend for the ParaS compiler. It registers a
`paras` device with PyTorch, so existing models run on the ParaS runtime
without code changes:

```python
import torch
import torch_paras

x = torch.randn(64, 64, device="paras")     # device 0 = host CPU engine
y = torch.randn(64, 64, device="paras:1")   # devices 1.. = NVIDIA or AMD GPUs
z = (x @ x).relu().cpu()
```

Device 0 always exists and runs on the host CPU through the ParaS
threadpool engine. In CUDA builds, devices 1..N map to the visible NVIDIA
GPUs; in HIP builds, to the visible AMD GPUs (each build targets one
vendor). Tested on Intel CPUs, NVIDIA GPUs, and AMD GPUs.

## What is implemented

- Tensor lifecycle: allocation, strided layouts, views, copies in every
  direction (host/device, cross-device, dtype conversion), resize.
- 215 native aten kernel registrations covering 170 distinct aten
  operators: tensor lifecycle and views, creation ops, elementwise math,
  activations (forward and backward), comparisons, bitwise ops,
  reductions, softmax/log-softmax, cumsum, sort/topk/argsort, indexing
  (gather, index_select/put/copy, where, triu/tril), scatter and
  masked_fill, embedding, matrix multiply (mm/bmm/addmm/linear),
  convolution, pooling, upsampling, batch/layer/RMS norm, losses, RNG
  (Philox counter-based), dropout, multinomial, scaled dot-product
  attention and multi-head attention.
- 7 custom fused ops outside aten: `torch.ops.paras.{rms_norm,
  swiglu, rotate_half}` with their backward kernels, and
  `torch.ops.torch_paras.swiglu`.
- See docs/OPERATOR_COVERAGE.md for the per-category breakdown and test
  status.
- Anything not implemented natively falls back to CPU through the boxed
  fallback, so models keep working while coverage grows.
- A binned memory pool per device on top of CUDA or HIP unified memory
  (GPU) or aligned host memory (CPU).

## Repository layout

```
csrc/compat/      compatibility layer, the only code that knows about the
                  ParaS compiler and CUDA/HIP 
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
scripts/build.sh cuda          # CPU + NVIDIA GPU build, or:
scripts/build.sh hip           # CPU + AMD GPU build

python tests/run_all.py --all-devices
```

See docs/INSTALL.md for the full toolchain setup, docs/TESTING.md




