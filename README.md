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
- Over 160 aten operators (198 registrations counting overloads):
  elementwise math, activations, comparisons,
  reductions, matrix multiply (mm/bmm/addmm), convolution, pooling,
  upsampling, batch/layer norm, losses, RNG (Philox counter-based),
  dropout, multi-head attention.
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
benchmark/        reproducible benchmark suites
tests/            test suite, run with tests/run_all.py
docs/             install, testing, backend design, operator coverage,
                  troubleshooting
```

## Quick start

```bash
source scripts/env.sh          # adjust paths for your machine first
scripts/build.sh cpu           # CPU-only build, or:
scripts/build.sh cuda          # CPU + NVIDIA GPU build, or:
scripts/build.sh hip           # CPU + AMD GPU build

python tests/run_all.py --all-devices
```

## Documentation

| document | what it covers |
|---|---|
| [docs/INSTALL.md](docs/INSTALL.md) | full toolchain setup for a source build |
| [docs/TESTING.md](docs/TESTING.md) | running the operator suite |
| [docs/BACKEND_DESIGN.md](docs/BACKEND_DESIGN.md) | how the PrivateUse1 backend is put together |
| [docs/OPERATOR_COVERAGE.md](docs/OPERATOR_COVERAGE.md) | which aten operators are implemented natively |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | common build and runtime failures |
| [benchmark/super-resolution/GUIDE.md](benchmark/super-resolution/GUIDE.md) | running the super-resolution benchmark |




