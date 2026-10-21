# Torch-ParaS

A PyTorch out-of-tree backend for the ParaS compiler. It registers a
`paras` device with PyTorch, so existing models run on the ParaS runtime
without code changes:

```python
import torch
import torch_paras

x = torch.randn(64, 64, device="paras")     # default paras device (paras:0)
y = torch.randn(64, 64, device="paras:0")   # devices 0..N-1 = NVIDIA or AMD GPUs
z = (x @ x).relu().cpu()
```

In GPU-enabled builds (CUDA or HIP), devices 0..N-1 map to visible GPUs,
and device N maps to the host CPU engine. In CPU-only builds, device 0 maps
to the host CPU. Tested on Intel CPUs, NVIDIA GPUs, and AMD GPUs.

## What is implemented

- Tensor lifecycle: allocation, strided layouts, views, copies in every
  direction (host/device, cross-device, dtype conversion), resize.
- 237 native aten kernel registrations covering 184 distinct aten
  operators: tensor lifecycle and views, creation ops, elementwise math,
  activations (forward and backward), comparisons, bitwise and logical
  ops, reductions, softmax/log-softmax, cumsum, sort/topk/argsort,
  nonzero, indexing (gather, index_select/put/copy/add, where,
  triu/tril), scatter/scatter_add and masked_fill, embedding, matrix
  multiply (mm/bmm/addmm), convolution, pooling, upsampling, batch/layer
  norm, losses, vector norms, RNG (Philox counter-based), dropout,
  multinomial. Composite operators such as `linear`, `reshape`,
  `contiguous` and `rms_norm` decompose into these kernels with autograd
  intact. Scaled dot-product attention runs fused inference kernels (on
  AMD GPUs a matrix-core flash-attention kernel for prefill; decode
  kernels on every GPU) and PyTorch's math decomposition for training.
- 7 custom fused ops outside aten: `torch.ops.paras.{rms_norm,
  swiglu, rotate_half}` with their backward kernels, and
  `torch.ops.torch_paras.swiglu`.
- See docs/OPERATOR_COVERAGE.md for the per-category breakdown and test
  status.
- On GPU builds with CLAP (`CLAP_HOME`), `mm`/`addmm`/`bmm` -- and so
  `linear` and its backward -- run on the device through the vendor BLAS
  behind CLAP, enqueued on the paras queue's own stream through a ParaS
  custom operation (no host wait).
- Anything not implemented natively falls back to CPU through the boxed
  fallback, so models keep working while coverage grows. Every fallback
  and every host transfer is counted (`torch_paras.runtime_stats()`), and
  strict mode (`PTSYCL_STRICT=1` / `torch_paras.set_strict(True)`) turns a
  fallback into an error.
- A binned memory pool per device on top of device memory
  (`sycl::malloc_device`, GPU) or aligned host memory (CPU).
- A graph optimizer for `torch.compile` (`backend="paras"`), for
  inference and training -- see docs/GRAPH_OPTIMIZER.md.
- Graph capture and replay (`torch_paras.graphs`, and
  `torch_paras.compile(..., graphs=True)` for compiled inference) on ParaS
  record-and-replay command graphs -- see docs/GRAPHS.md.

## Graph optimizer

```python
import torch, torch_paras

model = model.to("paras:0")
fast = torch.compile(model, backend="paras")      # or torch_paras.compile(model)
loss = fast(x).sum(); loss.backward()             # training works too

for report in torch_paras.optimizer.reports():
    print(report.summary())                       # fusions, where every op runs
```

TorchDynamo graphs are rewritten by fusion passes (RMSNorm, SwiGLU,
rotate-half -> `paras::` fused kernels that reproduce the eager result bit
for bit), traced by AOTAutograd so that every forward and backward operator
is checked for a ParaS kernel (`torch_paras.compile(model, strict=True)`
rejects any that would fall back to the CPU), and executed like eager code.

## Repository layout

```
csrc/compat/      compatibility layer, the only code that knows about the
                  ParaS compiler and CUDA/HIP 
csrc/core/        device context, allocator, kernel launch helpers, registration
csrc/ops/         the aten kernels, plain C++ lambdas, compiler agnostic
python/torch_paras/  the python package (optimizer/: the graph optimizer)
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

See docs/INSTALL.md for the full toolchain setup, docs/TESTING.md for running the tests,
and docs/GRAPH_OPTIMIZER.md for torch.compile.
