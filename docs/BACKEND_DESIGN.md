# Backend Design

How torch-paras plugs into PyTorch and the ParaS runtime. This is the
"why does it work this way" doc — see `docs/INSTALL.md` for build setup
and `README.md` for the repository layout.

## Overview

Torch-ParaS is an out-of-tree PyTorch backend. It registers a `paras`
device with PyTorch using the `PrivateUse1` dispatch key, so tensors on
`paras` (and `paras:1..N`) go through the same dispatcher mechanism as
`cpu`/`cuda` tensors, without patching PyTorch itself.

```
PyTorch op call (e.g. torch.add(x, y))
        |
        v
PyTorch dispatcher --- looks at tensor device ---
        |
        +-- paras kernel registered? --> csrc/ops kernel runs on ParaS runtime
        |
        +-- not registered? --> boxed fallback --> runs on CPU, result copied back
```

## Device Registration

- Device 0 (`paras` / `paras:0`) always exists and executes on the host
  CPU. Large flat kernels use an OpenMP team; small kernels stay serial.
- In CUDA builds, devices `paras:1..N` map to the visible NVIDIA GPUs; in
  HIP builds, to the visible AMD GPUs (enumerated the same way the vendor
  runtime enumerates them). A single build targets one vendor.
- Device registration and enumeration lives in `csrc/core`; the
  PyTorch-facing dispatcher glue (the `PrivateUse1` hooks) lives in
  `csrc/compat`, kept separate so PyTorch-version-specific code doesn't
  leak into the core runtime logic.

## Compatibility Layer (`csrc/compat`)

This is the only part of the codebase that's aware of PyTorch version
differences and the ParaS compiler (`parascc`) specifically. It's
responsible for:

- Registering the `paras` `PrivateUse1` backend with the PyTorch
  dispatcher.
- Bridging PyTorch's tensor/storage APIs to the ParaS runtime's memory
  and execution primitives.
- Isolating any PyTorch-version-specific behavior behind named
  compatibility checks, so upgrading
  the supported PyTorch version doesn't require touching kernel code.

## Memory Management

- Each device gets its own **binned memory pool**, sized in classes to
  reduce allocator overhead for the small/medium tensor sizes common in
  model workloads.
- On GPU builds, the pool sits on top of **CUDA or HIP unified memory**
  (`cudaMallocManaged` / `hipMallocManaged`), so host
  and device can both address the same allocation without an explicit
  copy step (the runtime still manages transfers/coherency as needed for
  correctness and performance).
- On CPU-only builds (device 0 always, or CPU-only configurations), the
  pool sits on top of **aligned host memory** allocations sized for the
  CPU threadpool engine's access patterns.
- Pool sizing/reuse behavior is currently fixed at build time; making it
  tunable at runtime is not yet implemented.

## Streams and Execution

- CPU flat-kernel launches use the host OpenMP runtime to reuse worker
  threads instead of creating threads per operation. `PTSYCL_CPU_THREADS`
  overrides the default PyTorch intra-op thread count and
  `PTSYCL_CPU_GRAIN_SIZE` controls the parallelization threshold.
- GPU launches use ParaS compiler-generated device code.
- Device-guard and stream/event abstractions in `csrc/core` are kept
  symmetric between CPU and CUDA code paths, so code above this layer
  doesn't need to branch on device type to launch work correctly.

## Operator Fallback

- Operators without a native `paras` kernel automatically fall back to
  CPU via PyTorch's **boxed fallback** mechanism: inputs are copied to
  CPU, the CPU implementation runs, and the result is copied back to the
  original `paras` device.
- This means models run correctly even before every operator has a native
  kernel — coverage grows incrementally without breaking existing model
  code. See `docs/OPERATOR_COVERAGE.md` for what's currently native vs.
  fallback.
- The tradeoff is performance: any fallback op incurs a host round-trip.

## Runtime Integration (ParaS Compiler)

- Torch-ParaS does **not** modify the ParaS compiler (`parascc`) — it
  consumes a finished installation (`$PARAS_HOME`), the same way any
  application using the compiler would.
- `parascc` is clang-based; kernel code in `csrc/ops` is written to be
  compiler-agnostic C++ where possible, with compiler-specific concerns
  isolated in `csrc/compat`.
- See `docs/INSTALL.md` for the expected `$PARAS_HOME` layout
  (`bin/parascc`, `include/`, `lib/`) and toolchain versions.

## Open Design Questions

Things not yet settled — flagged here so design discussions have a home:

- Whether/how to expose tunable memory pool sizing to end users.
- How autograd (backward-pass) registration will be organized as more
  operators grow backward support.
- Long-term plan for non-x86 CPU targets, and how much of `csrc/core` can
  stay backend-agnostic.
