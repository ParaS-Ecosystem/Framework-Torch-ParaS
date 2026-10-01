# Direct SYCL queues with GitHub ParaS main

Verified on 2026-09-30 against the compiler team's supplied response.

Compiler repository: https://github.com/ParaS-Ecosystem/ParaS-Compiler

Compiler commit: `279b10415943b285015bffbfb96d814c624ea412`.
Canonical install: `/storage/parikshit/torch_paras/ParaS-Compiler/build/install`.
Compiler tracked sources are unchanged. Older checkouts/builds are recoverable
under `/storage/parikshit/torch_paras/archive/2026-09-30-before-main`.

## Framework changes

`Queue::Impl` in `paras_compat.cpp` owns a typed `sycl::queue*`.
Construction, deletion, allocation, free, memcpy, memset and wait all use the
typed queue. No queue `void*`, `backend_handle()`, cast back to a SYCL queue,
explicit backend class, or direct CUDA runtime call is used by this adaptation.
Queue construction remains in `.cpp`; its typed accessor declaration and the
kernel template remain visible through headers.

ParaS main serializes only the main source file's rewrite buffer
(`src/driver/driver.cpp`, `EndSourceFileAction`). Ordinary project headers are
not emitted with their rewrites. A plain direct-queue build therefore fails:

```text
error: return type of out-of-line definition of 'ptsycl::compat::Queue::sycl_queue' differs from that in the declaration
cuda_threadpool& Queue::sycl_queue() const {
note: previous declaration is here
sycl::queue& sycl_queue() const;
```

The framework build now expands only its own `csrc` headers into generated
translation units before invoking ParaS. System, Torch and SYCL includes
remain normal includes. This lets the existing rewriter see queue declarations,
definitions and kernel templates together; it requires no compiler modification.

Generated inputs include SYCL before Torch. CUDA 12.2's
`cuda/std/detail/__config` defines `_Float16` as `__half`; if Torch brings that
header in first, ParaS's half comparisons fail with:

```text
half.hpp:147:18: error: constexpr function never produces a constant expression [-Winvalid-constexpr]
half.hpp:148:23: note: non-constexpr function 'operator==' cannot be used in a constant expression
```

Including SYCL first resolves this conflict. No diagnostic suppression was used
in the successful build.

## Reproduce the fast build

```bash
source /storage/parikshit/torch_paras/paras-main-env.sh
cd /storage/parikshit/torch_paras/Framework-Torch-ParaS
PTSYCL_FAST_BUILD=ON bash scripts/build.sh cuda
"$PTSYCL_PYTHON" tests/test_queue_smoke.py --expected-gpus 4
"$PTSYCL_PYTHON" tests/run_all.py --all-devices
```

The current CMake cache has both `PTSYCL_FAST_BUILD=ON` and
`PTSYCL_FLATTEN_HEADERS=ON`. Fast mode builds compatibility, core, Python bindings
and `tensor_ops.cpp`, omitting all other native operator source files.
To request a full operator build later, set `PTSYCL_FAST_BUILD=OFF`.

## Verified results

- Latest-main compiler built and installed with Clang 21.1.7 and GCC 14.3.0.
  GCC 14's library directory must precede the older GCC directory supplied by
  the LLVM CMake package at compiler link time.
- Reduced CUDA `sm_70` framework build and link passed.
- Built and installed `_C.so` SHA256:
  `7a549c434546b008f8fce13ffbe0b4deaf03cf54ea5011657b21372bab755721`.
  No unresolved `sycl::queue`, queue USM allocation, or queue free symbols.
- Standalone standard-SYCL allocation, memset, nonzero range kernel, memcpy and
  free passed on CPU and individually on all four GPUs.
- Fresh-process framework queue regressions passed: zeros, nonzero fill,
  float32/int64 copy round-trips, interleaved GPU operations, worker-thread queue
  reuse, synchronization and cache release.
- Device numbering is GPU-first: `paras:0` through `paras:3` are V100 GPUs;
  `paras:4` is CPU. Generally, N GPUs occupy 0..N-1 and CPU occupies N.
- Direct multi-queue SYCL probe passed GPU memory operations after other queue
  construction/kernel execution and managed cross-device copies.
- `tests/run_all.py --all-devices`: **910 passed, 45 failed**, exit 45, 14.8 seconds.
  Forty failures are missing custom schemas from omitted `fused_ops.cpp` and
  `rms_swiglu_ops.cpp`. Five are `test_index_put_accumulate_unsupported_dtypes`:
  the omitted native indexing implementation normally rejects half accumulation,
  whereas the generic CPU fallback accepts it. Passing fallback tests do not
  prove native GPU implementation or native GPU atomic correctness.

Logs are in `build/cuda/plain-build.log`, `flattened-build.log`,
`sycl-first-build.log`, `queue-smoke.log`, and `run-all.log`; multi-queue results
are in `build/queue_probes/multi-queue-runtime.log`.

## Remaining CPU architecture limitation

An explicit `cpu_selector_v` queue in a CUDA-target translation unit is rejected:

```text
Error: Found cpu queue but compiling for gpu (gave -parasdevice flag)
```

Passing an enumerated CPU device avoids that selector check but still lowers the
queue to the CUDA backend. Its memset/wait fail at runtime:

```text
CUDA operation requested without a valid CUDA stream
```

Therefore the combined GPU/CPU framework build retains its existing CPU path.
The successful CPU-only standalone probe proves ParaS CPU execution separately,
not mixed runtime dispatch inside a CUDA-target translation unit. Replacing the
combined build's CPU path would require separate CPU-target integration or
general mixed-device support from ParaS.

No compiler runtime patches are necessary for the tested GPU queue operations.
The header rewrite limitation and mixed CPU/GPU lowering remain generally
applicable compiler constraints, with reproducible examples above.
