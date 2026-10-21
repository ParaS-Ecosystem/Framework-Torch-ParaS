# Testing torch-paras

The suite lives in `tests/` and requires the built package and the dependencies
in `requirements-test.txt`, including pytest. Operator tests compare the
backend against native CPU PyTorch.

## Running

```bash
source scripts/env.sh

# Use the same Python environment as PyTorch and the extension build.
"$PTSYCL_PYTHON" -m pip install -r requirements-test.txt
"$PTSYCL_PYTHON" tests/run_all.py                    # paras:0 (first GPU if visible, otherwise CPU)
"$PTSYCL_PYTHON" tests/run_all.py --device paras:0   # first visible GPU in a GPU build
"$PTSYCL_PYTHON" tests/run_all.py --all-devices      # every enumerated device
```

In a GPU-enabled build with `N` visible GPUs, `paras:0` through
`paras:N-1` select GPUs and `paras:N` selects the host CPU. If no GPU is
visible (including a CPU-only build), the host CPU is `paras:0`.

The exit code is the number of failed tests, so it slots into CI as is.
`TORCH_PARAS_DEVICE` selects the device when you run a module directly.

## CLAP activation checks

Build with `PTSYCL_ENABLE_CLAP=ON` (default: OFF), set `CLAP_HOME` to your
CLAP installation, and make the vendor libraries available on
`LD_LIBRARY_PATH`:

```bash
PTSYCL_ENABLE_CLAP=ON scripts/build.sh cuda
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
CLAP_DNN_VERBOSE=1 "$PTSYCL_PYTHON" -m pytest -q tests/test_clap_dnn_multigpu.py
env -u CLAP_DNN_VERBOSE "$PTSYCL_PYTHON" -m pytest -q tests/test_clap_dnn_multigpu.py
"$PTSYCL_PYTHON" -m pytest -q tests/test_clap_result.py
```

The result-parser checks need only pytest. They cover vendor logs before
and after the marked JSON result, missing/malformed/repeated results,
incorrect counts, and subprocess failures. Setting `CLAP_DNN_VERBOSE=0`
still enables logging in runtimes that check for the variable's presence;
unset it for the quiet check.

Current limits remain:

- CPU Float32 and BFloat16 activations use oneDNN. CPU Float16 Sigmoid and
  SiLU (functional, out, and inplace) raise the expected unsupported
  datatype/layout error on the tested Xeon/oneDNN setup. Passing that
  error check does not mean CPU Float16 is supported.
- Scalar, empty, Float64, and strided inputs use the ParaS fallback.
  Contiguous CUDA tensors with rank above eight raise a cuDNN error.
- Backend overrides must match the tensor device. Conflicting
  `CLAP_BACKEND` and command-line flags can select the fallback even when
  the environment setting matches; option precedence is unchanged.

### Activation availability traces

Sigmoid and SiLU share an internal resolver in `clap_dnn_adapter.cpp`.
It runs once for each call reaching the existing activation implementation,
including functional, `out=`, and in-place forms. Set `PTSYCL_LOG=2` before
starting Python to see its decision on stderr:

```text
[ptsycl op] clap_dnn.resolve op=sigmoid route=clap_cuda reason=eligible
[ptsycl op] clap_dnn.resolve op=silu route=paras reason=dtype_unsupported
```

The routes are `paras`, `clap_cpu`, `clap_cuda`, and `clap_rocm`. The first
rejected check determines the reason, in this order:

| Reason | Meaning |
|---|---|
| `clap_disabled` | This build disables CLAP-DNN. |
| `input_unsupported` | Scalar, empty, non-contiguous input/output, or incompatible input/output shape, device, or dtype. |
| `dtype_unsupported` | The matching input/output dtype is not Float32, Float16, or BFloat16. |
| `device_unsupported` | No compiled CLAP-DNN backend matches the GPU queue. |
| `override_mismatch` | Environment or command-line settings reject the tensor backend, including invalid nonempty `CLAP_BACKEND`. |
| `eligible` | Existing checks permit an attempt on the selected CLAP backend. |

An eligible decision precedes backend creation and execution; it does not
guarantee vendor support. CPU Float16 and CUDA rank errors still reach the
caller after an `eligible` trace. They do not trigger a ParaS retry. Rejected
decisions increment the CLAP fallback counter once; completed vendor calls
increment the vendor counters. These counters do not describe PyTorch's
general CPU fallback. The resolver covers only these two activations and
does not change dispatcher registrations, other operators, or CLAP-BLAS.
Resolver traces are absent with `PTSYCL_LOG` unset, `0`, or `1`.

The resolver checks run in fresh subprocesses and compare marked stderr
lines independently of surrounding vendor logs. They also check numerical
results, output storage, vendor/fallback counters, input-check precedence,
backend settings, and error propagation. Failures include captured output.
Run them with a normal build (fast builds omit activation registrations):

```bash
"$PTSYCL_PYTHON" -m pytest -q tests/test_clap_dnn_multigpu.py -k resolver
```

The input and override resolver checks also work with CLAP OFF and require
`route=paras reason=clap_disabled`. The vendor-error check skips in that build.
`tests/run_all.py --all-devices` runs these checks on every enumerated device.
The GPU tests in this module target NVIDIA/cuDNN; they do not validate AMD.

## Build smoke check

`tests/smoke_build.py` checks package import, the CLAP enable status and
all stats bindings, and a CPU → ParaS → CPU tensor copy. It works with
`PTSYCL_FAST_BUILD=ON`, which omits activation operator registrations.
Run it in fresh processes for both CLAP ON and OFF builds:

```bash
"$PTSYCL_PYTHON" tests/smoke_build.py --package-root build/fast-on-package --expect-clap on
"$PTSYCL_PYTHON" tests/smoke_build.py --package-root build/fast-off-package --expect-clap off
```

Each command must point to the corresponding build's package. For separate
CMake build trees (`cmake -S . -B <build-dir>`), copy `python/torch_paras`
into separate package directories (such as `build/fast-on-package` and
`build/fast-off-package`) and replace each `_C.so` with its build tree's
`lib/_C.so`. Pass `--package-root <package-dir>` to check that exact package
without overwriting another build. Configure each tree with the toolchain
arguments used by `scripts/build.sh`, `-DPTSYCL_FAST_BUILD=ON`, and the
appropriate `-DPTSYCL_ENABLE_CLAP=ON` or `OFF`. Build serially (`-j1`),
waiting for each build to finish before starting another.

## What is covered

- `test_tensor_ops.py` — allocation, dtypes, fill/zero, copies in all
  directions (including strided and cross-device), view/reshape/
  as_strided, masked_select, item, resize, storage sharing, shape-operation
  edge cases, a 1M element round trip, and repeated large parallel dispatch.
- `test_pointwise_ops.py` — unary math, activations, binary ops with
  tensor and scalar operands, reductions, bitwise ops, cat, matmul.
- `test_random_ops.py` — uniform/bernoulli distributions, dropout in
  train/eval, dropout backward, RNG state advancement. These assert
  statistical properties, not exact values.
- `test_vision_ops.py` — adaptive pooling, linear, and nearest-neighbor
  upsampling, ensuring the large vision translation unit is linked.
- `test_indexing_ops.py` — gather, index_select, index_copy_, index_put_
  (with and without accumulate), advanced indexing, where, triu/tril.
- `test_fused_ops.py` — `torch.ops.paras` rms_norm, swiglu and
  rotate_half, forward and backward.
- `test_attention_norm_ops.py` — scaled dot-product attention (scale,
  causal, bool/additive masks, GQA), `aten::rms_norm`,
  `torch_paras::swiglu`, and copy edge cases.
- `test_matmul_sort_creation_ops.py` — mm/bmm/matmul shape variants,
  argsort, and creation ops (zeros/ones/full, `*_like`, arange).
- `test_param2_17b.py` — model-level check for the Param2 profiling
  workload.
- `test_clap_dnn_multigpu.py` — vendor call counts, interleaved GPU calls,
  CPU dtype limits, input fallbacks, backend overrides, and activation resolver
  traces (including quiet logging and vendor errors).

See `docs/OPERATOR_COVERAGE.md` for which native kernels still lack a
parity test.

Each test function takes the device string as its one argument, so adding
a test means adding a `test_*` function to the right module. The harness
in `common.py` seeds torch before each test and prints a per-test
PASS/FAIL line with a short traceback on failure.
