# Troubleshooting

Common build, import, runtime, and device issues, and what to check first.
If you hit something not covered here, please open an issue rather than a
duplicate — and consider adding what you learned back to this file in the
same PR.

## Build Issues

### Build fails partway through with a compiler/parascc error

- Confirm you're using the exact toolchain versions in `docs/INSTALL.md`
  (LLVM/clang 21.1, GCC 13.4, CUDA 12.2 for CUDA builds). `parascc` is
  clang-based and sensitive to toolchain mismatches.
- Confirm `scripts/env.sh` has been sourced *after* setting
  `PARAS_HOME`, `PTSYCL_GCC_TOOLCHAIN`, `PTSYCL_LLVM_DIR`,
  `PTSYCL_CUDA_HOME`, and `PTSYCL_PYTHON` — a stale environment from a
  previous shell session is a common cause of confusing errors.
- The build is intentionally single-threaded (`-j1`) because `parascc`
  names temporary files with second-resolution timestamps; a full CUDA
  build taking ~15 minutes is expected, not a sign something is stuck.

### Wrong GPU architecture / CUDA build targets the wrong compute capability

- Set `PTSYCL_CUDA_ARCH` explicitly for your GPU (e.g. `cuda:sm_80` for
  A100) rather than relying on a default — see `docs/INSTALL.md` section 1.

### CMake can't find the ParaS compiler / runtime

- Confirm `$PARAS_HOME` points at a real install with `bin/parascc`,
  `include/`, and `lib/` present. Torch-ParaS doesn't build or modify the
  ParaS compiler itself — it only consumes an existing installation.

## Import Issues

### `import torch_paras` fails with a GLIBCXX error

This means your `LD_LIBRARY_PATH` is missing the GCC 13.4 `lib64`
directory. Re-source `scripts/env.sh` in the shell you're running Python
from (not just the shell you built in).

### `import torch_paras` loads, but behaves like an old/wrong build

Check whether an older copy was ever installed with `pip` — a pip-installed
copy will shadow the local build. Run:

```bash
pip uninstall torch_paras
```

and make sure the repo's `python/` directory is first on `PYTHONPATH`
(the test and benchmark scripts already do this for you, but a manual
`python -c "import torch_paras"` from an arbitrary directory may not).

### Wrong build flavor loads (CPU build when you expected CUDA, or vice versa)

The built extension is copied to `python/torch_paras/_C.so`, and whichever
flavor (`scripts/build.sh cpu` vs `scripts/build.sh cuda`) you built last
is the one that gets imported. Rebuild with the flavor you actually want,
or check `build/cpu` vs `build/cuda` to see which was built most recently.

## Runtime Issues

### Tests fail with `RESULTS: N passed, M failed` where M > 0

- Confirm you sourced `scripts/env.sh` in the current shell before running
  `tests/run_all.py`.
- Run with `--device paras:0` explicitly first to isolate whether the
  failure is CPU-engine-specific or GPU-specific, then try
  `--all-devices`.
- Check `docs/TESTING.md` for what each test module covers, and
  `docs/OPERATOR_COVERAGE.md` for whether the failing area has known
  gaps already.

### A model produces different output on `paras` vs `cpu`/`cuda`

- Check `docs/OPERATOR_COVERAGE.md` for whether the ops involved are
  native or fallback — a fallback op should match CPU behavior exactly
  (it *is* CPU behavior); a mismatch there is more likely a copy/dtype
  bug at the fallback boundary than a kernel bug.
- For native kernels, check whether the operator has a parity test yet.
  If not, that's the gap — file a bug report with a minimal repro,
  including exact shapes and dtypes.

### Performance is much worse than expected

- Check the fallback ratio for your model/workload — a high proportion
  of fallback ops (each incurring a host round-trip) is a common and
  expected cause during early coverage, not necessarily a bug. See
  `docs/OPERATOR_COVERAGE.md`.
- If a *native* kernel regressed, see
  `benchmark/super-resolution/GUIDE.md` and consider filing a
  performance-regression issue.

## Device Issues

### Only device 0 (`paras:0`) is available, no GPU devices show up

- Confirm you built with `scripts/build.sh cuda` (NVIDIA) or
  `scripts/build.sh hip` (AMD), not `scripts/build.sh cpu` — the CPU-only
  build only ever exposes device 0.
- Confirm the vendor driver and toolkit are visible in the environment
  `scripts/env.sh` configured (`PTSYCL_CUDA_HOME`, or `PTSYCL_ROCM_HOME`
  for HIP builds).

### Wrong number of GPU devices enumerated, or wrong device mapping

- Devices `paras:1..N` map to whatever GPUs are visible in the process's
  environment, the same way the vendor runtime enumerates them — check
  `CUDA_VISIBLE_DEVICES` on NVIDIA, or `ROCR_VISIBLE_DEVICES` /
  `HIP_VISIBLE_DEVICES` on AMD, if the mapping looks different from what
  you expect.

## Still Stuck?

Open an issue and include your environment details — see `docs/INSTALL.md`
section 1 for the hardware and toolchain that are currently tested.
