# Troubleshooting

Common build, import, runtime, and device issues, and what to check first.
If you hit something not covered here, please open an issue using the
`build_failure` or `bug_report` template (see `.github/ISSUE_TEMPLATE/`)
rather than a duplicate — and consider adding what you learned back to
this file in the same PR.

## Build Issues

### Build fails partway through with a compiler/parascc error

- Confirm you're using the exact toolchain versions in `docs/INSTALL.md`
  (LLVM/clang 21.1, GCC 13.4, CUDA 12.2 for CUDA builds). `parascc` is
  clang-based and sensitive to toolchain mismatches.
- Confirm `scripts/env.sh` has been sourced *after* setting
  `PARAS_HOME`, `PTSYCL_GCC_TOOLCHAIN`, `PTSYCL_LLVM_DIR`,
  `PTSYCL_CUDA_HOME`, and `PTSYCL_PYTHON` — a stale environment from a
  previous shell session is a common cause of confusing errors.
- The build uses all cores (`-j$(nproc)`). Override with `JOBS`, e.g.
  `JOBS=1 scripts/build.sh cpu`, if you need a serial build.

### Objects built from the wrong source file

The `cpu` and `hip` flavors go through `parascc`'s source-transformation
stage, which writes an intermediate `.cpp` per translation unit. Older
`parascc` builds name that file with a second-resolution timestamp, so
two compiles starting in the same second write to the same path and the
objects that come out are built from whichever source won the race. The
build still succeeds and produces a working `_C.so` that behaves like a
different translation unit — no error is reported. Symptoms are tests
failing in ops you did not touch.

Rebuild with `JOBS=1`. If that fixes it, your `parascc` has this race;
it needs a build whose intermediates carry the pid. The `cuda` flavor is
unaffected because it does not use that stage.

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
  If not, that's the gap — file a bug report with a minimal repro (see
  `.github/ISSUE_TEMPLATE/bug_report.md`) including exact shapes/dtypes.

### Performance is much worse than expected

- Check the fallback ratio for your model/workload — a high proportion
  of fallback ops (each incurring a host round-trip) is a common and
  expected cause during early coverage, not necessarily a bug. See
  `docs/MODEL_VALIDATION.md` and `docs/OPERATOR_COVERAGE.md`.
- If a *native* kernel regressed, see `docs/BENCHMARKS.md` and consider
  filing a performance-regression issue.

## Device Issues

### Only device 0 (`paras:0`) is available, no GPU devices show up

- Confirm you built with `scripts/build.sh cuda`, not `scripts/build.sh
  cpu` — the CPU-only build only ever exposes device 0.
- Confirm the NVIDIA driver and CUDA toolkit are visible in the
  environment `scripts/env.sh` configured (`PTSYCL_CUDA_HOME` etc.).

### Wrong number of GPU devices enumerated, or wrong device mapping

- Devices `paras:1..N` map to whatever NVIDIA GPUs are visible in the
  process's environment, the same way CUDA enumerates them — check
  `CUDA_VISIBLE_DEVICES` if the mapping looks different from what you
  expect.

## Still Stuck?

Open an issue with the `build_failure`, `bug_report`, or `model_failure`
template (whichever fits) and include your environment details — see
`docs/SUPPORT_MATRIX.md` for what's currently tested and what's best-effort.
