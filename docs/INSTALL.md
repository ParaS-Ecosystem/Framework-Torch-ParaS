# Building torch-paras

This guide goes from a bare Linux machine to a working `paras` device in
PyTorch. The exact versions below are the ones it was built and tested
with; newer ones may work but are untested.

## 1. Hardware and OS

- Linux x86_64 (built on RHEL 8, kernel 4.18)
- Intel CPU (any reasonably modern Xeon/Core works)
- Optional: NVIDIA GPU for the CUDA flavor. Tested on Tesla V100 (sm_70).
  Set `PTSYCL_CUDA_ARCH` if your GPU has a different compute capability,
  e.g. `cuda:sm_80` for A100.
- Optional: AMD GPU for the HIP flavor. `scripts/build.sh hip` auto-detects
  your GPU's architecture via `rocm_agent_enumerator` (confirm your own
  card's ISA with `rocminfo | grep gfx` if you want to double check, e.g.
  `gfx942` for MI300-class, `gfx90a` for MI200-class). Set
  `PTSYCL_HIP_ARCH=hip:gfxXXX` explicitly to override auto-detection, or
  if you have multiple GPU architectures visible and need to pick one.
  A single build targets CUDA or HIP, not both.

## 2. Toolchain

The ParaS compiler driver (`parascc`) is clang-based and needs:

| component | version used | why |
|---|---|---|
| LLVM/clang | 21.1 | parascc execs clang++ for its sub-steps |
| GCC | 13.4 | C++ standard headers and libstdc++ (GLIBCXX_3.4.30) |
| CUDA toolkit | 12.2 | CUDA flavor only |
| ROCm | 6.x | HIP flavor only |
| CMake | >= 3.18 | build system |
| Python | 3.10 / 3.11 / 3.12 | 3.10 for V100 and CPU, 3.11 for H200, 3.12 for MI300X |
| PyTorch | 2.5.1 | the backend links against libtorch from this install |

PyTorch comes from a normal conda env:

```bash
conda create -n pytorch_build python=3.10
conda activate pytorch_build
pip install torch==2.5.1 numpy matplotlib
```

## 3. The ParaS compiler

torch-paras does not modify the ParaS compiler in any way; it consumes a
finished installation. Get the compiler from wherever your team hosts it
(git clone and build, or a shared install on your cluster), then point
`PARAS_HOME` at the install prefix. The layout it expects:

```
$PARAS_HOME/bin/parascc      the compiler driver
$PARAS_HOME/include/         kem/ (CPU threadpool) and runtime headers
$PARAS_HOME/lib/             runtime libraries
```

If you build the compiler from source it is the usual CMake dance; follow
the compiler's own README. The backend only cares that `parascc` runs and
that the include/lib trees above exist.

## 4. Configure paths

All paths live in one file, `scripts/env.sh`. Either edit it or export
overrides before sourcing it:

```bash
export PARAS_HOME=/opt/paras/install
export PTSYCL_GCC_TOOLCHAIN=/opt/gcc-13.4.0
export PTSYCL_LLVM_DIR=/opt/llvm-21.1
export PTSYCL_CUDA_HOME=/usr/local/cuda-12.2
export PTSYCL_ROCM_HOME=/opt/rocm-6.2.0
export PTSYCL_PYTHON=$(which python3)        # the conda env's python
source scripts/env.sh
```

`env.sh` derives the PyTorch location from the python you point it at and
sets PATH / LD_LIBRARY_PATH so parascc and the runtime resolve.

## 5. Build

```bash
scripts/build.sh cpu             # CPU engine only
scripts/build.sh cuda            # CPU engine + NVIDIA GPUs
scripts/build.sh cuda Debug      # debug build
scripts/build.sh hip             # CPU engine + AMD GPUs
```

Notes:

- The build is intentionally serial (`-j1`). parascc names its temporary
  files with second resolution, so parallel compiles would race on them.
  A full CUDA or HIP build takes around 15 minutes on the reference machine.
- The built extension is copied to `python/torch_paras/_C.so`. Whichever
  flavor you built last is the one `import torch_paras` loads.
- Build trees live under `build/cpu`, `build/cuda`, and `build/hip`;
  incremental builds only recompile what changed.
- `PTSYCL_DEVICE` values are `cpu`, `cuda:sm_XX`, or `hip:gfxXXX` —
  `scripts/build.sh` picks the right one for `cuda`/`hip` from
  `PTSYCL_CUDA_ARCH` (default: `cuda:sm_70`) or `PTSYCL_HIP_ARCH`
  (auto-detected from the visible GPU via `rocm_agent_enumerator` if
  unset).

## 6. Check it works

```bash
python tests/run_all.py                 # device paras:0 (CPU engine)
python tests/run_all.py --all-devices   # every device, GPUs included
```

You should see `RESULTS: N passed, 0 failed`. If `import torch_paras`
fails with a GLIBCXX error, your LD_LIBRARY_PATH is missing the GCC 13.4
lib64 directory; re-source `scripts/env.sh`.

A common pitfall: if you ever installed an older copy of the package with
pip, it will shadow the local one. `pip uninstall torch_paras` and run
with the repo's `python/` directory first on PYTHONPATH (the test and
benchmark scripts already do this themselves).

## 7. Run the benchmarks

See [benchmark/super-resolution/GUIDE.md](../benchmark/super-resolution/GUIDE.md)
for the super-resolution suite: datasets, the three measured arms, and how
to read the output.
