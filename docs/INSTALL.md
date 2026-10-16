# Building torch-paras

This guide goes from a bare Linux machine to a working `paras` device in
PyTorch. I keep the exact versions I built and tested with; newer ones may
work but I have not tried them.

## 1. Hardware and OS

- Linux x86_64 (built on RHEL 8, kernel 4.18)
- Intel CPU (any reasonably modern Xeon/Core works)
- Optional: NVIDIA GPU for the CUDA flavor. I test on Tesla V100 (sm_70).
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
| Python | 3.10 | matches the PyTorch build below |
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

- The build runs in parallel (`-j$(nproc)`, override with
  `PTSYCL_BUILD_JOBS`): current `parascc` gives every intermediate file a
  unique name. A full HIP build takes about 2.5 minutes on a 256-core node.
- Set `CLAP_HOME` to a CLAP install to build with CLAP BLAS. On GPU builds
  this is what runs `mm`/`addmm`/`bmm` (and therefore `linear` and its
  backward) on the device through the vendor BLAS behind CLAP; without it
  those operators stage their operands through host memory.
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

From the repository root, install the test dependencies with the same
Python environment used for PyTorch and the build (`PTSYCL_PYTHON`):

```bash
"$PTSYCL_PYTHON" -m pip install -r requirements-test.txt
"$PTSYCL_PYTHON" tests/run_all.py                 # paras:0 (first GPU if visible; otherwise CPU)
"$PTSYCL_PYTHON" tests/run_all.py --all-devices   # every device, GPUs included
```

CLAP tests require a build with `PTSYCL_ENABLE_CLAP=ON` and a CLAP
installation, for example:

```bash
export CLAP_HOME=/opt/clap/install
PTSYCL_ENABLE_CLAP=ON scripts/build.sh cuda
```

Make the vendor libraries (oneDNN for CPU, cuDNN for NVIDIA, MIOpen for AMD)
available on `LD_LIBRARY_PATH`. MIOpen also needs a writable kernel
database: if its default location is read-only (for example inside a Python
environment), set `MIOPEN_USER_DB_PATH` and `MIOPEN_CUSTOM_CACHE_DIR` to a
writable directory. MIOpen has no bfloat16 activation kernels, so bfloat16
sigmoid/SiLU on AMD GPUs always run the ParaS kernel. See [Testing](TESTING.md) for focused CLAP and fast-build
checks and the current activation limits.

In a GPU-enabled build with `N` visible GPUs, `paras:0` through
`paras:N-1` are GPUs and `paras:N` is the host CPU. With no visible GPUs,
the host CPU is `paras:0`.

You should see `RESULTS: N passed, 0 failed`. If `import torch_paras`
fails with a GLIBCXX error, your LD_LIBRARY_PATH is missing the GCC 13.4
lib64 directory; re-source `scripts/env.sh`.

A common pitfall: if you ever installed an older copy of the package with
pip, it will shadow the local one. `pip uninstall torch_paras` and run
with the repo's `python/` directory first on PYTHONPATH (the test and
benchmark scripts already do this themselves).

## 7. Run the benchmarks

```bash
python benchmarks/bench_ops.py --device paras:1 --json ops.json
python benchmarks/bench_models.py --json models.json
python benchmarks/make_report.py --ops ops.json --models models.json
```

The last command regenerates the figures and tables in docs/benchmarks/.
See docs/BENCHMARKS.md for the numbers from our reference machine.
