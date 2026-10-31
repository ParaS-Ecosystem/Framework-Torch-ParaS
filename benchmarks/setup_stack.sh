#!/bin/bash
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Centre for Development of Advanced Computing (C-DAC)
#
# This file is part of Torch_ParaS, a component of the ParaS Ecosystem
#
# This library is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License (LGPL)
# version 3 as published by the Free Software Foundation.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library. If not, see <https://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------
# Builds everything the reproduction needs, from public sources, into one work directory:
#   setup_stack.sh [hip|cuda]            REPRO_WORK (default ./torch-paras-repro) holds all of it
# 1. ParaS-Compiler, CLAP-Library and Framework-Torch-ParaS at the pinned commits; their source trees are checked
# 2. Python 3.12 environments with the pinned packages (PyTorch 2.10.0 for ROCm 7.0 or for CUDA 12.8)
# 3. LLVM/Clang 20.1.0 built from the official source release (unless LLVM_DIR points at a 20.1 install)
# 4. the three builds (ParaS, CLAP, Torch-ParaS with CLAP)
# 5. Param2-17B at the pinned revision (every file checked against the Hub), WikiText-2, InferenceX and its client
# 6. $REPRO_WORK/env.sh, which reproduce_param2.sh and reproduce_inferencex.sh source
# Re-running skips the steps that are already done. No GPU is needed for this step.
# REPRO_HF_HOME: a Hugging Face cache that already holds the model (default $REPRO_WORK/hf); REPRO_JOBS: build jobs;
# REPRO_PYTHON: the Python 3.12 interpreter (default python3.12).
set -euo pipefail
if [ -n "${1:-}" ]; then FLAVOR=$1
elif [ -e /dev/kfd ] || [ -x /opt/rocm/bin/rocminfo ]; then FLAVOR=hip
else FLAVOR=cuda; fi
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORK=${REPRO_WORK:-$PWD/torch-paras-repro}
mkdir -p "$WORK"
WORK=$(cd "$WORK" && pwd)
J=${REPRO_JOBS:-$(nproc)}
SRC=$WORK/src; BUILD=$WORK/build; INST=$WORK/install; LOG=$WORK/logs
mkdir -p "$SRC" "$BUILD" "$INST" "$LOG"
step() { echo "=== $(date '+%F %T') $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

PARAS_REPO=https://github.com/ParaS-Ecosystem/ParaS-Compiler.git
PARAS_REF=refs/pull/44/head PARAS_COMMIT=bd84ff8a617049dc265dcef6219079491090d5df PARAS_TREE=f3f0e7b
CLAP_REPO=https://github.com/ParaS-Ecosystem/CLAP-Library.git
CLAP_REF=refs/pull/24/head CLAP_COMMIT=13793a4b9208cdae58fe7b5f6b65e6f203c0fa95 CLAP_TREE=228ef9c
TP_REPO=https://github.com/ParaS-Ecosystem/Framework-Torch-ParaS.git
TP_REF=refs/pull/35/head TP_COMMIT=f99a8c5f800e80ddc0d945277f1777c748e06c17 TP_TREE=4d03e22
IX_REPO=https://github.com/SemiAnalysisAI/InferenceX.git IX_COMMIT=4ee45310344ae1041635348671f173228079fa55
LLVM_VER=20.1.0 LLVM_SRC_SHA=4579051e3c255fb4bb795d54324f5a7f3ef79bd9181e44293d7ee9a7f62aad9a
case $(uname -m) in
  x86_64)  LLVM_TARGETS="X86;AMDGPU;NVPTX" ;;
  aarch64) LLVM_TARGETS="AArch64;AMDGPU;NVPTX" ;;
  *) die "unsupported CPU $(uname -m)" ;;
esac
PY312=${REPRO_PYTHON:-python3.12}

step "checks ($FLAVOR, work directory $WORK)"
for t in git curl tar make "$PY312"; do command -v "$t" >/dev/null || die "$t not found"; done
"$PY312" -c "import sys; assert sys.version_info[:2] == (3, 12)" || die "$PY312 is not Python 3.12 (set REPRO_PYTHON)"
GCCTC=${PTSYCL_GCC_TOOLCHAIN:-/usr}
GCCV=$("$GCCTC/bin/gcc" -dumpversion | cut -d. -f1)
[ "$GCCV" -ge 12 ] || die "GCC >= 12 is needed for its libstdc++ (found $GCCV in $GCCTC): set PTSYCL_GCC_TOOLCHAIN"
ROCM="" CUDA=""
if [ "$FLAVOR" = hip ]; then
  ROCM=${PTSYCL_ROCM_HOME:-/opt/rocm}
  [ -d "$ROCM" ] || die "ROCm not found in $ROCM (set PTSYCL_ROCM_HOME)"
  if [ -z "${PTSYCL_HIP_ARCH:-}" ]; then
    G=$("$ROCM/bin/rocm_agent_enumerator" 2>/dev/null | grep -v '^gfx000$' | sort -u | head -1 || true)
    PTSYCL_HIP_ARCH=hip:${G:-gfx942}
  fi
  DEV=$PTSYCL_HIP_ARCH TORCH_INDEX=https://download.pytorch.org/whl/rocm7.0 TORCH_WHEEL=torch-2.10.0%2Brocm7.0
elif [ "$FLAVOR" = cuda ]; then
  CUDA=${PTSYCL_CUDA_HOME:-${CUDA_HOME:-/usr/local/cuda}}
  [ -x "$CUDA/bin/nvcc" ] || die "CUDA toolkit not found in $CUDA (set PTSYCL_CUDA_HOME)"
  if [ -z "${PTSYCL_CUDA_ARCH:-}" ]; then
    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.' || true)
    [ -n "$CC" ] || die "no NVIDIA GPU visible: set PTSYCL_CUDA_ARCH, e.g. cuda:sm_90"
    PTSYCL_CUDA_ARCH=cuda:sm_$CC
  fi
  DEV=$PTSYCL_CUDA_ARCH TORCH_INDEX=https://download.pytorch.org/whl/cu128 TORCH_WHEEL=torch-2.10.0%2Bcu128
else
  die "flavor must be hip or cuda"
fi
echo "target device: $DEV"

step "sources at the pinned commits"
fetch() {
  local d=$SRC/$1
  [ -d "$d/.git" ] || git init -q "$d"
  if [ "$(git -C "$d" rev-parse -q --verify HEAD 2>/dev/null || true)" != "$4" ]; then
    git -C "$d" fetch -q --depth 1 "$2" "$3" || git -C "$d" fetch -q --depth 1 "$2" "$4"
    git -C "$d" checkout -q --detach "$4"
  fi
  [ "$(git -C "$d" rev-parse HEAD)" = "$4" ] || die "$1 is not at $4"
  if [ -n "$5" ]; then
    [ "$(git -C "$d" rev-parse --short=7 'HEAD^{tree}')" = "$5" ] || die "$1: source tree is not $5"
  fi
  echo "$1 $(git -C "$d" rev-parse --short HEAD), tree $(git -C "$d" rev-parse --short=7 'HEAD^{tree}')"
}
fetch ParaS-Compiler $PARAS_REPO $PARAS_REF $PARAS_COMMIT $PARAS_TREE
fetch CLAP-Library $CLAP_REPO $CLAP_REF $CLAP_COMMIT $CLAP_TREE
fetch Framework-Torch-ParaS $TP_REPO $TP_REF $TP_COMMIT $TP_TREE
fetch InferenceX $IX_REPO $IX_COMMIT $IX_COMMIT ""

step "Python environments"
export PIP_CACHE_DIR=$WORK/pip-cache PIP_RETRIES=10 PIP_TIMEOUT=120 PIP_DISABLE_PIP_VERSION_CHECK=1
pipi() { local py=$1; shift; for a in 1 2 3; do "$py" -m pip install -q "$@" && return 0; echo "pip retry $a"; done; return 1; }
fetch_wheel() {
  local href url sha file
  href=$(curl -fsSL --retry 5 "$TORCH_INDEX/torch/" | grep -oE "href=\"[^\"]*$TORCH_WHEEL-cp312-cp312-manylinux_2_28_$(uname -m)\.whl#sha256=[0-9a-f]+\"" | head -1)
  [ -n "$href" ] || die "$TORCH_WHEEL for $(uname -m) not found in $TORCH_INDEX"
  href=${href#href=\"}; href=${href%\"}; url=${href%%#*}; sha=${href##*sha256=}
  case $url in http*) ;; *) url=$TORCH_INDEX/torch/$url ;; esac
  mkdir -p "$WORK/wheels"; file=$WORK/wheels/$(basename "${url//%2B/+}")
  if ! echo "$sha  $file" | sha256sum -c --quiet - > /dev/null 2>&1; then
    curl -fL -C - --retry 30 --retry-all-errors --retry-delay 5 -sS -o "$file" "$url" >&2 || true
    echo "$sha  $file" | sha256sum -c --quiet - >&2 || { rm -f "${file:?}"; die "PyTorch wheel download failed: $url"; }
  fi
  echo "$file"
}
PY=$WORK/venv/bin/python
if ! "$PY" -c "import torch, transformers, pyarrow, cmake, ninja" 2>/dev/null; then
  "$PY312" -m venv "$WORK/venv"
  pipi "$PY" --upgrade pip==25.2
  TW=$(fetch_wheel) || exit 1
  pipi "$PY" "$TW" --extra-index-url "$TORCH_INDEX"
  pipi "$PY" transformers==4.55.2 tokenizers==0.21.4 accelerate==1.14.0 safetensors==0.8.0 \
    huggingface_hub==0.36.2 hf-xet==1.6.0 pyarrow==26.0.0 psutil==7.2.2 cmake==3.31.6 ninja==1.13.0
fi
"$PY" -c "import torch, transformers; print('PyTorch', torch.__version__, '| transformers', transformers.__version__)"
IXPY=$WORK/ixclient/bin/python
if ! "$IXPY" -c "import aiohttp, transformers" 2>/dev/null; then
  "$PY312" -m venv "$WORK/ixclient"
  pipi "$IXPY" --upgrade pip==25.2
  pipi "$IXPY" aiohttp==3.14.5 transformers==4.55.2 tokenizers==0.21.4 huggingface_hub==0.36.2 \
    numpy==2.5.4 requests==2.34.2 tqdm==4.70.1 pillow==13.0.0 PyYAML==6.0.3 safetensors==0.8.0
fi
CMAKE=$WORK/venv/bin/cmake
TORCH_DIR=$("$PY" -c 'import torch, os; print(os.path.dirname(torch.__file__))')

step "LLVM $LLVM_VER"
if [ -n "${LLVM_DIR:-}" ]; then
  LLVM=$LLVM_DIR
else
  LLVM=$INST/llvm-$LLVM_VER
  if [ ! -x "$LLVM/bin/clang" ]; then
    ARCHIVE=${WORK:?}/llvm-project-$LLVM_VER.src.tar.xz
    curl -fL --retry 3 -sS -o "$ARCHIVE" "https://github.com/llvm/llvm-project/releases/download/llvmorg-$LLVM_VER/llvm-project-$LLVM_VER.src.tar.xz"
    echo "$LLVM_SRC_SHA  $ARCHIVE" | sha256sum -c - || die "LLVM source archive checksum mismatch"
    tar -xf "$ARCHIVE" -C "$SRC"
    rm -f "${ARCHIVE:?}"
    { "$CMAKE" -S "$SRC/llvm-project-$LLVM_VER.src/llvm" -B "$BUILD/llvm" -G Ninja -DCMAKE_MAKE_PROGRAM="$WORK/venv/bin/ninja" \
        -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang;lld" -DLLVM_TARGETS_TO_BUILD="$LLVM_TARGETS" \
        -DLLVM_ENABLE_ASSERTIONS=OFF -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF \
        -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_ENABLE_LIBXML2=OFF \
        -DCMAKE_C_COMPILER="$GCCTC/bin/gcc" -DCMAKE_CXX_COMPILER="$GCCTC/bin/g++" -DCMAKE_INSTALL_PREFIX="$LLVM" \
      && "$CMAKE" --build "$BUILD/llvm" -j"$J" --target install; } > "$LOG/llvm.log" 2>&1 \
      || { grep -m10 -iE "error|FAILED" "$LOG/llvm.log" || true; die "LLVM build failed: $LOG/llvm.log"; }
  fi
fi
"$LLVM/bin/clang" --version | head -1 | grep -q "version 20\.1" || die "$LLVM/bin/clang is not Clang 20.1"
"$LLVM/bin/clang" --version | head -1
export PATH=$LLVM/bin:$PATH
export LD_LIBRARY_PATH=$LLVM/lib:${LD_LIBRARY_PATH:-}

PARAS_HOME=$INST/paras CLAP_HOME=$INST/clap
step "ParaS-Compiler"
if [ ! -x "$PARAS_HOME/bin/parascc" ]; then
  { "$CMAKE" -S "$SRC/ParaS-Compiler" -B "$BUILD/paras" -DLLVM_DIR="$LLVM/lib/cmake/llvm" -DClang_DIR="$LLVM/lib/cmake/clang" \
      -DPARAS_INSTALL_PREFIX="$PARAS_HOME" -DPARAS_BASE_COMPILER=clang++ -DCMAKE_C_COMPILER="$LLVM/bin/clang" \
      -DCMAKE_CXX_COMPILER="$LLVM/bin/clang++" -DCMAKE_CXX_FLAGS="-std=c++17 --gcc-toolchain=$GCCTC" \
      -DCMAKE_C_FLAGS="--gcc-toolchain=$GCCTC" \
    && "$CMAKE" --build "$BUILD/paras" -j"$J" && "$CMAKE" --install "$BUILD/paras"; } > "$LOG/paras.log" 2>&1 \
    || { grep -m10 -i error "$LOG/paras.log" || true; die "ParaS build failed: $LOG/paras.log"; }
fi
echo "parascc: $PARAS_HOME/bin/parascc"
export PATH=$PARAS_HOME/bin:$PATH
export LD_LIBRARY_PATH=$PARAS_HOME/lib:${LD_LIBRARY_PATH:-}

step "CLAP-Library"
if [ ! -d "$CLAP_HOME/include" ]; then
  { "$CMAKE" -S "$SRC/CLAP-Library" -B "$BUILD/clap" -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="$CLAP_HOME" -DCMAKE_C_COMPILER="$LLVM/bin/clang" -DCMAKE_CXX_COMPILER="$LLVM/bin/clang++" \
      -DCMAKE_CXX_FLAGS="--gcc-toolchain=$GCCTC" -DCMAKE_C_FLAGS="--gcc-toolchain=$GCCTC" \
      -DBUILD_BLAS_EXAMPLES=ON -DBUILD_LAPACK_EXAMPLES=OFF -DBUILD_DNN_EXAMPLES=OFF \
    && "$CMAKE" --build "$BUILD/clap" -j"$J" && "$CMAKE" --install "$BUILD/clap"; } > "$LOG/clap.log" 2>&1 \
    || { grep -m10 -i error "$LOG/clap.log" || true; die "CLAP build failed: $LOG/clap.log"; }
fi
echo "CLAP: $CLAP_HOME"

step "Framework-Torch-ParaS ($DEV)"
TPS=$SRC/Framework-Torch-ParaS
if [ ! -f "$TPS/python/torch_paras/_C.so" ]; then
  EXTRA=()
  [ "$FLAVOR" = cuda ] && EXTRA=(-DPTSYCL_CUDA_HOME="$CUDA")
  { PTSYCL_ROCM_HOME=$ROCM PTSYCL_CUDA_HOME=$CUDA "$CMAKE" -S "$TPS" -B "$BUILD/torch-paras" -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER="$PARAS_HOME/bin/parascc" -DPTSYCL_DEVICE="sycl:$DEV" -DPTSYCL_GCC_TOOLCHAIN="$GCCTC" \
      -DPTSYCL_LLVM_DIR="$LLVM" -DPTSYCL_TORCH_DIR="$TORCH_DIR" -DPARAS_HOME="$PARAS_HOME" -DPython3_EXECUTABLE="$PY" \
      -DPTSYCL_ENABLE_CLAP_BLAS=ON -DPTSYCL_ENABLE_CLAP=ON -DPTSYCL_CLAP_ROOT="$CLAP_HOME" "${EXTRA[@]}" \
    && "$CMAKE" --build "$BUILD/torch-paras" -j"$J"; } > "$LOG/torch-paras.log" 2>&1 \
    || { grep -m15 "error:" "$LOG/torch-paras.log" || true; die "Torch-ParaS build failed: $LOG/torch-paras.log"; }
  cp -f "$(find "$BUILD/torch-paras" -name "_C*.so" -type f | head -1)" "$TPS/python/torch_paras/_C.so"
fi
echo "torch_paras: $TPS/python/torch_paras"

step "environment file, model and data"
LIBS=$GCCTC/lib64:$LLVM/lib:$PARAS_HOME/lib:$CLAP_HOME/lib64:$CLAP_HOME/lib:$TORCH_DIR/lib
[ -n "$ROCM" ] && LIBS=$LIBS:$ROCM/lib
[ -n "$CUDA" ] && LIBS=$LIBS:$CUDA/lib64
cat > "$WORK/env.sh" <<ENV
export REPRO_WORK=$WORK REPRO_FLAVOR=$FLAVOR REPRO_DEVICE_ARCH=$DEV
export PTSYCL_PYTHON=$PY IX_CLIENT_PYTHON=$IXPY INFERENCEX_DIR=$SRC/InferenceX
export PARAS_HOME=$PARAS_HOME CLAP_HOME=$CLAP_HOME HF_HOME=${REPRO_HF_HOME:-$WORK/hf} TOKENIZERS_PARALLELISM=false
export PATH=$LLVM/bin:$PARAS_HOME/bin:\$PATH
export LD_LIBRARY_PATH=$LIBS:\${LD_LIBRARY_PATH:-}
export PYTHONPATH=$TPS/python\${PYTHONPATH:+:\$PYTHONPATH}
ENV
source "$WORK/env.sh"
MODEL_DIR=$(env -u PARAM2_MODEL "$PY" -c "
import sys; sys.path.insert(0, '$HERE/common')
from harness import MODEL; print(MODEL)" | tail -1)
echo "export PARAM2_MODEL=$MODEL_DIR" >> "$WORK/env.sh"
export PARAM2_MODEL=$MODEL_DIR
echo "Param2-17B: $MODEL_DIR"
"$PY" -c "
import sys; sys.path.insert(0, '$HERE/common')
from harness import verify_model; print('model files identical to the pinned revision:', verify_model())"
BENCH_OUT=$WORK/cache "$PY" -c "
import sys; sys.path.insert(0, '$HERE/common')
from harness import tokens; print('WikiText-2 test tokens:', tokens().numel())"
step "SETUP_DONE: on the GPU machine run benchmarks/reproduce_param2.sh and benchmarks/reproduce_inferencex.sh"
