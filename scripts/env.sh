#!/usr/bin/env bash

# ==========================================================
# ParaS PyTorch Backend Environment
#
# Usage:
#   export PARAS_HOME=/path/to/paras
#   export PTSYCL_GCC_TOOLCHAIN=/path/to/gcc
#   export PTSYCL_LLVM_DIR=/path/to/llvm
#   export PTSYCL_CUDA_HOME=/path/to/cuda
#   export PTSYCL_ROCM_HOME=/path/to/rocm
#
#   source scripts/env.sh
#
# PTSYCL_GCC_LDPATH=0 keeps the GCC toolchain off LD_LIBRARY_PATH; see below.
# PTSYCL_PYTHON_INCLUDE / PTSYCL_PYTHON_LIBRARY point cmake at the Python
# headers and library when the interpreter is a venv without dev files.
# ==========================================================

: "${PARAS_HOME:?Please set PARAS_HOME}"
: "${PTSYCL_GCC_TOOLCHAIN:?Please set PTSYCL_GCC_TOOLCHAIN}"
: "${PTSYCL_LLVM_DIR:?Please set PTSYCL_LLVM_DIR}"

export PTSYCL_CUDA_HOME="${PTSYCL_CUDA_HOME:-/usr/local/cuda}"
export PTSYCL_ROCM_HOME="${PTSYCL_ROCM_HOME:-/opt/rocm}"

export PTSYCL_PYTHON="${PTSYCL_PYTHON:-python3}"

PTSYCL_TORCH_DIR_DEFAULT="$(${PTSYCL_PYTHON} -c 'import torch, os; print(os.path.dirname(torch.__file__))' 2>/dev/null)"
export PTSYCL_TORCH_DIR="${PTSYCL_TORCH_DIR:-${PTSYCL_TORCH_DIR_DEFAULT}}"

export PATH="${PTSYCL_LLVM_DIR}/bin:${PARAS_HOME}/bin:${PTSYCL_CUDA_HOME}/bin:${PTSYCL_ROCM_HOME}/bin:${PATH}"

# ROCm's lib dir is deliberately NOT on LD_LIBRARY_PATH. A full ROCm lib dir
# here shadows the ROCm libs bundled with a ROCm PyTorch wheel -- an
# libamd_smi.so undefined-symbol clash breaks `import torch` outright. ROCm is
# resolved for the build through -L / --rocm-path and at runtime through the
# extension's RPATH, so it is not needed here.
#
# The GCC toolchain lib dir is on by default and can be turned off with
# PTSYCL_GCC_LDPATH=0. On boxes where the system cmake was linked against an
# older libstdc++, a newer gcc's libstdc++ ahead of it makes cmake fail to
# start; gcc reaches the compiler via --gcc-toolchain and the extension via
# RPATH either way.
_ptsycl_gcc_ld=""
if [ "${PTSYCL_GCC_LDPATH:-1}" = "1" ]; then
    _ptsycl_gcc_ld="${PTSYCL_GCC_TOOLCHAIN}/lib64:"
fi
export LD_LIBRARY_PATH="${_ptsycl_gcc_ld}${PTSYCL_LLVM_DIR}/lib:${PARAS_HOME}/lib:${PTSYCL_CUDA_HOME}/lib64:${PTSYCL_TORCH_DIR}/lib:${LD_LIBRARY_PATH:-}"

echo "[env.sh] PARAS_HOME = ${PARAS_HOME}"
echo "[env.sh] LLVM       = ${PTSYCL_LLVM_DIR}"
echo "[env.sh] GCC        = ${PTSYCL_GCC_TOOLCHAIN}"
echo "[env.sh] CUDA       = ${PTSYCL_CUDA_HOME}"
echo "[env.sh] ROCm       = ${PTSYCL_ROCM_HOME}"
echo "[env.sh] Torch      = ${PTSYCL_TORCH_DIR}"