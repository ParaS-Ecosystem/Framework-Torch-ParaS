#!/usr/bin/env bash

# ==========================================================
# ParaS PyTorch Backend Environment
#
# Usage:
#   export PARAS_HOME=/path/to/paras
#   export PTSYCL_GCC_TOOLCHAIN=/path/to/gcc
#   export PTSYCL_LLVM_DIR=/path/to/llvm
#   export PTSYCL_CUDA_HOME=/path/to/cuda
#
#   source scripts/env.sh
# ==========================================================

: "${PARAS_HOME:?Please set PARAS_HOME}"
: "${PTSYCL_GCC_TOOLCHAIN:?Please set PTSYCL_GCC_TOOLCHAIN}"
: "${PTSYCL_LLVM_DIR:?Please set PTSYCL_LLVM_DIR}"

export PTSYCL_CUDA_HOME="${PTSYCL_CUDA_HOME:-/usr/local/cuda}"

export PTSYCL_PYTHON="${PTSYCL_PYTHON:-python3}"

PTSYCL_TORCH_DIR_DEFAULT="$(${PTSYCL_PYTHON} -c 'import torch, os; print(os.path.dirname(torch.__file__))' 2>/dev/null)"
export PTSYCL_TORCH_DIR="${PTSYCL_TORCH_DIR:-${PTSYCL_TORCH_DIR_DEFAULT}}"

export PATH="${PTSYCL_LLVM_DIR}/bin:${PARAS_HOME}/bin:${PTSYCL_CUDA_HOME}/bin:${PATH}"

export LD_LIBRARY_PATH="${PTSYCL_GCC_TOOLCHAIN}/lib64:${PTSYCL_LLVM_DIR}/lib:${PARAS_HOME}/lib:${PTSYCL_CUDA_HOME}/lib64:${PTSYCL_TORCH_DIR}/lib:${LD_LIBRARY_PATH:-}"

echo "[env.sh] PARAS_HOME = ${PARAS_HOME}"
echo "[env.sh] LLVM       = ${PTSYCL_LLVM_DIR}"
echo "[env.sh] GCC        = ${PTSYCL_GCC_TOOLCHAIN}"
echo "[env.sh] CUDA       = ${PTSYCL_CUDA_HOME}"
echo "[env.sh] Torch      = ${PTSYCL_TORCH_DIR}"