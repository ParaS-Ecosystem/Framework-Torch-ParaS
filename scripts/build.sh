#!/usr/bin/env bash
# Builds the backend with the ParaS compiler.
#
#   scripts/build.sh [cpu|cuda]
#
# Build trees: build/<flavor>. The produced extension is copied to
# python/torch_paras/_C.so, so the most recently built flavor is the one
# `import torch_paras` picks up.
set -euo pipefail

FLAVOR="${1:-cpu}"
BUILD_TYPE="${2:-Release}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/env.sh"

case "${FLAVOR}" in
    cpu)  DEVICE="cpu" ;;
    cuda) DEVICE="${PTSYCL_CUDA_ARCH:-cuda:sm_70}" ;;
    *) echo "usage: build.sh [cpu|cuda] [Release|Debug]" >&2; exit 1 ;;
esac

BUILD_DIR="${REPO_ROOT}/build/${FLAVOR}"
mkdir -p "${BUILD_DIR}"

CMAKE_BIN="${CMAKE_BIN:-cmake}"

if ! command -v "${CMAKE_BIN}" >/dev/null 2>&1; then
    echo "Error: CMake not found."
    echo "Please install CMake or set CMAKE_BIN to the CMake executable."
    echo "Example: export CMAKE_BIN=/path/to/cmake"
    exit 1
fi

"${CMAKE_BIN}" -S "${REPO_ROOT}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_CXX_COMPILER="${PARAS_HOME}/bin/parascc" \
    -DPTSYCL_DEVICE="${DEVICE}" \
    -DPTSYCL_GCC_TOOLCHAIN="${PTSYCL_GCC_TOOLCHAIN}" \
    -DPTSYCL_TORCH_DIR="${PTSYCL_TORCH_DIR}" \
    -DPTSYCL_CUDA_HOME="${PTSYCL_CUDA_HOME}" \
    -DPARAS_HOME="${PARAS_HOME}" \
    -DPython3_EXECUTABLE="${PTSYCL_PYTHON}"

# parascc names its /tmp intermediates with second resolution; parallel
# compiles would race on them, so the build is serial by design.
"${CMAKE_BIN}" --build "${BUILD_DIR}" -- -j1

cp "${BUILD_DIR}/lib/_C.so" "${REPO_ROOT}/python/torch_paras/_C.so"
echo "[build.sh] installed ${FLAVOR} flavor -> python/torch_paras/_C.so"
