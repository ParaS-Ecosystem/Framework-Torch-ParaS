#!/usr/bin/env bash
# Builds the backend with the ParaS compiler.
#
#   scripts/build.sh [cpu|cuda|hip]
#
# Build trees: build/<flavor>. The produced extension is copied to
# python/torch_paras/_C.so, so the most recently built flavor is the one
# `import torch_paras` picks up.
set -euo pipefail

FLAVOR="${1:-cpu}"
BUILD_TYPE="${2:-Release}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/env.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${FLAVOR}" in
    # CPU flavor: parascc wrapper (adds clang resource-dir / include -isystem).
    cpu)  DEVICE="cpu"
          CXX_COMPILER="${SCRIPT_DIR}/parascc-cpu"
          # parascc names its /tmp intermediates with second resolution; parallel
          # compiles would race on them, so the CPU build is serial by design.
          BUILD_JOBS="1" ;;
    # CUDA flavor: clang++ driven directly (bypasses parascc's expfinder, which
    # miscompiles X-macro lambdas). No /tmp race, so it can build in parallel.
    cuda) DEVICE="${PTSYCL_CUDA_ARCH:-cuda:sm_70}"
          CXX_COMPILER="${SCRIPT_DIR}/parascc-cuda"
          BUILD_JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}" ;;
    hip)
        if [[ -n "${PTSYCL_HIP_ARCH:-}" ]]; then
            DEVICE="${PTSYCL_HIP_ARCH}"
        elif command -v rocm_agent_enumerator >/dev/null 2>&1; then
            # rocm_agent_enumerator lists one gfx target per visible agent,
            # including a "gfx000" line for the host CPU which we must skip.
            mapfile -t _ptsycl_gfx_agents < <(rocm_agent_enumerator | grep -v '^gfx000$' | sort -u)
            if [[ ${#_ptsycl_gfx_agents[@]} -eq 0 ]]; then
                echo "error: rocm_agent_enumerator found no AMD GPU agents." >&2
                echo "       Set PTSYCL_HIP_ARCH=hip:gfxXXX explicitly." >&2
                exit 1
            elif [[ ${#_ptsycl_gfx_agents[@]} -gt 1 ]]; then
                echo "error: multiple distinct GPU architectures detected (${_ptsycl_gfx_agents[*]})." >&2
                echo "       This build targets one architecture; set PTSYCL_HIP_ARCH=hip:gfxXXX" >&2
                echo "       to pick which one, e.g. PTSYCL_HIP_ARCH=hip:${_ptsycl_gfx_agents[0]}" >&2
                exit 1
            fi
            DEVICE="hip:${_ptsycl_gfx_agents[0]}"
            echo "[build.sh] auto-detected HIP arch: ${DEVICE}"
        else
            echo "error: rocm_agent_enumerator not found and PTSYCL_HIP_ARCH not set." >&2
            echo "       Set PTSYCL_HIP_ARCH=hip:gfxXXX explicitly (check with: rocminfo | grep gfx)." >&2
            exit 1
        fi
        CXX_COMPILER="${PARAS_HOME}/bin/parascc"
        # HIP still goes through parascc -parasdevice, so keep it serial.
        BUILD_JOBS="1"
        ;;
    *) echo "usage: build.sh [cpu|cuda|hip] [Release|Debug]" >&2; exit 1 ;;
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
    -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" \
    -DPTSYCL_DEVICE="${DEVICE}" \
    -DPTSYCL_GCC_TOOLCHAIN="${PTSYCL_GCC_TOOLCHAIN}" \
    -DPTSYCL_TORCH_DIR="${PTSYCL_TORCH_DIR}" \
    -DPTSYCL_CUDA_HOME="${PTSYCL_CUDA_HOME}" \
    -DPTSYCL_ROCM_HOME="${PTSYCL_ROCM_HOME}" \
    -DPARAS_HOME="${PARAS_HOME}" \
    -DPython3_EXECUTABLE="${PTSYCL_PYTHON}"

"${CMAKE_BIN}" --build "${BUILD_DIR}" -- -j"${BUILD_JOBS}"

cp "${BUILD_DIR}/lib/_C.so" "${REPO_ROOT}/python/torch_paras/_C.so"
echo "[build.sh] installed ${FLAVOR} flavor -> python/torch_paras/_C.so"
