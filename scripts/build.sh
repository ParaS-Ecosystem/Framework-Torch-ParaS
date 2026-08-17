#!/usr/bin/env bash
# Builds the backend with the ParaS compiler.
#
#   scripts/build.sh [auto|cpu|cuda|hip]
#
# "auto" (the default) picks the flavor from what is actually installed on the
# machine: an NVIDIA GPU with a CUDA toolkit gives cuda, an AMD GPU with ROCm
# gives hip, and anything else gives cpu. Every flavor exposes paras:0 as the
# host CPU engine, so a cuda or hip build still runs on a machine with no GPU
# visible -- it just reports device_count() == 1.
#
# Build trees: build/<flavor>. The produced extension is copied to
# python/torch_paras/_C.so, so the most recently built flavor is the one
# `import torch_paras` picks up.
set -euo pipefail

FLAVOR="${1:-auto}"
BUILD_TYPE="${2:-Release}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/env.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Both a GPU and its toolchain have to be present: a CUDA toolkit on a machine
# with no NVIDIA card builds fine but has nothing to run on, and a card with no
# toolkit cannot be compiled for at all.
detect_flavor() {
    if command -v nvidia-smi >/dev/null 2>&1 &&
       nvidia-smi -L 2>/dev/null | grep -q '^GPU' &&
       [[ -x "${PTSYCL_CUDA_HOME}/bin/nvcc" || -d "${PTSYCL_CUDA_HOME}/include" ]]; then
        echo cuda
    elif command -v rocm_agent_enumerator >/dev/null 2>&1 &&
         rocm_agent_enumerator 2>/dev/null | grep -qv '^gfx000$' &&
         [[ -d "${PTSYCL_ROCM_HOME}" ]]; then
        echo hip
    else
        echo cpu
    fi
}

if [[ "${FLAVOR}" == "auto" ]]; then
    FLAVOR="$(detect_flavor)"
    echo "[build.sh] auto-detected flavor: ${FLAVOR}"
fi

# Every flavor builds on all cores. The cpu and hip flavors go through
# parascc's source-transformation stage, which needs a parascc that names its
# intermediate files per-process -- older ones use a second-resolution
# timestamp, so concurrent compiles overwrite each other's transformed source
# and silently produce objects built from the wrong translation unit. Set
# JOBS=1 to fall back to a serial build (see docs/TROUBLESHOOTING.md).
BUILD_JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

case "${FLAVOR}" in
    # CPU flavor: parascc wrapper (adds clang resource-dir / include -isystem).
    cpu)  DEVICE="cpu"
          CXX_COMPILER="${SCRIPT_DIR}/parascc-cpu" ;;
    # CUDA flavor: clang++ driven directly (bypasses parascc's expfinder, which
    # miscompiles X-macro lambdas), so it never touches that stage.
    cuda) DEVICE="${PTSYCL_CUDA_ARCH:-cuda:sm_70}"
          CXX_COMPILER="${SCRIPT_DIR}/parascc-cuda" ;;
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
        ;;
    *) echo "usage: build.sh [auto|cpu|cuda|hip] [Release|Debug]" >&2; exit 1 ;;
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
