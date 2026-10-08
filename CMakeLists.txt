
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

project(torch_paras LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ------------------------------------------------------------------------------
# Required configuration (normally provided by scripts/build.sh)
# ------------------------------------------------------------------------------
set(PTSYCL_DEVICE "cpu" CACHE STRING "Target: cpu, or sycl:cuda:sm_XX / sycl:hip:gfxXXX")
set(PTSYCL_GCC_TOOLCHAIN "" CACHE PATH "GCC toolchain root for parascc")
set(PTSYCL_TORCH_DIR "" CACHE PATH "PyTorch installation (site-packages/torch)")
set(PARAS_HOME "$ENV{PARAS_HOME}" CACHE PATH "ParaS compiler installation")
set(PTSYCL_LLVM_DIR "$ENV{PTSYCL_LLVM_DIR}" CACHE PATH "LLVM/clang installation used by parascc")

foreach(var PTSYCL_GCC_TOOLCHAIN PTSYCL_TORCH_DIR PARAS_HOME)
    if(NOT ${var})
        message(FATAL_ERROR "${var} must be set (use scripts/build.sh)")
    endif()
endforeach()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
    OUTPUT_VARIABLE PTSYCL_CXX11_ABI
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT PTSYCL_CXX11_ABI MATCHES "^[01]$")
    message(FATAL_ERROR "Could not determine torch._C._GLIBCXX_USE_CXX11_ABI from "
                         "${Python3_EXECUTABLE} (got '${PTSYCL_CXX11_ABI}'). Is "
                         "PTSYCL_TORCH_DIR's python env active / importable?")
endif()
message(STATUS "Torch C++11 ABI : ${PTSYCL_CXX11_ABI}")

# ------------------------------------------------------------------------------
# Flavor selection
# ------------------------------------------------------------------------------
# PTSYCL_DEVICE_FLAGS carries -parasdevice <target> so parascc's SYCL
# codegen knows which underlying GPU to lower to.
if(PTSYCL_DEVICE STREQUAL "cpu")
    set(PTSYCL_FLAVOR_DEFINE PTSYCL_BACKEND_CPU)
    set(PTSYCL_DEVICE_FLAGS "")
elseif(PTSYCL_DEVICE MATCHES "^sycl:(cuda:sm_[0-9]+|hip:gfx[0-9a-zA-Z]+)$")
    set(PTSYCL_FLAVOR_DEFINE PTSYCL_BACKEND_SYCL)
    set(PTSYCL_PARASDEVICE_TARGET "${CMAKE_MATCH_1}")
    set(PTSYCL_DEVICE_FLAGS "-parasdevice;${PTSYCL_PARASDEVICE_TARGET}")
else()
    message(FATAL_ERROR "PTSYCL_DEVICE must be 'cpu', 'sycl:cuda:sm_XX', or "
                         "'sycl:hip:gfxXXX', got '${PTSYCL_DEVICE}'")
endif()

message(STATUS "ParaS home      : ${PARAS_HOME}")
message(STATUS "Device flavor   : ${PTSYCL_DEVICE} (${PTSYCL_FLAVOR_DEFINE})")
message(STATUS "Torch           : ${PTSYCL_TORCH_DIR}")
message(STATUS "Python          : ${Python3_EXECUTABLE}")

# ------------------------------------------------------------------------------
# CUDA runtime (required for PTSYCL_BACKEND_SYCL targeting cuda:sm_XX)
# ------------------------------------------------------------------------------
if(PTSYCL_FLAVOR_DEFINE STREQUAL "PTSYCL_BACKEND_SYCL" AND
   PTSYCL_PARASDEVICE_TARGET MATCHES "^cuda:")
    set(PTSYCL_CUDA_HOME "$ENV{PTSYCL_CUDA_HOME}" CACHE PATH
        "CUDA toolkit root (needed to locate libcudart for the cuda: SYCL flavor)")
    if(NOT PTSYCL_CUDA_HOME)
        message(FATAL_ERROR "PTSYCL_CUDA_HOME must be set when PTSYCL_DEVICE "
                            "targets sycl:cuda:sm_XX -- parascc's SYCL layer "
                            "for NVIDIA links against the real CUDA runtime "
                            "underneath (see cuda_threadpool in "
                            "kem_gpu/gpu_threadpool.hpp)")
    endif()
    find_library(PTSYCL_CUDART_LIBRARY NAMES cudart
        HINTS "${PTSYCL_CUDA_HOME}/lib64" "${PTSYCL_CUDA_HOME}/lib")
    if(NOT PTSYCL_CUDART_LIBRARY)
        message(FATAL_ERROR "libcudart not found under PTSYCL_CUDA_HOME=${PTSYCL_CUDA_HOME}")
    endif()
    message(STATUS "CUDA runtime    : ${PTSYCL_CUDART_LIBRARY}")
endif()
# ------------------------------------------------------------------------------
# Sources
# ------------------------------------------------------------------------------
option(PTSYCL_FAST_BUILD "Build only tensor operators for queue diagnostics" OFF)
option(PTSYCL_FLATTEN_HEADERS "Expose framework headers to ParaS source rewriting" ON)
set(PTSYCL_SOURCES
    csrc/compat/paras_compat.cpp
    csrc/core/context.cpp
    csrc/core/allocator.cpp
    csrc/core/registration.cpp
    csrc/ops/tensor_ops.cpp
    csrc/python/module.cpp
)
if(NOT PTSYCL_FAST_BUILD)
    list(APPEND PTSYCL_SOURCES
    csrc/ops/pointwise_ops.cpp
    csrc/ops/clap_dnn_adapter.cpp
    csrc/ops/scatter_ops.cpp
    csrc/ops/indexing_ops.cpp
    csrc/ops/fused_ops.cpp
    csrc/ops/random_ops.cpp
    csrc/ops/loss_ops.cpp
    csrc/ops/norm_ops.cpp
    csrc/ops/vision_ops.cpp
    csrc/ops/attention_ops.cpp
    csrc/ops/rms_swiglu_ops.cpp
    csrc/ops/sort_ops.cpp
    csrc/ops/creation_ops.cpp
    )
else()
    message(STATUS "Fast build: only tensor_ops.cpp; other operator registrations are omitted")
endif()

if(PTSYCL_FLATTEN_HEADERS AND PTSYCL_FLAVOR_DEFINE STREQUAL "PTSYCL_BACKEND_SYCL")
    file(GLOB_RECURSE PTSYCL_FRAMEWORK_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.h")
    set(PTSYCL_GENERATED_SOURCES "")
    foreach(source IN LISTS PTSYCL_SOURCES)
        set(generated "${CMAKE_BINARY_DIR}/paras_sources/${source}")
        add_custom_command(OUTPUT "${generated}"
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/flatten_framework_headers.py"
                --root "${CMAKE_CURRENT_SOURCE_DIR}/csrc"
                --source "${CMAKE_CURRENT_SOURCE_DIR}/${source}" --output "${generated}"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${source}" ${PTSYCL_FRAMEWORK_HEADERS}
                "${CMAKE_CURRENT_SOURCE_DIR}/scripts/flatten_framework_headers.py"
            VERBATIM)
        list(APPEND PTSYCL_GENERATED_SOURCES "${generated}")
    endforeach()
    set(PTSYCL_SOURCES ${PTSYCL_GENERATED_SOURCES})
endif()

add_library(torch_paras_C SHARED ${PTSYCL_SOURCES})

set_target_properties(torch_paras_C PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_C"
    SUFFIX ".so"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

target_compile_definitions(torch_paras_C PRIVATE
    ${PTSYCL_FLAVOR_DEFINE}
    _GLIBCXX_USE_CXX11_ABI=${PTSYCL_CXX11_ABI}
    TORCH_API_INCLUDE_EXTENSION_H
    TORCH_EXTENSION_NAME=_C
)

target_include_directories(torch_paras_C PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc"
)

target_include_directories(torch_paras_C SYSTEM PRIVATE
    "${PTSYCL_TORCH_DIR}/include"
    "${PTSYCL_TORCH_DIR}/include/torch/csrc/api/include"
    "${Python3_INCLUDE_DIRS}"
    "${PARAS_HOME}/include"
)

target_compile_options(torch_paras_C PRIVATE
    --gcc-toolchain=${PTSYCL_GCC_TOOLCHAIN}
    ${PTSYCL_DEVICE_FLAGS}
    -Wall
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Debug>:-O0;-g>
)

target_link_directories(torch_paras_C PRIVATE
    "${PTSYCL_TORCH_DIR}/lib"
    "${PTSYCL_GCC_TOOLCHAIN}/lib64"
)

file(GLOB _ptsycl_gcc_lib_dirs "${PTSYCL_GCC_TOOLCHAIN}/lib/gcc/*/*")
find_library(PTSYCL_GOMP_LIBRARY NAMES gomp
    HINTS "${PTSYCL_GCC_TOOLCHAIN}/lib64" "${PTSYCL_GCC_TOOLCHAIN}/lib" ${_ptsycl_gcc_lib_dirs})
if(NOT PTSYCL_GOMP_LIBRARY)
    message(FATAL_ERROR "GNU OpenMP runtime (libgomp) not found under "
                        "PTSYCL_GCC_TOOLCHAIN=${PTSYCL_GCC_TOOLCHAIN}")
endif()
message(STATUS "GNU OpenMP     : ${PTSYCL_GOMP_LIBRARY}")

# parascc's own clang resource dir may not ship compiler-rt (a "clang+llvm
# only" build skips the separate runtimes project), and older GCC libgcc
# archives are missing some newer soft-float builtins (e.g. __truncsfbf2,
# for float->bfloat16 truncation, only landed in GCC 13). If so, fall back
# to any system/vendor clang's compiler-rt builtins archive of a matching
# major version to backfill just those symbols.
set(PTSYCL_CLANG_RT_BUILTINS "" CACHE FILEPATH
    "Override: path to a libclang_rt.builtins*.a providing soft-float builtins")
if(NOT PTSYCL_CLANG_RT_BUILTINS)
    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _ptsycl_host_processor)
    if(_ptsycl_host_processor MATCHES "^(x86_64|amd64)$")
        set(_ptsycl_rt_arch "x86_64*")
    elseif(_ptsycl_host_processor MATCHES "^(aarch64|arm64)$")
        set(_ptsycl_rt_arch "aarch64*")
    elseif(_ptsycl_host_processor MATCHES "^(i[3-6]86|x86)$")
        set(_ptsycl_rt_arch "i386*")
    else()
        message(WARNING "Unknown host processor '${CMAKE_SYSTEM_PROCESSOR}'; "
                        "set PTSYCL_CLANG_RT_BUILTINS explicitly if linking fails")
        set(_ptsycl_rt_arch "*")
    endif()
    file(GLOB _ptsycl_rt_candidates
        "${PTSYCL_LLVM_DIR}/lib/clang/*/lib/${_ptsycl_rt_arch}/libclang_rt.builtins*.a"
        "/usr/lib/clang/*/lib/${_ptsycl_rt_arch}/libclang_rt.builtins*.a"
        "/usr/lib64/clang/*/lib/${_ptsycl_rt_arch}/libclang_rt.builtins*.a"
    )
    if(_ptsycl_rt_candidates)
        list(GET _ptsycl_rt_candidates 0 PTSYCL_CLANG_RT_BUILTINS)
    endif()
endif()
if(PTSYCL_CLANG_RT_BUILTINS)
    message(STATUS "clang_rt.builtins : ${PTSYCL_CLANG_RT_BUILTINS}")
else()
    message(STATUS "clang_rt.builtins : not found (ok unless you hit missing "
                    "soft-float builtin symbols like __truncsfbf2 at link time)")
endif()

find_library(PTSYCL_PYTHON_LIBRARY NAMES python3.12 python3.12d python3
    HINTS "${PTSYCL_GCC_TOOLCHAIN}/lib64" "${PTSYCL_GCC_TOOLCHAIN}/lib" /usr/lib64 /usr/lib)

target_link_libraries(torch_paras_C PRIVATE
    torch torch_cpu torch_python c10
    ${PTSYCL_PYTHON_LIBRARY}
    pthread
    "${PTSYCL_GOMP_LIBRARY}"
)

if(PTSYCL_CLANG_RT_BUILTINS)
    target_link_libraries(torch_paras_C PRIVATE "${PTSYCL_CLANG_RT_BUILTINS}")
endif()

if(PTSYCL_CUDART_LIBRARY)
    target_link_libraries(torch_paras_C PRIVATE "${PTSYCL_CUDART_LIBRARY}")
endif()

# Note: still no amdhip64 link line, and no PTSYCL_ROCM_HOME include/link
# directories -- the hip: SYCL flavor's own runtime dependency (if it has
# one analogous to cuda_threadpool's libcudart need) hasn't been verified
# the way the cuda: flavor's has been here. Revisit if/when hip:gfxXXX is
# actually exercised on hardware -- see the earlier compiler-support-
# matrix finding that this parascc install's public backend list is only
# {host, cuda}, with AMD/HIP-ROCm still in Laxmikant's internal fork.

# ------------------------------------------------------------------------------
# Optional CLAP integration
# ------------------------------------------------------------------------------
option(PTSYCL_ENABLE_CLAP "Enable CLAP integration" OFF)

set(PTSYCL_CLAP_ROOT "$ENV{CLAP_HOME}" CACHE PATH
    "CLAP installation prefix")

if(PTSYCL_ENABLE_CLAP)
    set(_ptsycl_clap_include_hints)
    set(_ptsycl_clap_library_hints)

    if(PTSYCL_CLAP_ROOT)
        list(APPEND _ptsycl_clap_include_hints
            "${PTSYCL_CLAP_ROOT}/include")
        list(APPEND _ptsycl_clap_library_hints
            "${PTSYCL_CLAP_ROOT}/lib64"
            "${PTSYCL_CLAP_ROOT}/lib")
    endif()

    find_path(PTSYCL_CLAP_INCLUDE_DIR
        NAMES clap/dnn_factory.hpp
        HINTS ${_ptsycl_clap_include_hints})

    find_library(PTSYCL_CLAP_LIBRARY
        NAMES clap
        HINTS ${_ptsycl_clap_library_hints})

    if(NOT PTSYCL_CLAP_INCLUDE_DIR)
        message(FATAL_ERROR
            "CLAP headers not found. Set PTSYCL_CLAP_ROOT or CLAP_HOME.")
    endif()

    if(NOT PTSYCL_CLAP_LIBRARY)
        message(FATAL_ERROR
            "libclap not found. Set PTSYCL_CLAP_ROOT or CLAP_HOME.")
    endif()

    target_compile_definitions(torch_paras_C PRIVATE USE_CLAP_BACKEND)

    if(PTSYCL_FLAVOR_DEFINE STREQUAL "PTSYCL_BACKEND_SYCL")
        if(PTSYCL_PARASDEVICE_TARGET MATCHES "^cuda:")
            target_compile_definitions(
                torch_paras_C PRIVATE PTSYCL_CLAP_DNN_CUDA)
        elseif(PTSYCL_PARASDEVICE_TARGET MATCHES "^hip:")
            target_compile_definitions(
                torch_paras_C PRIVATE PTSYCL_CLAP_DNN_ROCM)
        endif()
    endif()

    target_include_directories(torch_paras_C SYSTEM PRIVATE
        "${PTSYCL_CLAP_INCLUDE_DIR}")

    target_link_libraries(torch_paras_C PRIVATE
        "${PTSYCL_CLAP_LIBRARY}")

    get_filename_component(
        PTSYCL_CLAP_LIBRARY_DIR
        "${PTSYCL_CLAP_LIBRARY}"
        DIRECTORY)

    message(STATUS "CLAP include     : ${PTSYCL_CLAP_INCLUDE_DIR}")
    message(STATUS "CLAP library     : ${PTSYCL_CLAP_LIBRARY}")
endif()

set(PTSYCL_PARAS_RUNTIME_DIR "" CACHE PATH
    "Optional ParaS runtime library override")

set(PTSYCL_RPATHS
    "${PTSYCL_TORCH_DIR}/lib"
)

if(PTSYCL_PARAS_RUNTIME_DIR)
    list(APPEND PTSYCL_RPATHS "${PTSYCL_PARAS_RUNTIME_DIR}")
endif()

list(APPEND PTSYCL_RPATHS
    "${PARAS_HOME}/lib"
    "${PTSYCL_GCC_TOOLCHAIN}/lib64"
)

if(PTSYCL_ENABLE_CLAP)
    list(APPEND PTSYCL_RPATHS "${PTSYCL_CLAP_LIBRARY_DIR}")
endif()
if(PTSYCL_CUDART_LIBRARY)
    get_filename_component(_ptsycl_cudart_dir "${PTSYCL_CUDART_LIBRARY}" DIRECTORY)
    list(APPEND PTSYCL_RPATHS "${_ptsycl_cudart_dir}")
endif()

if(PTSYCL_FLAVOR_DEFINE STREQUAL "PTSYCL_BACKEND_SYCL")
    target_link_options(torch_paras_C PRIVATE ${PTSYCL_DEVICE_FLAGS})
endif()

list(JOIN PTSYCL_RPATHS ";" PTSYCL_RPATH_STR)
set_target_properties(torch_paras_C PROPERTIES
    BUILD_RPATH "${PTSYCL_RPATH_STR}"
    INSTALL_RPATH "${PTSYCL_RPATH_STR}"
)

target_link_options(torch_paras_C PRIVATE
    --gcc-toolchain=${PTSYCL_GCC_TOOLCHAIN}
)
