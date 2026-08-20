# PyTorch backend for the ParaS compiler.
#
# The C++ compiler must be parascc; scripts/build.sh sets it together with
# the toolchain environment. Two build flavors exist:
#
#   -DPTSYCL_DEVICE=cpu          ParaS CPU engine (kem threadpool)
#   -DPTSYCL_DEVICE=cuda:sm_70   CUDA kernels through -parasdevice
#   -DPTSYCL_DEVICE=hip:gfx90a   HIP/ROCm kernels through -parasdevice

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
set(PTSYCL_DEVICE "cpu" CACHE STRING "Target: cpu or cuda:sm_XX")
set(PTSYCL_GCC_TOOLCHAIN "" CACHE PATH "GCC toolchain root for parascc")
set(PTSYCL_TORCH_DIR "" CACHE PATH "PyTorch installation (site-packages/torch)")
set(PTSYCL_CUDA_HOME "" CACHE PATH "CUDA toolkit root (cuda flavor only)")
set(PTSYCL_ROCM_HOME "" CACHE PATH "ROCm installation root (hip flavor only)")
set(PARAS_HOME "$ENV{PARAS_HOME}" CACHE PATH "ParaS compiler installation")
set(PTSYCL_LLVM_DIR "$ENV{PTSYCL_LLVM_DIR}" CACHE PATH "LLVM/clang installation used by parascc")

foreach(var PTSYCL_GCC_TOOLCHAIN PTSYCL_TORCH_DIR PARAS_HOME)
    if(NOT ${var})
        message(FATAL_ERROR "${var} must be set (use scripts/build.sh)")
    endif()
endforeach()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

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
if(PTSYCL_DEVICE STREQUAL "cpu")
    set(PTSYCL_FLAVOR_DEFINE PTSYCL_BACKEND_CPU)
    set(PTSYCL_DEVICE_FLAGS "")
elseif(PTSYCL_DEVICE MATCHES "^cuda:sm_[0-9]+$")
    set(PTSYCL_FLAVOR_DEFINE PTSYCL_BACKEND_CUDA)
    set(PTSYCL_DEVICE_FLAGS "-parasdevice;${PTSYCL_DEVICE}")
    if(NOT PTSYCL_CUDA_HOME)
        message(FATAL_ERROR "PTSYCL_CUDA_HOME required for ${PTSYCL_DEVICE}")
    endif()
elseif(PTSYCL_DEVICE MATCHES "^hip:gfx[0-9a-zA-Z]+$")
    set(PTSYCL_FLAVOR_DEFINE PTSYCL_BACKEND_HIP)
    set(PTSYCL_DEVICE_FLAGS "-parasdevice;${PTSYCL_DEVICE}")
    if(NOT PTSYCL_ROCM_HOME)
        message(FATAL_ERROR "PTSYCL_ROCM_HOME required for ${PTSYCL_DEVICE}")
    endif()
else()
    message(FATAL_ERROR "PTSYCL_DEVICE must be 'cpu', 'cuda:sm_XX', or 'hip:gfxXXX', got '${PTSYCL_DEVICE}'")
endif()

message(STATUS "ParaS home      : ${PARAS_HOME}")
message(STATUS "Device flavor   : ${PTSYCL_DEVICE} (${PTSYCL_FLAVOR_DEFINE})")
message(STATUS "Torch           : ${PTSYCL_TORCH_DIR}")
message(STATUS "Python          : ${Python3_EXECUTABLE}")
if(PTSYCL_FLAVOR_DEFINE STREQUAL "PTSYCL_BACKEND_CUDA")
    message(STATUS "CUDA home       : ${PTSYCL_CUDA_HOME}")
elseif(PTSYCL_FLAVOR_DEFINE STREQUAL "PTSYCL_BACKEND_HIP")
    message(STATUS "ROCm home       : ${PTSYCL_ROCM_HOME}")
endif()

# ------------------------------------------------------------------------------
# Sources
# ------------------------------------------------------------------------------
add_library(torch_paras_C SHARED
    csrc/compat/paras_compat.cpp
    csrc/core/context.cpp
    csrc/core/allocator.cpp
    csrc/core/registration.cpp
    csrc/ops/tensor_ops.cpp
    csrc/ops/pointwise_ops.cpp
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
    csrc/python/module.cpp
)

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

if(PTSYCL_FLAVOR_DEFINE STREQUAL "PTSYCL_BACKEND_HIP")
    target_include_directories(torch_paras_C SYSTEM PRIVATE "${PTSYCL_ROCM_HOME}/include")
endif()

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

find_library(PTSYCL_GOMP_LIBRARY NAMES gomp
    HINTS "${PTSYCL_GCC_TOOLCHAIN}/lib64" "${PTSYCL_GCC_TOOLCHAIN}/lib")
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

target_link_libraries(torch_paras_C PRIVATE
    torch torch_cpu torch_python c10
    ${Python3_LIBRARIES}
    pthread
    "${PTSYCL_GOMP_LIBRARY}"
)

if(PTSYCL_CLANG_RT_BUILTINS)
    target_link_libraries(torch_paras_C PRIVATE "${PTSYCL_CLANG_RT_BUILTINS}")
endif()

set(PTSYCL_RPATHS
    "${PTSYCL_TORCH_DIR}/lib"
    "${PARAS_HOME}/lib"
    "${PTSYCL_GCC_TOOLCHAIN}/lib64"
)

if(PTSYCL_FLAVOR_DEFINE STREQUAL "PTSYCL_BACKEND_CUDA")
    target_link_directories(torch_paras_C PRIVATE "${PTSYCL_CUDA_HOME}/lib64")
    target_link_libraries(torch_paras_C PRIVATE cudart)
    list(APPEND PTSYCL_RPATHS "${PTSYCL_CUDA_HOME}/lib64")
    target_link_options(torch_paras_C PRIVATE ${PTSYCL_DEVICE_FLAGS})
endif()

if(PTSYCL_FLAVOR_DEFINE STREQUAL "PTSYCL_BACKEND_HIP")
    target_link_directories(torch_paras_C PRIVATE "${PTSYCL_ROCM_HOME}/lib")
    target_link_libraries(torch_paras_C PRIVATE amdhip64)
    list(APPEND PTSYCL_RPATHS "${PTSYCL_ROCM_HOME}/lib")
    target_link_options(torch_paras_C PRIVATE ${PTSYCL_DEVICE_FLAGS})
endif()

list(JOIN PTSYCL_RPATHS ";" PTSYCL_RPATH_STR)
set_target_properties(torch_paras_C PROPERTIES
    BUILD_RPATH "${PTSYCL_RPATH_STR}"
    INSTALL_RPATH "${PTSYCL_RPATH_STR}"
)

target_link_options(torch_paras_C PRIVATE
    --gcc-toolchain=${PTSYCL_GCC_TOOLCHAIN}
    -Wl,--no-undefined
)
