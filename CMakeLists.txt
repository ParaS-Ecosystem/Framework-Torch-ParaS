# PyTorch backend for the ParaS compiler.
#
# The C++ compiler must be parascc; scripts/build.sh sets it together with
# the toolchain environment. Two build flavors exist:
#
#   -DPTSYCL_DEVICE=cpu          ParaS CPU engine (kem threadpool)
#   -DPTSYCL_DEVICE=cuda:sm_70   CUDA kernels through -parasdevice

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
set(PARAS_HOME "$ENV{PARAS_HOME}" CACHE PATH "ParaS compiler installation")

foreach(var PTSYCL_GCC_TOOLCHAIN PTSYCL_TORCH_DIR PARAS_HOME)
    if(NOT ${var})
        message(FATAL_ERROR "${var} must be set (use scripts/build.sh)")
    endif()
endforeach()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

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
else()
    message(FATAL_ERROR "PTSYCL_DEVICE must be 'cpu' or 'cuda:sm_XX', got '${PTSYCL_DEVICE}'")
endif()

message(STATUS "ParaS home      : ${PARAS_HOME}")
message(STATUS "Device flavor   : ${PTSYCL_DEVICE} (${PTSYCL_FLAVOR_DEFINE})")
message(STATUS "Torch           : ${PTSYCL_TORCH_DIR}")
message(STATUS "Python          : ${Python3_EXECUTABLE}")

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
    csrc/ops/random_ops.cpp
    csrc/ops/loss_ops.cpp
    csrc/ops/norm_ops.cpp
    csrc/ops/vision_ops.cpp
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
    _GLIBCXX_USE_CXX11_ABI=0      # match the PyTorch build
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

target_link_libraries(torch_paras_C PRIVATE
    torch torch_cpu torch_python c10
    ${Python3_LIBRARIES}
    pthread
)

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

list(JOIN PTSYCL_RPATHS ";" PTSYCL_RPATH_STR)
set_target_properties(torch_paras_C PROPERTIES
    BUILD_RPATH "${PTSYCL_RPATH_STR}"
    INSTALL_RPATH "${PTSYCL_RPATH_STR}"
)

target_link_options(torch_paras_C PRIVATE
    --gcc-toolchain=${PTSYCL_GCC_TOOLCHAIN}
    -Wl,--no-undefined
)
