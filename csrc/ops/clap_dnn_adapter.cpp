// -----------------------------------------------------------------------------
// Copyright (c) 2026 Centre for Development of Advanced Computing (C-DAC)
//
// This file is part of Torch_ParaS, a component of the ParaS Ecosystem.
//
// This library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License (LGPL)
// version 3 as published by the Free Software Foundation.
// -----------------------------------------------------------------------------

#include "ops/clap_dnn_adapter.h"

#include "core/kernels.h"

#include <atomic>
#include <mutex>

#ifdef USE_CLAP_BACKEND

#include <clap/dnn_factory.hpp>
#include <clap/dnn_types.hpp>
#include <clap/idnn_backend.hpp>

#if defined(PTSYCL_CLAP_DNN_CUDA)
#include <cuda_runtime_api.h>
#endif

#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#endif


namespace ptsycl::clap_dnn {

namespace {

std::atomic<std::int64_t> g_total_calls{0};
std::atomic<std::int64_t> g_cpu_calls{0};
std::atomic<std::int64_t> g_cuda_calls{0};
std::atomic<std::int64_t> g_rocm_calls{0};
std::atomic<std::int64_t> g_fallback_calls{0};

std::mutex g_last_backend_mutex;
std::string g_last_backend;

void record_clap_execution(
    const char* backend_name,
    int backend_kind) noexcept
{
    g_total_calls.fetch_add(1, std::memory_order_relaxed);

    if (backend_kind == 0) {
        g_cpu_calls.fetch_add(1, std::memory_order_relaxed);
    } else if (backend_kind == 1) {
        g_cuda_calls.fetch_add(1, std::memory_order_relaxed);
    } else if (backend_kind == 2) {
        g_rocm_calls.fetch_add(1, std::memory_order_relaxed);
    }

    if (backend_name != nullptr) {
        std::lock_guard<std::mutex> lock(g_last_backend_mutex);
        g_last_backend = backend_name;
    }
}

void record_fallback() noexcept
{
    g_fallback_calls.fetch_add(1, std::memory_order_relaxed);
}

void sync_all_devices_noexcept() noexcept
{
    try {
        ptsycl::Context::instance().synchronize_all();
    } catch (...) {
    }
}

} // namespace

#ifdef USE_CLAP_BACKEND

namespace {


#if defined(PTSYCL_CLAP_DNN_CUDA)

class CudaDeviceGuard {
public:
    explicit CudaDeviceGuard(int target_device)
        : previous_device_(-1),
          restore_(false)
    {
        const auto get_status =
            cudaGetDevice(&previous_device_);

        TORCH_CHECK(
            get_status == cudaSuccess,
            "Torch-ParaS CLAP-DNN cudaGetDevice failed: ",
            cudaGetErrorString(get_status));

        if (previous_device_ == target_device)
            return;

        const auto set_status =
            cudaSetDevice(target_device);

        TORCH_CHECK(
            set_status == cudaSuccess,
            "Torch-ParaS CLAP-DNN cudaSetDevice(",
            target_device,
            ") failed: ",
            cudaGetErrorString(set_status));

        restore_ = true;
    }

    ~CudaDeviceGuard() noexcept
    {
        if (restore_) {
            (void)cudaSetDevice(previous_device_);
        }
    }

    CudaDeviceGuard(const CudaDeviceGuard&) = delete;
    CudaDeviceGuard& operator=(const CudaDeviceGuard&) = delete;

private:
    int previous_device_;
    bool restore_;
};


int cuda_device_for_pointer(const void* ptr)
{
    cudaPointerAttributes attributes{};

    const auto status =
        cudaPointerGetAttributes(
            &attributes,
            ptr);

    TORCH_CHECK(
        status == cudaSuccess,
        "Torch-ParaS CLAP-DNN cudaPointerGetAttributes failed: ",
        cudaGetErrorString(status));

    TORCH_CHECK(
        attributes.type == cudaMemoryTypeDevice ||
        attributes.type == cudaMemoryTypeManaged,
        "Torch-ParaS CLAP-DNN expected CUDA device memory");

    TORCH_CHECK(
        attributes.device >= 0,
        "Torch-ParaS CLAP-DNN received an invalid CUDA device ordinal");

    return attributes.device;
}

#endif


bool select_backend(
    const at::Tensor& tensor,
    clap::DnnBackendType& requested)
{
    auto& q = queue_for(tensor);

    if (!q.is_gpu()) {
        requested = clap::DnnBackendType::CPU;
        return true;
    }

#if defined(PTSYCL_CLAP_DNN_CUDA)

    requested = clap::DnnBackendType::CUDA;
    return true;

#elif defined(PTSYCL_CLAP_DNN_ROCM)

    requested = clap::DnnBackendType::ROCM;
    return true;

#else

    return false;

#endif
}


bool override_matches_requested(
    clap::DnnBackendType override_backend,
    clap::DnnBackendType requested) noexcept
{
    switch (override_backend) {

        case clap::DnnBackendType::CPU:
            return requested == clap::DnnBackendType::CPU;

        case clap::DnnBackendType::CUDA:
            return requested == clap::DnnBackendType::CUDA;

        case clap::DnnBackendType::ROCM:
            return requested == clap::DnnBackendType::ROCM;

        case clap::DnnBackendType::GPU:
            return requested == clap::DnnBackendType::CUDA ||
                   requested == clap::DnnBackendType::ROCM;
    }

    return false;
}


bool parse_env_backend(
    const std::string& backend,
    clap::DnnBackendType& parsed) noexcept
{
    if (backend == "CPU") {
        parsed = clap::DnnBackendType::CPU;
        return true;
    }

    if (backend == "CUDA") {
        parsed = clap::DnnBackendType::CUDA;
        return true;
    }

    if (backend == "ROCM" || backend == "AMD") {
        parsed = clap::DnnBackendType::ROCM;
        return true;
    }

    if (backend == "GPU") {
        parsed = clap::DnnBackendType::GPU;
        return true;
    }

    return false;
}


bool parse_cmdline_backend_arg(
    const std::string& arg,
    clap::DnnBackendType& parsed) noexcept
{
    if (arg == "-cuda" || arg == "--cuda") {
        parsed = clap::DnnBackendType::CUDA;
        return true;
    }

    if (arg == "-rocm" || arg == "--rocm" ||
        arg == "-amd" || arg == "--amd") {
        parsed = clap::DnnBackendType::ROCM;
        return true;
    }

    if (arg == "-cpu" || arg == "--cpu") {
        parsed = clap::DnnBackendType::CPU;
        return true;
    }

    if (arg == "-gpu" || arg == "--gpu") {
        parsed = clap::DnnBackendType::GPU;
        return true;
    }

    return false;
}


bool cmdline_override_matches(
    clap::DnnBackendType requested)
{
    std::ifstream cmdline("/proc/self/cmdline", std::ios::in | std::ios::binary);

    if (!cmdline.is_open())
        return true;

    std::string arg;

    while (std::getline(cmdline, arg, '\0')) {
        clap::DnnBackendType cli_backend{};
        if (parse_cmdline_backend_arg(arg, cli_backend)) {
            return override_matches_requested(cli_backend, requested);
        }
    }

    return true;
}


bool backend_override_matches(
    clap::DnnBackendType requested)
{
    const char* value = std::getenv("CLAP_BACKEND");

    if (value != nullptr && *value != '\0') {
        clap::DnnBackendType env_backend{};

        if (!parse_env_backend(std::string(value), env_backend))
            return false;

        if (!override_matches_requested(env_backend, requested))
            return false;
    }

    return cmdline_override_matches(requested);
}


clap::IDnnBackend& backend_for(
    clap::DnnBackendType requested)
{
    struct Cache {
        std::unique_ptr<clap::IDnnBackend> cpu;
        std::unique_ptr<clap::IDnnBackend> cuda;
        std::unique_ptr<clap::IDnnBackend> rocm;
    };

    thread_local Cache cache;

    std::unique_ptr<clap::IDnnBackend>* slot = nullptr;
    clap::ExecutionBackend expected_exec = clap::ExecutionBackend::CPU;

    switch (requested) {

        case clap::DnnBackendType::CPU:
            slot = &cache.cpu;
            expected_exec = clap::ExecutionBackend::CPU;
            break;

        case clap::DnnBackendType::CUDA:
            slot = &cache.cuda;
            expected_exec = clap::ExecutionBackend::CUDA;
            break;

        case clap::DnnBackendType::ROCM:
            slot = &cache.rocm;
            expected_exec = clap::ExecutionBackend::ROCM;
            break;

        case clap::DnnBackendType::GPU:
            TORCH_CHECK(
                false,
                "Torch-ParaS CLAP-DNN requires an explicit backend");
    }

    if (!*slot)
        *slot = clap::DnnFactory::create(requested);

    TORCH_CHECK(
        *slot,
        "CLAP-DNN backend creation failed");

    if ((*slot)->executionBackend() != expected_exec) {
        slot->reset();
        TORCH_CHECK(
            false,
            "Torch-ParaS CLAP-DNN backend mismatch for tensor device: expected ",
            clap::executionBackendName(expected_exec));
    }

    return **slot;
}


bool map_dtype(
    c10::ScalarType dtype,
    clap::DataType& result)
{
    switch (dtype) {

        case at::kFloat:
            result = clap::DataType::Float32;
            return true;

        case at::kHalf:
            result = clap::DataType::Float16;
            return true;

        case at::kBFloat16:
            result = clap::DataType::BFloat16;
            return true;

        default:
            return false;
    }
}


bool try_activation_forward(
    const at::Tensor& self,
    at::Tensor& out,
    clap::ActivationMode mode)
{
    if (self.numel() == 0 ||
        self.numel() != out.numel() ||
        self.sizes() != out.sizes() ||
        self.device() != out.device() ||
        self.scalar_type() != out.scalar_type() ||
        !self.is_contiguous() ||
        !out.is_contiguous() ||
        self.dim() == 0) {

        record_fallback();
        return false;
    }

    clap::DataType dtype;

    if (!map_dtype(self.scalar_type(), dtype)) {
        record_fallback();
        return false;
    }

    clap::DnnBackendType requested;

    if (!select_backend(out, requested) ||
        !backend_override_matches(requested)) {

        record_fallback();
        return false;
    }

    std::vector<std::int64_t> dims;

    dims.reserve(self.dim());

    for (const auto size : self.sizes())
        dims.push_back(size);

    const clap::TensorDesc tensor_desc(
        dtype,
        std::move(dims));

    clap::ActivationDesc activation;
    activation.mode = mode;

    auto& backend = backend_for(requested);

    const void* input_ptr = self.data_ptr();
    void* output_ptr = out.data_ptr();

    if (requested == clap::DnnBackendType::CPU) {

        const auto ctx =
            clap::ExecutionContext::cpu();

        backend.activationForward(
            activation,
            tensor_desc,
            input_ptr,
            tensor_desc,
            output_ptr,
            ctx);

        record_clap_execution(backend.name(), 0);
        return true;
    }

    auto& q = queue_for(out);

#if defined(PTSYCL_CLAP_DNN_CUDA)

    if (requested == clap::DnnBackendType::CUDA) {

        auto* backend_ptr = &backend;

        q.sycl_queue().parasSYCL_enqueue_custom_operation(
            [backend_ptr,
             activation,
             tensor_desc,
             input_ptr,
             output_ptr](sycl::interop_handle ih) {

                auto native_stream =
                    ih.get_native_queue<
                        sycl::backend::cuda>();

                TORCH_CHECK(
                    native_stream != nullptr,
                    "Torch-ParaS CLAP-DNN received a null CUDA stream");

                const int input_device =
                    cuda_device_for_pointer(input_ptr);

                const int output_device =
                    cuda_device_for_pointer(output_ptr);

                TORCH_CHECK(
                    input_device == output_device,
                    "Torch-ParaS CLAP-DNN input/output CUDA device mismatch: input=",
                    input_device,
                    ", output=",
                    output_device);

                CudaDeviceGuard device_guard(input_device);

                auto ctx =
                    clap::ExecutionContext::cuda(
                        reinterpret_cast<void*>(
                            native_stream));

                backend_ptr->activationForward(
                    activation,
                    tensor_desc,
                    input_ptr,
                    tensor_desc,
                    output_ptr,
                    ctx);

                record_clap_execution(backend_ptr->name(), 1);
            });

        return true;
    }

#endif


#if defined(PTSYCL_CLAP_DNN_ROCM)

    if (requested == clap::DnnBackendType::ROCM) {

        auto* backend_ptr = &backend;

        q.sycl_queue().parasSYCL_enqueue_custom_operation(
            [backend_ptr,
             activation,
             tensor_desc,
             input_ptr,
             output_ptr](sycl::interop_handle ih) {

                auto native_stream =
                    ih.get_native_queue<
                        sycl::backend::hip>();

                TORCH_CHECK(
                    native_stream != nullptr,
                    "Torch-ParaS CLAP-DNN received a null HIP stream");

                auto ctx =
                    clap::ExecutionContext::rocm(
                        reinterpret_cast<void*>(
                            native_stream));

                backend_ptr->activationForward(
                    activation,
                    tensor_desc,
                    input_ptr,
                    tensor_desc,
                    output_ptr,
                    ctx);

                record_clap_execution(backend_ptr->name(), 2);
            });

        return true;
    }

#endif

    record_fallback();
    return false;
}


} // namespace

#endif // USE_CLAP_BACKEND


bool try_sigmoid_forward(
    const at::Tensor& self,
    at::Tensor& out)
{
#ifdef USE_CLAP_BACKEND

    return try_activation_forward(
        self,
        out,
        clap::ActivationMode::Sigmoid);

#else

    (void)self;
    (void)out;

    record_fallback();
    return false;

#endif
}


bool try_silu_forward(
    const at::Tensor& self,
    at::Tensor& out)
{
#ifdef USE_CLAP_BACKEND

    return try_activation_forward(
        self,
        out,
        clap::ActivationMode::SiLU);

#else

    (void)self;
    (void)out;

    record_fallback();
    return false;

#endif
}


bool is_enabled() noexcept
{
#ifdef USE_CLAP_BACKEND
    return true;
#else
    return false;
#endif
}


void reset_stats() noexcept
{
    sync_all_devices_noexcept();
    g_total_calls.store(0, std::memory_order_relaxed);
    g_cpu_calls.store(0, std::memory_order_relaxed);
    g_cuda_calls.store(0, std::memory_order_relaxed);
    g_rocm_calls.store(0, std::memory_order_relaxed);
    g_fallback_calls.store(0, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(g_last_backend_mutex);
    g_last_backend.clear();
}


std::int64_t call_count() noexcept
{
    sync_all_devices_noexcept();
    return g_total_calls.load(std::memory_order_relaxed);
}


std::int64_t backend_call_count(const std::string& backend) noexcept
{
    sync_all_devices_noexcept();
    if (backend == "CPU" || backend == "oneDNN") {
        return g_cpu_calls.load(std::memory_order_relaxed);
    }
    if (backend == "CUDA" || backend == "cuDNN") {
        return g_cuda_calls.load(std::memory_order_relaxed);
    }
    if (backend == "ROCM" || backend == "AMD" || backend == "MIOpen") {
        return g_rocm_calls.load(std::memory_order_relaxed);
    }
    return 0;
}


std::int64_t fallback_count() noexcept
{
    sync_all_devices_noexcept();
    return g_fallback_calls.load(std::memory_order_relaxed);
}


std::string last_backend_name()
{
    sync_all_devices_noexcept();
    std::lock_guard<std::mutex> lock(g_last_backend_mutex);
    return g_last_backend;
}


} // namespace ptsycl::clap_dnn
