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

#ifdef USE_CLAP_BACKEND

#include <clap/dnn_factory.hpp>
#include <clap/dnn_types.hpp>
#include <clap/idnn_backend.hpp>

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#endif


namespace ptsycl::clap_dnn {

#ifdef USE_CLAP_BACKEND

namespace {


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


bool backend_override_matches(
    clap::DnnBackendType requested)
{
    const char* value = std::getenv("CLAP_BACKEND");

    if (value == nullptr || *value == '\0')
        return true;

    const std::string backend(value);

    switch (requested) {

        case clap::DnnBackendType::CPU:
            return backend == "CPU";

        case clap::DnnBackendType::CUDA:
            return backend == "CUDA";

        case clap::DnnBackendType::ROCM:
            return backend == "ROCM" ||
                   backend == "AMD";

        case clap::DnnBackendType::GPU:
            return backend == "GPU";
    }

    return false;
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

    switch (requested) {

        case clap::DnnBackendType::CPU:
            slot = &cache.cpu;
            break;

        case clap::DnnBackendType::CUDA:
            slot = &cache.cuda;
            break;

        case clap::DnnBackendType::ROCM:
            slot = &cache.rocm;
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
    /*
     * CLAP receives the actual Torch-ParaS device pointers.
     *
     * No:
     *   GPU -> CPU copy
     *   CPU temporary
     *   CPU -> GPU copy
     *
     * The vendor operation is submitted onto the same native stream
     * backing the ParaS SYCL queue.
     */

    if (self.numel() == 0 ||
        self.numel() != out.numel() ||
        self.sizes() != out.sizes() ||
        self.device() != out.device() ||
        self.scalar_type() != out.scalar_type() ||
        !self.is_contiguous() ||
        !out.is_contiguous() ||
        self.dim() == 0) {

        return false;
    }

    clap::DataType dtype;

    if (!map_dtype(self.scalar_type(), dtype))
        return false;

    clap::DnnBackendType requested;

    if (!select_backend(out, requested) ||
        !backend_override_matches(requested)) {

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

    /*
     * CPU ParaS path.
     */
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
            });

        return true;
    }

#endif

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

    return false;

#endif
}


} // namespace ptsycl::clap_dnn
