// -----------------------------------------------------------------------------
// Copyright (c) 2026 Centre for Development of Advanced Computing (C-DAC)
//
// This file is part of Torch_ParaS, a component of the ParaS Ecosystem.
//
// This library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License (LGPL)
// version 3 as published by the Free Software Foundation.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this library. If not, see <https://www.gnu.org/licenses/>.
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
#endif

namespace ptsycl::clap_dnn {

#ifdef USE_CLAP_BACKEND
namespace {

bool select_backend(
    at::Tensor& tensor,
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

bool backend_override_matches(clap::DnnBackendType requested)
{
    const char* value = std::getenv("CLAP_BACKEND");
    if (value == nullptr || *value == '\0') {
        return true;
    }

    const std::string backend(value);

    switch (requested) {
        case clap::DnnBackendType::CPU:
            return backend == "CPU";
        case clap::DnnBackendType::CUDA:
            return backend == "CUDA";
        case clap::DnnBackendType::ROCM:
            return backend == "ROCM" || backend == "AMD";
        case clap::DnnBackendType::GPU:
            return backend == "GPU";
    }

    return false;
}

clap::IDnnBackend& backend_for(clap::DnnBackendType requested)
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
            TORCH_CHECK(false, "Torch-ParaS CLAP-DNN requires an explicit backend");
    }

    if (!*slot) {
        *slot = clap::DnnFactory::create(requested);
    }

    TORCH_CHECK(*slot, "CLAP-DNN backend creation failed");
    return **slot;
}

} // namespace
#endif

bool try_sigmoid_forward(const at::Tensor& self, at::Tensor& out)
{
#ifdef USE_CLAP_BACKEND
    if (self.scalar_type() != at::kFloat ||
        out.scalar_type() != at::kFloat ||
        self.numel() == 0 ||
        self.numel() != out.numel() ||
        self.sizes() != out.sizes() ||
        self.device() != out.device() ||
        !out.is_contiguous()) {
        return false;
    }

    clap::DnnBackendType requested;
    if (!select_backend(out, requested) ||
        !backend_override_matches(requested)) {
        return false;
    }

    // Current CLAP-DNN execution accepts host pointers. Stage through CPU
    // until CLAP-DNN exposes direct device-pointer and stream execution.
    at::Tensor cpu_input =
        self.contiguous().to(c10::Device(c10::kCPU));
    at::Tensor cpu_output = at::empty_like(cpu_input);

    const clap::TensorDesc tensor_desc{
        1, 1, 1, cpu_input.numel()
    };

    clap::ActivationDesc activation;
    activation.mode = clap::ActivationMode::Sigmoid;

    backend_for(requested).activationForward(
        activation,
        tensor_desc,
        cpu_input.data_ptr<float>(),
        tensor_desc,
        cpu_output.data_ptr<float>());

    auto& q = queue_for(out);
    q.copy(
        out.data_ptr(),
        cpu_output.data_ptr(),
        static_cast<std::size_t>(cpu_output.nbytes()),
        true);

    return true;
#else
    (void)self;
    (void)out;
    return false;
#endif
}

} // namespace ptsycl::clap_dnn
