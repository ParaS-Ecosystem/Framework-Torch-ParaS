// -----------------------------------------------------------------------------
// Copyright (c) 2026 Centre for Development of Advanced Computing (C-DAC)
//
// This file is part of Torch_ParaS, a component of the ParaS Ecosystem
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

#include <torch/version.h>

#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>

#include "core/common.h"        // kParasDevice, data_ptr<T>, queue_for, is_paras_tensor
#include "core/kernel_utils.h"  // StridedSpec, make_spec, launch_flat, launch_strided_store, atomic_add
#include "core/log.h"           // PTSYCL_TRACE_OP, PTSYCL_WARN, PTSYCL_INFO

#if TORCH_VERSION_MAJOR < 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR < 4)
#error "ptsycl requires PyTorch >= 2.4"
#endif

namespace ptsycl {
namespace {

// Default device for any paras tensor created without an explicit index
// (e.g. torch.device('paras')). Index 0 here means "the first entry
// enumerate_devices() returns" -- since that ordering is now GPU-first
// (CPU appended last as a fallback), this defaults to the first GPU, not
// the host. See paras_compat.cpp::enumerate_devices() for the ordering
// contract this relies on.
thread_local c10::Device tl_current_device{kParasDevice, 0};

struct ParasGuardImpl final : public c10::impl::DeviceGuardImplInterface {
    c10::DeviceType type() const override { return kParasDevice; }

    c10::Device exchangeDevice(c10::Device d) const override {
        c10::Device prev = tl_current_device;
        tl_current_device = d;
        return prev;
    }

    c10::Device getDevice() const override { return tl_current_device; }

    void setDevice(c10::Device d) const override { tl_current_device = d; }

    void uncheckedSetDevice(c10::Device d) const noexcept override {
        tl_current_device = d;
    }

    // There is exactly one c10::Stream per device -- Context maps each
    // device index to a single in-order compat::Queue, and nothing in
    // this backend creates a second queue for the same device. So a
    // fixed stream id of 0 always denotes "the queue for this device";
    // there's no real multi-stream support to select between.
    c10::Stream getStream(c10::Device d) const noexcept override {
        return c10::Stream(c10::Stream::UNSAFE, d, 0);
    }

    c10::Stream getDefaultStream(c10::Device d) const override {
        return getStream(d);
    }

    // exchangeStream() ignores its argument and returns the current
    // device's one stream. Safe only because of the single-queue-per-
    // device invariant above -- MemoryPool::release() (core/allocator.cpp)
    // recycles freed GPU memory without waiting for device completion,
    // relying on the queue's in-order guarantee (paras_compat.cpp:
    // sycl::queue constructed with sycl::property::queue::in_order{}) to
    // keep any reuse strictly ordered after prior work on that queue. If
    // real multi-stream support is ever added (Queue::stream() support
    // was already reverted once per the push-package audit -- see that
    // note), this method, queryStream() below, and the allocator all need
    // to change together: the allocator would need actual stream-
    // completion tracking before it could be safe to hand freed memory
    // to a different stream than the one that last used it.
    c10::Stream exchangeStream(c10::Stream) const noexcept override {
        return getStream(tl_current_device);
    }

    c10::DeviceIndex deviceCount() const noexcept override {
        try {
            return static_cast<c10::DeviceIndex>(Context::instance().device_count());
        } catch (...) {
            return 0;
        }
    }

    // Always reports the stream as done. Correct today only because
    // there's no async stream state to actually query -- see the
    // single-in-order-queue note on exchangeStream() above. If
    // multi-stream support is added, this needs a real completion check.
    bool queryStream(const c10::Stream&) const override { return true; }

    void synchronizeStream(const c10::Stream& stream) const override {
        Context::instance().synchronize(
            static_cast<int>(stream.device().index()));
    }
};

ParasGuardImpl guard_impl;
c10::impl::DeviceGuardImplRegistrar guard_registrar(kParasDevice, &guard_impl);

struct ParasHooks final : public at::PrivateUse1HooksInterface {
    bool hasPrimaryContext(c10::DeviceIndex device_index) const override {
        return device_index >= 0 &&
               device_index < Context::instance().device_count();
    }
};

struct HooksRegistrar {
    ParasHooks hooks;
    HooksRegistrar() { at::RegisterPrivateUse1HooksInterface(&hooks); }
};
HooksRegistrar hooks_registrar;

} // namespace
} // namespace ptsycl