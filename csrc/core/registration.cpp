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

#include "core/allocator.h"
#include "core/common.h"

#if TORCH_VERSION_MAJOR < 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR < 4)
#error "ptsycl requires PyTorch >= 2.4"
#endif

namespace ptsycl {
namespace {

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

    static c10::Stream stream_of(c10::Device d, int id) { return c10::Stream(c10::Stream::UNSAFE, d, id); }
    static int index_of(c10::Device d) { return d.has_index() ? static_cast<int>(d.index()) : static_cast<int>(tl_current_device.index()); }

    c10::Stream getStream(c10::Device d) const noexcept override {
        return stream_of(d, Context::current_stream(index_of(d)));
    }

    c10::Stream getDefaultStream(c10::Device d) const override { return stream_of(d, 0); }

    c10::Stream getStreamFromGlobalPool(c10::Device d, bool ) const override {
        return stream_of(d, Context::instance().pool_stream(index_of(d)));
    }

    c10::Stream getNewStream(c10::Device d, int ) const override {
        return stream_of(d, Context::instance().new_stream(index_of(d)));
    }

    c10::Stream exchangeStream(c10::Stream s) const noexcept override {
        const int index = index_of(s.device());
        const c10::Stream prev = stream_of(s.device(), Context::current_stream(index));
        try {
            Context::set_current_stream(index, static_cast<int>(s.id()));
        } catch (...) {
        }
        return prev;
    }

    static compat::Queue& queue_of(const c10::Stream& s) {
        return Context::instance().stream_queue(index_of(s.device()), static_cast<int>(s.id()));
    }

    void record(void** event, const c10::Stream& stream, const c10::DeviceIndex ,
                const c10::EventFlag flag) const override {
        queue_of(stream).record_event(event, flag == c10::EventFlag::BACKEND_DEFAULT);
    }

    void block(void* event, const c10::Stream& stream) const override { queue_of(stream).wait_event(event); }

    bool queryEvent(void* event) const override { return compat::Queue::event_query(event); }

    void destroyEvent(void* event, const c10::DeviceIndex ) const noexcept override {
        compat::Queue::event_destroy(event);
    }

    void synchronizeEvent(void* event) const override { compat::Queue::event_synchronize(event); }

    double elapsedTime(void* start, void* end, const c10::DeviceIndex ) const override {
        return static_cast<double>(compat::Queue::event_elapsed_ms(start, end));
    }

    void synchronizeDevice(const c10::DeviceIndex device_index) const override {
        Context::instance().synchronize(static_cast<int>(device_index));
    }

    void recordDataPtrOnStream(const c10::DataPtr& data, const c10::Stream& stream) const override {
        if (data.get() != nullptr)
            Context::instance().device(index_of(stream.device())).pool->record_stream(data.get(), static_cast<int>(stream.id()));
    }

    c10::DeviceIndex deviceCount() const noexcept override {
        try {
            return static_cast<c10::DeviceIndex>(Context::instance().device_count());
        } catch (...) {
            return 0;
        }
    }

    bool queryStream(const c10::Stream& stream) const override { return queue_of(stream).query(); }

    void synchronizeStream(const c10::Stream& stream) const override { queue_of(stream).synchronize(); }
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
