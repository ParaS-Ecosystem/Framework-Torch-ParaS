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

#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/util/ApproximateClock.h>
#include <torch/csrc/profiler/stubs/base.h>

#include "core/context.h"

namespace ptsycl {
namespace {

using torch::profiler::impl::ProfilerVoidEventStub;

c10::DeviceIndex current_index() {
    return c10::impl::getDeviceGuardImpl(c10::DeviceType::PrivateUse1)->getDevice().index();
}

struct ParasProfilerStubs final : torch::profiler::impl::ProfilerStubs {
    void record(c10::DeviceIndex* device, ProfilerVoidEventStub* event, int64_t* cpu_ns) const override {
        const c10::DeviceIndex index = current_index();
        if (device != nullptr) *device = index;
        void* ev = nullptr;
        Context::instance().current_queue(index).record_event(&ev, true);
        *event = std::shared_ptr<void>(ev, [](void* e) {
            if (e != nullptr) compat::Queue::event_destroy(e);
        });
        if (cpu_ns != nullptr) *cpu_ns = c10::getTime();
    }

    float elapsed(const ProfilerVoidEventStub* start, const ProfilerVoidEventStub* end) const override {
        void* s = start->get();
        void* e = end->get();
        if (s == nullptr || e == nullptr) return 0.f;
        compat::Queue::event_synchronize(e);
        return compat::Queue::event_elapsed_ms(s, e) * 1000.f;
    }

    void mark(const char*) const override {}
    void rangePush(const char*) const override {}
    void rangePop() const override {}
    bool enabled() const override { return true; }

    void onEachDevice(std::function<void(int)> op) const override {
        auto* impl = c10::impl::getDeviceGuardImpl(c10::DeviceType::PrivateUse1);
        const c10::Device prev = impl->getDevice();
        for (int i = 0; i < Context::instance().device_count(); ++i) {
            impl->setDevice(c10::Device(c10::DeviceType::PrivateUse1, static_cast<c10::DeviceIndex>(i)));
            op(i);
        }
        impl->setDevice(prev);
    }

    void synchronize() const override { Context::instance().synchronize_all(); }
};

struct Registrar {
    Registrar() {
        static ParasProfilerStubs stubs;
        torch::profiler::impl::registerPrivateUse1Methods(&stubs);
    }
} registrar;

}
}
