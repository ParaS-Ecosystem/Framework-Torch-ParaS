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


#include "core/allocator.h"

#include <cstdlib>

#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/util/Exception.h>

#include "compat/paras_compat.h"
#include "core/context.h"
#include "core/log.h"

namespace ptsycl {

namespace {
constexpr c10::DeviceType kParasDevice = c10::DeviceType::PrivateUse1;

bool env_flag(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && std::atoi(v) != 0;
}
} // namespace

// -----------------------------------------------------------------------------
// MemoryPool
// -----------------------------------------------------------------------------
MemoryPool::MemoryPool(compat::Queue& queue, int device_index)
    : queue_(queue),
      device_index_(device_index),
      caching_enabled_(!env_flag("PTSYCL_NO_CACHE")),
      debug_(env_flag("PTSYCL_DEBUG_ALLOC")) {}

MemoryPool::~MemoryPool() { clear(); }

std::size_t MemoryPool::round_size(std::size_t v) {
    constexpr std::size_t kMin = 256;
    if (v <= kMin) return kMin;
    std::size_t r = kMin;
    while (r < v) r <<= 1;
    return r;
}

void* MemoryPool::acquire(std::size_t nbytes) {
    const std::size_t rounded = round_size(nbytes);
    std::uint64_t pool = 0;
    {
        std::lock_guard<std::mutex> g(mutex_);
        pool = active_private_;
        auto& lists = pool ? private_[pool].free_lists : free_lists_;
        auto it = lists.find(rounded);
        if (it != lists.end() && !it->second.empty()) {
            void* p = it->second.front();
            it->second.pop_front();
            if (pool) {
                ++private_[pool].live;
            } else {
                cached_ -= rounded;
            }
            live_[p] = Live{rounded, pool};
            if (debug_)
                PTSYCL_WARN("pool[%d] reuse %zu bytes @%p", device_index_, rounded, p);
            return p;
        }
    }
    void* p = queue_.alloc(rounded);
    {
        std::lock_guard<std::mutex> g(mutex_);
        live_[p] = Live{rounded, pool};
        if (pool) ++private_[pool].live;
        allocated_ += rounded;
    }
    if (debug_)
        PTSYCL_WARN("pool[%d] alloc %zu bytes @%p", device_index_, rounded, p);
    return p;
}

void MemoryPool::release(void* ptr) {
    if (ptr == nullptr) return;
    std::size_t rounded = 0;
    {
        std::lock_guard<std::mutex> g(mutex_);
        auto it = live_.find(ptr);
        TORCH_CHECK(it != live_.end(),
                    "paras allocator: release of unknown pointer");
        rounded = it->second.rounded;
        const std::uint64_t pool = it->second.pool;
        live_.erase(it);
        if (pool) {
            auto pit = private_.find(pool);
            Private& pr = pit->second;
            --pr.live;
            if (!pr.released) {
                pr.free_lists[rounded].push_back(ptr);
                return;
            }
            if (pr.live == 0) private_.erase(pit);
            allocated_ -= rounded;
        } else if (caching_enabled_) {
            free_lists_[rounded].push_back(ptr);
            cached_ += rounded;
            if (debug_)
                PTSYCL_WARN("pool[%d] cache %zu bytes @%p", device_index_, rounded, ptr);
            return;
        } else {
            allocated_ -= rounded;
        }
    }
    if (debug_)
        PTSYCL_WARN("pool[%d] free %zu bytes @%p", device_index_, rounded, ptr);
    queue_.dealloc(ptr);
}

void MemoryPool::clear() {
    std::lock_guard<std::mutex> g(mutex_);
    for (auto& [size, list] : free_lists_) {
        for (void* p : list) {
            queue_.dealloc(p);
            allocated_ -= size;
        }
        list.clear();
    }
    free_lists_.clear();
    cached_ = 0;
}

void MemoryPool::begin_private(std::uint64_t pool) {
    TORCH_CHECK(pool != 0, "paras allocator: graph pool 0 is the shared cache");
    std::lock_guard<std::mutex> g(mutex_);
    TORCH_CHECK(active_private_ == 0, "paras allocator: a graph pool is already active");
    auto& pr = private_[pool];
    TORCH_CHECK(!pr.released, "paras allocator: graph pool was released");
    active_private_ = pool;
}

void MemoryPool::end_private() {
    std::lock_guard<std::mutex> g(mutex_);
    active_private_ = 0;
}

void MemoryPool::release_private(std::uint64_t pool) {
    std::lock_guard<std::mutex> g(mutex_);
    auto it = private_.find(pool);
    if (it == private_.end()) return;
    TORCH_CHECK(active_private_ != pool, "paras allocator: release of the active graph pool");
    for (auto& [size, list] : it->second.free_lists) {
        for (void* p : list) {
            queue_.dealloc(p);
            allocated_ -= size;
        }
    }
    it->second.free_lists.clear();
    it->second.released = true;
    if (it->second.live == 0) private_.erase(it);
}

// -----------------------------------------------------------------------------
// c10::Allocator binding
// -----------------------------------------------------------------------------
namespace {

struct AllocRecord {
    void* ptr;
    int   device;
};

void delete_alloc_record(void* opaque) {
    if (opaque == nullptr) return;
    auto* rec = static_cast<AllocRecord*>(opaque);
    Context::instance().device(rec->device).pool->release(rec->ptr);
    delete rec;
}

int current_device_index() {
    const c10::impl::DeviceGuardImplInterface* impl =
        c10::impl::getDeviceGuardImpl(kParasDevice);
    return static_cast<int>(impl->getDevice().index());
}

} // namespace

c10::DataPtr ParasAllocator::allocate(size_t nbytes) {
    const int dev = current_device_index();
    auto& state = Context::instance().device(dev);
    void* p = state.pool->acquire(nbytes);
    auto* rec = new AllocRecord{p, dev};
    return c10::DataPtr(p, rec, &delete_alloc_record,
                        c10::Device(kParasDevice, dev));
}

c10::DeleterFnPtr ParasAllocator::raw_deleter() const {
    return &delete_alloc_record;
}

void ParasAllocator::copy_data(void* dest, const void* src,
                               std::size_t count) const {
    const int dev = current_device_index();
    Context::instance().queue(dev).copy(dest, src, count, /*blocking=*/true);
}

ParasAllocator& allocator_instance() {
    static ParasAllocator alloc;
    return alloc;
}

REGISTER_ALLOCATOR(kParasDevice, &allocator_instance());

} // namespace ptsycl
