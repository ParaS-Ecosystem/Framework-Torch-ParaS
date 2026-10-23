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


#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <map>
#include <mutex>

#include <c10/core/Allocator.h>
#include <c10/core/CachingDeviceAllocator.h>
#include <c10/core/Device.h>

namespace ptsycl {

namespace compat { class Queue; }

class MemoryPool {
public:
    MemoryPool(compat::Queue& queue, int device_index);
    ~MemoryPool();

    void* acquire(std::size_t nbytes);
    void  release(void* ptr);
    void  clear();

    void begin_private(std::uint64_t pool);
    void end_private();
    void release_private(std::uint64_t pool);

    std::size_t allocated_bytes() const { return allocated_; }
    std::size_t cached_bytes()    const { return cached_; }

    c10::CachingDeviceAllocator::DeviceStats stats() const;
    void reset_peak_stats();
    void reset_accumulated_stats();

private:
    struct Live {
        std::size_t   rounded;
        std::uint64_t pool;
    };
    struct Private {
        std::map<std::size_t, std::list<void*>> free_lists;
        std::size_t live     = 0;
        bool        released = false;
    };

    static std::size_t round_size(std::size_t v);
    void sync_before_free() noexcept;

    compat::Queue& queue_;
    int            device_index_;
    bool           caching_enabled_;
    bool           debug_;

    mutable std::mutex                      mutex_;
    std::map<std::size_t, std::list<void*>> free_lists_;
    std::map<void*, Live>                   live_;
    std::map<std::uint64_t, Private>        private_;
    std::uint64_t                           active_private_ = 0;
    std::size_t                             allocated_ = 0;
    std::size_t                             cached_    = 0;
    c10::CachingAllocator::Stat                               live_bytes_, live_blocks_;
    c10::CachingAllocator::Stat                               reserved_bytes_, segments_;
};


struct ParasAllocator final : public c10::DeviceAllocator {
    c10::DataPtr allocate(size_t nbytes) override;
    c10::DeleterFnPtr raw_deleter() const override;
    void copy_data(void* dest, const void* src, std::size_t count) const override;

    bool initialized() override { return true; }
    void emptyCache(c10::MempoolId_t mempool_id = {0, 0}) override;
    void recordStream(const c10::DataPtr&, c10::Stream) override {}
    c10::CachingDeviceAllocator::DeviceStats getDeviceStats(c10::DeviceIndex device) override;
    void resetAccumulatedStats(c10::DeviceIndex device) override;
    void resetPeakStats(c10::DeviceIndex device) override;
};

ParasAllocator& allocator_instance();

} // namespace ptsycl
