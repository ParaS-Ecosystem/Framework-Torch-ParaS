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

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "compat/paras_compat.h"

namespace ptsycl {

class MemoryPool; // allocator.h

struct DeviceState {
    compat::DeviceInfo          info;
    compat::Queue               queue;
    std::unique_ptr<MemoryPool> pool;
    std::mutex                  rng_mutex;
    std::uint64_t               rng_seed     = 0x5deece66dULL;
    std::uint64_t               rng_sequence = 0;
};

class Context {
public:
    static Context& instance();

    int device_count() const { return static_cast<int>(devices_.size()); }

    DeviceState& device(int index);
    compat::Queue& queue(int index) { return device(index).queue; }

   
    void synchronize(int index);
    void synchronize_all();

 
    std::pair<std::uint64_t, std::uint64_t> rng_draw(int index, std::int64_t items);
    void set_seed(int index, std::uint64_t seed);
    void set_seed_all(std::uint64_t seed);

   
    void empty_cache(int index);
    void empty_cache_all();

    static bool bad_fork();

private:
    Context();
    ~Context() = default;
    Context(const Context&)            = delete;
    Context& operator=(const Context&) = delete;

    std::vector<std::unique_ptr<DeviceState>> devices_;
};

} // namespace ptsycl
