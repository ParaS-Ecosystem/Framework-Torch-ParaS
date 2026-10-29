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

#include "core/telemetry.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace ptsycl {
namespace telemetry {
namespace {

std::mutex fallback_mutex;
std::map<std::string, std::uint64_t> fallback_counts;

std::atomic<std::uint64_t> d2h_count{0};
std::atomic<std::uint64_t> d2h_bytes{0};
std::atomic<std::uint64_t> h2d_count{0};
std::atomic<std::uint64_t> h2d_bytes{0};
std::atomic<std::uint64_t> scalar_reads{0};

std::atomic<bool>& strict_flag() {
    static std::atomic<bool> flag{[] {
        const char* env = std::getenv("PTSYCL_STRICT");
        return env != nullptr && std::strcmp(env, "0") != 0 && env[0] != '\0';
    }()};
    return flag;
}

}

void record_fallback(const std::string& op) {
    std::lock_guard<std::mutex> lock(fallback_mutex);
    ++fallback_counts[op];
}

void record_transfer(Transfer direction, std::size_t bytes) {
    if (direction == Transfer::DeviceToHost) {
        d2h_count.fetch_add(1, std::memory_order_relaxed);
        d2h_bytes.fetch_add(bytes, std::memory_order_relaxed);
    } else {
        h2d_count.fetch_add(1, std::memory_order_relaxed);
        h2d_bytes.fetch_add(bytes, std::memory_order_relaxed);
    }
}

void record_scalar_read() {
    scalar_reads.fetch_add(1, std::memory_order_relaxed);
}

Snapshot snapshot() {
    Snapshot s;
    {
        std::lock_guard<std::mutex> lock(fallback_mutex);
        s.fallbacks = fallback_counts;
    }
    s.d2h_count = d2h_count.load(std::memory_order_relaxed);
    s.d2h_bytes = d2h_bytes.load(std::memory_order_relaxed);
    s.h2d_count = h2d_count.load(std::memory_order_relaxed);
    s.h2d_bytes = h2d_bytes.load(std::memory_order_relaxed);
    s.scalar_reads = scalar_reads.load(std::memory_order_relaxed);
    return s;
}

void reset() {
    {
        std::lock_guard<std::mutex> lock(fallback_mutex);
        fallback_counts.clear();
    }
    d2h_count = 0;
    d2h_bytes = 0;
    h2d_count = 0;
    h2d_bytes = 0;
    scalar_reads = 0;
}

bool strict() { return strict_flag().load(std::memory_order_relaxed); }

void set_strict(bool enabled) {
    strict_flag().store(enabled, std::memory_order_relaxed);
}

}
}
