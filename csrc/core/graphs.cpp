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


#include "core/graphs.h"

#include <atomic>
#include <map>
#include <memory>
#include <mutex>

#include <c10/util/Exception.h>

#include "core/allocator.h"
#include "core/context.h"

namespace ptsycl {
namespace graphs {

namespace {

struct Recorded {
    int device;
    std::shared_ptr<compat::Graph> graph;
};

std::mutex& registry_mutex() {
    static std::mutex m;
    return m;
}

std::map<std::int64_t, Recorded>& registry() {
    static auto* r = new std::map<std::int64_t, Recorded>;
    return *r;
}

}

bool supported() { return compat::Queue::graphs_supported(); }

bool capturing(int device) { return Context::instance().queue(device).capturing(); }

std::uint64_t new_pool() {
    static std::atomic<std::uint64_t> next{1};
    return next++;
}

void capture_begin(int device, std::uint64_t pool) {
    auto& state = Context::instance().device(device);
    TORCH_CHECK(state.queue.is_gpu(), "graph capture needs a GPU paras device");
    state.queue.synchronize();
    state.pool->begin_private(pool);
    try {
        state.queue.begin_capture();
    } catch (...) {
        state.pool->end_private();
        throw;
    }
}

std::int64_t capture_end(int device) {
    auto& state = Context::instance().device(device);
    std::shared_ptr<compat::Graph> graph;
    try {
        graph = state.queue.end_capture();
    } catch (...) {
        state.pool->end_private();
        throw;
    }
    state.pool->end_private();
    static std::atomic<std::int64_t> next{1};
    const std::int64_t id = next++;
    std::lock_guard<std::mutex> g(registry_mutex());
    registry()[id] = Recorded{device, std::move(graph)};
    return id;
}

namespace {
std::atomic<std::uint64_t>& replays() {
    static std::atomic<std::uint64_t> n{0};
    return n;
}
}

std::uint64_t replay_epoch() { return replays().load(std::memory_order_relaxed); }

void replay(std::int64_t graph, bool advance_epoch) {
    if (advance_epoch) replays().fetch_add(1, std::memory_order_relaxed);
    Recorded rec;
    {
        std::lock_guard<std::mutex> g(registry_mutex());
        auto it = registry().find(graph);
        TORCH_CHECK(it != registry().end(), "paras graph ", graph, " does not exist");
        rec = it->second;
    }
    Context::instance().queue(rec.device).replay(*rec.graph);
}

void release(std::int64_t graph) {
    std::lock_guard<std::mutex> g(registry_mutex());
    registry().erase(graph);
}

void release_pool(int device, std::uint64_t pool) {
    Context::instance().device(device).pool->release_private(pool);
}

}
}
