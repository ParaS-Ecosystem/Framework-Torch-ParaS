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


#include "core/context.h"

#include <pthread.h>

#include <c10/util/CallOnce.h>
#include <c10/util/Exception.h>

#include "core/allocator.h"
#include "core/log.h"

namespace ptsycl {

namespace {
bool g_bad_fork = false;
void forked_child() { g_bad_fork = true; }
void poison_fork() {
    static c10::once_flag flag;
    c10::call_once(flag, [] { pthread_atfork(nullptr, nullptr, forked_child); });
}
} // namespace

Context::Context() {
    poison_fork();
    auto infos = compat::enumerate_devices();
    devices_.reserve(infos.size());
    for (auto& info : infos) {
        auto state = std::make_unique<DeviceState>();
        state->info = info;
        state->queue.init(info);
        state->streams[0].store(&state->queue);
        state->pool = std::make_unique<MemoryPool>(
            state->queue, static_cast<int>(devices_.size()));
        PTSYCL_INFO("device %zu: %s%s", devices_.size(), info.name.c_str(),
                    info.is_gpu ? " [gpu]" : " [cpu]");
        devices_.push_back(std::move(state));
    }
}

Context& Context::instance() {
    static Context ctx;
    return ctx;
}

DeviceState& Context::device(int index) {
    TORCH_CHECK(index >= 0 && index < device_count(),
                "paras: invalid device index ", index,
                " (device count: ", device_count(), ")");
    return *devices_[index];
}

namespace {
thread_local std::array<std::int16_t, 256> tl_current_stream{};
}

int Context::current_stream(int index) {
    return index >= 0 && index < static_cast<int>(tl_current_stream.size()) ? tl_current_stream[index] : 0;
}

void Context::set_current_stream(int index, int id) {
    TORCH_CHECK(index >= 0 && index < static_cast<int>(tl_current_stream.size()) && id >= 0 && id < kMaxStreams,
                "paras: invalid stream ", id, " of device ", index);
    tl_current_stream[index] = static_cast<std::int16_t>(id);
}

compat::Queue& Context::stream_queue(int index, int id) {
    auto& d = device(index);
    TORCH_CHECK(id >= 0 && id < kMaxStreams, "paras: invalid stream id ", id);
    if (compat::Queue* q = d.streams[id].load(std::memory_order_acquire)) return *q;
    std::lock_guard<std::mutex> g(d.streams_mutex);
    if (compat::Queue* q = d.streams[id].load(std::memory_order_acquire)) return *q;
    auto q = std::make_unique<compat::Queue>();
    q->init(d.info);
    compat::Queue* raw = q.get();
    d.owned[id] = std::move(q);
    d.streams[id].store(raw, std::memory_order_release);
    if (id + 1 > d.created_streams) d.created_streams = id + 1;
    return *raw;
}

int Context::pool_stream(int index) {
    auto& d = device(index);
    const int id = 1 + d.next_pool_stream.fetch_add(1) % kPoolStreams;
    stream_queue(index, id);
    return id;
}

int Context::new_stream(int index) {
    auto& d = device(index);
    int id;
    {
        std::lock_guard<std::mutex> g(d.streams_mutex);
        id = d.created_streams > kPoolStreams ? d.created_streams : kPoolStreams + 1;
        TORCH_CHECK(id < kMaxStreams, "paras: more than ", kMaxStreams, " streams on device ", index);
        d.created_streams = id + 1;
    }
    stream_queue(index, id);
    return id;
}

int Context::stream_count(int index) {
    auto& d = device(index);
    std::lock_guard<std::mutex> g(d.streams_mutex);
    return d.created_streams;
}

void Context::synchronize(int index) {
    auto& d = device(index);
    for (auto& s : d.streams)
        if (compat::Queue* q = s.load(std::memory_order_acquire)) q->synchronize();
}

void Context::synchronize_all() {
    for (int i = 0; i < device_count(); ++i) synchronize(i);
}

std::pair<std::uint64_t, std::uint64_t>
Context::rng_draw(int index, std::int64_t items) {
    auto& d = device(index);
    std::lock_guard<std::mutex> g(d.rng_mutex);
    std::uint64_t seq = d.rng_sequence;
    // Philox2x32 yields two 32-bit values per counter tick.
    d.rng_sequence += static_cast<std::uint64_t>((items + 1) / 2) + 1;
    return {d.rng_seed, seq};
}

void Context::set_seed(int index, std::uint64_t seed) {
    auto& d = device(index);
    std::lock_guard<std::mutex> g(d.rng_mutex);
    d.rng_seed     = seed;
    d.rng_sequence = 0;
}

void Context::set_seed_all(std::uint64_t seed) {
    for (int i = 0; i < device_count(); ++i) set_seed(i, seed);
}

void Context::empty_cache(int index) {
    auto& d = device(index);
    synchronize(index);
    d.pool->clear();
}

void Context::empty_cache_all() {
    for (int i = 0; i < device_count(); ++i) empty_cache(i);
}

bool Context::bad_fork() {
    instance();
    return g_bad_fork;
}

} // namespace ptsycl
