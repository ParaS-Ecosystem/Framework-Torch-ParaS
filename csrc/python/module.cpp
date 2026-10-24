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


// Python extension module (_C). Thin bindings only: all logic lives in C++.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core/common.h"
#include "core/allocator.h"
#include "core/graphs.h"
#include "ops/clap_dnn_adapter.h"
#include "core/telemetry.h"

namespace py = pybind11;

PYBIND11_MODULE(_C, m) {
    m.doc() = "PyTorch backend for the ParaS SYCL compiler";

    m.def("device_count",
          [] { return ptsycl::Context::instance().device_count(); });

    m.def("backend_name", [] { return ptsycl::compat::backend_name(); });

    m.def("device_name", [](int index) {
        return ptsycl::Context::instance().device(index).info.name;
    });

    m.def("is_gpu_device", [](int index) {
        return ptsycl::Context::instance().device(index).info.is_gpu;
    });

    m.def("device_kernel_caps", [](int index) {
        const auto& info = ptsycl::Context::instance().device(index).info;
        py::dict d;
        d["wave_size"] = info.is_gpu ? info.wave_size : 1;
        d["mfma"] = info.is_gpu && info.mfma;
        return d;
    });

    m.def("synchronize", [](int index) {
        py::gil_scoped_release no_gil;
        ptsycl::Context::instance().synchronize(index);
    });

    m.def("synchronize_all", [] {
        py::gil_scoped_release no_gil;
        ptsycl::Context::instance().synchronize_all();
    });

    m.def("manual_seed", [](int index, uint64_t seed) {
        ptsycl::Context::instance().set_seed(index, seed);
    });

    m.def("manual_seed_all", [](uint64_t seed) {
        ptsycl::Context::instance().set_seed_all(seed);
    });

    m.def("empty_cache", [] {
        py::gil_scoped_release no_gil;
        ptsycl::Context::instance().empty_cache_all();
    });

    m.def("allocated_bytes", [](int index) {
        return ptsycl::Context::instance().device(index).pool->allocated_bytes();
    });

    m.def("memory_stats", [](int index) {
        const auto st = ptsycl::Context::instance().device(index).pool->stats();
        py::dict d;
        auto put = [&](const char* name, const c10::CachingAllocator::Stat& s) {
            d[name] = py::make_tuple(s.current, s.peak, s.allocated, s.freed);
        };
        put("allocated_bytes", st.allocated_bytes[0]);
        put("reserved_bytes", st.reserved_bytes[0]);
        put("active_bytes", st.active_bytes[0]);
        put("requested_bytes", st.requested_bytes[0]);
        put("allocation", st.allocation[0]);
        put("active", st.active[0]);
        put("segment", st.segment[0]);
        return d;
    });
    m.def("reset_peak_stats", [](int index) {
        ptsycl::Context::instance().device(index).pool->reset_peak_stats();
    });
    m.def("reset_accumulated_stats", [](int index) {
        ptsycl::Context::instance().device(index).pool->reset_accumulated_stats();
    });

    m.def("cached_bytes", [](int index) {
        return ptsycl::Context::instance().device(index).pool->cached_bytes();
    });

    m.def("is_bad_fork", [] { return ptsycl::Context::bad_fork(); });

    m.def("_graphs_supported", [] { return ptsycl::graphs::supported(); });
    m.def("_graph_capturing", [](int index) { return ptsycl::graphs::capturing(index); });
    m.def("_graph_pool_new", [] { return ptsycl::graphs::new_pool(); });
    m.def("_graph_capture_begin", [](int index, uint64_t pool) {
        py::gil_scoped_release no_gil;
        ptsycl::graphs::capture_begin(index, pool);
    });
    m.def("_graph_capture_end", [](int index) { return ptsycl::graphs::capture_end(index); });
    m.def("_graph_replay", [](int64_t graph, bool advance_epoch) {
        py::gil_scoped_release no_gil;
        ptsycl::graphs::replay(graph, advance_epoch);
    }, py::arg("graph"), py::arg("advance_epoch") = true);
    m.def("_graph_release", [](int64_t graph) { ptsycl::graphs::release(graph); });
    m.def("_graph_pool_release", [](int index, uint64_t pool) {
        ptsycl::graphs::release_pool(index, pool);
    });

    m.def("clap_dnn_enabled", [] {
        return ptsycl::clap_dnn::is_enabled();
    });

    m.def("clap_dnn_activation_uses_vendor", [](int device_index, const std::string& dtype) {
        return ptsycl::clap_dnn::activation_uses_vendor(device_index, dtype);
    });

    m.def("clap_dnn_reset_stats", [] {
        py::gil_scoped_release no_gil;
        ptsycl::clap_dnn::reset_stats();
    });

    m.def("clap_dnn_call_count", [] {
        py::gil_scoped_release no_gil;
        return ptsycl::clap_dnn::call_count();
    });

    m.def("clap_dnn_backend_call_count", [](const std::string& backend) {
        py::gil_scoped_release no_gil;
        return ptsycl::clap_dnn::backend_call_count(backend);
    });

    m.def("clap_dnn_fallback_count", [] {
        py::gil_scoped_release no_gil;
        return ptsycl::clap_dnn::fallback_count();
    });

    m.def("clap_dnn_last_backend", [] {
        py::gil_scoped_release no_gil;
        return ptsycl::clap_dnn::last_backend_name();
    });

    m.def("clap_blas_enabled", [] {
#ifdef PTSYCL_USE_CLAP_BLAS
        return true;
#else
        return false;
#endif
    });

    m.def("runtime_stats", [] {
        const auto s = ptsycl::telemetry::snapshot();
        py::dict d;
        d["cpu_fallbacks"] = s.fallbacks;
        d["device_to_host_copies"] = s.d2h_count;
        d["device_to_host_bytes"] = s.d2h_bytes;
        d["host_to_device_copies"] = s.h2d_count;
        d["host_to_device_bytes"] = s.h2d_bytes;
        d["scalar_reads"] = s.scalar_reads;
        return d;
    });

    m.def("reset_runtime_stats", [] { ptsycl::telemetry::reset(); });

    m.def("strict", [] { return ptsycl::telemetry::strict(); });

    m.def("set_strict", [](bool enabled) {
        ptsycl::telemetry::set_strict(enabled);
    });
}
