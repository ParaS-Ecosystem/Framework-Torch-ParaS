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

    m.def("cached_bytes", [](int index) {
        return ptsycl::Context::instance().device(index).pool->cached_bytes();
    });

    m.def("is_bad_fork", [] { return ptsycl::Context::bad_fork(); });
}
