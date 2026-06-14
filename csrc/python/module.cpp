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
