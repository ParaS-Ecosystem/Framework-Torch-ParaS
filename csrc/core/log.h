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


#include <atomic>
#include <cstdio>
#include <cstdlib>

namespace ptsycl {
namespace log {

enum Level : int { kWarn = 0, kInfo = 1, kTrace = 2 };

inline int level() {
    static const int lvl = [] {
        const char* env = std::getenv("PTSYCL_LOG");
        return env ? std::atoi(env) : 0;
    }();
    return lvl;
}

#define PTSYCL_WARN(...)                                            \
    do {                                                            \
        std::fprintf(stderr, "[ptsycl warn] " __VA_ARGS__);         \
        std::fprintf(stderr, "\n");                                 \
    } while (0)

#define PTSYCL_INFO(...)                                            \
    do {                                                            \
        if (::ptsycl::log::level() >= ::ptsycl::log::kInfo) {       \
            std::fprintf(stderr, "[ptsycl info] " __VA_ARGS__);     \
            std::fprintf(stderr, "\n");                             \
        }                                                           \
    } while (0)

#define PTSYCL_TRACE_OP(name)                                       \
    do {                                                            \
        if (::ptsycl::log::level() >= ::ptsycl::log::kTrace) {      \
            std::fprintf(stderr, "[ptsycl op] %s\n", (name));       \
        }                                                           \
    } while (0)

} // namespace log
} // namespace ptsycl
