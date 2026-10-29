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
#include <map>
#include <string>

namespace ptsycl {
namespace telemetry {

enum class Transfer { DeviceToHost, HostToDevice };

void record_fallback(const std::string& op);
void record_transfer(Transfer direction, std::size_t bytes);
void record_scalar_read();

struct Snapshot {
    std::map<std::string, std::uint64_t> fallbacks;
    std::uint64_t d2h_count = 0;
    std::uint64_t d2h_bytes = 0;
    std::uint64_t h2d_count = 0;
    std::uint64_t h2d_bytes = 0;
    std::uint64_t scalar_reads = 0;
};

Snapshot snapshot();
void reset();

bool strict();
void set_strict(bool enabled);

}
}
