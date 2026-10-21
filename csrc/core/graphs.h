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

namespace ptsycl {
namespace graphs {

bool supported();
bool capturing(int device);

std::uint64_t new_pool();

void capture_begin(int device, std::uint64_t pool);
std::int64_t capture_end(int device);

void replay(std::int64_t graph);
void release(std::int64_t graph);
void release_pool(int device, std::uint64_t pool);

}
}
