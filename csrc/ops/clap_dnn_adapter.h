// -----------------------------------------------------------------------------
// Copyright (c) 2026 Centre for Development of Advanced Computing (C-DAC)
//
// This file is part of Torch_ParaS, a component of the ParaS Ecosystem.
//
// This library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License (LGPL)
// version 3 as published by the Free Software Foundation.
// -----------------------------------------------------------------------------

#pragma once

#include <ATen/ATen.h>

namespace ptsycl::clap_dnn {

bool try_sigmoid_forward(const at::Tensor& self, at::Tensor& out);
bool try_silu_forward(const at::Tensor& self, at::Tensor& out);

} // namespace ptsycl::clap_dnn
