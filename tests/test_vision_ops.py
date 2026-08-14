# -----------------------------------------------------------------------------
# Copyright (c) 2026 Centre for Development of Advanced Computing (C-DAC)
#
# This file is part of Torch_ParaS, a component of the ParaS Ecosystem
#
# This library is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License (LGPL)
# version 3 as published by the Free Software Foundation.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library. If not, see <https://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------

import torch
import torch.nn.functional as F

from common import assert_close


def test_adaptive_avg_pool2d(device):
    x = torch.randn(2, 3, 8, 10, dtype=torch.float32)
    out = F.adaptive_avg_pool2d(x.to(device), (1, 1))
    assert_close(out, F.adaptive_avg_pool2d(x, (1, 1)),
                 "adaptive_avg_pool2d")


def test_linear(device):
    x = torch.randn(4, 8, dtype=torch.float32)
    weight = torch.randn(6, 8, dtype=torch.float32)
    bias = torch.randn(6, dtype=torch.float32)
    out = F.linear(x.to(device), weight.to(device), bias.to(device))
    assert_close(out, F.linear(x, weight, bias), "linear")


def test_upsample_nearest2d(device):
    x = torch.randn(1, 2, 5, 7, dtype=torch.float32)
    out = F.interpolate(x.to(device), size=(9, 11), mode="nearest")
    assert_close(out, F.interpolate(x, size=(9, 11), mode="nearest"),
                 "upsample_nearest2d")
