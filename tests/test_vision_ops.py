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

import unittest

import torch
import torch.nn.functional as F
import torch_paras

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


def test_linear_bf16_bias_precision(device):
    index = int(device.split(":")[1])
    if not torch_paras.is_gpu_device(index):
        raise unittest.SkipTest("requires a ParaS GPU device")

    x = torch.tensor([[1.0, 1.0]], dtype=torch.bfloat16)

    cases = [
        (
            torch.tensor(
                [[1.0, 0.00390625]],
                dtype=torch.bfloat16,
            ),
            torch.tensor([-1.0], dtype=torch.bfloat16),
            0.00390625,
        ),
        (
            torch.tensor(
                [[torch.finfo(torch.bfloat16).max, 2.0**119]],
                dtype=torch.bfloat16,
            ),
            torch.tensor(
                [-torch.finfo(torch.bfloat16).max],
                dtype=torch.bfloat16,
            ),
            2.0**119,
        ),
    ]

    for weight, bias, expected_value in cases:
        out = F.linear(
            x.to(device),
            weight.to(device),
            bias.to(device),
        ).cpu().float()

        expected = torch.tensor(
            [[expected_value]],
            dtype=torch.float32,
        )

        torch.testing.assert_close(
            out,
            expected,
            rtol=0,
            atol=0,
        )


def test_upsample_nearest2d(device):
    x = torch.randn(1, 2, 5, 7, dtype=torch.float32)
    out = F.interpolate(x.to(device), size=(9, 11), mode="nearest")
    assert_close(out, F.interpolate(x, size=(9, 11), mode="nearest"),
                 "upsample_nearest2d")
