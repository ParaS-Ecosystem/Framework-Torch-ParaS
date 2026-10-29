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


def test_max_pool2d_forward_backward_and_compile(device):
    torch.manual_seed(0)
    x = torch.randn(2, 3, 17, 15, requires_grad=True)
    for kw in ({"kernel_size": 3, "stride": 2, "padding": 1},
               {"kernel_size": 2, "stride": 1, "dilation": 2},
               {"kernel_size": 3, "stride": 2, "ceil_mode": True}):
        dx = x.detach().to(device).requires_grad_()
        y = torch.nn.functional.max_pool2d(x, **kw)
        dy = torch.nn.functional.max_pool2d(dx, **kw)
        assert torch.equal(dy.detach().cpu(), y.detach())
        g = torch.randn(y.shape)
        y.backward(g)
        dy.backward(g.to(device))
        assert torch.allclose(dx.grad.cpu(), x.grad, atol=1e-6)
        x.grad = None
    f = torch.compile(lambda t: torch.nn.functional.max_pool2d(t, 3, 2, 1), backend="paras")
    t = torch.randn(2, 4, 16, 16)
    assert torch.equal(f(t.to(device)).cpu(), torch.nn.functional.max_pool2d(t, 3, 2, 1))
