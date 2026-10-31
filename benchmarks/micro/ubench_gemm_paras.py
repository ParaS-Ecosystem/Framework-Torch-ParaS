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

"""Runs InferenceX's ubenchX GEMM benchmark (ubenchX/gemm/bench.py) unchanged on a Torch-ParaS device.

bench.py looks its device API up as torch.<kind> and needs set_device and get_device_properties; torch.paras
(registered by torch_paras) has synchronize, Event(enable_timing) and empty_cache but not those two, so this shim
supplies them and hands bench.main the paras device. bench.py's measure() -- the L2 flush before every call,
device events around every torch.matmul, median of 10 after 5 warm-ups, TFLOP/s = 2MKN / latency -- is not touched.

  INFERENCEX_DIR=<InferenceX checkout> ubench_gemm_paras.py <bench.py arguments, with --device cuda and an explicit --flush-mib>
"""
import os
import sys
import types

import torch
import torch_paras

BENCH = os.path.join(os.environ["INFERENCEX_DIR"], "ubenchX", "gemm")
sys.path.insert(0, BENCH)
import bench  # noqa: E402


def select_device(kind, index):
    torch.accelerator.set_device_index(index)
    props = types.SimpleNamespace(name=torch_paras.device_name(index), L2_cache_size=0)
    return torch.paras, f"paras:{index}", props


bench.select_device = select_device
sys.exit(bench.main(sys.argv[1:]))
