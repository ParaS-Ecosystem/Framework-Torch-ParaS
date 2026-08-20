"""
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
"""


"""PyTorch backend for the ParaS SYCL compiler.

Importing this package registers the ``paras`` device with PyTorch:

    import torch
    import torch_paras

    x = torch.randn(64, 64, device="paras")        # device 0 = host CPU engine
    y = torch.randn(64, 64, device="paras:1")      # device 1.. = CUDA/HIP GPUs
    z = (x @ x).relu().cpu()

Device 0 always exists and executes on the host through the ParaS CPU
engine. In CUDA- or HIP-enabled builds, devices 1..N map to the visible
NVIDIA or AMD GPUs, respectively (a single build is one flavor or the
other, never both).
"""

import types

import torch

from . import _C
from . import nn  # noqa: F401

__all__ = [
    "device_count",
    "is_available",
    "synchronize",
    "manual_seed",
    "empty_cache",
    "device_name",
    "backend_name",
    "get_amp_supported_dtype",
    "nn",
]


def device_count() -> int:
    """Number of paras devices (host CPU + visible GPUs)."""
    return _C.device_count()


def is_available() -> bool:
    return device_count() > 0


def backend_name() -> str:
    """Active compat flavor: 'paras-cpu', 'paras-cuda', or 'paras-hip'."""
    return _C.backend_name()


def device_name(device: int = 0) -> str:
    return _C.device_name(device)


def is_gpu_device(device: int) -> bool:
    return _C.is_gpu_device(device)


def synchronize(device: int | None = None) -> None:
    """Blocks until queued work on one device (or all devices) completes."""
    if device is None:
        _C.synchronize_all()
    else:
        _C.synchronize(device)


def manual_seed(seed: int, device: int | None = None) -> None:
    if device is None:
        _C.manual_seed_all(seed)
    else:
        _C.manual_seed(device, seed)


def manual_seed_all(seed: int) -> None:
    _C.manual_seed_all(seed)


def empty_cache() -> None:
    """Releases cached blocks held by the device memory pools."""
    _C.empty_cache()


def memory_allocated(device: int = 0) -> int:
    return _C.allocated_bytes(device)


def memory_cached(device: int = 0) -> int:
    return _C.cached_bytes(device)


def get_amp_supported_dtype() -> list[torch.dtype]:
    return [torch.float16, torch.float32]


def _register() -> None:
    torch.utils.rename_privateuse1_backend("paras")

    mod = types.ModuleType("torch.paras")
    mod.device_count = device_count
    mod.is_available = is_available
    mod.synchronize = synchronize
    mod.manual_seed = manual_seed
    mod.manual_seed_all = manual_seed_all
    mod.empty_cache = empty_cache
    mod.device_name = device_name
    mod.is_bad_fork = _C.is_bad_fork
    mod._is_in_bad_fork = _C.is_bad_fork
    mod.get_amp_supported_dtype = get_amp_supported_dtype
    torch._register_device_module("paras", mod)

    # tensor.paras() / module.paras() / tensor.is_paras
    torch.utils.generate_methods_for_privateuse1_backend(
        for_tensor=True, for_module=True, for_storage=False
    )


_register()