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

    x = torch.randn(64, 64, device="paras")        # default paras device (paras:0, primary GPU or CPU)
    y = torch.randn(64, 64, device="paras:0")      # device 0..N-1 = GPUs (in GPU builds)
    z = (x @ x).relu().cpu()

In GPU-enabled builds (CUDA or HIP), devices 0..N-1 map to visible GPUs,
and device N maps to the host CPU engine. In CPU-only builds, device 0 maps
to the host CPU.
"""

import contextlib
import types

import torch

from . import _C
from . import graphs  # noqa: F401
from . import nn  # noqa: F401

__all__ = [
    "device_count",
    "is_available",
    "synchronize",
    "manual_seed",
    "empty_cache",
    "memory_allocated",
    "max_memory_allocated",
    "memory_reserved",
    "max_memory_reserved",
    "memory_cached",
    "memory_stats",
    "reset_peak_memory_stats",
    "reset_accumulated_memory_stats",
    "device_name",
    "backend_name",
    "get_amp_supported_dtype",
    "Stream",
    "Event",
    "current_stream",
    "default_stream",
    "set_stream",
    "stream",
    "runtime_stats",
    "reset_runtime_stats",
    "set_strict",
    "is_strict",
    "compile",
    "graphs",
    "nn",
    "optimizer",
]


def device_count() -> int:
    """Number of paras devices (host CPU + visible GPUs)."""
    return _C.device_count()


def is_available() -> bool:
    """True when at least one paras device exists (the host CPU engine counts)."""
    return device_count() > 0


def backend_name() -> str:
    """Active compat flavor: 'paras-cpu', 'paras-cuda', or 'paras-hip'."""
    return _C.backend_name()


def device_name(device: int = 0) -> str:
    """Name of paras device `device` as the vendor runtime reports it (e.g. the GPU's marketing
    name, or the host CPU)."""
    return _C.device_name(device)


def is_gpu_device(device: int) -> bool:
    return _C.is_gpu_device(device)


def device_kernel_caps(device: int) -> dict:
    """What the device's hand-written kernels may use: ``wave_size`` (64 on
    AMD GCN / CDNA, 32 on RDNA and NVIDIA, 1 for the host engine) and ``mfma``
    (AMD matrix-core instructions, CDNA 2 and later). Devices without them
    run the generic kernels and the vendor libraries instead."""
    return dict(_C.device_kernel_caps(device))


def synchronize(device: int | torch.device | None = None) -> None:
    """Blocks until queued work on one device (or all devices) completes."""
    if device is None:
        _C.synchronize_all()
    elif isinstance(device, torch.device):
        _C.synchronize(device.index if device.index is not None else 0)
    else:
        _C.synchronize(int(device))


def manual_seed(seed: int, device: int | None = None) -> None:
    """Seeds the device random generators: every paras device, or only `device` when given
    (torch.manual_seed seeds them too)."""
    if device is None:
        _C.manual_seed_all(seed)
    else:
        _C.manual_seed(device, seed)


def manual_seed_all(seed: int) -> None:
    _C.manual_seed_all(seed)


def empty_cache() -> None:
    """Releases cached blocks held by the device memory pools."""
    _C.empty_cache()


def _index(device) -> int:
    if device is None:
        return 0
    if isinstance(device, (str, torch.device)):
        return torch.device(device).index or 0
    return int(device)


def memory_allocated(device=None) -> int:
    """Bytes in live tensors' blocks (torch.cuda.memory_allocated's meaning)."""
    return _C.memory_stats(_index(device))["allocated_bytes"][0]


def max_memory_allocated(device=None) -> int:
    """Peak of memory_allocated() since start-up or reset_peak_memory_stats()."""
    return _C.memory_stats(_index(device))["allocated_bytes"][1]


def memory_reserved(device=None) -> int:
    """Bytes the pool holds from the device: live blocks plus cached ones."""
    return _C.memory_stats(_index(device))["reserved_bytes"][0]


def max_memory_reserved(device=None) -> int:
    """Peak of memory_reserved() since start-up or reset_peak_memory_stats()."""
    return _C.memory_stats(_index(device))["reserved_bytes"][1]


def memory_cached(device=None) -> int:
    """Cached free bytes of the shared pool (released by empty_cache())."""
    return _C.cached_bytes(_index(device))


def reset_peak_memory_stats(device=None) -> None:
    """Restarts the peaks reported by max_memory_allocated() / max_memory_reserved() at the
    current values."""
    _C.reset_peak_stats(_index(device))


def reset_accumulated_memory_stats(device=None) -> None:
    """Zeroes the accumulated counters of memory_stats() (allocation and free counts, bytes)."""
    _C.reset_accumulated_stats(_index(device))


def memory_stats(device=None) -> dict:
    """torch.cuda.memory_stats-style keys, e.g. "allocated_bytes.all.peak"."""
    out = {}
    for name, (cur, peak, alloc, freed) in _C.memory_stats(_index(device)).items():
        for k, v in (("current", cur), ("peak", peak), ("allocated", alloc), ("freed", freed)):
            out[f"{name}.all.{k}"] = v
    return dict(sorted(out.items()))


def get_amp_supported_dtype() -> list[torch.dtype]:
    """Dtypes torch.autocast(device_type="paras") accepts (the policy is
    registered in csrc/core/autocast.cpp)."""
    return [torch.float16, torch.bfloat16, torch.float32]


def runtime_stats() -> dict:
    """Execution counters since start-up or the last reset_runtime_stats().

    ``cpu_fallbacks`` maps each operator that ran on the boxed CPU fallback
    (no ParaS kernel) to its call count. The copy counters cover every tensor
    transfer between a paras device and host memory, including transfers
    made inside operator implementations; ``scalar_reads`` counts
    single-value reads such as ``tensor.item()``.
    """
    return _C.runtime_stats()


def reset_runtime_stats() -> None:
    """Zeroes the counters of runtime_stats() (CPU fallbacks, host-device copies and bytes,
    scalar reads)."""
    _C.reset_runtime_stats()


def set_strict(enabled: bool = True) -> None:
    """Make operators without a ParaS kernel raise instead of falling back to
    the CPU. Also enabled at start-up by ``PTSYCL_STRICT=1``."""
    _C.set_strict(bool(enabled))


def is_strict() -> bool:
    """True when CPU fallbacks raise instead of running (PTSYCL_STRICT, or set_strict(True))."""
    return _C.strict()


def _dev(device=None) -> torch.device:
    if device is None:
        return torch.device("paras", torch.accelerator.current_device_index())
    d = torch.device(device) if not isinstance(device, int) else torch.device("paras", device)
    return d if d.index is not None else torch.device("paras", 0)


def Stream(device=None, priority: int = 0) -> torch.Stream:
    """A new stream of a paras device (from the device's pool of streams)."""
    return torch.Stream(device=_dev(device), priority=priority)


def Event(enable_timing: bool = False, device=None) -> torch.Event:
    """A torch.Event on a paras device (`device`: index or torch.device; default the current
    one); enable_timing for elapsed_time()."""
    return torch.Event(device=_dev(device), enable_timing=enable_timing)


def current_stream(device=None) -> torch.Stream:
    """The current torch.Stream of a paras device (default the current device)."""
    return torch.accelerator.current_stream(_dev(device))


def default_stream(device=None) -> torch.Stream:
    """The default stream (id 0) of a paras device: the queue work goes to unless another stream
    is made current."""
    d = _dev(device)
    cur = torch.accelerator.current_stream(d)
    return torch.Stream(stream_id=0, device_index=d.index, device_type=cur.device_type)


def set_stream(stream: torch.Stream) -> None:
    """Makes `stream` the current stream of its device (as torch.accelerator.set_stream)."""
    torch.accelerator.set_stream(stream)


@contextlib.contextmanager
def stream(s: torch.Stream):
    """Makes `s` the current stream of its device inside the block."""
    prev = torch.accelerator.current_stream(s.device)
    torch.accelerator.set_stream(s)
    try:
        yield s
    finally:
        torch.accelerator.set_stream(prev)


def _register() -> None:
    torch.utils.rename_privateuse1_backend("paras")

    mod = types.ModuleType("torch.paras")
    mod.device_count = device_count
    mod.is_available = is_available
    mod.synchronize = synchronize
    mod.manual_seed = manual_seed
    mod.manual_seed_all = manual_seed_all
    mod.empty_cache = empty_cache
    for fn in (memory_allocated, max_memory_allocated, memory_reserved, max_memory_reserved,
               memory_stats, reset_peak_memory_stats, reset_accumulated_memory_stats):
        setattr(mod, fn.__name__, fn)
    mod.device_name = device_name
    mod.is_bad_fork = _C.is_bad_fork
    mod._is_in_bad_fork = _C.is_bad_fork
    mod.get_amp_supported_dtype = get_amp_supported_dtype
    for fn in (Stream, Event, current_stream, default_stream, set_stream, stream):
        setattr(mod, fn.__name__, fn)
    torch._register_device_module("paras", mod)

    # tensor.paras() / module.paras() / tensor.is_paras
    torch.utils.generate_methods_for_privateuse1_backend(
        for_tensor=True, for_module=True, for_storage=False
    )


_register()

from . import optimizer  # noqa: E402
from .optimizer import compile  # noqa: E402,F401

optimizer.register()
