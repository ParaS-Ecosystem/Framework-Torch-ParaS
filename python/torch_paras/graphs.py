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


"""Record and replay of paras device work, in the style of torch.cuda.CUDAGraph.

A capture records the kernels, device copies and vendor-library calls that
the code inside it queues on one GPU device, without running them; a replay
runs all of them with one launch, which takes the per-operator host cost
(Python, dispatcher, Torch-ParaS, kernel launch) out of the loop:

    g = torch_paras.graphs.Graph()
    static_x = torch.zeros(8, 4096, device="paras")
    model(static_x)                          # warm up outside the capture
    with torch.no_grad(), torch_paras.graphs.graph(g):
        static_y = model(static_x)
    static_x.copy_(x)
    g.replay()                               # static_y now holds model(x)

As with CUDA graphs, a replay uses the memory and the kernel arguments of
the capture: inputs go in by copying into the captured tensors, results come
out in the captured output tensors (overwritten by the next replay), and
host-side decisions made during the capture -- shapes, Python control flow,
random seeds -- are fixed. Allocations made during the capture come from the
graph's own pool, so its memory is never given to other tensors; several
graphs may share a pool when they are replayed in capture order. Work that
needs the host while capturing (synchronize, .item(), copies to or from CPU
tensors) raises, and the capture fails.

Needs a ParaS build with record-and-replay command graphs (CUDA or HIP);
see is_supported().
"""

import torch

from . import _C

__all__ = ["Graph", "graph", "is_supported", "is_capturing", "new_pool"]


def is_supported() -> bool:
    """Whether this build can capture graphs (ParaS command graphs, GPU devices)."""
    return bool(_C._graphs_supported())


def _index(device) -> int:
    if device is None:
        return 0
    if isinstance(device, int):
        return device
    device = torch.device(device)
    if device.type != "paras":
        raise ValueError(f"graphs capture paras devices, not {device}")
    return device.index or 0


def is_capturing(device=None) -> bool:
    return bool(_C._graph_capturing(_index(device)))


def new_pool() -> int:
    """A graph-pool id to share between graphs replayed in capture order."""
    return int(_C._graph_pool_new())


class Graph:
    """A recording of the work queued on one paras GPU device."""

    def __init__(self):
        self._device = None
        self._handle = None
        self._pool = None
        self._owns_pool = False

    def capture_begin(self, device=None, pool: int | None = None) -> None:
        if self._handle is not None or self._device is not None:
            raise RuntimeError("this Graph is already captured; reset() it first")
        self._device = _index(device)
        self._owns_pool = pool is None
        self._pool = new_pool() if pool is None else int(pool)
        try:
            _C._graph_capture_begin(self._device, self._pool)
        except Exception:
            self._release_pool()
            self._device = None
            raise

    def capture_end(self) -> None:
        if self._device is None or self._handle is not None:
            raise RuntimeError("capture_end without capture_begin")
        try:
            self._handle = int(_C._graph_capture_end(self._device))
        except Exception:
            self._release_pool()
            self._device = None
            raise

    def replay(self) -> None:
        if self._handle is None:
            raise RuntimeError("replay of a Graph that was not captured")
        _C._graph_replay(self._handle)

    def pool(self) -> int | None:
        return self._pool

    def reset(self) -> None:
        """Frees the recording (and its pool, unless the pool was passed in)."""
        if self._handle is not None:
            _C._graph_release(self._handle)
            self._handle = None
        self._release_pool()
        self._device = None

    def _release_pool(self) -> None:
        if self._owns_pool and self._pool is not None and self._device is not None:
            _C._graph_pool_release(self._device, self._pool)
        self._pool = None
        self._owns_pool = False

    def __del__(self):
        try:
            self.reset()
        except Exception:
            pass


class graph:
    """Context manager capturing the work queued inside it into `g`.

    If the body raises, or something in it cannot be captured, the capture
    is ended and the exception propagates; `g` stays uncaptured."""

    def __init__(self, g: Graph, device=None, pool: int | None = None):
        self.g, self.device, self.pool = g, device, pool

    def __enter__(self):
        self.g.capture_begin(self.device, self.pool)
        return self.g

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.g.capture_end()
            return False
        try:
            self.g.capture_end()
        except Exception:
            pass
        self.g.reset()
        return False
