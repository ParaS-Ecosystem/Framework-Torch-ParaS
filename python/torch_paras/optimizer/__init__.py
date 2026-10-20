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

"""ParaS graph optimizer: a TorchDynamo backend for paras devices.

    import torch, torch_paras

    model = model.to("paras:0")
    fast = torch_paras.compile(model)                 # or
    fast = torch.compile(model, backend="paras")      # default options

Works for inference and training: fused subgraphs carry ParaS backward
kernels, and every other operator runs on its ParaS kernel exactly as in
eager mode. ``reports()`` describes what was fused and where each operator of
the compiled graphs executes.
"""

from typing import Iterable

import torch

from .backend import (
    ALL_FUSIONS,
    GraphReport,
    Options,
    ParasCompileError,
    backend,
    make_backend,
    register,
    reports,
    reset_reports,
)

__all__ = [
    "ALL_FUSIONS",
    "GraphReport",
    "Options",
    "ParasCompileError",
    "backend",
    "compile",
    "make_backend",
    "register",
    "reports",
    "reset_reports",
]


def compile(model, *, fusions: Iterable[str] | None = None, strict: bool = False,
            min_dynamic_graph_ops: int | None = None,
            eager_at_recompile_limit: bool | None = None,
            fullgraph: bool = False, dynamic: bool | None = None):
    """``torch.compile`` with the paras backend.

    fusions:   fusion passes to apply (default: all of ``ALL_FUSIONS``).
    strict:    raise ParasCompileError if a compiled graph contains an
               operator without a ParaS kernel.
    min_dynamic_graph_ops: dynamic-shape graphs with fewer tensor operators
               run eagerly (default 8; 0 compiles every graph).
    eager_at_recompile_limit: frames that reach Dynamo's recompile limit run
               eagerly with Dynamo off for their callees (default True).
    fullgraph, dynamic: forwarded to torch.compile.
    """
    options = Options.make(fusions=fusions, strict=strict,
                           min_dynamic_graph_ops=min_dynamic_graph_ops,
                           eager_at_recompile_limit=eager_at_recompile_limit)
    return torch.compile(model, backend=make_backend(options),
                         fullgraph=fullgraph, dynamic=dynamic)
