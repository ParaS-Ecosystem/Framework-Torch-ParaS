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

"""Where each operator of a compiled graph executes on a paras device.

Classification uses the dispatcher's registration tables:

* ``paras``     - ParaS registers a PrivateUse1 kernel for the overload, or
                  the overload is a structured functional/in-place form whose
                  out= variant has one (PyTorch's generated wrapper calls it).
* ``composite`` - a PyTorch composite that decomposes into other operators
                  (each of which is dispatched, and counted by
                  torch_paras.runtime_stats(), when it runs).
* ``cpu``       - no ParaS kernel: the operator would run on the boxed CPU
                  fallback.
"""

from collections import Counter
from functools import lru_cache

import torch
from torch import fx

_COMPOSITE_KEYS = (
    "CompositeExplicitAutograd",
    "CompositeImplicitAutograd",
)


def _qualified(op: torch._ops.OpOverload) -> str:
    schema = op._schema
    return schema.name + ("." + schema.overload_name if schema.overload_name else "")


def _has_kernel(name: str, key: str) -> bool:
    return torch._C._dispatch_has_kernel_for_dispatch_key(name, key)


def _out_variant(op: torch._ops.OpOverload):
    """The out= overload computing the same function as `op`, matched by
    schema (e.g. sum.dim_IntList <-> sum.IntList_out), or None."""
    signature = [(a.name, str(a.type)) for a in op._schema.arguments]
    packet = op.overloadpacket
    for name in packet.overloads():
        candidate = getattr(packet, name)
        args = candidate._schema.arguments
        outs = [a for a in args if a.is_out]
        if not outs:
            continue
        if [(a.name, str(a.type)) for a in args if not a.is_out] == signature:
            return candidate
    return None


@lru_cache(maxsize=None)
def classify(op: torch._ops.OpOverload) -> str:
    name = _qualified(op)
    if _has_kernel(name, "PrivateUse1"):
        return "paras"
    if _has_kernel(name, "CompositeExplicitAutogradNonFunctional"):
        out = _out_variant(op)
        if out is not None:
            return "paras" if _has_kernel(_qualified(out), "PrivateUse1") else "cpu"
        return "composite"
    if any(_has_kernel(name, k) for k in _COMPOSITE_KEYS):
        return "composite"
    return "cpu"


def analyse(gm: fx.GraphModule) -> tuple[Counter, Counter]:
    """Returns (count per class, count per CPU-fallback operator)."""
    kinds: Counter = Counter()
    cpu_ops: Counter = Counter()
    for node in gm.graph.nodes:
        if node.op != "call_function" or not isinstance(node.target, torch._ops.OpOverload):
            continue
        kind = classify(node.target)
        kinds[kind] += 1
        if kind == "cpu":
            cpu_ops[_qualified(node.target)] += 1
    return kinds, cpu_ops
