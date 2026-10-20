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

"""The ``paras`` TorchDynamo backend.

Pipeline for every graph Dynamo captures:

1. fusion passes on the Dynamo graph (passes.py),
2. analysis at compile time: AOTAutograd traces the functionalised ATen
   forward and backward graphs (the backward of a fused operator is its ParaS
   backward kernel, see ops.py) and residency.py classifies every operator;
   with ``strict`` a graph containing an operator without a ParaS kernel is
   rejected,
3. execution: the fused Dynamo graph runs through the dispatcher and the
   autograd engine exactly like eager code, on the same ParaS kernels.

Running the Dynamo graph rather than AOTAutograd's compiled function avoids
AOTAutograd's per-call runtime wrapper and its Python-level backward, which
dominate for models that call many small compiled regions (an MoE's expert
MLPs called from an eager expert loop: ~40 us per call).
"""

import itertools
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from torch import fx

from . import ops, passes, residency

ALL_FUSIONS = tuple(passes.PASSES)


class ParasCompileError(RuntimeError):
    pass


@dataclass(frozen=True)
class Options:
    """Options of the paras backend.

    fusions: fusion passes to apply (subset of ``ALL_FUSIONS``).
    strict:  reject graphs containing operators without a ParaS kernel
             instead of letting them run on the CPU fallback.
    """

    fusions: tuple = ALL_FUSIONS
    strict: bool = False

    @classmethod
    def make(cls, fusions: Iterable[str] | None = None, strict: bool = False) -> "Options":
        chosen = ALL_FUSIONS if fusions is None else tuple(fusions)
        unknown = set(chosen) - set(ALL_FUSIONS)
        if unknown:
            raise ValueError(f"unknown fusions {sorted(unknown)}; available: {list(ALL_FUSIONS)}")
        return cls(fusions=chosen, strict=bool(strict))


@dataclass
class GraphReport:
    """What the backend did with one captured graph."""

    graph_id: int
    fusions: Counter = field(default_factory=Counter)
    residency: dict = field(default_factory=dict)
    cpu_fallback_ops: Counter = field(default_factory=Counter)

    @property
    def fully_on_paras(self) -> bool:
        return not self.cpu_fallback_ops

    def summary(self) -> str:
        parts = [f"graph {self.graph_id}"]
        if self.fusions:
            parts.append("fused " + ", ".join(f"{k} x{v}" for k, v in sorted(self.fusions.items())))
        for stage, kinds in self.residency.items():
            parts.append(f"{stage}: " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
        if self.cpu_fallback_ops:
            parts.append("CPU fallback: " + ", ".join(sorted(self.cpu_fallback_ops)))
        return " | ".join(parts)


_reports: list[GraphReport] = []
_lock = threading.Lock()
_ids = itertools.count()


def reports() -> list[GraphReport]:
    """Reports of the graphs compiled since start-up or reset_reports()."""
    with _lock:
        return list(_reports)


def reset_reports() -> None:
    with _lock:
        _reports.clear()


def make_backend(options: Options = Options()):
    """Returns a TorchDynamo backend callable configured by `options`."""

    def paras_backend(gm: fx.GraphModule, example_inputs: Sequence[object]):
        from functorch.compile import make_boxed_func
        from torch._dynamo.backends.common import aot_autograd
        from torch._functorch import config as functorch_config

        ops.register()
        report = GraphReport(graph_id=next(_ids))
        report.fusions.update(passes.run(gm, set(options.fusions)))

        def analyser(stage):
            def analyse_stage(aten_gm: fx.GraphModule, _inputs):
                kinds, cpu_ops = residency.analyse(aten_gm)
                report.residency[stage] = dict(kinds)
                report.cpu_fallback_ops.update(cpu_ops)
                if options.strict and cpu_ops:
                    raise ParasCompileError(
                        f"{stage} graph uses operators without a ParaS kernel: "
                        + ", ".join(sorted(cpu_ops)))
                return make_boxed_func(aten_gm.forward)
            return analyse_stage

        with functorch_config.patch(force_non_lazy_backward_lowering=True):
            aot_autograd(
                fw_compiler=analyser("forward"),
                bw_compiler=analyser("backward"),
                inference_compiler=analyser("inference"),
            )(gm, example_inputs)
        with _lock:
            _reports.append(report)
        gm.recompile()
        return gm.forward

    return paras_backend


backend = make_backend()


def register() -> None:
    """Registers ``backend`` under the TorchDynamo name "paras"."""
    from torch._dynamo import list_backends, register_backend

    if "paras" not in list_backends(exclude_tags=()):
        register_backend(backend, name="paras")
