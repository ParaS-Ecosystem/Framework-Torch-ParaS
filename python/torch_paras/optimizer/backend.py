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
   rejected. Graphs that are the same code on the same input metadata --
   the layers of a transformer compiled one by one (see blocks.py) -- are
   analysed once,
3. execution: the fused Dynamo graph runs through the dispatcher and the
   autograd engine exactly like eager code, on the same ParaS kernels.

Running the Dynamo graph rather than AOTAutograd's compiled function avoids
AOTAutograd's per-call runtime wrapper and its Python-level backward, which
dominate for models that call many small compiled regions (an MoE's expert
MLPs called from an eager expert loop: ~40 us per call).

Some frames are handed back to the interpreter: Dynamo skips them from then
on, and they run as plain eager code -- which is what their graphs compute,
bit for bit. Two cases:

* small dynamic-shape graphs. When Dynamo recompiles a frame with dynamic
  shapes, every later call pays guard evaluation over several cache entries
  and a SymInt prologue (~10-20 us). A frame captured whole (no graph break)
  whose graph has a few operators cannot win that back -- an MoE's expert
  MLP, called once per expert with a different token count, is the typical
  case. Dynamo is also switched off for what such a frame calls.
* frames Dynamo refuses to recompile (recompile limit). Dynamo itself then
  keeps evaluating the frame's cached guards on every call without ever
  compiling a matching graph again. A decoder layer reading a growing KV cache
  without per-layer code objects (see blocks.py), or a model's top-level
  forward called by transformers' generate() with ever new keyword sets, are
  the typical cases. The frame alone is skipped; the functions it calls keep
  their compiled graphs. (A frame that needs exactly the limit keeps its
  graphs.)
"""

import hashlib
import itertools
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch
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
    min_dynamic_graph_ops: dynamic-shape graphs with fewer tensor operators
             than this run eagerly (see the module docstring); 0 disables.
    eager_at_recompile_limit: frames Dynamo refuses to recompile run
             eagerly (their callees keep their graphs; see the module
             docstring); False keeps Dynamo's own behaviour.
    """

    fusions: tuple = ALL_FUSIONS
    strict: bool = False
    min_dynamic_graph_ops: int = 8
    eager_at_recompile_limit: bool = True

    @classmethod
    def make(cls, fusions: Iterable[str] | None = None, strict: bool = False,
             min_dynamic_graph_ops: int | None = None,
             eager_at_recompile_limit: bool | None = None) -> "Options":
        chosen = ALL_FUSIONS if fusions is None else tuple(fusions)
        unknown = set(chosen) - set(ALL_FUSIONS)
        if unknown:
            raise ValueError(f"unknown fusions {sorted(unknown)}; available: {list(ALL_FUSIONS)}")
        limit = cls.min_dynamic_graph_ops if min_dynamic_graph_ops is None else int(min_dynamic_graph_ops)
        if limit < 0:
            raise ValueError("min_dynamic_graph_ops must be >= 0")
        at_limit = (cls.eager_at_recompile_limit if eager_at_recompile_limit is None
                    else bool(eager_at_recompile_limit))
        return cls(fusions=chosen, strict=bool(strict), min_dynamic_graph_ops=limit,
                   eager_at_recompile_limit=at_limit)


@dataclass
class GraphReport:
    """What the backend did with one captured graph."""

    graph_id: int
    frame: str = ""
    fusions: Counter = field(default_factory=Counter)
    residency: dict = field(default_factory=dict)
    cpu_fallback_ops: Counter = field(default_factory=Counter)
    runs_eagerly: str = ""

    @property
    def fully_on_paras(self) -> bool:
        return not self.cpu_fallback_ops

    def summary(self) -> str:
        parts = [f"graph {self.graph_id}" + (f" [{self.frame}]" if self.frame else "")]
        if self.fusions:
            parts.append("fused " + ", ".join(f"{k} x{v}" for k, v in sorted(self.fusions.items())))
        for stage, kinds in self.residency.items():
            parts.append(f"{stage}: " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
        if self.cpu_fallback_ops:
            parts.append("CPU fallback: " + ", ".join(sorted(self.cpu_fallback_ops)))
        if self.runs_eagerly:
            parts.append(f"runs eagerly ({self.runs_eagerly})")
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


_analyses: dict[bytes, list] = {}


def _symbolic(s) -> object:
    if not isinstance(s, torch.SymInt):
        return s
    node = s.node
    ranges = node.shape_env.var_to_range
    return (str(node.expr), sorted((str(v), str(ranges.get(v))) for v in node.expr.free_symbols))


def _analysis_key(gm: fx.GraphModule, example_inputs: Sequence[object]) -> bytes:
    """What the ATen graphs depend on: the graph's code, grad mode and, per
    input, its kind, dtype, device, requires_grad and (symbolic) sizes."""
    def meta(x):
        if isinstance(x, torch.Tensor):
            return ("tensor", x.dtype, x.device.type, x.requires_grad,
                    tuple(_symbolic(d) for d in x.shape))
        if isinstance(x, torch.SymInt):
            return ("symint", _symbolic(x))
        return (type(x).__name__, x if isinstance(x, (bool, int, float, str)) else None)

    key = (gm.code, torch.is_grad_enabled(), tuple(meta(x) for x in example_inputs))
    return hashlib.sha256(repr(key).encode()).digest()


def _tensor_op_count(gm: fx.GraphModule) -> int:
    """Operators of a Dynamo graph that produce tensors (not size arithmetic)."""
    import torch

    return sum(1 for n in gm.graph.nodes
               if n.op in ("call_function", "call_method", "call_module")
               and isinstance(n.meta.get("example_value"), torch.Tensor))


def _has_dynamic_shapes(example_inputs: Sequence[object]) -> bool:
    import torch

    for value in example_inputs:
        if isinstance(value, torch.SymInt):
            return True
        if isinstance(value, torch.Tensor) and any(
                isinstance(d, torch.SymInt) for d in (*value.shape, value.storage_offset())):
            return True
    return False


def _is_whole_frame(gm: fx.GraphModule) -> bool:
    """The graph covers its whole frame (no graph break). Handing back a
    frame that broke would also run its later, separately compiled part
    eagerly."""
    reason = getattr(gm, "compile_subgraph_reason", None)
    return reason is not None and not reason.graph_break


def _root_code():
    """Code object of the frame Dynamo is compiling, or None."""
    from torch._guards import TracingContext

    ctx = TracingContext.try_get()
    return ctx.traced_code[0] if ctx is not None and ctx.traced_code else None


def _frame_name(code) -> str:
    if code is None:
        return ""
    import os
    return f"{code.co_name} ({os.path.basename(code.co_filename)}:{code.co_firstlineno})"


def _hand_back(code, recursive: bool) -> None:
    """Dynamo skips `code` from now on -- with `recursive`, also everything
    it calls; otherwise the functions it calls keep their compiled graphs."""
    from torch._dynamo.eval_frame import set_code_exec_strategy
    from torch._dynamo.types import FrameAction, FrameExecStrategy

    callees = FrameAction.SKIP if recursive else FrameAction.DEFAULT
    set_code_exec_strategy(code, FrameExecStrategy(FrameAction.SKIP, callees))


_frames: dict = {}
_exceeded: list = []


def _on_recompile() -> str:
    import torch
    from torch._guards import CompileContext

    cid = CompileContext.current_compile_id()
    if cid is None or cid.frame_compile_id is None:
        return ""
    if cid.frame_compile_id >= torch._dynamo.config.recompile_limit:
        entry = _frames.get(cid.frame_id)
        if entry is not None and entry[1]:
            with _lock:
                _exceeded.append((cid.frame_id, entry))
    return ""


def _install_recompile_hook() -> None:
    from torch._dynamo.utils import (get_hook_for_recompile_user_context,
                                     register_hook_for_recompile_user_context)

    if _on_recompile not in (get_hook_for_recompile_user_context() or []):
        register_hook_for_recompile_user_context(_on_recompile)


def _hand_back_exceeded() -> None:
    with _lock:
        pending = list(_exceeded)
        _exceeded.clear()
    for frame_id, entry in pending:
        if _frames.get(frame_id) is not entry:
            continue
        code, _, reports = entry
        _hand_back(code, recursive=False)
        for r in reports:
            r.runs_eagerly = "recompile limit reached"


def _graph_callable(gm: fx.GraphModule, code, hand_back_now: bool):
    """The graph's callable. With `hand_back_now`, the frame is handed back
    after this call. Dynamo sets the code's execution strategy itself when
    the compile returns, so the first call is the earliest point."""
    forward = gm.forward

    def run(*args):
        if _exceeded:
            _hand_back_exceeded()
        if hand_back_now and code is not None:
            _hand_back(code, recursive=True)
        return forward(*args)

    return run


def make_backend(options: Options = Options()):
    """Returns a TorchDynamo backend callable configured by `options`."""

    def paras_backend(gm: fx.GraphModule, example_inputs: Sequence[object]):
        from functorch.compile import make_boxed_func
        from torch._dynamo.backends.common import aot_autograd
        from torch._functorch import config as functorch_config

        from torch._guards import CompileContext

        ops.register()
        code = _root_code()
        report = GraphReport(graph_id=next(_ids), frame=_frame_name(code))
        cid = CompileContext.current_compile_id()
        if cid is not None and code is not None:
            _install_recompile_hook()
            with _lock:
                entry = _frames.get(cid.frame_id)
                if entry is None or cid.frame_compile_id == 0 or entry[0] is not code:
                    entry = (code, options.eager_at_recompile_limit, [])
                    _frames[cid.frame_id] = entry
                entry[2].append(report)
        report.fusions.update(passes.run(gm, set(options.fusions)))

        def record(stage, kinds, cpu_ops):
            report.residency[stage] = dict(kinds)
            report.cpu_fallback_ops.update(cpu_ops)
            if options.strict and cpu_ops:
                raise ParasCompileError(
                    f"{stage} graph uses operators without a ParaS kernel: "
                    + ", ".join(sorted(cpu_ops)))

        key = _analysis_key(gm, example_inputs)
        with _lock:
            cached = _analyses.get(key)
        if cached is not None:
            for stage, kinds, cpu_ops in cached:
                record(stage, kinds, cpu_ops)
        else:
            stages = []

            def analyser(stage):
                def analyse_stage(aten_gm: fx.GraphModule, _inputs):
                    kinds, cpu_ops = residency.analyse(aten_gm)
                    stages.append((stage, kinds, cpu_ops))
                    record(stage, kinds, cpu_ops)
                    return make_boxed_func(aten_gm.forward)
                return analyse_stage

            with functorch_config.patch(force_non_lazy_backward_lowering=True):
                aot_autograd(
                    fw_compiler=analyser("forward"),
                    bw_compiler=analyser("backward"),
                    inference_compiler=analyser("inference"),
                )(gm, example_inputs)
            gm.recompile()
            with _lock:
                _analyses[key] = stages
        if (options.min_dynamic_graph_ops and _is_whole_frame(gm)
                and _has_dynamic_shapes(example_inputs)
                and _tensor_op_count(gm) < options.min_dynamic_graph_ops):
            report.runs_eagerly = "small dynamic-shape graph"
        compiled = _graph_callable(gm, code, bool(report.runs_eagerly))
        with _lock:
            _reports.append(report)
        return compiled

    return paras_backend


backend = make_backend()


def register() -> None:
    """Registers ``backend`` under the TorchDynamo name "paras"."""
    from torch._dynamo import list_backends, register_backend

    if "paras" not in list_backends(exclude_tags=()):
        register_backend(backend, name="paras")
