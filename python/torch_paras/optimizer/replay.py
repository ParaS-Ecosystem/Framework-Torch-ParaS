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

"""Graph replay for compiled inference (``torch_paras.compile(..., graphs=True)``).

An inference graph is captured (torch_paras.graphs) and then replayed: one
launch runs all of its kernels, so the per-operator host cost -- Python,
dispatcher, Torch-ParaS, kernel launch -- is paid once per graph instead of
once per operator.

A capture holds for one set of input shapes. For each new shape a graph
(``Replayed``) first runs normally -- lazy initialisation such as vendor
library handles happens outside any capture -- and is captured when the
shape comes again; up to ``MAX_SHAPES`` shapes are kept, beyond that the
graph runs normally for new shapes. A replay copies the inputs into the
capture's input buffers, replays, copies mutated inputs back and returns
copies of the capture's outputs (the next replay overwrites them).
Captures that take the same tensor share its input buffer, and a replay
skips the copy when the buffer already holds that tensor (same object, same
version): the RoPE tables, mask and cache position a model computes once per
step are copied once, not once per layer.
Parameters and buffers Dynamo treats as static (``static_input_indices``:
module parameters, tensors passed to ``torch._dynamo.mark_static_address``)
are used in place; a call with such an input at another address runs the
graph normally. A graph that fails to capture (it needs the host:
``.item()``, a copy to the CPU) runs normally from then on.

A graph with dynamic shapes is split (``Split``). Its first
``OBSERVE_CALLS`` calls run normally while the values of its symbolic sizes
are recorded; a size whose value changes between those calls is volatile
-- in a cached decode step the KV length, which grows every token, while
the batch size and the query length stay. The graph is then split into runs
of operators whose shapes involve no volatile size, which are replayed (per
shape, as above), and the rest, which runs normally: the projections, norms,
RoPE and MLP of a decoder layer are replayed, the concatenation with the KV
cache and attention over it are not. A run must have at least
``MIN_PIECE_OPS`` tensor operators; it contains no in-place operation and
returns no view of its inputs.

Not replayed: graphs that need autograd, use random operators (their seeds
are drawn on the host), return their inputs or aliases of them, mutate
input metadata, or have inputs on several devices or on the host. All
replayed graphs of a device share one graph pool: they run in program
order on the device's queue and their outputs are copied out before the
next replay. A piece whose outputs the next piece reads in place (not
copied) has a pool of its own.
"""

import collections
import operator
import os
import weakref
from dataclasses import dataclass

import torch
from torch import fx

from .. import graphs

_pools: dict = {}

MIN_PIECE_OPS = 6
MAX_SHAPES = 8
OBSERVE_CALLS = 3

VERIFY = os.environ.get("PTSYCL_GRAPHS_VERIFY", "0") == "1"
mismatches: list = []

stats = collections.Counter()

_output_ptrs: set = set()


@dataclass(frozen=True)
class GraphMeta:
    """What AOTAutograd's analysis says about a graph's inputs and outputs."""

    static_inputs: tuple = ()
    mutated_inputs: tuple = ()
    private_outputs: tuple = ()
    mutates_metadata: bool = False
    output_aliases_input: bool = False
    random_ops: tuple = ()


def meta_from_aot(fw_metadata, random_ops) -> GraphMeta:
    from torch._functorch._aot_autograd.schemas import OutputType

    inputs = fw_metadata.input_info
    aliased = {OutputType.alias_of_input, OutputType.is_input}
    return GraphMeta(
        static_inputs=tuple(fw_metadata.static_input_indices or ()),
        mutated_inputs=tuple(i for i, info in enumerate(inputs) if info.mutates_data),
        mutates_metadata=any(info.mutates_metadata for info in inputs),
        output_aliases_input=any(o.output_type in aliased for o in fw_metadata.output_info),
        random_ops=tuple(sorted(random_ops)))


def _draws_random(node) -> bool:
    if torch.Tag.nondeterministic_seeded not in getattr(node.target, "tags", ()):
        return False
    for i, arg in enumerate(node.target._schema.arguments):
        if arg.name == "dropout_p":
            p = node.kwargs.get("dropout_p", node.args[i] if i < len(node.args) else arg.default_value)
            return p != 0
    return True


def random_ops(aten_gm) -> set:
    """Operators of an ATen graph that draw random numbers."""
    return {str(n.target) for n in aten_gm.graph.nodes
            if n.op == "call_function" and _draws_random(n)}


def _symbols(value) -> set:
    """The size symbols `value` (an example value) involves."""
    if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
        return set(value.node.expr.free_symbols)
    if isinstance(value, torch.Tensor):
        out = set()
        for d in (*value.shape, *value.stride(), value.storage_offset()):
            if isinstance(d, torch.SymInt):
                out |= d.node.expr.free_symbols
        return out
    if isinstance(value, (list, tuple)):
        return set().union(*map(_symbols, value)) if value else set()
    return set()


def has_dynamic_shapes(example_inputs) -> bool:
    return any(_symbols(x) for x in example_inputs)


def not_replayable(meta: GraphMeta | None, example_inputs) -> str:
    """Why a graph cannot be replayed ("" if it can)."""
    if not graphs.is_supported():
        return "this ParaS build has no command graphs"
    if meta is None:
        return "no analysis"
    tensors = [x for x in example_inputs if isinstance(x, torch.Tensor)]
    if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        return "needs autograd"
    if meta.random_ops:
        return "random operators " + ", ".join(meta.random_ops)
    if meta.output_aliases_input:
        return "returns an input or a view of one"
    if meta.mutates_metadata:
        return "changes an input's metadata"
    devices = {t.device for t in tensors}
    if len(devices) != 1 or next(iter(devices)).type != "paras":
        return "inputs not on one paras device"
    import torch_paras
    if not torch_paras.is_gpu_device(next(iter(devices)).index or 0):
        return "not a GPU device"
    return ""


def _buffer_like(t: torch.Tensor) -> torch.Tensor:
    from torch._prims_common import is_non_overlapping_and_dense

    if is_non_overlapping_and_dense(t):
        return torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device=t.device)
    return torch.empty(t.shape, dtype=t.dtype, device=t.device)


_staged: dict = {}
_held: dict = {}


def _holds(buf, src) -> bool:
    held = _held.get(id(buf))
    return held is not None and held[0]() is src and held[1] == src._version


def _load(buf, src):
    if _holds(buf, src):
        stats["inputs already loaded"] += 1
        return
    buf.copy_(src)
    _held[id(buf)] = (weakref.ref(src), src._version)


def _input_buffer(t: torch.Tensor) -> torch.Tensor:
    """The input buffer for `t` at capture: the one another capture made for
    this very tensor, or a new one."""
    entry = _staged.get(id(t))
    if entry is not None:
        ref, version, buf = entry
        if (ref() is t and version == t._version and buf.shape == t.shape
                and buf.stride() == t.stride() and buf.dtype == t.dtype and buf.device == t.device):
            return buf
    for k in [k for k, (r, _, _) in _staged.items() if r() is None]:
        del _staged[k]
    buf = _buffer_like(t)
    _staged[id(t)] = (weakref.ref(t), t._version, buf)
    return buf


def _shape_key(args):
    """What a capture depends on besides tensor contents: tensor layouts and
    the values of other arguments."""
    key = []
    for a in args:
        if isinstance(a, torch.Tensor):
            key.append((a.shape, a.stride(), a.storage_offset(), a.dtype, a.device))
        elif isinstance(a, (tuple, list)):
            key.append(_shape_key(a))
        else:
            key.append(a)
    return tuple(key)


class _Capture:
    """One capture of a graph, for one set of input shapes."""

    def __init__(self, owner, args, graph, outputs, buffers, captured, in_place):
        meta = owner.meta
        static = set(meta.static_inputs)
        self.graph = graph
        self.buffers = tuple(buffers)
        self.captured = captured
        self.static_ptrs = tuple((i, args[i].data_ptr()) for i in sorted(static | in_place)
                                 if isinstance(args[i], torch.Tensor))
        top = {i: buf for i, j, buf in buffers if j is None}
        self.copy_back = tuple((i, top[i]) for i in meta.mutated_inputs if i in top)
        self.single = not isinstance(outputs, (tuple, list))
        self.outputs = (outputs,) if self.single else outputs
        self.is_tuple = isinstance(outputs, tuple)
        private = set(meta.private_outputs)
        self.tensor_outputs = tuple(i for i, o in enumerate(self.outputs)
                                    if isinstance(o, torch.Tensor) and i not in private)
        self.private_tensors = tuple(o for i, o in enumerate(self.outputs)
                                     if i in private and isinstance(o, torch.Tensor) and not o.is_inference())
        self.checked_outputs = tuple(i for i, o in enumerate(self.outputs)
                                     if isinstance(o, torch.Tensor))
        for o in self.outputs:
            if isinstance(o, torch.Tensor):
                _output_ptrs.add(o.data_ptr())

    def inputs_in_place(self, args) -> bool:
        return all(args[i].data_ptr() == ptr for i, ptr in self.static_ptrs)

    def run(self, args):
        stats["replays"] += 1
        for i, j, buf in self.buffers:
            _load(buf, args[i] if j is None else args[i][j])
        self.graph._replay_versioned()
        for t in self.private_tensors:
            torch.autograd.graph.increment_version(t)
        return self.results(args)

    def results(self, args):
        for i, buf in self.copy_back:
            args[i].copy_(buf)
        out = list(self.outputs)
        for i in self.tensor_outputs:
            out[i] = out[i].clone()
        if self.single:
            return out[0]
        return tuple(out) if self.is_tuple else out


class Replayed:
    """Callable running `forward` through captures, one per input shape;
    see the module docstring."""

    def __init__(self, forward, meta: GraphMeta, report, piece=False):
        self.forward = forward
        self.meta = meta
        self.report = report
        self.piece = piece
        self.captures = {}
        self.seen = collections.Counter()
        self.failed = False

    def __call__(self, *args):
        if self.failed:
            return self.forward(*args)
        key = _shape_key(args)
        capture = self.captures.get(key)
        if capture is not None:
            if not capture.inputs_in_place(args):
                stats["inputs moved"] += 1
                return self.forward(*args)
            if VERIFY:
                return self._verify(capture, args)
            return capture.run(args)
        self.seen[key] += 1
        if self.seen[key] < 2 or len(self.captures) >= MAX_SHAPES:
            stats["new shape" if self.seen[key] < 2 else "too many shapes"] += 1
            return self.forward(*args)
        return self._capture(key, args)

    def _capture(self, key, args):
        static = set(self.meta.static_inputs)
        tensors = []
        for i, a in enumerate(args):
            if isinstance(a, torch.Tensor):
                tensors.append((i, None, a))
            elif isinstance(a, (tuple, list)):
                for j, e in enumerate(a):
                    if isinstance(e, torch.Tensor):
                        tensors.append((i, j, e))
                    elif isinstance(e, (tuple, list, dict)):
                        return self._fail("nested input", args)
            elif isinstance(a, dict):
                return self._fail("dict input", args)
        devices = {t.device for _, _, t in tensors}
        if len(devices) != 1 or next(iter(devices)).type != "paras":
            return self._fail("inputs not on one paras device", args)
        device = next(iter(devices))
        index = device.index or 0
        pool = _pools.get(index)
        if pool is None:
            pool = _pools[index] = graphs.new_pool()
        if self.meta.private_outputs:
            pool = None
        buffers = []
        in_place = set()
        captured = [list(a) if isinstance(a, (tuple, list)) else a for a in args]
        for i, j, t in tensors:
            if j is None and i in static:
                continue
            if j is None and t.data_ptr() in _output_ptrs:
                in_place.add(i)
                continue
            buf = _input_buffer(t)
            _load(buf, t)
            buffers.append((i, j, buf))
            if j is None:
                captured[i] = buf
            else:
                captured[i][j] = buf
        captured = [type(a)(c) if isinstance(a, (tuple, list)) else c
                    for a, c in zip(args, captured)]
        g = graphs.Graph()
        try:
            with graphs.graph(g, device=device, pool=pool):
                outputs = self.forward(*captured)
        except Exception as e:
            return self._fail(f"capture failed ({str(e).splitlines()[0][:160]})", args)
        capture = _Capture(self, args, g, outputs, buffers, captured, in_place)
        self.captures[key] = capture
        stats["captures"] += 1
        g.replay()
        return capture.results(args)

    def _fail(self, why, args):
        self.failed = True
        if self.piece:
            self.report.replay += f"; a piece is not replayed: {why}"
        else:
            self.report.replay = f"not replayed: {why}"
        return self.forward(*args)

    def _verify(self, capture, args):
        replayed = capture.run(args)
        normal = self.forward(*args)
        normal = (normal,) if capture.single else normal
        got = (replayed,) if capture.single else replayed
        for i in capture.checked_outputs:
            if not torch.equal(got[i], normal[i]):
                diff = (got[i].float() - normal[i].float()).abs().max().item()
                mismatches.append((self.report.frame, self.report.graph_id, self.piece, i,
                                   tuple(got[i].shape), diff))
        return replayed


def _is_tensor_op(node) -> bool:
    value = node.meta.get("example_value")
    return isinstance(value, torch.Tensor) and not (
        node.op == "call_function" and node.target is operator.getitem)


def _replayable_node(node, volatile) -> bool:
    """`node` computes tensors or sizes, changes nothing else, and its shapes
    involve no volatile size -- so a capture of it holds for every call with
    the same input shapes."""
    from .passes import _may_mutate

    if node.op not in ("call_function", "call_method") or _may_mutate(node):
        return False
    value = node.meta.get("example_value")
    if value is None or not isinstance(
            value, (torch.Tensor, int, float, bool, torch.SymInt, torch.Size, list, tuple)):
        return False
    used = _symbols(value)
    for arg in (*node.args, *node.kwargs.values()):
        for a in (arg if isinstance(arg, (list, tuple)) else (arg,)):
            if isinstance(a, fx.Node):
                used |= _symbols(a.meta.get("example_value"))
            else:
                used |= _symbols(a)
    return not (used & volatile)


_UNKNOWN = "unknown storage"


def _storages(value):
    """Storages of the tensors in `value`; _UNKNOWN for a tensor whose storage
    cannot be inspected (a vmap BatchedTensor), which counts as shared."""
    from torch.multiprocessing.reductions import StorageWeakRef

    values = value if isinstance(value, (list, tuple)) else (value,)
    out = set()
    for v in values:
        if isinstance(v, torch.Tensor):
            try:
                out.add(StorageWeakRef(v.untyped_storage()))
            except (NotImplementedError, RuntimeError):
                out.add(_UNKNOWN)
    return out


def split_pieces(gm: fx.GraphModule, meta: GraphMeta, report, volatile,
                 min_ops: int = MIN_PIECE_OPS):
    """`gm` split into runs of operators whose shapes involve no `volatile`
    size, each replayed, and the rest; None if no run has `min_ops` tensor
    operators."""
    from torch.fx.passes.split_module import split_module

    nodes = [n for n in gm.graph.nodes if n.op not in ("placeholder", "output")]
    flags = {n: _replayable_node(n, volatile) for n in nodes}
    part, current, assign, runs = 0, None, {}, {}
    for n in nodes:
        if flags[n] != current:
            part += 1
            current = flags[n]
        assign[n] = part
        if current:
            runs.setdefault(part, []).append(n)
    worth = {p for p, members in runs.items() if sum(map(_is_tensor_op, members)) >= min_ops}
    if not worth:
        return None
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    static_names = {placeholders[i].name for i in meta.static_inputs if i < len(placeholders)}
    graph_out = next(n for n in gm.graph.nodes if n.op == "output").args[0]
    escaping = set()
    for o in (graph_out if isinstance(graph_out, (list, tuple)) else (graph_out,)):
        if isinstance(o, fx.Node):
            escaping |= _storages(o.meta.get("example_value"))
    by_name = {n.name: n for n in gm.graph.nodes}
    split = split_module(gm, None, lambda n: assign[n], keep_original_order=True)
    pieces = 0
    for name, sub in list(split.named_children()):
        if int(name.rsplit("_", 1)[1]) not in worth:
            continue
        inputs = [n for n in sub.graph.nodes if n.op == "placeholder"]
        outputs = next(n for n in sub.graph.nodes if n.op == "output").args[0]
        outputs = outputs if isinstance(outputs, (list, tuple)) else (outputs,)
        in_storages = set()
        for p in inputs:
            orig = by_name.get(p.name)
            if orig is not None:
                in_storages |= _storages(orig.meta.get("example_value"))
        out_storages = set()
        for o in outputs:
            if isinstance(o, fx.Node):
                out_storages |= _storages(o.meta.get("example_value"))
        if in_storages & out_storages:
            continue
        piece_meta = GraphMeta(
            static_inputs=tuple(i for i, p in enumerate(inputs) if p.name in static_names),
            private_outputs=tuple(k for k, o in enumerate(outputs) if isinstance(o, fx.Node)
                                  and not (_storages(o.meta.get("example_value")) & escaping)))
        delattr(split, name)
        object.__setattr__(split, name, Replayed(sub.forward, piece_meta, report, piece=True))
        pieces += 1
    if not pieces:
        return None
    split.recompile()
    report.replay = f"replayed in {pieces} pieces"
    return split


class Split:
    """Callable for a dynamic-shape graph: runs `run` while it observes which
    sizes change between calls, then the graph split into replayed pieces;
    see the module docstring."""

    def __init__(self, gm, run, example_inputs, meta: GraphMeta, report):
        self.gm, self.run, self.meta, self.report = gm, run, meta, report
        self.split = None
        self.calls = 0
        self.sources = {}
        for i, x in enumerate(example_inputs):
            if isinstance(x, torch.SymInt) and x.node.expr.is_symbol:
                self.sources.setdefault(x.node.expr, (i, None))
            elif isinstance(x, torch.Tensor):
                for d, size in enumerate(x.shape):
                    if isinstance(size, torch.SymInt) and size.node.expr.is_symbol:
                        self.sources.setdefault(size.node.expr, (i, d))
        self.symbols = set().union(*map(_symbols, example_inputs))
        self.values = {s: set() for s in self.sources}
        report.replay = "not replayed yet: observing sizes"

    def __call__(self, *args):
        if self.split is not None:
            return self.split(*args)
        self.calls += 1
        for s, (i, d) in self.sources.items():
            self.values[s].add(int(args[i]) if d is None else args[i].shape[d])
        if self.calls >= OBSERVE_CALLS:
            volatile = {s for s, v in self.values.items() if len(v) > 1}
            volatile |= self.symbols - set(self.sources)
            split = split_pieces(self.gm, self.meta, self.report, volatile)
            if split is None:
                self.report.replay = "not replayed: no piece without changing sizes"
                self.split = self.run
            else:
                self.split = split
        return self.run(*args)
