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

"""Fusion passes over TorchDynamo graphs.

Each pass recognises one subgraph by its semantics (accepting the method,
function and operator spellings Dynamo emits, e.g. ``x.pow(2)`` and
``torch.pow(x, 2)``) and replaces it with a single fused ParaS kernel. A
pattern is only rewritten when the fused kernel computes the same function:
shapes, dtypes and devices are read from Dynamo's FakeTensor metadata, and
every intermediate must be private to the pattern.

The passes run before AOTAutograd, so the backward of a fused operator is the
registered ParaS backward kernel (see ops.py) rather than the backward of the
original decomposition.
"""

import operator
from collections import Counter
from typing import Optional

import torch
import torch.nn.functional as F
from torch import fx

_FUNCTIONS = {
    operator.mul: "mul", torch.mul: "mul",
    operator.add: "add", torch.add: "add",
    operator.pow: "pow", torch.pow: "pow",
    operator.neg: "neg", torch.neg: "neg",
    operator.getitem: "getitem",
    torch.rsqrt: "rsqrt",
    torch.mean: "mean",
    torch.cat: "cat", torch.concat: "cat",
    F.silu: "silu",
}


def _op(node) -> Optional[str]:
    if not isinstance(node, fx.Node):
        return None
    if node.op == "call_method":
        return node.target
    if node.op == "call_function":
        return _FUNCTIONS.get(node.target)
    return None


def _value(node):
    """FakeTensor (or real tensor for get_attr) a node produces, if known."""
    if not isinstance(node, fx.Node):
        return None
    value = node.meta.get("example_value")
    if value is None and node.op == "get_attr":
        obj = node.graph.owning_module
        for atom in node.target.split("."):
            obj = getattr(obj, atom, None)
        value = obj
    return value if isinstance(value, torch.Tensor) else None


def _on_paras(*nodes) -> bool:
    for n in nodes:
        v = _value(n)
        if v is None or v.device.type != "paras":
            return False
    return True


def _private(node, allowed_users) -> bool:
    """True when every user of `node` is one of `allowed_users`."""
    return all(u in allowed_users for u in node.users)


_INPLACE_FUNCTIONS = {
    operator.iadd, operator.isub, operator.imul, operator.itruediv,
    operator.ifloordiv, operator.ipow, operator.imod, operator.iand,
    operator.ior, operator.ixor, operator.ilshift, operator.irshift,
    operator.setitem,
}


def _may_mutate(node) -> bool:
    """Conservatively true when `node` may write to an existing tensor."""
    if node.op == "call_module":
        return True
    if "out" in node.kwargs:
        return True
    if node.op == "call_method":
        name = node.target
        return name.endswith("_") and not name.startswith("__")
    if node.op == "call_function":
        if node.target in _INPLACE_FUNCTIONS:
            return True
        if node.kwargs.get("inplace", False):
            return True
        if isinstance(node.target, torch._ops.OpOverload):
            return node.target._schema.is_mutable
        name = getattr(node.target, "__name__", "")
        return name.endswith("_") and not name.startswith("__")
    return False


def _no_mutation_within(nodes) -> bool:
    """True when no node outside `nodes` that may mutate a tensor lies between
    the first and the last of `nodes` in graph order. The fused operator is
    inserted at the last node and reads its operands there, so an in-place
    update of an operand inside the pattern's span would change its result."""
    members = set(nodes)
    graph_nodes = list(next(iter(members)).graph.nodes)
    positions = [graph_nodes.index(n) for n in members]
    return not any(_may_mutate(n) and n not in members
                   for n in graph_nodes[min(positions):max(positions) + 1])


def _erase(gm: fx.GraphModule, nodes) -> None:
    """Erases the replaced pattern's nodes, users first. Only the pattern is
    touched: a graph-wide dead-code pass could drop in-place operations whose
    results are unused."""
    pending = list(dict.fromkeys(nodes))
    while pending:
        progress = False
        for n in list(pending):
            if not n.users:
                gm.graph.erase_node(n)
                pending.remove(n)
                progress = True
        if not progress:
            raise RuntimeError("paras fusion: replaced pattern still has users")


def _operands(node):
    """Positional operands of a binary op, for both spellings."""
    if len(node.args) == 2 and not node.kwargs:
        return node.args
    return None


def _cast_source(node, dtype):
    """If `node` casts its input to `dtype` (x.to(dtype) / x.float()),
    return the input; otherwise None."""
    name = _op(node)
    if name == "to" and len(node.args) == 2 and node.args[1] == dtype and not node.kwargs:
        return node.args[0]
    if name == "float" and dtype == torch.float32 and len(node.args) == 1:
        return node.args[0]
    return None


def _is_last_dim(dim, ndim) -> bool:
    if isinstance(dim, (list, tuple)):
        if len(dim) != 1:
            return False
        dim = dim[0]
    return dim == -1 or dim == ndim - 1


def _match_variance(var, xf):
    """var == mean(xf ** 2, dim=-1, keepdim=True); returns the pattern nodes."""
    if _op(var) != "mean":
        return None
    args = list(var.args)
    kwargs = dict(var.kwargs)
    if len(args) < 1 or args[0] is None:
        return None
    sq = args[0]
    dim = args[1] if len(args) > 1 else kwargs.pop("dim", None)
    keepdim = args[2] if len(args) > 2 else kwargs.pop("keepdim", False)
    if kwargs or keepdim is not True or dim is None:
        return None
    xv = _value(xf)
    if xv is None or not _is_last_dim(dim, xv.dim()):
        return None
    name = _op(sq)
    ops = _operands(sq)
    if ops is None:
        return None
    if name == "pow" and ops[0] is xf and ops[1] == 2:
        return [var, sq]
    if name == "mul" and ops[0] is xf and ops[1] is xf:
        return [var, sq]
    return None


def _match_rms_norm(out):
    """`out` is the final `weight * normed` node."""
    if _op(out) != "mul":
        return None
    ops = _operands(out)
    if ops is None:
        return None
    out_v = _value(out)
    if out_v is None:
        return None
    for weight, cast_back in (ops, ops[::-1]):
        if not isinstance(weight, fx.Node) or not isinstance(cast_back, fx.Node):
            continue
        w_v = _value(weight)
        if w_v is None or w_v.dim() != 1:
            continue
        dtype = out_v.dtype
        normed = _cast_source(cast_back, dtype)
        cast_nodes = [cast_back]
        if normed is None:
            if dtype != torch.float32:
                continue
            normed, cast_nodes = cast_back, []
        if _op(normed) != "mul" or _operands(normed) is None:
            continue
        for xf, rs in (_operands(normed), _operands(normed)[::-1]):
            if _op(rs) != "rsqrt" or len(rs.args) != 1 or rs.kwargs:
                continue
            add = rs.args[0]
            if _op(add) != "add" or _operands(add) is None:
                continue
            var, eps = _operands(add)
            if isinstance(var, (int, float)) and isinstance(eps, fx.Node):
                var, eps = eps, var
            if not isinstance(eps, (int, float)) or isinstance(eps, bool):
                continue
            var_nodes = _match_variance(var, xf)
            if var_nodes is None:
                continue
            x = _cast_source(xf, torch.float32)
            xf_nodes = [xf]
            if x is None:
                x, xf_nodes = xf, []
            x_v, xf_v = _value(x), _value(xf)
            if x_v is None or xf_v is None or xf_v.dtype != torch.float32:
                continue
            if x_v.dtype != dtype or w_v.dtype != dtype:
                continue
            if tuple(w_v.shape) != (x_v.shape[-1],) or tuple(out_v.shape) != tuple(x_v.shape):
                continue
            if not _on_paras(x, weight):
                continue
            inner = [normed, rs, add] + var_nodes + cast_nodes
            members = set(inner) | set(xf_nodes) | {out}
            if not all(_private(n, members) for n in inner):
                continue
            if xf_nodes and not _private(xf, members):
                continue
            if not _no_mutation_within(members):
                continue
            return x, weight, float(eps), inner + xf_nodes
    return None


def fuse_rms_norm(gm: fx.GraphModule) -> int:
    count = 0
    for node in list(gm.graph.nodes):
        match = _match_rms_norm(node)
        if match is None:
            continue
        x, weight, eps, pattern = match
        with gm.graph.inserting_before(node):
            fused = gm.graph.call_function(torch.ops.paras.rms_norm.default, (x, weight, eps))
            out = gm.graph.call_function(operator.getitem, (fused, 0))
        out.meta["example_value"] = node.meta.get("example_value")
        node.replace_all_uses_with(out)
        _erase(gm, [node] + pattern)
        count += 1
    return count


def _silu_input(node):
    if _op(node) != "silu" or len(node.args) != 1:
        return None
    if node.kwargs.get("inplace", False):
        return None
    if any(k != "inplace" for k in node.kwargs):
        return None
    return node.args[0]


def _eager_silu_is_paras(value) -> bool:
    """Whether eager silu on tensors like ``value`` runs the ParaS kernel.
    Where CLAP-DNN runs it in the vendor DNN library (e.g. fp32 through
    oneDNN or cuDNN; on ROCm SiLU stays on ParaS), the fused kernel would not
    reproduce its rounding, so fusing would change results."""
    from .. import _C
    if not (hasattr(_C, "clap_dnn_enabled") and _C.clap_dnn_enabled()):
        return True
    query = getattr(_C, "clap_dnn_activation_uses_vendor", None)
    if query is None:
        return False
    dtype = str(value.dtype).replace("torch.", "")
    return not query(value.device.index or 0, dtype)


def fuse_swiglu(gm: fx.GraphModule) -> int:
    count = 0
    for node in list(gm.graph.nodes):
        if _op(node) != "mul" or _operands(node) is None:
            continue
        for act, up in (_operands(node), _operands(node)[::-1]):
            gate = _silu_input(act) if isinstance(act, fx.Node) else None
            if gate is None or not isinstance(up, fx.Node):
                continue
            g_v, u_v, o_v = _value(gate), _value(up), _value(node)
            if g_v is None or u_v is None or o_v is None:
                continue
            if g_v.shape != u_v.shape or g_v.dtype != u_v.dtype or o_v.dtype != g_v.dtype:
                continue
            if not g_v.dtype.is_floating_point or not _on_paras(gate, up):
                continue
            if not _eager_silu_is_paras(g_v):
                continue
            if not _private(act, {node}) or not _no_mutation_within([act, node]):
                continue
            with gm.graph.inserting_before(node):
                fused = gm.graph.call_function(torch.ops.paras.swiglu.default, (gate, up))
            fused.meta["example_value"] = node.meta.get("example_value")
            node.replace_all_uses_with(fused)
            _erase(gm, [node, act])
            count += 1
            break
    return count


def _half_slice(node, x_expected=None):
    """Returns (x, start, stop) for x[..., start:stop] over the last dim."""
    if _op(node) != "getitem" or len(node.args) != 2:
        return None
    x, index = node.args
    if x_expected is not None and x is not x_expected:
        return None
    if not isinstance(index, tuple) or len(index) != 2 or index[0] is not Ellipsis:
        return None
    sl = index[1]
    if not isinstance(sl, slice) or sl.step not in (None, 1):
        return None
    return x, sl.start, sl.stop


def fuse_rotate_half(gm: fx.GraphModule) -> int:
    count = 0
    for node in list(gm.graph.nodes):
        if _op(node) != "cat":
            continue
        tensors = node.args[0] if node.args else node.kwargs.get("tensors")
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
        if not isinstance(tensors, (list, tuple)) or len(tensors) != 2:
            continue
        neg, first = tensors
        if _op(neg) != "neg" or len(neg.args) != 1:
            continue
        second = _half_slice(neg.args[0])
        if second is None:
            continue
        x, s_start, s_stop = second
        first_s = _half_slice(first, x)
        if first_s is None:
            continue
        _, f_start, f_stop = first_s
        x_v = _value(x)
        if x_v is None or x_v.dim() == 0 or not _is_last_dim(dim, x_v.dim()):
            continue
        width = x_v.shape[-1]
        if not isinstance(width, int) or width % 2:
            continue
        h = width // 2
        if not (f_start in (None, 0) and f_stop == h and s_start == h and s_stop in (None, width)):
            continue
        if not _on_paras(x) or not x_v.dtype.is_floating_point:
            continue
        if not (_private(neg, {node}) and _private(neg.args[0], {neg}) and _private(first, {node})):
            continue
        if not _no_mutation_within([node, neg, neg.args[0], first]):
            continue
        with gm.graph.inserting_before(node):
            fused = gm.graph.call_function(torch.ops.paras.rotate_half.default, (x,))
        fused.meta["example_value"] = node.meta.get("example_value")
        node.replace_all_uses_with(fused)
        _erase(gm, [node, neg, neg.args[0], first])
        count += 1
    return count


PASSES = {
    "rms_norm": fuse_rms_norm,
    "swiglu": fuse_swiglu,
    "rotate_half": fuse_rotate_half,
}


def run(gm: fx.GraphModule, enabled) -> Counter:
    """Applies the enabled fusion passes; returns fusions applied per pass."""
    applied = Counter()
    for name, fn in PASSES.items():
        if name in enabled:
            n = fn(gm)
            if n:
                applied[name] += n
    if applied:
        gm.graph.lint()
        gm.recompile()
    return applied
