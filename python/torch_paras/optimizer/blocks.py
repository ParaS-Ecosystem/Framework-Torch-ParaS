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

"""One Dynamo cache per block instance for repeated, per-instance blocks.

Dynamo keeps compiled graphs per *code object*. The layers of a transformer
share one ``forward`` code object, but each layer specialises its graphs on
its own Python attributes -- ``layer_idx``, and through it its own slot of a
KV cache (``past_key_value.layers[layer_idx]``). When the layers run as
separate frames (a graph break inside the layer, e.g. an MoE's host read),
all their variants compete in one cache: lookups scan dozens of entries and
the frame soon hits Dynamo's recompile limit.

``specialize_repeated_blocks`` gives each instance of such a block its own
copy of the ``forward`` code object (bound as an instance attribute; same
bytecode, same globals), so each layer keeps its own few graphs. A block
qualifies when it is one of two or more instances of a class inside an
``nn.ModuleList`` / ``nn.Sequential`` and the instances differ in a Python
scalar attribute (int, bool, float or str) of the block or one of its
submodules (Param2: ``attention.layer_idx``). Blocks that differ only in their
weights -- an MoE's experts -- share graphs anyway and are left alone.

Each layer then legitimately needs more graphs than Dynamo's default
recompile limit (8) allows for one code object: uncached and cached prefill,
decode, each first static and then dynamic, per batch size -- a
transformers ``generate()`` workload reaches ~12. When blocks are
specialised, ``torch._dynamo.config.recompile_limit`` is therefore raised to
at least ``MIN_RECOMPILE_LIMIT`` (process-wide; each layer keeps its own
cache, so a lookup scans only that layer's entries).
"""

import types
from collections import defaultdict

from torch import nn

_SCALARS = (bool, int, float, str)
MIN_RECOMPILE_LIMIT = 32
_MARK = "_torch_paras_specialized_forward"


def _scalar_attributes(module: nn.Module) -> dict:
    """Python scalar attributes of `module` and its submodules, by path --
    what Dynamo specialises a block's graphs on (e.g. attention.layer_idx)."""
    out = {}
    for path, sub in module.named_modules():
        for k, v in vars(sub).items():
            if isinstance(v, _SCALARS) and k != "training" and not k.startswith("_"):
                out[f"{path}.{k}" if path else k] = v
    return out


def _clone_forward(module: nn.Module) -> bool:
    if "forward" in vars(module):
        return False
    func = getattr(type(module), "forward", None)
    if not isinstance(func, types.FunctionType) or hasattr(func, "__wrapped__"):
        return False
    code = func.__code__.replace(co_name=func.__code__.co_name)
    clone = types.FunctionType(code, func.__globals__, func.__name__,
                               func.__defaults__, func.__closure__)
    clone.__kwdefaults__ = func.__kwdefaults__
    clone.__qualname__ = func.__qualname__
    clone.__dict__.update(func.__dict__)
    setattr(clone, _MARK, True)
    module.forward = types.MethodType(clone, module)
    return True


def specialize_repeated_blocks(model: nn.Module) -> int:
    """Gives each qualifying block instance its own forward code object and,
    if any was specialised, raises Dynamo's recompile limit to at least
    MIN_RECOMPILE_LIMIT; returns how many were specialised."""
    count = 0
    for container in model.modules():
        if not isinstance(container, (nn.ModuleList, nn.Sequential)):
            continue
        by_class = defaultdict(list)
        for child in container.children():
            by_class[type(child)].append(child)
        for blocks in by_class.values():
            if len(blocks) < 2:
                continue
            attrs = [_scalar_attributes(b) for b in blocks]
            if all(a == attrs[0] for a in attrs):
                continue
            count += sum(_clone_forward(b) for b in blocks)
    if count:
        import torch._dynamo

        config = torch._dynamo.config
        config.recompile_limit = max(config.recompile_limit, MIN_RECOMPILE_LIMIT)
    return count


def unspecialize(model: nn.Module) -> int:
    """Removes the per-instance forward copies (e.g. before pickling the
    module); returns how many were removed."""
    count = 0
    for module in model.modules():
        fwd = vars(module).get("forward")
        if fwd is not None and getattr(getattr(fwd, "__func__", None), _MARK, False):
            del module.forward
            count += 1
    return count
