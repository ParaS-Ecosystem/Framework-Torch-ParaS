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

"""Device-side expert dispatch for mixture-of-experts layers.

MoE layers usually run their routed experts in a loop over the experts that
received tokens, with sizes read on the host: one device synchronisation per
layer, and a Dynamo graph break (the loop depends on the routing).
``fuse_moe_experts(model)`` replaces that loop with ``paras::moe_experts``
(csrc/ops/moe_ops.cpp), one operator with a static output shape: no host
synchronisation for decode-sized batches, no graph break, and the layer can
be captured for graph replay. Recognised forms:

* a method ``moe_infer(x, topk_ids, topk_weight)`` (DeepSeek-V2/V3 and Param2
  remote code) or ``moe(...)`` (transformers' DeepSeek-V3 before 5.0) over a
  ModuleList ``experts`` of SwiGLU MLPs;
* transformers' ``*SparseMoeBlock.forward`` before 5.0 (Mixtral, Qwen2-MoE,
  Qwen3-MoE, OLMoE): softmax routing, top-k, optional renormalisation, the
  expert loop, and Qwen2-MoE's gated shared expert; the routing is computed
  as there;
* transformers' ``Experts`` modules from 5.0 (``forward(hidden_states,
  top_k_index, top_k_weights)``, parameters ``gate_up_proj`` [E, 2I, H] and
  ``down_proj`` [E, H, I]); the halves of ``gate_up_proj`` are used in place.

SwiGLU MLPs are ``gate_proj``, ``up_proj``, ``down_proj`` (or Mixtral's
``w1``, ``w3``, ``w2``) without bias, with SiLU. Their weights are stacked once
per layer ([E, I, H] and [E, H, I]) and every expert's ``nn.Linear`` weight
becomes a view of the stack, so no memory is added and training sees the same
parameters. While a block is in training mode or autograd needs its
gradients, the original method runs.

A layer is substituted only if its weights are on a paras device with one
dtype, it has no expert parallelism (``ep_size`` 1), and its own method and the
operator agree on a random input. Apply it after the model is on its device
and in its dtype (``model.to`` afterwards copies the parameters and the
stacks separately).
"""

import types
import warnings

import torch
import torch.nn.functional as F
from torch import nn

from . import ops

_ATTR = "_paras_moe_original"
_NAMES = (("gate_proj", "up_proj", "down_proj"), ("w1", "w3", "w2"))


def _silu(act) -> bool:
    return isinstance(act, nn.SiLU) or type(act).__name__ in ("SiLU", "SiLUActivation")


def _swiglu_experts(experts):
    """The (gate, up, down) attribute names if ``experts`` is a ModuleList of
    equal SwiGLU MLPs on a paras device, else None."""
    if not isinstance(experts, nn.ModuleList) or len(experts) == 0:
        return None
    for names in _NAMES:
        first = experts[0]
        if not all(isinstance(getattr(first, n, None), nn.Linear) for n in names):
            continue
        for e in experts:
            projs = [getattr(e, n, None) for n in names]
            if not all(isinstance(p, nn.Linear) and p.bias is None for p in projs):
                return None
            if not _silu(getattr(e, "act_fn", None)):
                return None
            for p, q in zip(projs, (getattr(first, n) for n in names)):
                if p.weight.shape != q.weight.shape or p.weight.dtype != q.weight.dtype \
                        or p.weight.device != q.weight.device:
                    return None
        g, u, d = (getattr(first, n).weight for n in names)
        if g.device.type == "paras" and u.shape == g.shape and d.shape == (g.shape[1], g.shape[0]):
            return names
        return None
    return None


def _experts_module(module: nn.Module) -> bool:
    gu, down = getattr(module, "gate_up_proj", None), getattr(module, "down_proj", None)
    if not (isinstance(gu, nn.Parameter) and isinstance(down, nn.Parameter)):
        return False
    if gu.dim() != 3 or down.dim() != 3 or gu.shape[1] % 2:
        return False
    experts, inter = gu.shape[0], gu.shape[1] // 2
    return (down.shape == (experts, gu.shape[2], inter) and gu.dtype == down.dtype
            and gu.device.type == "paras" and down.device == gu.device
            and getattr(module, "has_gate", True) and getattr(module, "is_concatenated", True)
            and not getattr(module, "is_transposed", False) and not getattr(module, "has_bias", False)
            and _silu(getattr(module, "act_fn", None)))


def _sparse_block(module: nn.Module) -> bool:
    gate = getattr(module, "gate", None)
    return (type(module).__name__.endswith("SparseMoeBlock") and isinstance(gate, nn.Linear)
            and gate.bias is None and isinstance(getattr(module, "top_k", None), int)
            and gate.out_features == len(getattr(module, "experts", ())))


def _stack(module: nn.Module, names) -> None:
    for slot, n in zip(("gate", "up", "down"), names):
        params = [getattr(e, n).weight for e in module.experts]
        stacked = torch.stack([p.data for p in params])
        for i, p in enumerate(params):
            p.data = stacked[i]
        module.register_buffer(f"_paras_moe_{slot}", stacked, persistent=False)


def _needs_grad(module: nn.Module, *tensors) -> bool:
    return torch.is_grad_enabled() and (any(t.requires_grad for t in tensors)
                                        or any(p.requires_grad for p in module.parameters()))


def _stacked_experts(self, x, topk_ids, topk_weight):
    return torch.ops.paras.moe_experts(x, topk_ids, topk_weight, self._paras_moe_gate,
                                       self._paras_moe_up, self._paras_moe_down)


@torch.no_grad()
def _fused_moe_infer(self, x, topk_ids, topk_weight):
    return _stacked_experts(self, x, topk_ids, topk_weight)


def _fused_method(self, x, topk_ids, topk_weight):
    if _needs_grad(self, x, topk_weight):
        return getattr(self, _ATTR)(x, topk_ids, topk_weight)
    return _stacked_experts(self, x, topk_ids, topk_weight)


def _fused_sparse_block(self, hidden_states):
    if self.training or _needs_grad(self, hidden_states):
        return getattr(self, _ATTR)(hidden_states)
    batch, seq, hidden = hidden_states.shape
    x = hidden_states.view(-1, hidden)
    router_logits = self.gate(x)
    weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    weights, selected = torch.topk(weights, self.top_k, dim=-1)
    if getattr(self, "norm_topk_prob", True):
        weights /= weights.sum(dim=-1, keepdim=True)
    out = _stacked_experts(self, x, selected, weights.to(x.dtype))
    if getattr(self, "shared_expert", None) is not None:
        out = out + F.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
    return out.reshape(batch, seq, hidden), router_logits


def _fused_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    if _needs_grad(self, hidden_states, top_k_weights):
        return getattr(self, _ATTR)(hidden_states, top_k_index, top_k_weights)
    inter = self.down_proj.shape[-1]
    return torch.ops.paras.moe_experts(hidden_states, top_k_index, top_k_weights,
                                       self.gate_up_proj[:, :inter], self.gate_up_proj[:, inter:],
                                       self.down_proj)


def _close(ref, got, dtype) -> bool:
    ref, got = ref.float(), got.float()
    if ref.shape != got.shape or not torch.isfinite(got).all():
        return False
    tol = 2e-2 if dtype in (torch.bfloat16, torch.float16) else 1e-4
    return bool((ref - got).norm() <= tol * ref.norm() + 1e-6)


@torch.no_grad()
def _agrees(kind, fused, original, w, k, tokens: int = 8) -> bool:
    experts, hidden = w.shape[0], w.shape[2]
    g = torch.Generator().manual_seed(0)
    x = (torch.randn(tokens, hidden, generator=g) * 0.5).to(w.device, w.dtype)
    if kind == "block":
        ref, got = original(x[None]), fused(x[None])
        return _close(ref[0], got[0], w.dtype) and torch.equal(ref[1], got[1])
    ids = torch.rand(tokens, experts, generator=g).topk(k, dim=-1).indices.to(w.device)
    weight = torch.rand(tokens, k, generator=g).softmax(-1).to(w.device)
    return _close(original(x, ids, weight), fused(x, ids, weight), w.dtype)


def _layer(module: nn.Module):
    """(kind, method name, fused function) for a recognised MoE layer, else None."""
    if _experts_module(module):
        return "experts", "forward", _fused_experts_forward
    names = _swiglu_experts(getattr(module, "experts", None))
    if names is None:
        return None
    if callable(getattr(module, "moe_infer", None)):
        return "method", "moe_infer", _fused_moe_infer
    if callable(getattr(module, "moe", None)):
        return "method", "moe", _fused_method
    if _sparse_block(module):
        return "block", "forward", _fused_sparse_block
    return None


def fuse_moe_experts(model: nn.Module) -> int:
    """Substitutes ``paras::moe_experts`` for the routed-expert loop of the
    MoE layers of ``model`` (see the module docstring); returns the number of
    layers substituted."""
    ops.register()
    done = 0
    for name, module in model.named_modules():
        if hasattr(module, _ATTR) or getattr(module, "ep_size", 1) != 1:
            continue
        found = _layer(module)
        if found is None:
            continue
        kind, method, fused = found
        if kind == "experts":
            w = module.gate_up_proj
        else:
            _stack(module, _swiglu_experts(module.experts))
            w = module._paras_moe_gate
        original = getattr(module, method)
        setattr(module, _ATTR, original)
        k = getattr(module, "top_k", None) or getattr(module, "num_experts_per_tok", None) \
            or getattr(getattr(module, "config", None), "num_experts_per_tok", 2)
        bound = types.MethodType(fused, module)
        if not _agrees(kind, bound, original, w, min(w.shape[0], int(k))):
            warnings.warn(f"torch_paras: MoE layer {name or type(module).__name__} kept its "
                          f"{method} (paras::moe_experts gives a different result)")
            continue
        setattr(module, method, bound)
        done += 1
    return done
