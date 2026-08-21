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

from __future__ import annotations

import importlib
from typing import Any

import torch
from torch import Tensor, nn

from .models import build_model


def _load_factory(spec: str):
    if ":" not in spec:
        raise ValueError("External factory must use 'python.module:function' syntax")
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name)
    if not callable(factory):
        raise TypeError(f"External factory is not callable: {spec}")
    return factory


def build_selected_model(
    name: str,
    scale: int,
    default_profile: str,
    entry: dict[str, Any] | None,
) -> tuple[nn.Module, str, dict[str, Any]]:
    if entry and entry.get("factory"):
        factory = _load_factory(str(entry["factory"]))
        kwargs = dict(entry.get("factory_kwargs", {}))
        if bool(entry.get("pass_scale", True)):
            scale_arg = str(entry.get("scale_arg", "scale"))
            kwargs.setdefault(scale_arg, scale)
        model = factory(**kwargs)
        if not isinstance(model, nn.Module):
            raise TypeError(f"External factory returned {type(model).__name__}, expected torch.nn.Module")
        setattr(model, "scale", scale)
        setattr(model, "expects_preupsampled", bool(entry.get("expects_preupsampled", False)))
        setattr(model, "architecture_name", str(entry.get("architecture", f"external:{entry['factory']}")))
        return model, "external", kwargs

    selected_profile = str(entry.get("profile", default_profile)) if entry else default_profile
    model_kwargs = dict(entry.get("model_kwargs", {})) if entry else {}
    return build_model(name, scale, selected_profile, model_kwargs), selected_profile, model_kwargs


def unwrap_output(output: Any) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if isinstance(output, dict):
        for key in ("sr", "output", "result", "image", "prediction", "pred"):
            value = output.get(key)
            if isinstance(value, Tensor):
                return value
    if isinstance(output, (tuple, list)):
        for value in output:
            if isinstance(value, Tensor):
                return value
    raise TypeError(f"Model output is {type(output).__name__}; no tensor output could be selected")
