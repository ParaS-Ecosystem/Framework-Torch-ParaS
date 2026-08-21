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

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass
class CheckpointResult:
    requested: bool
    loaded: bool
    path: str | None
    strict: bool
    missing_keys: list[str]
    unexpected_keys: list[str]
    error: str | None
    profile: str | None
    model_kwargs: dict[str, Any]


def load_checkpoint_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint config must be a JSON object")
    return payload


def get_entry(config: dict[str, Any], model: str, scale: int) -> dict[str, Any] | None:
    model_entry = config.get(model)
    if not isinstance(model_entry, dict):
        return None
    entry = model_entry.get(str(scale), model_entry.get("default"))
    return entry if isinstance(entry, dict) else None


def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("params_ema", "params", "state_dict", "model", "net_g", "generator"):
            candidate = payload.get(key)
            if isinstance(candidate, dict) and candidate:
                return candidate
        if payload and all(isinstance(value, torch.Tensor) for value in payload.values()):
            return payload
    raise ValueError("Unable to locate a tensor state_dict in checkpoint")


def _normalize_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = ("module.", "model.", "net_g.", "generator.")
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        normalized[new_key] = value
    return normalized


def load_into_model(model: nn.Module, entry: dict[str, Any] | None) -> CheckpointResult:
    if entry is None:
        return CheckpointResult(False, False, None, True, [], [], None, None, {})

    path_value = entry.get("path")
    strict = bool(entry.get("strict", True))
    profile = entry.get("profile")
    model_kwargs = entry.get("model_kwargs", {})
    if not isinstance(model_kwargs, dict):
        raise ValueError("model_kwargs must be a JSON object")
    if not path_value:
        return CheckpointResult(True, False, None, strict, [], [], "Checkpoint path is empty", profile, model_kwargs)

    path = Path(str(path_value)).expanduser()
    if not path.exists():
        return CheckpointResult(True, False, str(path), strict, [], [], "Checkpoint file does not exist", profile, model_kwargs)

    try:
        payload = torch.load(path, map_location="cpu")
        state = _normalize_keys(_extract_state_dict(payload))
        incompatible = model.load_state_dict(state, strict=strict)
        return CheckpointResult(
            True,
            True,
            str(path.resolve()),
            strict,
            list(incompatible.missing_keys),
            list(incompatible.unexpected_keys),
            None,
            profile,
            model_kwargs,
        )
    except Exception as exc:
        return CheckpointResult(
            True,
            False,
            str(path),
            strict,
            [],
            [],
            f"{type(exc).__name__}: {exc}",
            profile,
            model_kwargs,
        )
