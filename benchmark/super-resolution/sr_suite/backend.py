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
import os
import platform
import sys
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch


@dataclass
class BackendStatus:
    requested_device: str
    resolved_device: str
    imported_module: str | None
    available: bool
    error: str | None
    torch_version: str
    python_version: str
    platform: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _module_candidates(explicit: str | None = None) -> list[str]:
    values: list[str] = []
    if explicit:
        values.extend(item.strip() for item in explicit.split(",") if item.strip())
    env_value = os.environ.get("TORCH_PARAS_IMPORT", "")
    values.extend(item.strip() for item in env_value.split(",") if item.strip())
    values.extend(["torch_paras", "pytorch_sycl_backend"])
    seen: set[str] = set()
    return [value for value in values if not (value in seen or seen.add(value))]


def import_paras_backend(explicit: str | None = None) -> tuple[str | None, str | None]:
    errors: list[str] = []
    for module_name in _module_candidates(explicit):
        try:
            module = importlib.import_module(module_name)
            ensure = getattr(module, "ensure_backend_loaded", None)
            if callable(ensure):
                ensure()
            return module_name, None
        except Exception as exc:
            errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
    return None, " | ".join(errors)


def _candidate_devices(requested: str) -> Iterable[str]:
    if requested != "auto":
        yield requested
        return
    env_device = os.environ.get("TORCH_PARAS_DEVICE")
    if env_device:
        yield env_device
    yield "paras:0"
    yield "privateuseone:0"


def resolve_device(requested: str = "auto", backend_import: str | None = None) -> tuple[torch.device, BackendStatus]:
    requested = requested.strip().lower()
    if requested == "cpu":
        status = BackendStatus(
            requested_device=requested,
            resolved_device="cpu",
            imported_module=None,
            available=True,
            error=None,
            torch_version=torch.__version__,
            python_version=sys.version.split()[0],
            platform=platform.platform(),
        )
        return torch.device("cpu"), status

    module_name, import_error = import_paras_backend(backend_import)
    errors: list[str] = []
    if import_error:
        errors.append(import_error)

    for candidate in _candidate_devices(requested):
        try:
            device = torch.device(candidate)
            probe = torch.empty((1,), dtype=torch.float32, device=device)
            probe_cpu = probe.to("cpu")
            del probe, probe_cpu
            status = BackendStatus(
                requested_device=requested,
                resolved_device=str(device),
                imported_module=module_name,
                available=True,
                error=" | ".join(errors) if errors else None,
                torch_version=torch.__version__,
                python_version=sys.version.split()[0],
                platform=platform.platform(),
            )
            return device, status
        except Exception as exc:
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")

    status = BackendStatus(
        requested_device=requested,
        resolved_device="unavailable",
        imported_module=module_name,
        available=False,
        error=" | ".join(errors) if errors else "No ParaS device candidate succeeded",
        torch_version=torch.__version__,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
    )
    raise RuntimeError(status.error)


def synchronize(device: torch.device, imported_module: str | None = None) -> None:
    if device.type == "cpu":
        return


    names = [device.type, "paras", "privateuseone"]
    for name in names:
        namespace = getattr(torch, name, None)
        sync = getattr(namespace, "synchronize", None) if namespace is not None else None
        if callable(sync):
            sync()
            return

    if imported_module:
        try:
            module = importlib.import_module(imported_module)
            sync = getattr(module, "synchronize", None)
            if callable(sync):
                sync()
        except Exception:
            pass
