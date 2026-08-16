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

Installation report: ``python -m torch_paras``.

Prints JSON describing what this install can actually do, so a failure is
diagnosed before a long benchmark rather than during it. Exits non-zero when
the native extension did not load.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import torch

import torch_paras


def _compiler() -> dict:
    """Locate parascc and ask it to check itself."""
    home = os.environ.get("PARAS_HOME")
    path = Path(home) / "bin" / "parascc" if home else None
    if path is None or not path.is_file():
        found = shutil.which("parascc")
        path = Path(found) if found else None
    if path is None:
        return {"status": "missing",
                "error": "parascc not found; set PARAS_HOME or put it on PATH"}
    report = {"status": "ok", "path": str(path)}
    try:
        proc = subprocess.run([str(path), "--self-check"], check=False,
                              text=True, capture_output=True, timeout=120)
    except Exception as exc:
        report["status"] = "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report
    if proc.returncode != 0:
        report["status"] = "error"
        report["returncode"] = proc.returncode
        report["stderr"] = proc.stderr.strip()[:2000]
        return report
    try:
        report.update(json.loads(proc.stdout))
        report["status"] = "ok"
    except ValueError:
        report["self_check"] = proc.stdout.strip()[:2000]
    return report


def _extension() -> dict:
    """Whether the built _C extension is present, and what it enumerates."""
    try:
        loaded = torch_paras.is_available()
        count = torch_paras.device_count()
        return {
            "loaded": bool(loaded),
            "backend": torch_paras.backend_name(),
            "device_count": count,
            "devices": [torch_paras.device_name(i) for i in range(count)],
        }
    except Exception as exc:
        return {"loaded": False, "error": f"{type(exc).__name__}: {exc}"}


def report() -> dict:
    extension = _extension()
    return {
        "torch_paras": getattr(torch_paras, "__version__", None),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "hip": getattr(torch.version, "hip", None),
        "accelerator_available": torch.cuda.is_available(),
        "optimizer_available": hasattr(torch_paras, "autotune"),
        "extension": extension,
        "compiler": _compiler(),
    }


def main() -> int:
    value = report()
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if value["extension"].get("loaded") else 1


raise SystemExit(main())
