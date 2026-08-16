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

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from PIL import Image, ImageDraw

from .data import tensor_to_pil


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _safe_mean(values: list[float]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return mean(finite) if finite else None


def write_markdown_summary(path: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("model", "-")), str(row.get("dataset", "-")))].append(row)

    lines = [
        "# Torch-ParaS Super-Resolution Run Summary",
        "",
        f"- Mode: `{metadata.get('mode')}`",
        f"- Device: `{metadata.get('device')}`",
        f"- Scale: `x{metadata.get('scale')}`",
        f"- Profile: `{metadata.get('profile')}`",
        f"- Quality claims valid: `{metadata.get('quality_claims_valid')}`",
        "",
        "No values in this report are pre-filled. Every row is produced by the current execution.",
        "",
        "| Model | Dataset | Passed/Total | Mean latency (ms) | Mean PSNR-Y | Mean SSIM-Y | Checkpoint |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for (model, dataset), items in sorted(grouped.items()):
        passed = [item for item in items if item.get("status") == "PASS"]
        latency = _safe_mean([float(item["latency_ms"]) for item in passed if item.get("latency_ms") is not None])
        psnr_values = _safe_mean([float(item["psnr_y"]) for item in passed if item.get("psnr_y") is not None])
        ssim_values = _safe_mean([float(item["ssim_y"]) for item in passed if item.get("ssim_y") is not None])
        checkpoint = "yes" if any(item.get("checkpoint_loaded") for item in items) else "no"
        lines.append(
            "| {} | {} | {}/{} | {} | {} | {} | {} |".format(
                model,
                dataset,
                len(passed),
                len(items),
                f"{latency:.3f}" if latency is not None else "-",
                f"{psnr_values:.4f}" if psnr_values is not None else "-",
                f"{ssim_values:.6f}" if ssim_values is not None else "-",
                checkpoint,
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `compatibility` mode validates construction, device transfer, forward execution, timing and optional CPU parity. Random weights are not quality evidence.",
            "- `quality` mode runs only models with successfully loaded checkpoints. PSNR/SSIM are then eligible for reporting.",
            "- A failure can indicate a missing operator, unsupported tensor layout/dtype, device-copy issue, model bug or checkpoint mismatch. Inspect `failures.jsonl` and the traceback column.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def save_snapshot_set(output_dir: Path, lr, bicubic, sr, hr, labels: dict[str, str]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    images = {
        "lr": tensor_to_pil(lr),
        "bicubic": tensor_to_pil(bicubic),
        "sr": tensor_to_pil(sr),
        "hr": tensor_to_pil(hr),
    }
    paths: dict[str, str] = {}
    for name, image in images.items():
        path = output_dir / f"{name}.png"
        image.save(path)
        paths[name] = str(path)

    target_size = images["hr"].size
    resized = []
    resampling = getattr(Image, "Resampling", Image)
    for key in ("lr", "bicubic", "sr", "hr"):
        method = resampling.NEAREST if key == "lr" else resampling.BICUBIC
        resized.append(images[key].resize(target_size, method))

    top = 28
    bottom = 28
    canvas = Image.new("RGB", (target_size[0] * 4, top + target_size[1] + bottom), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (key, image) in enumerate(zip(("LR", "Bicubic", "Torch-ParaS SR", "HR"), resized)):
        canvas.paste(image, (index * target_size[0], top))
        draw.text((index * target_size[0] + 4, 7), key, fill="black")
    footer = " | ".join(f"{key}={value}" for key, value in labels.items())
    draw.text((4, top + target_size[1] + 7), footer[:260], fill="black")
    comparison_path = output_dir / "comparison.png"
    canvas.save(comparison_path)
    paths["comparison"] = str(comparison_path)
    return paths
