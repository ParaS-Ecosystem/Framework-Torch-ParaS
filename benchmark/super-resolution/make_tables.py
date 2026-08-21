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

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    payload = json.loads((path / "results.json").read_text())
    payload["_dir"] = str(path)
    return payload


def label(run: dict[str, Any]) -> str:
    env = run["environment"]
    if (env.get("compute_device") or "").startswith("cpu"):
        return "CPU"
    gpu = env.get("gpu") or "cpu"
    for short in ("V100", "H200", "MI300X", "A100"):
        if short.lower() in gpu.lower().replace("-", ""):
            return short
    return gpu


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(v) for v in values) / len(values))


def compiled_cells(run: dict[str, Any]) -> tuple[int, int]:
    rows = run["rows"]
    done = sum(1 for r in rows if r.get("pytorch_optimized_compiled"))
    return done, len(rows)


def pytorch_best_ms(model: dict[str, Any]) -> float:
    return min(model["pytorch_eager_total_ms"],
               model["pytorch_optimized_total_ms"])


def ratio_vs_best(model: dict[str, Any]) -> float:
    return pytorch_best_ms(model) / model["torch_paras_total_ms"]


def matrix(runs: list[dict[str, Any]]) -> tuple[int, int]:
    models = {row["model"] for run in runs for row in run["rows"]}
    datasets = {row["dataset"] for run in runs for row in run["rows"]}
    return len(models), len(datasets)


def table1(runs: list[dict[str, Any]]) -> str:
    out = ["## Table 1 — per-model latency", "",
           f"Summed median latency over the {matrix(runs)[1]} datasets. "
           "Lower is better."]
    for run in runs:
        done, total = compiled_cells(run)
        note = ("" if done == total else
                "  \n**All rows on this platform are marked ⚠: PyTorch's "
                "compiler did not run here, so \"PyTorch best\" is uncompiled "
                "PyTorch and the last column is not a compile-versus-compile "
                "ratio.**")
        out += ["", f"### {label(run)}{note}", "",
                "| Model | PyTorch eager (ms) | PyTorch tuned (ms) "
                "| PyTorch best (ms) | Torch-ParaS (ms) | vs eager | vs PyTorch best |",
                "|---|---|---|---|---|---|---|"]
        for m in run["model_summary"]:
            mark = "" if done == total else " ⚠"
            out.append(
                f"| {m['model'].upper()} | {m['pytorch_eager_total_ms']:.2f} "
                f"| {m['pytorch_optimized_total_ms']:.2f} "
                f"| {pytorch_best_ms(m):.2f} "
                f"| {m['torch_paras_total_ms']:.2f} "
                f"| {m['speedup']:.2f}x "
                f"| **{ratio_vs_best(m):.2f}x**{mark} |")
    return "\n".join(out)


def table2(runs: list[dict[str, Any]]) -> str:
    out = ["## Table 2 — Torch-ParaS vs the best PyTorch achieves", "",
           "Measured against whichever PyTorch configuration is faster on that "
           "machine, eager or tuned. This is the figure to quote.", "",
           "| Platform | Geomean | Median | Win rate | Compiled cells |",
           "|---|---|---|---|---|"]
    for run in runs:
        values = [ratio_vs_best(m) for m in run["model_summary"]]
        done, total = compiled_cells(run)
        note = f"{done}/{total}"
        if done < total:
            note += " ⚠"
        out.append(f"| {label(run)} | **{geomean(values):.2f}x** "
                   f"| {statistics.median(values):.2f}x "
                   f"| {100 * sum(1 for v in values if v > 1) / len(values):.0f}% | {note} |")
    if any(compiled_cells(r)[0] < compiled_cells(r)[1] for r in runs):
        out += ["", "⚠ Cells without a compiled PyTorch baseline: PyTorch's compiler "
                "could not run on that machine, so the comparison there is against "
                "the best uncompiled PyTorch, not against compiled PyTorch. Those "
                "rows are not a compile-versus-compile result."]
    return "\n".join(out)


def table3(runs: list[dict[str, Any]]) -> str:
    out = ["## Table 3 — where Torch-ParaS leads and where it does not", "",
           "Every model, sorted by margin over the best PyTorch achieves.", "",
           "| Platform | Model | PyTorch best (ms) | Torch-ParaS (ms) | ratio |",
           "|---|---|---|---|---|"]
    rows = [(label(r), m) for r in runs for m in r["model_summary"]]
    rows.sort(key=lambda item: ratio_vs_best(item[1]), reverse=True)
    for name, m in rows:
        out.append(f"| {name} | {m['model']} | {pytorch_best_ms(m):.2f} "
                   f"| {m['torch_paras_total_ms']:.2f} "
                   f"| {ratio_vs_best(m):.2f}x |")
    return "\n".join(out)


def table4(runs: list[dict[str, Any]]) -> str:
    out = ["## Table 4 — cross-platform summary", "",
           "Both ratios side by side, so neither can be read out of context.", "",
           "| Platform | GPU | vs eager (geomean) | vs PyTorch best (geomean) "
           "| Win rate | Fidelity | Max paras FP32 parity |", "|---|---|---|---|---|---|---|"]
    for run in runs:
        ms = run["model_summary"]
        eager = [m["speedup"] for m in ms]
        fair = [ratio_vs_best(m) for m in ms]
        rows = run["rows"]
        passed = sum(1 for r in rows if r["fidelity_pass"])
        parity = [r["direct_paras_fp32_parity_max_abs"] for r in rows
                  if r.get("direct_paras_fp32_parity_max_abs") is not None]
        out.append(
            f"| {label(run)} | {run['environment'].get('gpu')} "
            f"| {geomean(eager):.2f}x | **{geomean(fair):.2f}x** "
            f"| {100 * sum(1 for v in fair if v > 1) / len(fair):.0f}% "
            f"| {passed}/{len(rows)} PASS "
            f"| {max(parity):.2e} |" if parity else
            f"| {label(run)} | {run['environment'].get('gpu')} "
            f"| {geomean(eager):.2f}x | **{geomean(fair):.2f}x** "
            f"| {100 * sum(1 for v in fair if v > 1) / len(fair):.0f}% "
            f"| {passed}/{len(rows)} PASS | not measured |")
    return "\n".join(out)


def provenance(runs: list[dict[str, Any]]) -> str:
    out = ["## Provenance", "",
           "Which binaries were installed for each run. The plugin checksum "
           "identifies the backend that was loaded -- but note that the plugin "
           "is not in the timed path (see the header): it backs the "
           "`direct_paras_fp32_parity_max_abs` probe, not the latencies.", "",
           "| Platform | Host | torch | Python | torch-paras | plugin | plugin sha256 |",
           "|---|---|---|---|---|---|---|"]
    for run in runs:
        env = run["environment"]
        paras = env.get("torch_paras") or {}
        sha = (paras.get("plugin_sha256") or "")[:16] or "not recorded"
        plugin = Path(paras["plugin"]).name if paras.get("plugin") else "none"
        out.append(f"| {label(run)} | {env.get('hostname')} | {env.get('torch')} "
                   f"| {env.get('python')} | {paras.get('version') or '?'} "
                   f"| {plugin} | `{sha}` |")
    out += ["", "| Platform | Commit (detected) | Commit (declared) | Directory |",
            "|---|---|---|---|"]
    for run in runs:
        env = run["environment"]
        out.append(f"| {label(run)} | {(env.get('git_commit') or 'null')[:10]} "
                   f"| {(env.get('source_commit_declared') or 'null')[:10]} "
                   f"| `{run['_dir']}` |")
    out += ["", "A detected commit is read from a git checkout at run time. A "
            "declared commit was supplied by the operator with `--source-commit` "
            "for a run made outside a checkout, and is not independently "
            "verified — treat the plugin checksum above as the authoritative "
            "identifier."]
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Render benchmark result directories into markdown tables.")
    p.add_argument("results", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=Path("RESULTS.md"))
    args = p.parse_args()

    runs = [load(path) for path in args.results]
    incomplete = [r["_dir"] for r in runs if r.get("status") != "complete"]
    if incomplete:
        raise SystemExit(f"refusing to publish incomplete runs: {incomplete}")

    n_models, n_datasets = matrix(runs)
    body = "\n\n".join([
        "# Torch-ParaS super-resolution results",
        f"{n_models} models x {n_datasets} datasets = {n_models * n_datasets} "
        "cells per platform, measured with the pip-installed wheel. "
        "Generated by `make_tables.py` directly from the result files - no number "
        "here was transcribed by hand.",
        "**What these numbers are.** All three arms run on the same native "
        "`cuda`/`hip` device, at the same shapes, in the same process. The "
        "Torch-ParaS arm is `torch_paras.autotune`: autocast, channels-last, "
        "`torch.compile` and an explicit CUDA-graph capture, executing "
        "Inductor/Triton and cuDNN/cuBLAS kernels. The ratios therefore "
        "measure end-to-end inference latency of one optimization strategy "
        "against PyTorch's own on identical hardware.",
        table1(runs),
        table2(runs),
        table4(runs),
        table3(runs),
        provenance(runs),
    ])
    args.out.write_text(body + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
