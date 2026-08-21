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
import copy
import csv
import hashlib
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


sys.path.insert(0, str(HERE))

import torch_paras
from sr_suite.data import discover_images, load_dataset_config, prepare_pair
from sr_suite.models import MODEL_NAMES, build_model, model_info, parameter_count


MODELS = tuple(MODEL_NAMES)
DATASETS = ("Set5", "Set14", "BSD100", "Urban100", "Manga109")


def _cmd(argv: list[str]) -> str | None:
    try:
        return subprocess.check_output(argv, text=True, stderr=subprocess.DEVNULL,
                                       timeout=20).strip()
    except Exception:
        return None


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _telemetry() -> dict[str, Any]:
    if getattr(torch.version, "hip", None):
        return {"source": "rocm-smi", "raw": _cmd([
            "rocm-smi", "--showtemp", "--showpower", "--showclocks"])}
    raw = _cmd(["nvidia-smi", "-i", "0",
                "--query-gpu=temperature.gpu,power.draw,clocks.sm,clocks.mem,pstate",
                "--format=csv,noheader,nounits"])
    if not raw:
        return {}
    keys = ("temperature_c", "power_w", "sm_clock_mhz", "memory_clock_mhz", "pstate")
    return dict(zip(keys, (item.strip() for item in raw.split(","))))


def _paras_provenance() -> dict[str, Any]:
    info: dict[str, Any] = {"version": getattr(torch_paras, "__version__", None)}
    try:
        selection = torch_paras.extension_selection()
        plugin = getattr(selection, "plugin", None)
        info["platform"] = getattr(selection, "platform", None)
        info["python_abi"] = getattr(selection, "python_abi", None)
        info["plugin"] = plugin
        if plugin and Path(plugin).is_file():
            info["plugin_sha256"] = _sha(Path(plugin))
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def _paras_backend_name() -> str | None:
    try:
        return torch_paras.backend_name()
    except Exception:
        return None


def _paras_device_names() -> list[str]:
    try:
        return [torch_paras.device_name(i)
                for i in range(torch_paras.device_count())]
    except Exception:
        return []


def _environment(device: torch.device, source_commit: str | None = None) -> dict[str, Any]:
    try:
        registered = torch._C._dispatch_get_registrations_for_dispatch_key("PrivateUse1")
    except Exception:
        registered = []
    return {
        "hostname": socket.gethostname(), "platform": platform.platform(),
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "hip": getattr(torch.version, "hip", None),
        "gpu": (torch.cuda.get_device_name(device)
                if device.type != "cpu" else platform.processor() or "cpu"),
        "capability": (list(torch.cuda.get_device_capability(device))
                       if device.type != "cpu" else None),
        "device_count": (torch.cuda.device_count()
                         if torch.cuda.is_available() else 0),
        "cudnn": (torch.backends.cudnn.version()
                  if torch.cuda.is_available() else None),
        "compute_device": str(device),
        "torch_paras": _paras_provenance(),
        "torch_paras_backend": _paras_backend_name(),
        "torch_paras_devices": _paras_device_names(),
        "privateuse1_registered_operator_routes": len(registered),
        "privateuse1_registered_operators": sorted(registered),
        "git_commit": _cmd(["git", "-C", str(REPO), "rev-parse", "HEAD"]),


        "source_commit_declared": source_commit,
        "git_dirty": bool(_cmd(["git", "-C", str(REPO), "status", "--porcelain"])),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "telemetry_start": _telemetry(),
    }


def _pad(value: torch.Tensor, side: int) -> torch.Tensor:
    h, w = value.shape[-2:]
    if h > side or w > side:
        raise ValueError(f"input {h}x{w} exceeds fixed {side}x{side}")
    return F.pad(value, (0, side - w, 0, side - h), mode="replicate")


def _accelerated() -> bool:
    return torch.cuda.is_available()


def _sync(device: torch.device | None = None) -> None:
    if not _accelerated():
        return
    if device is None or device.type == "cpu":
        torch.cuda.synchronize()
    else:
        torch.cuda.synchronize(device)


def _reset_peak(device: torch.device) -> None:
    if _accelerated() and device.type != "cpu":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_allocated_mib(device: torch.device) -> float | None:
    if not _accelerated() or device.type == "cpu":
        return None
    return torch.cuda.max_memory_allocated(device) / 1048576.0


def _peak_reserved_mib(device: torch.device) -> float | None:
    if not _accelerated() or device.type == "cpu":
        return None
    return torch.cuda.max_memory_reserved(device) / 1048576.0


def _empty_cache() -> None:
    if _accelerated():
        torch.cuda.empty_cache()


def _events(call, repeats: int, warmup: int = 2) -> tuple[torch.Tensor, list[float]]:
    with torch.inference_mode():
        for _ in range(warmup):
            output = call()
        _sync()
        values: list[float] = []
        if not _accelerated():


            for _ in range(repeats):
                start = time.perf_counter()
                output = call()
                values.append((time.perf_counter() - start) * 1000.0)
            return output, values
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = call()
            end.record()
            end.synchronize()
            values.append(float(start.elapsed_time(end)))
    return output, values


def _copy_ms(value: torch.Tensor, device: torch.device, to_cpu: bool,
             repeats: int = 3) -> float:
    values = []
    for _ in range(repeats):
        _sync(device)
        t0 = time.perf_counter()
        copied = value.to("cpu" if to_cpu else device)
        _sync(device)
        values.append((time.perf_counter() - t0) * 1000.0)
        del copied
    return float(statistics.median(values))


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = float(torch.mean((a.float() - b.float()) ** 2))
    if mse == 0:
        return float("inf")
    return -10.0 * math.log10(mse)


def _sample_statistics(samples: list[float]) -> dict[str, float | list[float]]:
    ordered = sorted(float(value) for value in samples)
    count = len(ordered)
    mean = statistics.fmean(ordered)
    stdev = statistics.stdev(ordered) if count > 1 else 0.0
    pick = lambda q: ordered[max(0, min(count - 1, math.ceil(q * count) - 1))]
    half_width = 1.96 * stdev / math.sqrt(count) if count > 1 else 0.0
    return {
        "count": count, "median_ms": statistics.median(ordered), "mean_ms": mean,
        "stdev_ms": stdev, "p95_ms": pick(0.95), "p99_ms": pick(0.99),
        "cv_percent": stdev / mean * 100.0 if mean else 0.0,
        "mean_95ci_ms": [mean - half_width, mean + half_width],
    }


_PRECISION_DTYPE = {"fp32": torch.float32, "fp16": torch.float16,
                    "bf16": torch.bfloat16}


def _pytorch_optimized(cpu_model, selected: dict[str, Any],
                       device: torch.device, use_compile: bool):
    precision = selected.get("precision", "fp32")
    channels_last = bool(selected.get("channels_last", False))
    dtype = _PRECISION_DTYPE.get(precision, torch.float32)

    model = copy.deepcopy(cpu_model).to(device, dtype=torch.float32).eval()
    if channels_last:
        model = model.to(memory_format=torch.channels_last)


    if use_compile:
        mode = "reduce-overhead" if selected.get("graph", False)            else selected.get("compile_mode", "default") or "default"
        runner = torch.compile(model, mode=mode)
    else:
        mode = None
        runner = model

    def call(value: torch.Tensor) -> torch.Tensor:
        if channels_last:
            value = value.to(memory_format=torch.channels_last)
        if dtype is torch.float32:
            return runner(value)
        with torch.autocast(device.type, dtype=dtype):
            return runner(value)

    return call, {"precision": precision, "channels_last": channels_last,
                  "compile": use_compile, "compile_mode": mode}


def _candidates(device: torch.device) -> list[torch_paras.OptimizationSpec]:
    hip = bool(getattr(torch.version, "hip", None))
    if device.type == "cpu":


        return [
            torch_paras.OptimizationSpec("fp32", False, True, "default", False),
            torch_paras.OptimizationSpec("fp32", False, False, "default", False),
        ]
    major, _ = torch.cuda.get_device_capability(device)
    preferred = "bf16" if hip or major >= 9 else "fp16"
    values = [
        torch_paras.OptimizationSpec(preferred, True, True, "default", True),
        torch_paras.OptimizationSpec(preferred, True, False, "default", True),
        torch_paras.OptimizationSpec("fp32", True, True, "default", True),
        torch_paras.OptimizationSpec("fp32", True, False, "default", True),
    ]
    if hip or major >= 9:
        values.append(torch_paras.OptimizationSpec(
            preferred, True, True, "max-autotune", True, fp8=True))
    return values


def _prepare_inputs(paths: dict[str, Path], scale: int, max_hr: int,
                    side: int, preupsampled: bool) -> dict[str, dict[str, Any]]:
    prepared = {}
    for dataset, path in paths.items():
        lr, hr, bicubic = prepare_pair(path, scale, max_hr)
        source = bicubic if preupsampled else lr
        prepared[dataset] = {
            "path": str(path), "sha256": _sha(path), "hr_hw": list(hr.shape[-2:]),
            "input": _pad(source.unsqueeze(0), side),
        }
    return prepared


def _direct_paras_parity(cpu_model, prepared, baseline_outputs,
                         paras_device: torch.device | None) -> dict[str, float | None]:
    result: dict[str, float | None] = {name: None for name in prepared}
    if paras_device is None:
        return result
    try:
        model = copy.deepcopy(cpu_model).eval().to(paras_device)
        with torch.inference_mode():
            for dataset, item in prepared.items():
                value = item["input"].to(paras_device)
                out = model(value).detach().float().cpu()
                ref = baseline_outputs[dataset]
                result[dataset] = float((out - ref).abs().max())
                del value, out
        del model
    except Exception as exc:
        print(f"  direct ParaS FP32 parity unavailable: {type(exc).__name__}: {exc}", flush=True)
    return result


def _peak(values: list[dict[str, Any]], key: str) -> float | None:
    present = [row[key] for row in values if row.get(key) is not None]
    return max(present) if present else None


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = [key for key, value in rows[0].items()
              if not isinstance(value, (list, dict))]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def run(args) -> int:


    if args.device == "cpu" or (args.device == "auto"
                                and not torch.cuda.is_available()):
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA/ROCm accelerator visible; "
                               "pass --device cpu to benchmark on the CPU")
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)

    paras_device = None
    if getattr(torch_paras, "_C", None) is not None:
        try:
            count = torch_paras.device_count()
        except Exception:
            count = 0
        if device.type == "cpu":
            paras_index = 0 if count else None
        else:
            paras_index = args.gpu + 1 if count > args.gpu + 1 else None
        if paras_index is not None:
            paras_device = torch.device(f"paras:{paras_index}")
    if paras_device is None:
        print("note: no ParaS backend device for this hardware; the direct "
              "paras-tensor parity probe will be skipped", flush=True)
    models = MODELS if args.models == "all" else tuple(
        item.strip().lower() for item in args.models.split(",") if item.strip())
    datasets = DATASETS if args.datasets == "all" else tuple(
        item.strip() for item in args.datasets.split(",") if item.strip())
    unknown_models = sorted(set(models) - set(MODELS))
    unknown_datasets = sorted(set(datasets) - set(DATASETS))
    if unknown_models or unknown_datasets:
        raise ValueError(f"unknown models={unknown_models}; datasets={unknown_datasets}")

    config = load_dataset_config(args.dataset_config)
    discovered = {name: discover_images(args.datasets_root, name, config[name])
                  for name in datasets}
    missing = [name for name, values in discovered.items() if not values]
    if missing:
        raise RuntimeError("missing datasets: " + ", ".join(missing))
    selected_paths = {name: values[0] for name, values in discovered.items()}

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    partial_path = out / "partial_results.json"
    rows: list[dict[str, Any]] = []
    model_evidence: dict[str, Any] = {}
    if partial_path.exists():
        try:
            old = json.loads(partial_path.read_text())
            if old.get("status") == "running":
                rows = old.get("rows", [])
                model_evidence = old.get("model_evidence", {})
        except Exception:
            pass
    completed = {r["model"] for r in rows}
    env = _environment(device, args.source_commit)

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    def save_partial() -> None:
        temp = partial_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps({"status": "running", "rows": rows,
                                    "model_evidence": model_evidence}, indent=2,
                                   allow_nan=True))
        temp.replace(partial_path)

    for model_name in models:
        if model_name in completed:
            print(f"[{model_name}] resume: already complete", flush=True)
            continue
        print(f"[{model_name}] preparing stock eager baseline", flush=True)
        torch.manual_seed(args.seed)
        cpu_model = build_model(model_name, args.scale, profile="benchmark").eval()
        info = model_info(cpu_model, model_name, "benchmark")
        side = args.fixed_hr_side if info.expects_preupsampled else args.fixed_lr_side
        prepared = _prepare_inputs(selected_paths, args.scale, args.max_hr_side,
                                   side, info.expects_preupsampled)

        baseline = copy.deepcopy(cpu_model).to(device, dtype=torch.float32).eval()
        baseline_outputs: dict[str, torch.Tensor] = {}
        baseline_metrics: dict[str, Any] = {}
        for dataset, item in prepared.items():
            native_input = item["input"].to(device)
            _reset_peak(device)
            output, samples = _events(lambda: baseline(native_input), args.repeats)
            _sync(device)
            output_cpu = output.detach().float().cpu()
            baseline_outputs[dataset] = output_cpu
            baseline_metrics[dataset] = {
                "samples_ms": samples, "median_ms": float(statistics.median(samples)),
                "statistics": _sample_statistics(samples),
                "peak_allocated_mib": _peak_allocated_mib(device),
                "peak_reserved_mib": _peak_reserved_mib(device),
                "h2d_ms": _copy_ms(item["input"], device, False),
                "d2h_ms": _copy_ms(output, device, True),
            }
            del native_input, output
        del baseline
        _empty_cache()

        direct_parity = _direct_paras_parity(
            cpu_model, prepared, baseline_outputs, paras_device)
        _empty_cache()

        first = prepared[datasets[0]]["input"].to(device)
        first_ref = baseline_outputs[datasets[0]].to(device)
        torch.backends.cudnn.benchmark = True
        t0 = time.perf_counter()
        optimized, evidence = torch_paras.autotune(
            cpu_model, first, first_ref, _candidates(device),
            warmup=args.tune_warmup, iterations=args.tune_iters,
            fidelity_floor_db=args.fidelity_floor_db)
        tune_seconds = time.perf_counter() - t0
        torch.backends.cudnn.benchmark = False
        model_evidence[model_name] = {
            "parameters": parameter_count(cpu_model), "architecture": info.architecture,
            "expects_preupsampled": info.expects_preupsampled, "input_side": side,
            "tune_seconds": tune_seconds, "optimizer": asdict(evidence),
            "capabilities": torch_paras.capability_matrix(evidence),
        }

        print(f"[{model_name}] selected {evidence.selected}", flush=True)


        compile_status = "compiled"
        for attempt_compile in (True, False):
            compiled_call, compiled_spec = _pytorch_optimized(
                cpu_model, evidence.selected, device, attempt_compile)
            compiled_metrics: dict[str, Any] = {}
            try:
                for dataset, item in prepared.items():
                    native_input = item["input"].to(device)
                    output, samples = _events(lambda: compiled_call(native_input),
                                              args.repeats)
                    _sync(device)
                    compiled_metrics[dataset] = {
                        "median_ms": float(statistics.median(samples)),
                        "samples_ms": samples,
                        "statistics": _sample_statistics(samples),
                        "fidelity_psnr_db": _psnr(output.detach().float().cpu(),
                                                  baseline_outputs[dataset]),
                    }
                    del native_input, output
                break
            except Exception as exc:
                if not attempt_compile:
                    raise
                compile_status = f"compile unavailable: {type(exc).__name__}: {exc}"
                print(f"[{model_name}] torch.compile baseline failed, "
                      f"falling back to uncompiled: {type(exc).__name__}", flush=True)
                _empty_cache()
        compiled_spec["status"] = compile_status
        del compiled_call
        _empty_cache()
        model_evidence[model_name]["pytorch_optimized"] = compiled_spec
        print(f"[{model_name}] pytorch optimized {compiled_spec}", flush=True)

        model_rows = []
        for dataset, item in prepared.items():
            native_input = item["input"].to(device)
            _reset_peak(device)
            output, samples = _events(lambda: optimized(native_input), args.repeats)
            _sync(device)
            output_cpu = output.detach().float().cpu()
            ref = baseline_outputs[dataset]
            diff = (output_cpu - ref).abs()
            paras_ms = float(statistics.median(samples))
            eager_ms = baseline_metrics[dataset]["median_ms"]
            row = {
                "model": model_name, "dataset": dataset,
                "image": Path(item["path"]).name, "image_path": item["path"],
                "image_sha256": item["sha256"], "hr_hw": item["hr_hw"],
                "parameters": parameter_count(cpu_model),
                "pytorch_eager_ms": eager_ms, "torch_paras_optimized_ms": paras_ms,
                "speedup": eager_ms / paras_ms,
                "pytorch_optimized_ms": compiled_metrics[dataset]["median_ms"],
                "pytorch_optimized_samples_ms": compiled_metrics[dataset]["samples_ms"],
                "pytorch_optimized_statistics": compiled_metrics[dataset]["statistics"],
                "pytorch_optimized_img_s": 1000.0 / compiled_metrics[dataset]["median_ms"],
                "pytorch_optimized_fidelity_psnr_db": compiled_metrics[dataset]["fidelity_psnr_db"],
                "pytorch_optimized_compiled": compiled_spec["compile"],
                "pytorch_optimized_status": compile_status,
                "speedup_vs_pytorch_optimized":
                    compiled_metrics[dataset]["median_ms"] / paras_ms,
                "pytorch_eager_samples_ms": baseline_metrics[dataset]["samples_ms"],
                "torch_paras_samples_ms": samples,
                "pytorch_eager_statistics": baseline_metrics[dataset]["statistics"],
                "torch_paras_statistics": _sample_statistics(samples),
                "pytorch_eager_img_s": 1000.0 / eager_ms,
                "torch_paras_img_s": 1000.0 / paras_ms,
                "pytorch_peak_allocated_mib": baseline_metrics[dataset]["peak_allocated_mib"],
                "pytorch_peak_reserved_mib": baseline_metrics[dataset]["peak_reserved_mib"],
                "torch_paras_peak_allocated_mib": _peak_allocated_mib(device),
                "torch_paras_peak_reserved_mib": _peak_reserved_mib(device),
                "h2d_ms": baseline_metrics[dataset]["h2d_ms"],
                "pytorch_d2h_ms": baseline_metrics[dataset]["d2h_ms"],
                "torch_paras_d2h_ms": _copy_ms(output, device, True),
                "fidelity_psnr_db": _psnr(output_cpu, ref),
                "fidelity_max_abs": float(diff.max()),
                "fidelity_mean_abs": float(diff.mean()),
                "fidelity_pass": _psnr(output_cpu, ref) >= args.fidelity_floor_db,
                "direct_paras_fp32_parity_max_abs": direct_parity[dataset],
            }
            model_rows.append(row)
            print(f"  {dataset:11s} eager={eager_ms:8.3f} ms "
                  f"pt-opt={row['pytorch_optimized_ms']:8.3f} ms "
                  f"paras={paras_ms:8.3f} ms "
                  f"vs-eager={row['speedup']:.2f}x "
                  f"vs-pt-opt={row['speedup_vs_pytorch_optimized']:.2f}x "
                  f"fidelity={row['fidelity_psnr_db']:.1f} dB", flush=True)
            del native_input, output, output_cpu
        rows.extend(model_rows)
        completed.add(model_name)
        save_partial()
        del optimized, cpu_model, baseline_outputs, first, first_ref
        _empty_cache()

    env["telemetry_end"] = _telemetry()
    model_summary = []
    for model in models:
        values = [r for r in rows if r["model"] == model]
        eager = sum(r["pytorch_eager_ms"] for r in values)
        paras = sum(r["torch_paras_optimized_ms"] for r in values)
        ptopt = sum(r["pytorch_optimized_ms"] for r in values)
        model_summary.append({
            "model": model, "datasets": len(values), "pytorch_eager_total_ms": eager,
            "torch_paras_total_ms": paras, "speedup": eager / paras,
            "pytorch_optimized_total_ms": ptopt,
            "speedup_vs_pytorch_optimized": ptopt / paras,
            "pytorch_optimized_img_s": len(values) * 1000.0 / ptopt,
            "pytorch_eager_img_s": len(values) * 1000.0 / eager,
            "torch_paras_img_s": len(values) * 1000.0 / paras,
            "min_fidelity_psnr_db": min(r["fidelity_psnr_db"] for r in values),
            "max_direct_paras_fp32_parity_abs": max(
                (r["direct_paras_fp32_parity_max_abs"] for r in values
                 if r["direct_paras_fp32_parity_max_abs"] is not None), default=None),

            "pytorch_peak_allocated_mib": _peak(values, "pytorch_peak_allocated_mib"),
            "torch_paras_peak_allocated_mib": _peak(values, "torch_paras_peak_allocated_mib"),
        })
    status = "complete" if (len(rows) == len(models) * len(datasets)
                            and all(r["fidelity_pass"] for r in rows)) else "failed"
    payload = {
        "schema": "torch-paras-comprehensive-v1", "status": status,
        "environment": env,
        "protocol": {
            "models": list(models), "datasets": list(datasets),
            "matrix_cells": len(models) * len(datasets),
            "images_per_cell": 1, "scale": args.scale,
            "max_hr_side": args.max_hr_side, "fixed_lr_side": args.fixed_lr_side,
            "fixed_hr_side": args.fixed_hr_side, "warmup": 2,
            "repeats": args.repeats, "fidelity_floor_db": args.fidelity_floor_db,
            "baseline": "stock PyTorch eager fp32/NCHW; no compile, autocast, graph, channels-last, TF32 or vendor autotune",
            "baseline_optimized": "PyTorch optimized: the best PyTorch itself achieves on this machine, given the same budget Torch-ParaS selected for that model - same precision, same memory format, torch.compile with graph replay where available. pytorch_optimized_status records how each model was actually built",
            "proposed": "Torch-ParaS whole-graph optimizer; measured per-model candidate selection",
            "quality_scope": "compatibility models use deterministic random weights: fidelity/parity only; trained PSNR/SSIM comes from official SwinIR benchmark",
        },
        "model_evidence": model_evidence, "model_summary": model_summary, "rows": rows,
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2, allow_nan=True))
    _write_rows_csv(out / "results.csv", rows)
    (out / "environment.json").write_text(json.dumps(env, indent=2))
    partial_path.write_text(json.dumps({"status": status, "rows": len(rows)}, indent=2))
    (out / "results.json.sha256").write_text(f"{_sha(out / 'results.json')}  results.json\n")
    print(f"Wrote {out / 'results.json'} ({len(rows)} cells, status={status})")
    return 0 if status == "complete" else 2


def main() -> int:
    p = argparse.ArgumentParser(
        description="Super-resolution benchmark: PyTorch eager, PyTorch with "
                    "torch.compile, and Torch-ParaS.")
    p.add_argument("--datasets-root", type=Path,
                   default=HERE / "datasets" / "extracted")
    p.add_argument("--dataset-config", type=Path,
                   default=HERE / "configs" / "datasets.real-downloaded.json")
    p.add_argument("--out", type=Path,
                   default=HERE / "results" / "comprehensive-this-host")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--device", choices=("auto", "gpu", "cpu"), default="auto",
                   help="auto uses the GPU when one is visible, else the CPU")
    p.add_argument("--source-commit", default=None,
                   help="record the source revision when running outside a git "
                        "checkout; stored as source_commit_declared")
    p.add_argument("--models", default="all", help="all or comma-separated model names")
    p.add_argument("--datasets", default="all", help="all or comma-separated dataset names")
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--max-hr-side", type=int, default=256)
    p.add_argument("--fixed-lr-side", type=int, default=128)
    p.add_argument("--fixed-hr-side", type=int, default=256)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--tune-warmup", type=int, default=5)
    p.add_argument("--tune-iters", type=int, default=20)
    p.add_argument("--fidelity-floor-db", type=float, default=40.0)
    p.add_argument("--seed", type=int, default=2026)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
