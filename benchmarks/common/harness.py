# -----------------------------------------------------------------------------
# Copyright (c) 2026 Centre for Development of Advanced Computing (C-DAC)
#
# This file is part of Torch_ParaS, a component of the ParaS Ecosystem
#
# This library is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License (LGPL)
# version 3 as published by the Free Software Foundation.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library. If not, see <https://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------

"""Shared harness of the Param2-17B and InferenceX benchmarks.

Every observation is one long-form row (one metric per row, shared run
metadata) appended to $BENCH_OUT/<stage>/rows.jsonl: run_id, timestamp,
model_revision, backend, execution_mode, workload_type, device, precision,
batch_size, input_tokens, output_tokens, step, metric_name, metric_value,
unit, phase, success, error_message, config_hash.

Model: bharatgenai/Param2-17B-A2.4B-Thinking at a pinned revision (PARAM2_MODEL
may point at a local copy of that revision). Input text: the WikiText-2
(raw, v1) test split, concatenated, tokenised with the model's tokenizer.

Power, temperature and clocks are sampled every 50 ms from the AMD GPU's hwmon
files (power1_average, temp*_input, freq*_input) or from nvidia-smi.
"""
import contextlib
import glob
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import threading
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))

import torch                                        # noqa: E402

MODEL_ID = "bharatgenai/Param2-17B-A2.4B-Thinking"
MODEL_REVISION = "65373af52e60e57dcb1ef235ffb827b8abdb4551"
WIKITEXT_ID = "Salesforce/wikitext"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKITEXT_FILE = "wikitext-2-raw-v1/test-00000-of-00001.parquet"
WIKITEXT_TOKENS = 353839
OUT = os.environ.get("BENCH_OUT", os.path.join(os.getcwd(), "bench_results"))


def model_path():
    """Local directory of the pinned model revision (downloaded once into the Hugging Face cache)."""
    if os.environ.get("PARAM2_MODEL"):
        return os.environ["PARAM2_MODEL"]
    from huggingface_hub import snapshot_download
    return snapshot_download(MODEL_ID, revision=MODEL_REVISION)


def verify_model(path=None):
    """Check every file of the local model copy against the pinned revision on the Hub: SHA-256 for the
    weight files (Git LFS), the Git blob hash for the others. Returns the number of files checked."""
    from huggingface_hub import HfApi
    path = path or MODEL
    info = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION, files_metadata=True)
    checked = 0
    for f in info.siblings:
        local = os.path.join(path, f.rfilename)
        if not os.path.exists(local):
            raise SystemExit(f"model file missing: {local}")
        h = hashlib.sha256() if f.lfs else hashlib.sha1(b"blob %d\0" % os.path.getsize(local))
        with open(local, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        want = f.lfs.sha256 if f.lfs else f.blob_id
        if h.hexdigest() != want:
            raise SystemExit(f"model file differs from revision {MODEL_REVISION[:12]}: {f.rfilename}")
        checked += 1
    return checked


MODEL = model_path()


class Backend:
    """native (PyTorch ROCm, cuda:0) or paras (Torch-ParaS, paras:0)."""

    def __init__(self, name):
        self.name = name
        self.paras = name == "paras"
        if self.paras:
            import torch_paras
            self.tp = torch_paras
            self.dev = "paras:0"
        else:
            self.tp = None
            self.dev = "cuda:0"

    def sync(self):
        self.tp.synchronize() if self.paras else torch.cuda.synchronize()

    def allocated(self):
        return self.tp.memory_allocated(0) if self.paras else torch.cuda.memory_allocated()

    def reserved(self):
        if self.paras:
            return self.tp.memory_reserved(0) if hasattr(self.tp, "memory_reserved") \
                else self.tp.memory_allocated(0) + self.tp.memory_cached(0)
        return torch.cuda.memory_reserved()

    def reset_peak(self):
        if not self.paras:
            torch.cuda.reset_peak_memory_stats()
        elif hasattr(self.tp, "reset_peak_memory_stats"):
            self.tp.reset_peak_memory_stats(0)
        self._peak = self.allocated()

    def peak(self):
        if not self.paras:
            return torch.cuda.max_memory_allocated()
        if hasattr(self.tp, "max_memory_allocated"):
            return self.tp.max_memory_allocated(0)
        return max(getattr(self, "_peak", 0), self.allocated())

    def empty_cache(self):
        import gc
        gc.collect()
        self.tp.empty_cache() if self.paras else torch.cuda.empty_cache()

    def stats(self):
        return dict(self.tp.runtime_stats()) if self.paras else {}

    def load(self, dtype, mode="eager", train=False):
        from transformers import AutoModelForCausalLM
        t = time.perf_counter()
        m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype, trust_remote_code=True)
        cpu_s = time.perf_counter() - t
        m = m.to(self.dev)
        m.train(train)
        self.sync()
        load_s = time.perf_counter() - t
        if self.paras and mode in ("fused", "compiled"):
            self.tp.optimizer.fuse_moe_experts(m)
        if mode == "compiled":
            if self.paras:
                self.tp.optimizer.specialize_repeated_blocks(m)
                m.forward = torch.compile(m.forward, backend="paras")
            else:
                m.forward = torch.compile(m.forward)
        return m, {"checkpoint_load_s": cpu_s, "to_device_s": load_s - cpu_s, "load_s": load_s}


def tokens():
    """WikiText-2 test token ids (353,839 tokens), cached next to the results."""
    cache = os.path.join(OUT, "wikitext2_ids.pt")
    if os.path.exists(cache):
        return torch.load(cache)
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    path = hf_hub_download(WIKITEXT_ID, WIKITEXT_FILE, repo_type="dataset", revision=WIKITEXT_REVISION)
    text = "".join(pq.read_table(path).column("text").to_pylist())
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    ids = torch.tensor(tok(text).input_ids, dtype=torch.long)
    if ids.numel() != WIKITEXT_TOKENS:
        raise SystemExit(f"WikiText-2 gave {ids.numel()} tokens, expected {WIKITEXT_TOKENS}: wrong tokenizer or data")
    os.makedirs(OUT, exist_ok=True)
    torch.save(ids, cache)
    return ids


def _gpu_hwmon():
    """hwmon directory of the GPU this process uses (ROCR_VISIBLE_DEVICES)."""
    idx = os.environ.get("ROCR_VISIBLE_DEVICES", "0").split(",")[0]
    try:
        out = subprocess.run(["amd-smi", "list", "--json"], capture_output=True, text=True, timeout=30).stdout
        bdf = None
        for d in json.loads(out):
            if str(d.get("gpu")) == idx:
                bdf = d.get("bdf")
        if bdf:
            hw = glob.glob(f"/sys/bus/pci/devices/{bdf.lower()}/hwmon/hwmon*")
            if hw:
                return hw[0]
    except Exception:
        pass
    return None


class HwSampler:
    """Samples power (W), edge/junction/memory temperature (C) and clocks (MHz)
    every `period` seconds in a thread; energy is the trapezoid integral."""

    def __init__(self, period=0.05, alloc_fn=None):
        self.period = period
        self.dir = _gpu_hwmon()
        self.alloc_fn = alloc_fn
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def _read(self, name):
        try:
            with open(os.path.join(self.dir, name)) as f:
                return int(f.read().strip())
        except Exception:
            return None

    def _loop_nvidia(self):
        idx = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        while not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "-i", idx, "--query-gpu=power.draw,temperature.gpu,"
                                      "clocks.sm,memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout.split(",")
                pw, tc, clk, mem = (float(v) for v in out[:4])
                alloc = self.alloc_fn() if self.alloc_fn else None
                self.samples.append((time.perf_counter(), pw, None, int(tc * 1000), None, int(clk * 1e6),
                                     None, int(mem * 2**20), alloc))
            except Exception:
                pass
            self._stop.wait(max(self.period, 0.1))

    def _loop(self):
        while not self._stop.is_set():
            p = self._read("power1_average") or self._read("power1_input")
            vram = None
            try:
                with open(os.path.join(self.dir, "..", "..", "mem_info_vram_used")) as f:
                    vram = int(f.read().strip())
            except Exception:
                pass
            alloc = None
            if self.alloc_fn is not None:
                try:
                    alloc = self.alloc_fn()
                except Exception:
                    pass
            self.samples.append((time.perf_counter(), None if p is None else p / 1e6,
                                 self._read("temp1_input"), self._read("temp2_input"),
                                 self._read("temp3_input"), self._read("freq1_input"),
                                 self._read("freq2_input"), vram, alloc))
            self._stop.wait(self.period)

    def __enter__(self):
        target = self._loop if self.dir else (
            self._loop_nvidia if torch.version.cuda and shutil.which("nvidia-smi") else None)
        if target:
            self._thread = threading.Thread(target=target, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self._thread:
            self._thread.join()

    def summary(self):
        s = [x for x in self.samples if x[1] is not None]
        if len(s) < 2:
            return {"available": False}
        energy = sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(s, s[1:]))
        temps = [x[3] / 1000 for x in s if x[3] is not None]
        clocks = [x[5] / 1e6 for x in s if x[5] is not None]
        vram = [x[7] for x in s if x[7] is not None]
        alloc = [x[8] for x in s if x[8] is not None]
        return {"available": True, "samples": len(s), "seconds": s[-1][0] - s[0][0],
                "vram_used_peak_bytes": max(vram) if vram else None,
                "alloc_sampled_peak_bytes": max(alloc) if alloc else None,
                "energy_j": energy, "power_avg_w": statistics.mean(x[1] for x in s),
                "power_max_w": max(x[1] for x in s),
                "temp_junction_max_c": max(temps) if temps else None,
                "sclk_avg_mhz": statistics.mean(clocks) if clocks else None}


def device_label():
    """Short name of GPU 0: MI300X, GH200, or the name the driver reports."""
    try:
        name = torch.cuda.get_device_properties(0).name
    except Exception:
        return "unknown"
    for key, label in (("MI300X", "MI300X"), ("GH200", "GH200"), ("H100", "H100"), ("V100", "V100")):
        if key in name.replace(" ", ""):
            return label
    gcn = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "") or ""
    return "MI300X" if gcn.startswith("gfx942") else name


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def describe(xs):
    """n, mean, median, std, min, max, p50/p90/p95/p99, cv%, 95% CI of the mean."""
    if not xs:
        return {"n": 0}
    n = len(xs)
    mean = statistics.mean(xs)
    std = statistics.stdev(xs) if n > 1 else 0.0
    t = [12.71, 4.30, 3.18, 2.78, 2.57, 2.45, 2.36, 2.31, 2.26, 2.23, 2.20, 2.18, 2.16, 2.14,
         2.13, 2.12, 2.11, 2.10, 2.09, 2.09, 2.08, 2.07, 2.07, 2.06, 2.06, 2.06, 2.05, 2.05, 2.05,
         2.04]
    tc = t[n - 2] if 2 <= n <= 31 else 1.96
    half = tc * std / n ** 0.5 if n > 1 else 0.0
    return {"n": n, "mean": mean, "median": statistics.median(xs), "std": std, "min": min(xs),
            "max": max(xs), "p50": pct(xs, .5), "p90": pct(xs, .9), "p95": pct(xs, .95),
            "p99": pct(xs, .99), "cv_pct": 100 * std / mean if mean else None,
            "ci95_low": mean - half, "ci95_high": mean + half}


class Rows:
    """Long-form result rows for one stage and one run."""

    def __init__(self, stage, backend, mode, precision, **meta):
        self.dir = os.path.join(OUT, stage)
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, "rows.jsonl")
        self.base = {"run_id": uuid.uuid4().hex[:12], "model_revision": MODEL_REVISION,
                     "backend": backend, "execution_mode": mode, "precision": precision,
                     "device": os.environ.get("BENCH_DEVICE", device_label()), **meta}
        self.base["config_hash"] = hashlib.sha1(json.dumps(self.base, sort_keys=True,
                                                           default=str).encode()).hexdigest()[:12]

    def add(self, workload_type, metric_name, value, unit, *, phase="measured", success=True,
            error_message="", **dims):
        row = {**self.base, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "workload_type": workload_type, "metric_name": metric_name,
               "metric_value": value, "unit": unit, "phase": phase, "success": success,
               "error_message": error_message, **dims}
        with open(self.path, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def add_stats(self, workload_type, metric_name, xs, unit, **dims):
        for k, v in describe(xs).items():
            self.add(workload_type, f"{metric_name}.{k}", v, "count" if k == "n" else
                     ("%" if k == "cv_pct" else unit), **dims)

    def add_hw(self, workload_type, hw: dict, work_units=None, unit_name=None, **dims):
        if not hw.get("available"):
            self.add(workload_type, "power.available", 0, "bool", **dims)
            return
        self.add(workload_type, "energy", hw["energy_j"], "J", **dims)
        self.add(workload_type, "power.avg", hw["power_avg_w"], "W", **dims)
        self.add(workload_type, "power.max", hw["power_max_w"], "W", **dims)
        if hw.get("temp_junction_max_c") is not None:
            self.add(workload_type, "temperature.junction.max", hw["temp_junction_max_c"], "C", **dims)
        if hw.get("vram_used_peak_bytes") is not None:
            self.add(workload_type, "device_vram_used.peak", hw["vram_used_peak_bytes"] / 2**30, "GiB", **dims)
        if hw.get("alloc_sampled_peak_bytes") is not None:
            self.add(workload_type, "allocated.sampled_peak", hw["alloc_sampled_peak_bytes"] / 2**30, "GiB", **dims)
        if hw.get("sclk_avg_mhz") is not None:
            self.add(workload_type, "clock.sclk.avg", hw["sclk_avg_mhz"], "MHz", **dims)
        if work_units:
            self.add(workload_type, f"energy_per_{unit_name}", hw["energy_j"] / work_units,
                     f"J/{unit_name}", **dims)

    def blob(self, name, obj):
        with open(os.path.join(self.dir, f"{name}.json"), "w") as f:
            json.dump(obj, f, indent=1, default=str)


@contextlib.contextmanager
def timer(be, out: list):
    be.sync()
    t = time.perf_counter()
    yield
    be.sync()
    out.append(1000 * (time.perf_counter() - t))
