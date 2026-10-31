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

"""Compare a fresh run with the recorded results in benchmarks/results/ and print PASS / FAIL per metric.

  compare.py param2 <BENCH_OUT>            Param2-17B decode / prefill times, perplexity, greedy-token digests
  compare.py inferencex <IX output dir>     InferenceX serving: output tokens/s, median TPOT, ParaS-vs-native ratios
  compare.py micro <micro output dir>       GEMM geometric-mean speedup, peak copy bandwidth

A metric passes when it is within the tolerance of the recorded value (REPRO_TOL, a fraction, default 0.15 for
times and throughputs; perplexity 0.5 %). Greedy-token digests are shown, not judged: they are identical between
runs of the same build on the same GPU model, but bf16 reductions may break a near-tie differently elsewhere.
Writes compare_<kind>.md and compare_<kind>.json next to the results; exit status 1 if any metric fails.
"""
import csv
import glob
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REC = os.path.join(HERE, "results")
TOL = float(os.environ.get("REPRO_TOL", "0.15"))
PPL_TOL = 0.005
LABEL = {("native", "eager"): "native eager", ("native", "compiled"): "native compiled",
         ("paras", "fused"): "ParaS fused", ("paras", "compiled"): "ParaS compiled", ("paras", "eager"): "ParaS eager"}


def rows(path):
    if not os.path.exists(path):
        return []
    return [json.loads(line) for line in open(path) if line.strip()]


def verdict(new, ref, tol):
    if new is None:
        return "not run", ""
    if ref is None:
        return "INFO", ""
    d = new / ref - 1
    return ("PASS" if abs(d) <= tol else "FAIL"), f"{d * 100:+.1f}%"


def report(kind, out_dir, title, header, table, notes):
    n_pass = sum(1 for r in table if r[-2] == "PASS")
    n_fail = sum(1 for r in table if r[-2] == "FAIL")
    lines = [f"## {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in table]
    lines += ["", f"PASS {n_pass}, FAIL {n_fail}, other {len(table) - n_pass - n_fail}"] + notes
    text = "\n".join(lines)
    print(text)
    open(os.path.join(out_dir, f"compare_{kind}.md"), "w").write(text + "\n")
    json.dump({"pass": n_pass, "fail": n_fail, "rows": [dict(zip(header, r)) for r in table]},
              open(os.path.join(out_dir, f"compare_{kind}.json"), "w"), indent=1)
    return 1 if n_fail else 0


def pick(rs, backend, mode, wl, metric, **shape):
    for r in reversed(rs):
        if (r.get("backend"), r.get("execution_mode"), r.get("workload_type"), r.get("metric_name")) == \
                (backend, mode, wl, metric) and all(r.get(k) == v for k, v in shape.items()) and r.get("success", True):
            return r["metric_value"]
    return None


def param2(out_dir):
    rec = json.load(open(os.path.join(REC, "param2_recorded.json")))
    inf = rows(os.path.join(out_dir, "inference", "rows.jsonl"))
    ppl = rows(os.path.join(out_dir, "perplexity", "rows.jsonl"))
    machine = (inf or ppl or [{}])[0].get("device", "unknown")
    m = rec["machines"].get(machine)
    table = []
    for key in rec["metrics"]:
        for cfg in rec["configurations"]:
            b, md = cfg.split()
            if key.startswith("decode"):
                v = pick(inf, b, md, "decode", "tpot.median", batch_size=1 if key.endswith("b1") else 16, input_tokens=128)
            else:
                v = pick(inf, b, md, "prefill", "latency.median", batch_size=1, input_tokens=int(key.split()[1]))
            ref = m["reference"].get(f"{key}|{cfg}") if m else None
            if v is None:
                continue
            res, diff = verdict(v, ref, TOL)
            table.append([key, cfg, "ms", "" if ref is None else f"{ref:.2f}", f"{v:.2f}", diff, res, ""])
    for r in ppl:
        if r["metric_name"] != "perplexity":
            continue
        ref = (m or {}).get("perplexity", {}).get(f"{r['backend']} {r['precision']}")
        res, diff = verdict(r["metric_value"], ref, PPL_TOL)
        table.append(["perplexity", f"{r['backend']} eager {r['precision']}", "", "" if ref is None else f"{ref:.4f}",
                      f"{r['metric_value']:.4f}", diff, res, ""])
    for f in sorted(glob.glob(os.path.join(out_dir, "inference", "greedy_digests_*_bfloat16.json"))):
        b, md = os.path.basename(f).split("_")[2:4]
        got = json.load(open(f))
        ref = ((m or {}).get("greedy_digests") or {}).get(f"{b} {md}", {})
        for bs in ("1", "16"):
            if bs in got:
                same = "identical" if ref.get(bs) == got[bs] else ("no record" if bs not in ref else "different")
                table.append([f"greedy tokens b{bs}", f"{b} {md}", "sha1", ref.get(bs, ""), got[bs], "", "INFO", same])
    notes = ["", f"Machine: {machine}. Reference: the recorded clean-room run (benchmarks/results/param2_recorded.json); "
             f"tolerance ±{TOL * 100:.0f}% for times, ±{PPL_TOL * 100:.1f}% for perplexity."]
    if not m:
        notes.append(f"No recorded results for {machine}: values are shown without a verdict.")
    return report("param2", out_dir, f"Param2-17B on {machine}: this run against the recorded results",
                  ["Workload", "Configuration", "Unit", "Recorded", "This run", "Difference", "Verdict", "Note"], table, notes)


def ix_rows(out_dir, machine):
    e2e = []
    for d in sorted(glob.glob(os.path.join(out_dir, "*_*_*"))):
        if not os.path.isdir(d):
            continue
        b, md, s = os.path.basename(d).split("_")
        if s == "smoke":
            continue
        for f in sorted(glob.glob(os.path.join(d, "ix_c*.json")), key=lambda p: int(re.findall(r"c(\d+)", p)[-1])):
            c = int(re.findall(r"c(\d+)\.json", f)[0])
            r = json.load(open(f))
            st_p, pw_p = os.path.join(d, f"stats_c{c}.json"), os.path.join(d, f"power_c{c}.json")
            st = json.load(open(st_p)) if os.path.exists(st_p) else {}
            pw = json.load(open(pw_p)) if os.path.exists(pw_p) else {}
            toks = st.get("completion_tokens")
            row = {"machine": machine, "scenario": s, "config": LABEL[(b, md)], "backend": b, "mode": md, "concurrency": c,
                   "num_prompts": r["num_prompts"], "completed": r["completed"], "output_tps": r["output_throughput"],
                   "median_ttft_ms": r["median_ttft_ms"], "median_tpot_ms": r["median_tpot_ms"],
                   "p99_e2el_ms": r["p99_e2el_ms"]}
            if pw.get("available") and toks:
                row["gpu_joules_per_output_token"] = pw["energy_j"] / toks
            e2e.append(row)
    json.dump(e2e, open(os.path.join(out_dir, "e2e.json"), "w"), indent=1)
    return e2e


def best_ratio(rs, scenario, c, configs=None):
    rs = [r for r in rs if r["scenario"] == scenario and r["concurrency"] == c and (configs is None or r["config"] in configs)]
    ps = [r for r in rs if r["backend"] == "paras"]
    ns = [r for r in rs if r["backend"] == "native"]
    if not (ps and ns):
        return None, None, None
    P, N = max(ps, key=lambda r: r["output_tps"]), max(ns, key=lambda r: r["output_tps"])
    return P["output_tps"] / N["output_tps"], P["config"], N["config"]


def inferencex(out_dir):
    machine = os.environ.get("BENCH_DEVICE") or open(os.path.join(out_dir, "machine.txt")).read().strip()
    rec = [r for r in json.load(open(os.path.join(REC, "inferencex_recorded.json")))["rows"] if r["machine"] == machine]
    new = ix_rows(out_dir, machine)
    idx = {(r["scenario"], r["config"], r["concurrency"]): r for r in rec}
    table = []
    for r in new:
        ref = idx.get((r["scenario"], r["config"], r["concurrency"]))
        for metric, unit in (("output_tps", "tokens/s"), ("median_tpot_ms", "ms")):
            res, diff = verdict(r[metric], ref and ref[metric], TOL)
            table.append([r["scenario"], r["config"], r["concurrency"], metric, unit,
                          "" if not ref else f"{ref[metric]:.2f}", f"{r[metric]:.2f}", diff, res,
                          f"{r['completed']}/{r['num_prompts']} completed"])
    for s, c in sorted({(r["scenario"], r["concurrency"]) for r in new}):
        ran = {r["config"] for r in new if r["scenario"] == s and r["concurrency"] == c}
        x, pc, nc = best_ratio(new, s, c)
        if x is None:
            continue
        rx, _, _ = best_ratio(rec, s, c, ran)
        res, diff = verdict(x, rx, TOL)
        table.append([s, f"best ParaS ({pc}) / best native ({nc})", c, "output_tps ratio", "x",
                      "" if rx is None else f"{rx:.2f}", f"{x:.2f}", diff, res,
                      "recorded ratio over the same configurations" if rx is not None else ""])
    notes = ["", f"Machine: {machine}. Reference: benchmarks/results/inferencex_recorded.json (InferenceX client "
             f"unchanged, same settings); tolerance ±{TOL * 100:.0f}%."]
    if not rec:
        notes.append(f"No recorded serving results for {machine}: values are shown without a verdict.")
    return report("inferencex", out_dir, f"InferenceX serving on {machine}: this run against the recorded results",
                  ["Scenario", "Configuration", "Concurrency", "Metric", "Unit", "Recorded", "This run", "Difference",
                   "Verdict", "Note"], table, notes)


def micro(out_dir):
    rec = json.load(open(os.path.join(REC, "micro_recorded.json")))
    machine = open(os.path.join(out_dir, "machine.txt")).read().strip()
    same = machine == rec["machine"]
    table = []
    for sh in ("param2", "operatorx"):
        def load(p):
            return {(r["m"], r["k"], r["n"]): r for r in csv.DictReader(open(p))} if os.path.exists(p) else {}
        nat = load(os.path.join(out_dir, f"gemm_{sh}_native.csv"))
        par = load(os.path.join(out_dir, f"gemm_{sh}_paras.csv"))
        sp = [float(a["latency_us"]) / float(par[k]["latency_us"]) for k, a in nat.items()
              if k in par and a["status"] == "ok" and par[k]["status"] == "ok"]
        if not sp:
            continue
        g = statistics.geometric_mean(sp)
        ref = rec["gemm_geomean_speedup_paras_vs_native"][sh] if same else None
        res, diff = verdict(g, ref, TOL)
        crashed = sum(1 for r in par.values() if r["status"] == "crash")
        table.append([f"GEMM {sh} shapes", f"{len(sp)} of {len(nat)}", "geomean speedup ParaS/native",
                      "" if ref is None else f"{ref:.3f}", f"{g:.3f}", diff, res,
                      f"{crashed} shape(s) stopped the process (status=crash)" if crashed else ""])
    for k, stack in (("cuda", "native"), ("paras", "ParaS")):
        p = os.path.join(out_dir, f"mem_bw_{k}.json")
        if os.path.exists(p):
            peak = max(r["bandwidth_gbps"] for r in json.load(open(p)))
            ref = rec["mem_bw_peak_gbps"][stack] if same else None
            res, diff = verdict(peak, ref, TOL)
            table.append(["copy bandwidth", stack, "peak GB/s", "" if ref is None else f"{ref:.1f}", f"{peak:.1f}", diff, res, ""])
    notes = ["", f"Machine: {machine}. Reference: benchmarks/results/micro_recorded.json; tolerance ±{TOL * 100:.0f}%."]
    if not same:
        notes.append(f"Recorded micro-benchmarks exist for {rec['machine']} only: values are shown without a verdict.")
    return report("micro", out_dir, f"GEMM and bandwidth on {machine}: this run against the recorded results",
                  ["Benchmark", "Set", "Metric", "Recorded", "This run", "Difference", "Verdict", "Note"], table, notes)


if __name__ == "__main__":
    kind, d = sys.argv[1], sys.argv[2]
    sys.exit({"param2": param2, "inferencex": inferencex, "micro": micro}[kind](d))
