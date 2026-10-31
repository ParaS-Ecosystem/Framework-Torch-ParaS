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

"""Serving under load (spec section 4) for Param2-17B bf16 -- a static-batching
server: neither stack has a continuous-batching scheduler; both serve through
transformers-style batched decode. Requests arrive open-loop (Poisson, fixed
seed) with prompt lengths 64..512 and 32..128 new tokens; when the device is
free, up to MAX_BATCH waiting requests are padded to a common prompt length
(left padding, attention mask) and decoded together until the longest one
finishes. Per request: queueing delay, TTFT, end-to-end latency; per rate:
throughput, SLO attainment (TTFT <= 2 s and e2e <= 15 s), batch fill ratio,
padding overhead.

  serve.py <native|paras>
"""
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from harness import Backend, Rows, describe, tokens   # noqa: E402
import torch                                         # noqa: E402

be = Backend(sys.argv[1])
R = Rows("serving", sys.argv[1], "eager", "bfloat16", scheduler="static batching, open loop")
ids = tokens()
model, _ = be.load(torch.bfloat16, "eager")
MAX_BATCH, N = 8, 48
RATES = (0.25, 0.5, 1.0, 2.0)
if os.environ.get("BENCH_QUICK"):
    N, RATES = 4, (1.0,)
SLO_TTFT, SLO_E2E = 2000.0, 15000.0


def run_batch(reqs):
    L = max(r["plen"] for r in reqs)
    x = torch.zeros(len(reqs), L, dtype=torch.long)
    mask = torch.zeros(len(reqs), L, dtype=torch.long)
    for i, r in enumerate(reqs):
        x[i, L - r["plen"]:] = ids[r["off"]:r["off"] + r["plen"]]
        mask[i, L - r["plen"]:] = 1
    x, mask = x.to(be.dev), mask.to(be.dev)
    out = model(input_ids=x, attention_mask=mask, use_cache=True)
    past, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
    be.sync()
    first = time.perf_counter()
    longest = max(r["new"] for r in reqs)
    done_at = {}
    for step in range(1, longest):
        mask = torch.cat([mask, torch.ones(len(reqs), 1, dtype=mask.dtype, device=be.dev)], 1)
        out = model(input_ids=nxt, attention_mask=mask, past_key_values=past, use_cache=True)
        past, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
        for i, r in enumerate(reqs):
            if step + 1 == r["new"]:
                be.sync()
                done_at[i] = time.perf_counter()
    be.sync()
    end = time.perf_counter()
    pad = sum(L - r["plen"] for r in reqs) / (len(reqs) * L)
    return first, [done_at.get(i, end) for i in range(len(reqs))], pad


with torch.no_grad():
    run_batch([{"plen": 64, "new": 4, "off": 0}])
    for rate in RATES:
        rng = random.Random(1)
        t, reqs = 0.0, []
        for i in range(N):
            t += rng.expovariate(rate)
            reqs.append({"arrive": t, "plen": rng.randint(64, 512), "new": rng.randint(32, 128),
                         "off": rng.randint(0, 200000)})
        start = time.perf_counter()
        pending, results, fills, pads = list(reqs), [], [], []
        while pending:
            now = time.perf_counter() - start
            ready = [r for r in pending if r["arrive"] <= now]
            if not ready:
                time.sleep(min(r["arrive"] for r in pending) - now)
                continue
            batch = ready[:MAX_BATCH]
            for r in batch:
                pending.remove(r)
            t_start = time.perf_counter() - start
            first, done, pad = run_batch(batch)
            fills.append(len(batch) / MAX_BATCH)
            pads.append(pad)
            for r, d in zip(batch, done):
                results.append({"queue": 1000 * (t_start - r["arrive"]),
                                "ttft": 1000 * (first - start - r["arrive"]),
                                "e2e": 1000 * (d - start - r["arrive"]), "new": r["new"], "plen": r["plen"]})
        wall = time.perf_counter() - start
        d = dict(offered_rate_rps=rate, max_batch=MAX_BATCH, requests=N)
        for k in ("queue", "ttft", "e2e"):
            R.add_stats("serving", f"{k}_latency", [x[k] for x in results], "ms", **d)
        R.add("serving", "request_throughput", N / wall, "requests/s", **d)
        R.add("serving", "output_token_throughput", sum(x["new"] for x in results) / wall, "tokens/s", **d)
        R.add("serving", "slo_attainment", 100 * sum(x["ttft"] <= SLO_TTFT and x["e2e"] <= SLO_E2E
                                                    for x in results) / N, "%", **d)
        R.add("serving", "queueing_fraction", 100 * sum(x["queue"] for x in results) /
              sum(x["e2e"] for x in results), "%", **d)
        R.add("serving", "batch_fill_ratio", 100 * sum(fills) / len(fills), "%", **d)
        R.add("serving", "padding_overhead", 100 * sum(pads) / len(pads), "%", **d)
        R.add("serving", "requests_failed", 0, "count", **d)
print("SERVE_DONE", sys.argv[1])
