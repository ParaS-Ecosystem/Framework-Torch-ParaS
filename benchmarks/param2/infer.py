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

"""Param2-17B inference: prefill sweep, decode (TTFT / TPOT / ITL / throughput
by batch), decode by context length, energy, memory, host synchronisation and
greedy-token agreement -- one backend, execution mode and precision per
process (spec Part A, sections 3, 3.1, 3.2; INF-01..24).

  infer.py <native|paras> <eager|fused|compiled> <bfloat16|float32>

BENCH_SHAPES=repro limits the run to the shapes the reproduction compares
(prefill 2048 and 4096 tokens, decode batch 1 and 16 from 128 tokens, no
context sweep); BENCH_QUICK=1 is a seconds-long smoke run.

Timing: every interval starts and ends with a device synchronisation, so
asynchronous work is inside it. Request latency = prefill + all decode steps
of one request (tokenisation excluded: inputs are token ids); TTFT = prefill
+ the first sampled token; TPOT = mean of the later steps; ITL = each step
after the first. Greedy decoding, KV cache (DynamicCache), batch = identical
copies of distinct prompts.
"""
import hashlib
import os
import sys
import time

T_IMPORT = time.perf_counter()
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from harness import Backend, HwSampler, Rows, describe, timer, tokens   # noqa: E402
import torch                                                          # noqa: E402

backend_name, mode, dtype_name = sys.argv[1:4]
be = Backend(backend_name)
backend_init_s = time.perf_counter() - T_IMPORT
dtype = getattr(torch, dtype_name)
R = Rows("inference", backend_name, mode, dtype_name)
R.add("init", "backend_import_and_init", backend_init_s, "s")
t = time.perf_counter()
probe = torch.ones(1, device=be.dev)
be.sync()
R.add("init", "device_first_tensor", 1000 * (time.perf_counter() - t), "ms")
del probe

model, load = be.load(dtype, mode)
for k, v in load.items():
    R.add("init", k, v, "s")
R.add("init", "weights_allocated", be.allocated() / 2**30, "GiB")
R.add("init", "parameters", sum(p.numel() for p in model.parameters()), "count")
ids = tokens()
FP32 = dtype == torch.float32
PREFILL_LENS = [128, 2048] if FP32 else [128, 512, 2048, 4096]
DECODE_BS = [1, 4] if FP32 else [1, 2, 4, 8, 16]
NEW = 64 if FP32 else 128
CTX = [512, 2048] if FP32 else [512, 2048, 4032]
if os.environ.get("BENCH_SHAPES") == "repro" and not FP32:
    PREFILL_LENS, DECODE_BS, CTX = [2048, 4096], [1, 16], []
if os.environ.get("BENCH_QUICK"):
    PREFILL_LENS, DECODE_BS, NEW, CTX = [128], [1, 2], 8, [512]


def prompt(length, bs, offset=0):
    rows = [ids[offset + i * 4096: offset + i * 4096 + length] for i in range(bs)]
    return torch.stack(rows).to(be.dev)


def stats_snapshot():
    s = be.stats()
    return {k: v for k, v in s.items() if isinstance(v, (int, float))}


with torch.no_grad():
    for length in PREFILL_LENS:
        x = prompt(length, 1)
        cold = []
        with timer(be, cold):
            model(input_ids=x, use_cache=False)
        R.add("prefill", "first_call_latency", cold[0], "ms", phase="warmup", batch_size=1,
              input_tokens=length)

    for length in PREFILL_LENS:
        for bs in ([1, 4] if length == 512 else [1]):
            x = prompt(length, bs)
            model(input_ids=x, use_cache=False)
            be.reset_peak()
            times = []
            with HwSampler(alloc_fn=be.allocated if be.paras else None) as hw:
                for _ in range(7):
                    with timer(be, times):
                        model(input_ids=x, use_cache=False)
            d = dict(batch_size=bs, input_tokens=length)
            R.add_stats("prefill", "latency", times, "ms", **d)
            med = describe(times)["median"]
            R.add("prefill", "tokens_per_s", bs * length / (med / 1000), "tokens/s", **d)
            R.add("prefill", "latency_per_input_token", med / (bs * length), "ms/token", **d)
            R.add("prefill", "peak_allocated", be.peak() / 2**30, "GiB", **d)
            R.add_hw("prefill", hw.summary(), work_units=bs * length * len(times),
                     unit_name="input_token", **d)

    digests = {}
    for bs in DECODE_BS:
        if mode == "compiled":
            x = prompt(128, bs, offset=7 * 128)
            out = model(input_ids=x, use_cache=True)
            past, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
            for _ in range(NEW - 1):
                out = model(input_ids=nxt, past_key_values=past, use_cache=True)
                past, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
            be.sync()
            del out, past
        ttft, tpot, total, itl = [], [], [], []
        steps_per_req = []
        s0 = stats_snapshot()
        be.reset_peak()
        with HwSampler(alloc_fn=be.allocated if be.paras else None) as hw:
            for req in range(3 if bs <= 4 else 2):
                x = prompt(128, bs, offset=req * 128)
                be.sync()
                t0 = time.perf_counter()
                out = model(input_ids=x, use_cache=True)
                past = out.past_key_values
                nxt = out.logits[:, -1:].argmax(-1)
                be.sync()
                t_first = time.perf_counter()
                ttft.append(1000 * (t_first - t0))
                seq = [nxt]
                steps = []
                for _ in range(NEW - 1):
                    with timer(be, steps):
                        out = model(input_ids=nxt, past_key_values=past, use_cache=True)
                        past = out.past_key_values
                        nxt = out.logits[:, -1:].argmax(-1)
                    seq.append(nxt)
                total.append(1000 * (time.perf_counter() - t0))
                itl += steps
                tpot.append(sum(steps) / len(steps))
                steps_per_req.append(len(steps) + 1)
                if req == 0:
                    toks = torch.cat(seq, 1).cpu()
                    digests[bs] = hashlib.sha1(toks.numpy().tobytes()).hexdigest()[:12]
            cfg = model.config
            kv_bytes = (2 * cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.head_dim *
                        (128 + NEW) * bs * torch.tensor([], dtype=dtype).element_size())
        s1 = stats_snapshot()
        d = dict(batch_size=bs, input_tokens=128, output_tokens=NEW)
        R.add_stats("decode", "ttft", ttft, "ms", **d)
        R.add_stats("decode", "tpot", tpot, "ms/token", **d)
        R.add_stats("decode", "itl", itl, "ms", **d)
        R.add_stats("decode", "request_latency", total, "ms", **d)
        tot_tokens = bs * sum(steps_per_req)
        R.add("decode", "output_tokens_per_s", tot_tokens / (sum(total) / 1000), "tokens/s", **d)
        R.add("decode", "aggregate_tokens_per_s", bs * (sum(steps_per_req) + 128 * len(total)) /
              (sum(total) / 1000), "tokens/s", **d)
        R.add("decode", "requests_per_s", bs * len(total) / (sum(total) / 1000), "requests/s", **d)
        R.add("decode", "completion_success_rate", 100.0, "%", **d)
        R.add("decode", "peak_allocated", be.peak() / 2**30, "GiB", **d)
        if kv_bytes:
            R.add("decode", "kv_cache_size_at_end", kv_bytes / 2**30, "GiB", **d)
            R.add("decode", "kv_cache_read_per_step_avg", kv_bytes / 2 / 2**30 /
                  (describe(itl)["mean"] / 1000), "GiB/s", **d)
        R.add_hw("decode", hw.summary(), work_units=tot_tokens, unit_name="output_token", **d)
        for k in set(s0) | set(s1):
            if isinstance(s1.get(k, 0), (int, float)):
                R.add("decode", f"runtime_stats.{k}.per_step",
                      (s1.get(k, 0) - s0.get(k, 0)) / max(1, sum(steps_per_req)), "count/step", **d)
    R.blob(f"greedy_digests_{backend_name}_{mode}_{dtype_name}", digests)

    for ctx in CTX:
        x = prompt(ctx, 1)
        out = model(input_ids=x, use_cache=True)
        past, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
        steps = []
        for _ in range(32):
            with timer(be, steps):
                out = model(input_ids=nxt, past_key_values=past, use_cache=True)
                past, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
        R.add_stats("decode_context", "itl", steps[4:], "ms", batch_size=1, input_tokens=ctx,
                    output_tokens=32)
print("INFER_DONE", backend_name, mode, dtype_name)
