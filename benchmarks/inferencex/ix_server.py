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

"""OpenAI-compatible streaming completions server for Param2-17B on native PyTorch or Torch-ParaS, so that the
unmodified InferenceX client (inferencex-e2e/infx/bench_serving/benchmark_serving.py) can measure both stacks.

  ix_server.py <native|paras> <eager|fused|compiled> --port P --max-batch B --max-input N

The same code serves both stacks; only the device and the execution mode differ.
- Scheduling: static batching. When the GPU is free, up to B waiting requests (B = the client's concurrency) form
  one batch: left padding, attention mask, position ids from the mask. The batch decodes until its longest request
  is done; a request that reaches its own max_tokens stops streaming and its slot idles until the batch ends.
  Neither stack has a continuous-batching scheduler, so both are measured with this one.
- Each step's sampled tokens are copied to the host and streamed at once as server-sent events, one token per
  event, so the client's TTFT and inter-token latencies are real per-token times.
- Only the last position's logits are computed in prefill (the lm_head sees hidden[:, -1:]), as serving engines
  do; it is the same module on both stacks.
- Prompts longer than --max-input tokens after re-tokenisation are cut from the left; the count is reported.
- GPU power (the device's own sensor) is sampled every 50 ms with wall-clock time stamps; GET /ix/power?t0=&t1=
  returns energy and mean power between two time.time() values. GET /ix/stats returns batches, fallbacks, cuts.
"""
import argparse
import json
import os
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from harness import MODEL, Backend, _gpu_hwmon   # noqa: E402
import torch                                       # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("backend")
ap.add_argument("mode")
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--max-batch", type=int, default=8)
ap.add_argument("--max-input", type=int, default=3072)
ap.add_argument("--gather-ms", type=float, default=50.0)
args = ap.parse_args()

be = Backend(args.backend)
from transformers import AutoTokenizer        # noqa: E402
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model, load = be.load(torch.bfloat16, args.mode)


class LastTokenHead(torch.nn.Module):
    """lm_head on the last position only (the logits a server samples from)."""

    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, h):
        return self.head(h[:, -1:])


model.lm_head = LastTokenHead(model.lm_head)
NAME = "Param2-17B-A2.4B-Thinking"
STATS = {"backend": args.backend, "mode": args.mode, "load_s": load["load_s"], "batches": 0, "batch_sizes": [],
         "requests": 0, "completion_tokens": 0, "prompt_cuts": 0, "prefill_ms": [], "step_ms_by_batch": {}}
pending = queue.Queue()

POWER = []
_hw = _gpu_hwmon()


def _power_loop():
    import subprocess
    while True:
        w = None
        if _hw:
            for n in ("power1_average", "power1_input"):
                try:
                    w = int(open(os.path.join(_hw, n)).read()) / 1e6
                    break
                except Exception:
                    pass
        elif torch.version.cuda:
            try:
                idx = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
                w = float(subprocess.run(["nvidia-smi", "-i", idx, "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                                         capture_output=True, text=True, timeout=5).stdout)
            except Exception:
                pass
        if w is not None:
            POWER.append((time.time(), w))
            if len(POWER) > 2_000_000:
                del POWER[:1_000_000]
        time.sleep(0.05)


threading.Thread(target=_power_loop, daemon=True).start()


def power_window(t0, t1):
    s = [p for p in POWER if t0 <= p[0] <= t1]
    if len(s) < 2:
        return {"available": False}
    e = sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(s, s[1:]))
    return {"available": True, "energy_j": e, "power_avg_w": e / (s[-1][0] - s[0][0]), "samples": len(s),
            "seconds": s[-1][0] - s[0][0]}


def engine():
    while True:
        first = pending.get()
        batch = [first]
        deadline = time.time() + args.gather_ms / 1000
        while len(batch) < args.max_batch:
            try:
                batch.append(pending.get(timeout=max(0.0, deadline - time.time())))
            except queue.Empty:
                break
        try:
            run_batch(batch)
        except Exception as e:
            for r in batch:
                r["out"].put(("error", repr(e)[:300]))
            be.empty_cache()


@torch.no_grad()
def run_batch(batch):
    c = len(batch)
    L = max(len(r["ids"]) for r in batch)
    x = torch.full((c, L), tok.pad_token_id or 0, dtype=torch.long)
    mask = torch.zeros(c, L, dtype=torch.long)
    for i, r in enumerate(batch):
        n = len(r["ids"])
        x[i, L - n:] = torch.tensor(r["ids"])
        mask[i, L - n:] = 1
    t = time.perf_counter()
    x, mask = x.to(be.dev), mask.to(be.dev)
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    out = model(input_ids=x, attention_mask=mask, position_ids=pos, use_cache=True)
    past, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
    ids = nxt.cpu()
    STATS["prefill_ms"].append((c, L, 1000 * (time.perf_counter() - t)))
    longest = max(r["max_tokens"] for r in batch)
    emit(batch, ids, 1)
    last = pos[:, -1:]
    steps = []
    for k in range(2, longest + 1):
        t = time.perf_counter()
        mask = torch.cat([mask, torch.ones(c, 1, dtype=mask.dtype, device=be.dev)], 1)
        last = last + 1
        out = model(input_ids=nxt, attention_mask=mask, position_ids=last, past_key_values=past, use_cache=True)
        past, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
        ids = nxt.cpu()
        steps.append(1000 * (time.perf_counter() - t))
        emit(batch, ids, k)
    del past, out
    STATS["batches"] += 1
    STATS["batch_sizes"].append(c)
    STATS["step_ms_by_batch"].setdefault(str(c), []).extend(steps[-64:])


def emit(batch, ids, k):
    for i, r in enumerate(batch):
        if k <= r["max_tokens"]:
            r["out"].put(("tok", tok.decode([int(ids[i, 0])])))
            if k == r["max_tokens"]:
                STATS["completion_tokens"] += r["max_tokens"]
                r["out"].put(("done", r["max_tokens"]))


threading.Thread(target=engine, daemon=True).start()


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/v1/models":
            return self._json({"object": "list", "data": [{"id": NAME, "object": "model"}]})
        if u.path in ("/health", "/v1/health"):
            return self._json({"ok": True})
        if u.path == "/ix/power":
            q = parse_qs(u.query)
            return self._json(power_window(float(q["t0"][0]), float(q["t1"][0])))
        if u.path == "/ix/set":
            q = parse_qs(u.query)
            args.max_batch = int(q["max_batch"][0])
            return self._json({"max_batch": args.max_batch})
        if u.path == "/ix/stats":
            s = dict(STATS)
            s["cpu_fallbacks"] = be.stats().get("cpu_fallbacks", 0) if be.paras else 0
            s["peak_memory_gib"] = be.peak() / 2**30
            return self._json(s)
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if not self.path.endswith("/completions"):
            return self._json({"error": "not found"}, 404)
        prompt = body.get("prompt", "")
        ids = prompt if isinstance(prompt, list) else tok(prompt, add_special_tokens=False).input_ids
        if len(ids) > args.max_input:
            ids = ids[-args.max_input:]
            STATS["prompt_cuts"] += 1
        req = {"ids": ids or [0], "max_tokens": int(body.get("max_tokens", 16)), "out": queue.Queue()}
        STATS["requests"] += 1
        pending.put(req)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        rid = f"cmpl-{STATS['requests']}"
        while True:
            kind, val = req["out"].get()
            if kind == "tok":
                ev = {"id": rid, "object": "text_completion", "model": NAME,
                      "choices": [{"index": 0, "text": val, "finish_reason": None}]}
            elif kind == "done":
                ev = {"id": rid, "object": "text_completion", "model": NAME, "choices": [],
                      "usage": {"prompt_tokens": len(req["ids"]), "completion_tokens": val,
                                "total_tokens": len(req["ids"]) + val}}
            else:
                ev = {"error": {"message": val}}
            self.wfile.write(b"data: " + json.dumps(ev).encode() + b"\n\n")
            self.wfile.flush()
            if kind != "tok":
                break
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


print(f"IX_SERVER_READY {args.backend} {args.mode} port={args.port} load={load['load_s']:.0f}s", flush=True)
ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()
