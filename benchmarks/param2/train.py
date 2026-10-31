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

"""Param2-17B training (spec Part B, sections 5-7; TRN-01..20): per-phase step
time (forward, loss, backward, clip, optimizer; synchronised between phases),
memory per phase, loss / gradient-norm curves, NaN/Inf, update-to-weight
ratio, checkpoint save/load of a layer, tokens/s. Same data, order and seed for
every backend.

  train.py <native|paras> <eager|compiled> [steps]

bf16 weights, SGD with momentum 0.9 (AdamW's fp32 states for 17B parameters,
137 GB, plus weights and gradients exceed one 192 GB MI300X), lr 1e-4,
gradient clipping at 1.0, batch 1 x 256 tokens of WikiText-2.
"""
import math
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from harness import Backend, HwSampler, Rows, describe, tokens   # noqa: E402
import torch                                                   # noqa: E402

backend_name, mode = sys.argv[1:3]
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 12
SEQ = 256
be = Backend(backend_name)
R = Rows("training", backend_name, mode, "bfloat16")
torch.manual_seed(0)
model, load = be.load(torch.bfloat16, mode, train=True)
R.add("train_init", "load_s", load["load_s"], "s")
params = [p for p in model.parameters() if p.requires_grad]
R.add("train_init", "parameter_memory", sum(p.numel() * p.element_size() for p in params) / 2**30, "GiB")
opt = torch.optim.SGD(params, lr=1e-4, momentum=0.9)
ids = tokens()
mem = {}

phase_t = {k: [] for k in ("forward", "loss", "backward", "clip", "optimizer", "step")}
losses, gnorms, upd_ratio, nonfinite = [], [], [], 0
with HwSampler(alloc_fn=be.allocated if be.paras else None) as hw:
    for step in range(STEPS):
        x = ids[step * SEQ:(step + 1) * SEQ][None].to(be.dev)
        be.sync()
        t0 = time.perf_counter()
        out = model(input_ids=x, use_cache=False)
        be.sync(); t1 = time.perf_counter()
        mem.setdefault("after_forward", []).append(be.allocated())
        logits = out.logits[0, :-1].float()
        loss = torch.nn.functional.cross_entropy(logits, x[0, 1:])
        be.sync(); t2 = time.perf_counter()
        loss.backward()
        be.sync(); t3 = time.perf_counter()
        mem.setdefault("after_backward", []).append(be.allocated())
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        be.sync(); t4 = time.perf_counter()
        if step == STEPS - 1:
            before = [p.detach().float().norm() for p in params[:50]]
        opt.step()
        be.sync(); t5 = time.perf_counter()
        if step == STEPS - 1:
            after = [p.detach().float().norm() for p in params[:50]]
        opt.zero_grad(set_to_none=True)
        mem.setdefault("after_optimizer", []).append(be.allocated())
        for k, a, b in (("forward", t0, t1), ("loss", t1, t2), ("backward", t2, t3), ("clip", t3, t4),
                        ("optimizer", t4, t5), ("step", t0, t5)):
            phase_t[k].append(1000 * (b - a))
        lv, gv = loss.item(), float(gn)
        losses.append(lv)
        gnorms.append(gv)
        nonfinite += (not math.isfinite(lv)) + (not math.isfinite(gv))
        R.add("train_step", "loss", lv, "nats", step=step)
        R.add("train_step", "grad_norm", gv, "L2", step=step)
        for k in phase_t:
            R.add("train_step", f"{k}_ms", phase_t[k][-1], "ms", step=step)
        del out, logits, loss

hwsum = hw.summary()
measured = slice(2, None)
for k, v in phase_t.items():
    R.add_stats("train", f"{k}_latency", v[measured], "ms", batch_size=1, input_tokens=SEQ)
    R.add("train", f"{k}_first_steps_ms", sum(v[:2]) / 2, "ms", phase="warmup")
med = describe(phase_t["step"][measured])["median"]
R.add("train", "steps_per_s", 1000 / med, "steps/s")
R.add("train", "tokens_per_s", SEQ / (med / 1000), "tokens/s")
R.add("train", "nonfinite_events", nonfinite, "count")
R.add("train", "grad_clip_frequency", 100 * sum(g > 1.0 for g in gnorms) / len(gnorms), "%")
for k, v in mem.items():
    R.add("train", f"allocated_{k}.max", max(v) / 2**30, "GiB")
R.add("train", "memory_growth_per_step", (mem["after_optimizer"][-1] - mem["after_optimizer"][2]) /
      max(1, STEPS - 3), "bytes/step")
if not be.paras:
    R.add("train", "peak_allocated", torch.cuda.max_memory_allocated() / 2**30, "GiB")
R.add("train", "update_to_weight_ratio.median",
      describe([abs(a - b).item() / max(b.item(), 1e-12) for a, b in zip(after, before)])["median"], "ratio")
R.add_hw("train", hwsum, work_units=SEQ * STEPS, unit_name="token")

layer = model.model.layers[1] if hasattr(model, "model") else None
if layer is not None:
    path = os.path.join(tempfile.gettempdir(), f"book_layer_{os.getpid()}.pt")
    be.sync(); t = time.perf_counter()
    torch.save({k: v.detach().cpu() for k, v in layer.state_dict().items()}, path)
    R.add("checkpoint", "layer_save_s", time.perf_counter() - t, "s")
    t = time.perf_counter()
    sd = torch.load(path)
    layer.load_state_dict({k: v.to(be.dev) for k, v in sd.items()})
    be.sync()
    R.add("checkpoint", "layer_load_s", time.perf_counter() - t, "s")
    R.add("checkpoint", "layer_bytes", os.path.getsize(path) / 2**20, "MiB")
    os.remove(path)
R.blob(f"curves_{backend_name}_{mode}", {"loss": losses, "grad_norm": gnorms})
print("TRAIN_DONE", backend_name, mode)
