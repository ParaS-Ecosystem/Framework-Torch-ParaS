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

"""Device memory copy bandwidth, the method of InferenceX's ubenchX/mem_bw/bench.py, on either stack.

ubenchX times dst.copy_(src) on float32 tensors at every power-of-two size with triton.testing.do_bench, which
zeroes a 256 MB buffer before every repetition (cache flush), times each repetition with device events and reports
the mean; bandwidth counts one read and one write per byte. Triton does not target Torch-ParaS, so this script does
the same with the stack's own events: 25 ms of warm-up, then repetitions for about 100 ms (at least 10), each
preceded by the 256 MB flush, mean of the event times. The original bench.py is run unchanged on native PyTorch
in the same job as a check of this method.

  mem_bw_events.py <cuda|paras>
"""
import json
import sys
import time

import torch

kind = sys.argv[1]
if kind == "paras":
    import torch_paras  # noqa: F401
api = getattr(torch, kind)
dev = f"{kind}:0"
MAX = 8 * 1024**3
sizes = [1 << p for p in range(3, MAX.bit_length())]
a = torch.randn(MAX // 4, device=dev, dtype=torch.float32)
b = torch.empty_like(a)
flush = torch.empty(256 * 1024**2, dtype=torch.int8, device=dev)
rows = []
for size in sizes:
    n = size // 4
    src, dst = a[:n], b[:n]
    t = time.perf_counter()
    while time.perf_counter() - t < 0.025:
        dst.copy_(src)
    api.synchronize()
    s, e = api.Event(enable_timing=True), api.Event(enable_timing=True)
    s.record(); dst.copy_(src); e.record(); api.synchronize()
    reps = max(10, min(1000, int(100 / max(s.elapsed_time(e), 1e-3))))
    starts = [api.Event(enable_timing=True) for _ in range(reps)]
    ends = [api.Event(enable_timing=True) for _ in range(reps)]
    for st, en in zip(starts, ends):
        flush.zero_()
        st.record()
        dst.copy_(src)
        en.record()
    api.synchronize()
    ms = sum(st.elapsed_time(en) for st, en in zip(starts, ends)) / reps
    rows.append({"bytes": size, "time_ms": ms, "bandwidth_gbps": 2 * size / (ms * 1e-3) / 1e9, "reps": reps})
    print(f"{size:>14} B  {ms:10.4f} ms  {rows[-1]['bandwidth_gbps']:10.2f} GB/s", flush=True)
json.dump(rows, open(f"mem_bw_{kind}.json", "w"), indent=1)
