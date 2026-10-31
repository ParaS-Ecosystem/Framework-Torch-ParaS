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

"""Data movement that a single-node deployment depends on, on either stack (what CollectiveX would measure on top
of a communication library; Torch-ParaS has no torch.distributed backend yet, so collectives themselves are not
available to it and are not measured here):
host -> device and device -> host from pageable and from pinned host memory, and device -> device between two GPUs
of the node (the path that splits a model over GPUs, as the V100 x4 runs do). Each point: warm-up, then repeated
copies for about 200 ms between two synchronisations; median of 5 such loops; GB/s = bytes / time.

  transfers.py <cuda|paras>        (needs two visible GPUs for the device -> device rows)
"""
import json
import statistics
import sys
import time

import torch

kind = sys.argv[1]
if kind == "paras":
    import torch_paras  # noqa: F401
api = getattr(torch, kind)
ndev = api.device_count() if kind == "cuda" else sum(1 for i in range(api.device_count())
                                                      if torch_paras.is_gpu_device(i))
rows = []


def timeit(fn, nbytes):
    fn(); api.synchronize()
    t = time.perf_counter(); fn(); api.synchronize()
    one = max(time.perf_counter() - t, 1e-7)
    n = max(1, min(5000, int(0.2 / one)))
    out = []
    for _ in range(5):
        api.synchronize()
        t = time.perf_counter()
        for _ in range(n):
            fn()
        api.synchronize()
        out.append((time.perf_counter() - t) / n)
    s = statistics.median(out)
    return 1e6 * s, nbytes / s / 1e9


SIZES = [4 << 10, 64 << 10, 1 << 20, 16 << 20, 256 << 20, 1 << 30]
for size in SIZES:
    n = size
    d0 = torch.empty(n, dtype=torch.uint8, device=f"{kind}:0")
    for pin in (False, True):
        try:
            h = torch.empty(n, dtype=torch.uint8, pin_memory=pin) if pin else torch.empty(n, dtype=torch.uint8)
            if pin and not h.is_pinned():
                raise RuntimeError("pin_memory returned pageable memory")
            for name, fn in ((f"host_to_device_{'pinned' if pin else 'pageable'}", lambda: d0.copy_(h, non_blocking=pin)),
                             (f"device_to_host_{'pinned' if pin else 'pageable'}", lambda: h.copy_(d0, non_blocking=pin))):
                us, gbs = timeit(fn, size)
                rows.append({"path": name, "bytes": size, "time_us": us, "gbps": gbs, "ok": True})
        except Exception as e:
            rows.append({"path": f"host_device_{'pinned' if pin else 'pageable'}", "bytes": size, "ok": False,
                         "error": repr(e)[:200]})
    d0b = torch.empty_like(d0)
    us, gbs = timeit(lambda: d0b.copy_(d0), size)
    rows.append({"path": "device_to_device_same_gpu", "bytes": size, "time_us": us, "gbps": gbs, "ok": True})
    if ndev >= 2:
        d1 = torch.empty(n, dtype=torch.uint8, device=f"{kind}:1")
        try:
            us, gbs = timeit(lambda: d1.copy_(d0), size)
            rows.append({"path": "gpu0_to_gpu1", "bytes": size, "time_us": us, "gbps": gbs, "ok": True})
        except Exception as e:
            rows.append({"path": "gpu0_to_gpu1", "bytes": size, "ok": False, "error": repr(e)[:200]})
    print(size, [(r["path"], round(r.get("gbps", 0), 2)) for r in rows if r["bytes"] == size], flush=True)
json.dump({"gpus": ndev, "rows": rows}, open(f"transfers_{kind}.json", "w"), indent=1)
