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

"""Completes a ubenchX GEMM sweep on Torch-ParaS after a crash: runs ubench_gemm_paras.py on the shapes not yet in the
CSV; when a shape kills the process (no row written for it), that shape is recorded as status=crash with the
process's last error line, and the run resumes with the next shape. Rows already measured are kept as they are.

  gemm_resume.py <shapes.json> <out.csv> <flush MiB> [extra env NAME=VALUE ...]
"""
import csv
import json
import os
import subprocess
import sys

shapes_p, out_p, fl = sys.argv[1:4]
env = dict(os.environ, **dict(a.split("=", 1) for a in sys.argv[4:]))
shapes = [tuple(s) for s in json.load(open(shapes_p))]
H = os.path.dirname(os.path.abspath(__file__))
FIELDS = ["m", "k", "n", "dtype", "status", "latency_us", "tflops", "ai", "mfu_percent", "timings_us", "error"]


def done():
    if not os.path.exists(out_p):
        return []
    return [(int(r["m"]), int(r["k"]), int(r["n"])) for r in csv.DictReader(open(out_p))]


while True:
    have = done()
    i = len(have)
    if have != shapes[:i]:
        sys.exit("CSV rows are not a prefix of the shape list")
    if i >= len(shapes):
        break
    rest = os.path.join(os.path.dirname(out_p), "_rest_shapes.json")
    json.dump([list(s) for s in shapes[i:]], open(rest, "w"))
    p = subprocess.run([sys.executable, os.path.join(H, "ubench_gemm_paras.py"), "--device", "cuda", "--dtype", "bf16",
                        "--shapes", rest, "--flush-mib", fl], env=env, capture_output=True, text=True)
    lines = p.stdout.splitlines()
    rows = lines[1:] if lines and lines[0].startswith("m,") else lines
    with open(out_p, "a") as f:
        if not os.path.exists(out_p) or os.path.getsize(out_p) == 0:
            f.write(",".join(FIELDS) + "\n")
        for l in rows:
            f.write(l + "\n")
    got = len(rows)
    print(f"from shape {i}: {got} rows, rc={p.returncode}", flush=True)
    if p.returncode in (0, 1) and i + got >= len(shapes):
        break
    if i + got < len(shapes):
        m, k, n = shapes[i + got]
        err = (p.stderr.strip().splitlines() or ["(no message)"])[-1][:300].replace(",", ";")
        with open(out_p, "a", newline="") as f:
            csv.writer(f).writerow([m, k, n, "bf16", "crash", "", "", "", "", "", f"rc={p.returncode}: {err}"])
        print(f"  crash at {m}x{k}x{n}: {err}", flush=True)
print("GEMM_RESUME_DONE", len(done()), "of", len(shapes))
