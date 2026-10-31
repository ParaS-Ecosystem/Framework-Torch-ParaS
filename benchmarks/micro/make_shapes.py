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

"""GEMM shapes for the ubenchX run: Param2-17B's own matrix products at serving token counts, then every bf16 x bf16
shape of InferenceX's OperatorX GEMM testlist ($INFERENCEX_DIR/operatorx/testlists/gemm.json), as [M, K, N]."""
import json, os
H = os.path.dirname(os.path.abspath(__file__))
P2 = [(2048, 3072, "qkv"), (2048, 2048, "o_proj"), (2048, 9216, "dense gate/up"), (9216, 2048, "dense down"),
      (2048, 2048, "expert gate/up"), (2048, 2048, "expert down"), (2048, 64, "router"), (2048, 128008, "lm_head")]
param2 = [[m, k, n] for m in (1, 8, 16, 64, 256, 1024, 4096, 16384) for k, n, _ in P2]
t = json.load(open(os.path.join(os.environ["INFERENCEX_DIR"], "operatorx", "testlists", "gemm.json")))
ix = sorted({(e["args"]["m"], e["args"]["k"], e["args"]["n"]) for e in t
             if e["args"].get("a", {}).get("dtype") == "bf16" and e["args"].get("b", {}).get("dtype") == "bf16"})
json.dump(param2, open(os.path.join(H, "shapes_param2.json"), "w"))
json.dump([list(s) for s in ix], open(os.path.join(H, "shapes_operatorx.json"), "w"))
NAMES = {f"{k}x{n}": name for k, n, name in P2}
NAMES["2048x2048"] = "o_proj / expert"
json.dump(NAMES, open(os.path.join(H, "shapes_param2_names.json"), "w"))
print(len(param2), len(ix))
