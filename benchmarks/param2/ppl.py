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

"""Perplexity of Param2-17B on the whole WikiText-2 test set.

  ppl.py <native|paras> <bfloat16|float32>

Non-overlapping 2048-token windows over all 353,839 tokens (the last, shorter
window included); every predicted token counts once. Appends rows to
$BENCH_OUT/perplexity/rows.jsonl.
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from harness import Backend, Rows, tokens   # noqa: E402
import torch                                # noqa: E402

WINDOW = 2048


def main(backend_name, dtype_name):
    be = Backend(backend_name)
    model, _ = be.load(getattr(torch, dtype_name), "eager")
    ids = tokens()
    R = Rows("perplexity", backend_name, "eager", dtype_name)
    nll, count = 0.0, 0
    t = time.perf_counter()
    with torch.no_grad():
        for start in range(0, ids.numel() - 1, WINDOW):
            x = ids[start:start + WINDOW][None].to(be.dev)
            if x.shape[1] < 2:
                break
            lg = model(input_ids=x, use_cache=False).logits[0, :-1].float()
            nll += torch.nn.functional.cross_entropy(lg, x[0, 1:], reduction="sum").item()
            count += x.shape[1] - 1
    seconds = time.perf_counter() - t
    ppl = math.exp(nll / count)
    R.add("perplexity", "perplexity", ppl, "ppl", window=WINDOW, predicted_tokens=count)
    R.add("perplexity", "seconds", seconds, "s", window=WINDOW, predicted_tokens=count)
    print(f"PPL_DONE {backend_name} {dtype_name} perplexity {ppl:.4f} over {count} tokens in {seconds:.1f} s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
