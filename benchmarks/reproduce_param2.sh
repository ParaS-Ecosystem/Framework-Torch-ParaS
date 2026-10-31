#!/bin/bash
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
# Param2-17B on native PyTorch and on Torch-ParaS, then PASS / FAIL against the recorded results:
#   reproduce_param2.sh [output dir]
# Runs on GPU 0 (select it with ROCR_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES; the GPU should be otherwise idle).
#   REPRO_CONFIGS  configurations, comma separated (default: all five:
#                  native eager, paras eager, paras fused, native compiled, paras compiled)
#   REPRO_FULL=1   the whole inference sweep (all prefill lengths, batch sizes and contexts) instead of the
#                  shapes that are compared (prefill 2048 / 4096, decode batch 1 / 16)
#   REPRO_PPL=0    skip the perplexity runs (native and paras, bf16, all of WikiText-2)
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORK=${REPRO_WORK:-$PWD/torch-paras-repro}
[ -f "$WORK/env.sh" ] || { echo "run benchmarks/setup_stack.sh first (REPRO_WORK=$WORK)" >&2; exit 2; }
source "$WORK/env.sh"
OUT=${1:-$WORK/results/param2_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
export BENCH_OUT=$OUT
[ "${REPRO_FULL:-0}" = 1 ] || export BENCH_SHAPES=repro
IFS=, read -r -a CONFIGS <<< "${REPRO_CONFIGS:-native eager,paras eager,paras fused,native compiled,paras compiled}"
T0=$(date +%s)
{
  echo "host: $(hostname)  date: $(date '+%F %T %Z')"
  "$PTSYCL_PYTHON" -c "
import torch, torch_paras
print('GPU 0:', torch.cuda.get_device_properties(0).name, '| PyTorch', torch.__version__)
print('Torch-ParaS devices:', [torch_paras.device_name(i) for i in range(torch_paras.device_count())])"
} | tee "$OUT/machine_info.txt"
cd "$HERE/param2"
for cfg in "${CONFIGS[@]}"; do
  set -- $cfg
  echo "== $(date '+%T') infer.py $1 $2 bfloat16"
  "$PTSYCL_PYTHON" infer.py "$1" "$2" bfloat16 > "$OUT/infer_$1_$2.log" 2>&1
  echo "   rc=$? $(grep -E 'INFER_DONE|Error' "$OUT/infer_$1_$2.log" | tail -1)"
done
if [ "${REPRO_PPL:-1}" = 1 ]; then
  for b in native paras; do
    echo "== $(date '+%T') ppl.py $b bfloat16"
    "$PTSYCL_PYTHON" ppl.py $b bfloat16 > "$OUT/ppl_$b.log" 2>&1
    echo "   rc=$? $(grep -E 'PPL_DONE|Error' "$OUT/ppl_$b.log" | tail -1)"
  done
fi
echo "== run time $(( ($(date +%s) - T0) / 60 )) min; results in $OUT"
"$PTSYCL_PYTHON" "$HERE/compare.py" param2 "$OUT"
