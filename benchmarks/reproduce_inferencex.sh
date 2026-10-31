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
# InferenceX serving (its client unchanged) on native PyTorch and on Torch-ParaS, optionally the GEMM and
# bandwidth micro-benchmarks, then PASS / FAIL against the recorded results:
#   reproduce_inferencex.sh [output dir]
# Runs on GPU 0 (the GPU should be otherwise idle).
#   IX_CONFIGS   configurations, comma separated (default: native eager, paras fused, paras compiled;
#                add "native compiled" for all four; "none" runs only the micro-benchmarks)
#   IX_SCENARIO  1k1k (default) or 3k1k
#   IX_CONCS     concurrencies (default "1 16"; InferenceX's full sweep is "1 4 8 16 32 64")
#   IX_PORT      first local port for the server (default 8600)
#   REPRO_MICRO=1  also the GEMM (Param2 and OperatorX shapes) and copy-bandwidth micro-benchmarks
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORK=${REPRO_WORK:-$PWD/torch-paras-repro}
[ -f "$WORK/env.sh" ] || { echo "run benchmarks/setup_stack.sh first (REPRO_WORK=$WORK)" >&2; exit 2; }
source "$WORK/env.sh"
OUT=${1:-$WORK/results/inferencex_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
export BENCH_OUT=$OUT/rows IX_CONCS=${IX_CONCS:-1 16}
S=${IX_SCENARIO:-1k1k}
PORT=${IX_PORT:-8600}
IFS=, read -r -a CONFIGS <<< "${IX_CONFIGS:-native eager,paras fused,paras compiled}"
T0=$(date +%s)
"$PTSYCL_PYTHON" -c "
import sys; sys.path.insert(0, '$HERE/common')
from harness import device_label; print(device_label())" | tail -1 > "$OUT/machine.txt"
echo "machine: $(cat "$OUT/machine.txt") ($(hostname)); scenario $S; concurrencies $IX_CONCS"
[ "${IX_CONFIGS:-}" = none ] && CONFIGS=()
for cfg in "${CONFIGS[@]}"; do
  set -- $cfg
  bash "$HERE/inferencex/run_ix.sh" "$1" "$2" "$S" "$PORT" "$OUT/$1_$2_$S" 2>&1 | tee "$OUT/run_$1_$2_$S.log" | grep --line-buffered -E "^==|conc|IX_RUN_DONE|did not start"
  PORT=$((PORT + 1))
done
RC=0
[ ${#CONFIGS[@]} -gt 0 ] && { "$PTSYCL_PYTHON" "$HERE/compare.py" inferencex "$OUT" || RC=1; }
if [ "${REPRO_MICRO:-0}" = 1 ]; then
  bash "$HERE/micro/run_micro.sh" "$OUT/micro" 2>&1 | tee "$OUT/micro.log" | grep --line-buffered -E "^==|rc=|MICRO_DONE"
  cp "$OUT/machine.txt" "$OUT/micro/machine.txt"
  "$PTSYCL_PYTHON" "$HERE/compare.py" micro "$OUT/micro" || RC=1
fi
echo "== run time $(( ($(date +%s) - T0) / 60 )) min; results in $OUT"
exit $RC
