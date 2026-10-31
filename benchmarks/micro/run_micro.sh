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
# GEMM (InferenceX ubenchX gemm/bench.py: unchanged on native PyTorch, through ubench_gemm_paras.py on Torch-ParaS)
# for Param2's own shapes and every bf16 shape of InferenceX's OperatorX testlist, and device copy bandwidth
# (ubenchX's method with device events on both stacks), on GPU 0:
#   run_micro.sh <output dir>          MICRO_PARTS="gemm membw" by default; add "transfers" with two GPUs
# The Torch-ParaS GEMM sweep runs through gemm_resume.py: a shape that kills the process (a known intermittent GPU
# memory fault in the device GEMM path) is recorded as status=crash and the sweep continues with the next shape.
set -uo pipefail
O=$1; mkdir -p "$O"; O=$(cd "$O" && pwd)
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
UB=$INFERENCEX_DIR/ubenchX
cd "$HERE"
echo "== $(date '+%F %T') micro on $(hostname)"
L2=$("$PTSYCL_PYTHON" -c "import torch;print(getattr(torch.cuda.get_device_properties(0), 'L2_cache_size', 0) or 268435456)")
FL=$(( (L2 + 1048575) / 1048576 )); echo "L2 $L2 bytes -> flush $FL MiB (both stacks)"
has() { case " ${MICRO_PARTS:-gemm membw} " in *" $1 "*) return 0;; esac; return 1; }
has gemm && for sh in param2 operatorx; do
  "$PTSYCL_PYTHON" "$UB/gemm/bench.py" --device cuda --dtype bf16 --shapes shapes_$sh.json --flush-mib $FL \
     > "$O/gemm_${sh}_native.csv" 2> "$O/gemm_${sh}_native.meta"; echo "gemm $sh native rc=$? $(wc -l < "$O/gemm_${sh}_native.csv") rows"
  rm -f "$O/gemm_${sh}_paras.csv"
  "$PTSYCL_PYTHON" gemm_resume.py shapes_$sh.json "$O/gemm_${sh}_paras.csv" $FL > "$O/gemm_${sh}_paras.resume.log" 2>&1
  echo "gemm $sh paras rc=$? $(tail -1 "$O/gemm_${sh}_paras.resume.log"), crashed shapes: $(grep -c ',crash,' "$O/gemm_${sh}_paras.csv")"
done
has membw && ( cd "$O" && for k in cuda paras; do "$PTSYCL_PYTHON" "$HERE/mem_bw_events.py" $k > mem_bw_events_$k.txt 2>&1; echo "mem_bw $k rc=$?"; done )
has transfers && ( cd "$O" && for k in cuda paras; do "$PTSYCL_PYTHON" "$HERE/transfers.py" $k > transfers_$k.txt 2>&1; echo "transfers $k rc=$?"; done )
echo MICRO_DONE
