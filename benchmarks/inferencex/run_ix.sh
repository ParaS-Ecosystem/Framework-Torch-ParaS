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
# One stack through the InferenceX fixed-sequence sweep on GPU 0:
#   run_ix.sh <native|paras> <eager|fused|compiled> <1k1k|3k1k|smoke> <port> <output dir>
# Starts ix_server.py for the stack, then for each concurrency runs the unmodified InferenceX client with the
# arguments InferenceX's own fixed_seq.py uses (random dataset, range ratio 0.8, 10 x conc prompts, 2 x conc
# warm-ups, request rate inf, --ignore-eos, percentiles ttft,tpot,itl,e2el), then reads the GPU energy of the
# client's run window from the server. IX_CONCS overrides the concurrencies.
# Needs (from setup_stack.sh's env.sh): PTSYCL_PYTHON, IX_CLIENT_PYTHON, INFERENCEX_DIR, PARAM2_MODEL.
set -uo pipefail
B=$1 M=$2 S=$3 PORT=$4 OUT=$5
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
IX=$INFERENCEX_DIR/inferencex-e2e
case $S in
  1k1k)  ISL=1024 OSL=1024 CONCS="1 4 8 16 32 64" MAXIN=3072 ;;
  3k1k)  ISL=3072 OSL=1024 CONCS="1 4 8 16 32" MAXIN=3072 ;;
  smoke) ISL=128 OSL=16 CONCS="1 4" MAXIN=3072 ;;
  *) echo "scenario must be 1k1k, 3k1k or smoke" >&2; exit 2 ;;
esac
[ -n "${IX_CONCS:-}" ] && CONCS=$IX_CONCS
mkdir -p "$OUT"
echo "== $(date '+%F %T') $B $M $S on $(hostname), concurrencies: $CONCS"
cd "$HERE"
for C in $CONCS; do
  "$PTSYCL_PYTHON" ix_server.py $B $M --port $PORT --max-batch $C --max-input $MAXIN > "$OUT/server_c$C.log" 2>&1 &
  SP=$!
  for i in $(seq 1 360); do curl -sf http://127.0.0.1:$PORT/v1/models > /dev/null && break; kill -0 $SP 2>/dev/null || break; sleep 5; done
  grep -q IX_SERVER_READY "$OUT/server_c$C.log" || { echo "server did not start"; tail -20 "$OUT/server_c$C.log"; kill $SP 2>/dev/null; exit 1; }
  T0=$(date +%s.%N)
  ( cd "$IX" && "$IX_CLIENT_PYTHON" -m infx.bench_serving.benchmark_serving \
      --model "$PARAM2_MODEL" --served-model-name Param2-17B-A2.4B-Thinking --trust-remote-code --backend vllm \
      --base-url http://127.0.0.1:$PORT --dataset-name random --random-input-len $ISL --random-output-len $OSL \
      --random-range-ratio 0.8 --num-prompts $((10 * C)) --max-concurrency $C --request-rate inf --ignore-eos \
      --save-result --num-warmups $((2 * C)) --percentile-metrics ttft,tpot,itl,e2el \
      --result-dir "$OUT" --result-filename ix_c$C.json ) > "$OUT/client_c$C.log" 2>&1
  RC=$?
  T1=$(date +%s.%N)
  curl -s "http://127.0.0.1:$PORT/ix/stats" > "$OUT/stats_c$C.json"
  curl -s "http://127.0.0.1:$PORT/ix/power?t0=$T0&t1=$T1" > "$OUT/power_c$C.json"
  python3 - "$OUT" $C <<'PY'
import json, sys
out, c = sys.argv[1], sys.argv[2]
try:
    r = json.load(open(f"{out}/ix_c{c}.json"))
    st = json.load(open(f"{out}/stats_c{c}.json"))
    print(f"  conc {c}: completed {r.get('completed')} ok, out {r.get('output_throughput', 0):.1f} tok/s, "
          f"median TTFT {r.get('median_ttft_ms', 0):.0f} ms, median TPOT {r.get('median_tpot_ms', 0):.2f} ms, "
          f"fallbacks {st.get('cpu_fallbacks')}, cuts {st.get('prompt_cuts')}")
except Exception as e:
    print(f"  conc {c}: no result ({e})")
PY
  echo "  client rc=$RC"
  kill $SP; wait $SP 2>/dev/null
done
echo "IX_RUN_DONE $B $M $S $(date '+%T')"
