#!/bin/bash
set -e

export PYTHONUNBUFFERED=1
source /home/amdgpu02/parikshit/torch_paras/paras-envset.sh
source /home/amdgpu02/parikshit/torch_paras_venv/bin/activate
cd /home/amdgpu02/parikshit/torch_paras/Framework-Torch-ParaS

echo "=============================================================================="
echo "  Starting Param2-17B Full Profiling on AMD MI300X ($(date))"
echo "=============================================================================="

# 1. Run Torch-ParaS on MI300X (paras:1)
echo -e "\n>>> [1/3] Benchmarking Torch-ParaS (paras:1)..."
python -u small_param2_benchmark_cuda_paras.py --backend paras

# 2. Run Native PyTorch ROCm Baseline (cuda:0)
echo -e "\n>>> [2/3] Benchmarking Native PyTorch ROCm Baseline (cuda:0)..."
python -u small_param2_benchmark_cuda_paras.py --backend cuda

# 3. Compare Results & Generate Report
echo -e "\n>>> [3/3] Generating Comparison & Fidelity Report..."
python -u compare_param2_results.py

echo -e "\n=============================================================================="
echo "  Param2 MI300X Profiling Complete ($(date))"
echo "  Report: PARAM2_H200_BENCHMARK_REPORT.md / param2_comparison_report.json"
echo "=============================================================================="
