# Reproducing the Param2-17B and InferenceX results

This directory has everything needed to rebuild the Torch-ParaS stack from public sources, run the
Param2-17B benchmarks and the InferenceX serving benchmark on native PyTorch and on Torch-ParaS, and check
the numbers against the recorded results in `results/`. Each check prints PASS or FAIL per metric.

## What you need

| | AMD | NVIDIA |
|---|---|---|
| GPU | one MI300X (192 GB), otherwise idle | one GH200 (96 GB HBM), otherwise idle |
| Driver stack | ROCm 7.x (`/opt/rocm`, set `PTSYCL_ROCM_HOME` otherwise) | CUDA 12.x toolkit (`/usr/local/cuda`, set `PTSYCL_CUDA_HOME` otherwise) |
| Host | Linux x86_64 | Linux aarch64 |

Also on both: GCC 12 or newer (its libstdc++; point `PTSYCL_GCC_TOOLCHAIN` at it if it is not `/usr`), Python 3.12, git, curl, make, about 80 GB of free disk space
(45 GB without the model, see `REPRO_HF_HOME` below), and access to github.com, huggingface.co and download.pytorch.org.
Any other GPU builds and runs, but has no recorded results to compare with: its numbers are printed without a verdict.
The model needs 35 GB of GPU memory in bf16, so 32 GB GPUs (V100) cannot run the Param2 benchmarks.

## Steps

```bash
git clone https://github.com/ParaS-Ecosystem/Framework-Torch-ParaS.git
cd Framework-Torch-ParaS
export REPRO_WORK=$PWD/../torch-paras-repro     # any directory with ~80 GB free

benchmarks/setup_stack.sh                       # hip or cuda is detected; 25 to 60 minutes
benchmarks/reproduce_param2.sh                  # Param2-17B, five configurations + perplexity
benchmarks/reproduce_inferencex.sh              # InferenceX serving, three configurations
```

The environment also holds the Torch-ParaS test suite's requirements, so `tests/run_all.py` of the built source tree
(`$REPRO_WORK/src/Framework-Torch-ParaS`) runs with `$PTSYCL_PYTHON` after `source $REPRO_WORK/env.sh`.

`setup_stack.sh` needs no GPU, so it can run on a build node; the two `reproduce_*.sh` scripts must run on
the GPU machine with the same `REPRO_WORK` path. Options of `setup_stack.sh`:
- `REPRO_HF_HOME`: a Hugging Face cache that already holds the model, to avoid a second 35 GB download. Every model file is checked against the pinned revision either way: SHA-256 for the weights, the Git blob hash for the rest.
- `REPRO_JOBS`: the number of parallel build jobs (default: all cores).
- `LLVM_DIR`: an existing LLVM/Clang 20.1 install (with lld), instead of building it from source.
- `REPRO_PYTHON`: the Python 3.12 interpreter, if it is not `python3.12` on the `PATH`.
- `PTSYCL_GCC_TOOLCHAIN`: the GCC installation whose libstdc++ the builds use (default `/usr`; GCC 12 or newer). It must be one that Clang recognises: a distribution GCC or a `gcc-toolset` (for example `/opt/rh/gcc-toolset-13/root/usr`), not a conda compiler package.
- `PTSYCL_HIP_ARCH` / `PTSYCL_CUDA_ARCH`: the GPU target, if it cannot be detected (default `hip:gfx942` on a build node without a GPU). Each `reproduce_*.sh` ends with a table and exits with status 1
if any metric fails. Results, logs and the tables (`compare_*.md`, `compare_*.json`) are written to
`$REPRO_WORK/results/<name>_<date>/`.

### What `setup_stack.sh` builds

| Component | Version | Source |
|---|---|---|
| LLVM / Clang / lld | 20.1.0, built from the official source release (checksum verified) | github.com/llvm/llvm-project releases |
| ParaS-Compiler | `bd84ff8`, source tree `f3f0e7b` | ParaS-Ecosystem/ParaS-Compiler, pull request 44 |
| CLAP-Library | `13793a4`, source tree `228ef9c` | ParaS-Ecosystem/CLAP-Library, pull request 24 |
| Framework-Torch-ParaS | `f99a8c5`, source tree `4d03e22` | ParaS-Ecosystem/Framework-Torch-ParaS, pull request 35 (stacked on 30 to 34) |
| PyTorch | 2.10.0 for ROCm 7.0, or for CUDA 12.8 | download.pytorch.org |
| transformers | 4.55.2 (and the other pins in `setup_stack.sh`) | PyPI |
| Param2-17B | `bharatgenai/Param2-17B-A2.4B-Thinking`, revision `65373af52e60` | Hugging Face |
| WikiText-2 | `Salesforce/wikitext`, `wikitext-2-raw-v1` test split, revision `b08601e04326` | Hugging Face |
| InferenceX | `4ee4531` (serving client `inferencex-e2e/infx/bench_serving`, unchanged) | SemiAnalysisAI/InferenceX |

The source-tree hashes are checked, so a stack built here is byte-for-byte the code that was measured.
Once the pull requests are merged, the same trees are on `main`.

## What is measured

**Param2-17B** (`param2/infer.py`, bf16, one process per configuration):
- prefill of 2,048 and 4,096 tokens, batch 1: median of 7 timed calls;
- decode from 128 prompt tokens, 128 new tokens, batch 1 and 16: median time per output token;
- every interval starts and ends with a device synchronisation; greedy decoding with a KV cache;
- inputs are real WikiText-2 token ids;
- the hash of the greedy tokens is recorded, to compare outputs exactly.

The configurations:
- **native eager** and **native compiled**: PyTorch on ROCm or CUDA, the second with `torch.compile`;
- **paras eager**: Torch-ParaS;
- **paras fused**: Torch-ParaS with fused MoE experts;
- **paras compiled**: Torch-ParaS with fused MoE experts and `torch.compile(backend="paras")`.

`REPRO_FULL=1` runs the complete sweep: all prefill lengths, batch sizes 1 to 16, and decode by context length.
`param2/ppl.py` measures perplexity over the whole WikiText-2 test set (353,839 tokens, 2,048-token windows).

**InferenceX** (`inferencex/run_ix.sh`):
- `inferencex/ix_server.py` serves Param2-17B through an OpenAI-compatible streaming endpoint on either stack: static batching, the same code for both.
- InferenceX's own client is run unchanged, with the arguments of its fixed-sequence sweep:
  - random dataset, range ratio 0.8;
  - 10 × concurrency prompts, 2 × concurrency warm-ups;
  - `--ignore-eos`;
  - percentiles of TTFT, TPOT, ITL and E2EL.
- GPU energy comes from the GPU's own power sensor, sampled every 50 ms during the client's window.
- The default is 1k/1k at concurrency 1 and 16. `IX_CONCS="1 4 8 16 32 64"` runs InferenceX's full 1k/1k sweep, and `IX_SCENARIO=3k1k` the 3k/1k one. `IX_CONFIGS=none REPRO_MICRO=1` runs only the micro-benchmarks.

**Micro-benchmarks** (`REPRO_MICRO=1`, `micro/run_micro.sh`):
- **GEMM:** InferenceX's ubenchX GEMM benchmark, run unchanged on native and through a 15-line device shim on Torch-ParaS. It covers Param2's own matrix shapes and all 1,635 bf16 shapes of InferenceX's OperatorX testlist.
- **Copy bandwidth:** ubenchX's method with device events, on both stacks.
- **A known gap:** a few OperatorX shapes can stop the Torch-ParaS process with a GPU memory fault in the device GEMM path, at a different shape from run to run. The sweep records such a shape as `status=crash` and goes on (`micro/gemm_resume.py`). The geometric mean covers the shapes measured on both stacks; the recorded run had 2 crashed shapes out of 1,635.

## Expected results

The comparison uses the recorded run made with exactly this stack, as listed in `results/param2_recorded.json`,
`results/inferencex_recorded.json` and `results/micro_recorded.json`.

A metric passes within ±15% (`REPRO_TOL`), or ±0.5% for perplexity. Why ±15%:
- Two of our own campaigns on the same machines differed by up to 13% on MI300X and 17% on GH200, where other users' jobs shared the node.
- Decode at batch 1 is bound by the host CPU (one small step per token), so it moves most with CPU load. In our independent check, where a second benchmark shared the node's CPUs, decode b1 moved by up to 12%; prefill and batch-16 decode stayed within 9%.

Greedy-token hashes are shown, not judged. Native eager gave identical tokens in every run, including the independent check. For the ParaS configurations and native compiled, the hashes matched in some runs and not in others: a greedy choice between two nearly equal bf16 logits flips if the last bits of a sum change. The perplexity over all of WikiText-2 is the accuracy check. It reproduced to 0.1%.

Param2-17B, ms (lower is better):

| Workload | native eager | paras eager | paras fused | native compiled | paras compiled |
|---|---|---|---|---|---|
| MI300X decode b1 | 41.44 | 30.11 | 17.32 | 45.86 | 9.90 |
| MI300X decode b16 | 118.85 | 66.04 | 21.01 | 149.78 | 17.24 |
| MI300X prefill 2048 | 163.84 | 137.14 | 102.52 | 195.64 | 96.70 |
| MI300X prefill 4096 | 180.93 | 158.55 | 188.26 | 237.75 | 180.52 |
| GH200 decode b1 | 53.82 | 44.12 | 31.22 | 61.05 | 17.10 |
| GH200 decode b16 | 167.06 | 127.63 | 38.11 | 196.82 | 21.68 |
| GH200 prefill 2048 | 218.48 | 255.56 | 171.35 | 258.09 | 163.45 |
| GH200 prefill 4096 | 225.75 | 317.67 | 221.04 | 298.14 | 261.48 |

Perplexity on WikiText-2, MI300X, bf16: native 17.5206, Torch-ParaS 17.5452.

InferenceX 1k/1k, output tokens/s and median TPOT (ms):

| Machine, concurrency | native eager | native compiled | ParaS fused | ParaS compiled |
|---|---|---|---|---|
| MI300X, 1 | 24.2 / 40.71 | 21.9 / 44.96 | 49.0 / 20.42 | 71.6 / 11.55 |
| MI300X, 16 | 149.2 / 95.52 | 116.8 / 124.24 | 540.4 / 26.09 | 654.7 / 19.51 |
| GH200, 1 | 17.9 / 55.72 | 15.3 / 65.18 | 33.6 / 29.60 | 51.9 / 16.59 |
| GH200, 16 | 86.4 / 157.03 | 70.7 / 194.68 | 370.5 / 36.09 | 613.9 / 21.63 |

The full recorded sweeps (MI300X 1k/1k and 3k/1k at 1 to 64 users, GH200 1k/1k at 1 to 64 users) are in
`results/inferencex_recorded.json`. The comparison also checks the best ParaS configuration against the best
native one at each concurrency, as an output-tokens/s ratio.

## Running pieces by hand

```bash
source $REPRO_WORK/env.sh
export BENCH_OUT=/tmp/my_run
python benchmarks/param2/infer.py paras compiled bfloat16      # BENCH_SHAPES=repro for the compared shapes only
python benchmarks/param2/ppl.py native bfloat16
python benchmarks/compare.py param2 /tmp/my_run
benchmarks/inferencex/run_ix.sh paras fused 1k1k 8600 /tmp/my_ix && python benchmarks/compare.py inferencex /tmp/my_ix
```

`param2/train.py` (training steps: per-phase step time, loss, memory) and `param2/serve.py` (open-loop serving
with SLOs) are the other two workloads of the same campaign. They have no recorded comparison here.

## Files

| Path | What it is |
|---|---|
| `setup_stack.sh` | toolchain, sources, Python environments, builds, model and data |
| `reproduce_param2.sh`, `reproduce_inferencex.sh` | one command each: run and compare |
| `compare.py` | PASS / FAIL against `results/` |
| `common/harness.py` | timing, statistics, power sampling, result rows, model and data loading |
| `param2/` | `infer.py`, `ppl.py`, `train.py`, `serve.py` |
| `inferencex/` | `ix_server.py` (serving endpoint), `run_ix.sh` (client sweep) |
| `micro/` | GEMM shim and shapes, copy bandwidth, host/device transfers |
| `results/` | the recorded results the comparison uses |
