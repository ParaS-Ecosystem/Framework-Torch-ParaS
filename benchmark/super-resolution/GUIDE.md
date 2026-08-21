# Super-resolution benchmark

Ten super-resolution models measured three ways on the same machine: PyTorch
eager, PyTorch with `torch.compile`, and Torch-ParaS.

Torch-ParaS is an out-of-tree PyTorch backend for the ParaS compiler. It
registers a `paras` device so existing models run on the ParaS runtime without
code changes, and it ships a whole-graph optimizer, `torch_paras.autotune`,
which tunes a model's execution configuration for the machine it runs on.

`autotune` is what this benchmark measures. For each model it searches a small
space of execution configurations — fp32 or reduced precision, NCHW or
channels-last, `torch.compile` on or off, CUDA-graph capture on or off — runs
each one, and keeps the fastest that still agrees with the eager reference
above a PSNR floor. The chosen configuration is recorded per model in
`results.json`, so every number can be traced back to what actually ran. The
gain comes from that per-model search: the same model can want a different
precision, memory format and capture strategy on a V100 than on an H200 or an
MI300X, and a fixed recipe leaves that on the table.

## 1. Install

Either method works; the benchmark imports `torch_paras` and does not care how
it got there.

### Method A — the wheel

Download:
**[torch_paras-2026.8.0.2.0-py3-none-linux_x86_64.whl](https://github.com/ParaS-Ecosystem/Framework-Torch-ParaS/releases/download/v2026.8.0.2.0/torch_paras-2026.8.0.2.0-py3-none-linux_x86_64.whl)**
(101 MB, from [Releases](https://github.com/ParaS-Ecosystem/Framework-Torch-ParaS/releases)).
The wheel is a release asset, not on PyPI, so it installs by path.

**1. Download it.**

```bash
curl -LO https://github.com/ParaS-Ecosystem/Framework-Torch-ParaS/releases/download/v2026.8.0.2.0/torch_paras-2026.8.0.2.0-py3-none-linux_x86_64.whl
```

**2. Check it arrived intact.**

```bash
echo "462910e466fdd01d7ea34718a44affc87aaf8531838a95e275b91161d4ce3986  torch_paras-2026.8.0.2.0-py3-none-linux_x86_64.whl" | sha256sum -c
```

Expect `OK`. Anything else means a truncated or wrong file — download it again.

**3. Make a clean environment.** Python 3.10 for V100 and CPU, 3.11 for H200,
3.12 for MI300X. Other versions are rejected with a diagnostic rather than
silently loading the wrong plugin.

```bash
python3.10 -m venv .venv && source .venv/bin/activate
```

**4. Install it.** This pulls `torch==2.5.1` for you. On an AMD machine, do
[On an AMD machine](#on-an-amd-machine) *first*.

```bash
pip install torch_paras-2026.8.0.2.0-py3-none-linux_x86_64.whl
```

**5. Add what the benchmark itself needs.**

```bash
pip install "numpy>=1.24" "Pillow>=9.0"
```

The wheel carries the ParaS compiler, its runtime and prebuilt backend plugins,
so there is no `PARAS_HOME` to set. Linux x86_64 only.

### Method B — build from source

Build the ParaS compiler and the backend per [docs/INSTALL.md](../../docs/INSTALL.md),
then:

```bash
source scripts/env.sh
scripts/build.sh cuda          # or: hip, cpu
pip install "numpy>=1.24" "Pillow>=9.0"
```

Take this path when you are changing the compiler or the backend, when your GPU
is not one of the prebuilt targets, or when policy forbids a binary wheel.

### On an AMD machine

Install the ROCm build of PyTorch **first**, or the CPU plugin is selected:

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/rocm6.2
```

### Check

```bash
python -c "import torch_paras; print(torch_paras.is_available(), torch_paras.device_count())"
```

Expect `True` and a device count of at least 1. For a fuller report:

```bash
python -m torch_paras
```

`extension.loaded` must be `true` and `optimizer_available` must be `true` —
those two decide whether the benchmark can run. `compiler.status` is `ok` when
`parascc` is on PATH or `PARAS_HOME` is set; the benchmark does not need it.

## 2. Datasets

Four datasets download directly. Each archive extracts to one `<Name>_HR/`
directory of PNGs.

```bash
mkdir -p datasets/extracted
for d in Set5 Set14 BSD100 Urban100; do
  curl -sL -o $d.tar.gz \
    "https://huggingface.co/datasets/eugenesiow/$d/resolve/main/data/${d}_HR.tar.gz"
  tar xzf $d.tar.gz -C datasets/extracted
done
```

Manga109 needs an academic agreement from <http://www.manga109.org/en/>.
Without it, run the four datasets above.

Two glob configs ship: `configs/datasets.huggingface.json` matches the layout
above, `configs/datasets.real-downloaded.json` matches `Set5/GTmod12/`,
`BSDS100/`, `urban100/`. They are plain JSON — edit either if your paths differ.

## 3. Run

```bash
python bench_comprehensive.py --gpu 0 \
  --datasets-root datasets/extracted \
  --dataset-config configs/datasets.huggingface.json \
  --datasets Set5,Set14,BSD100,Urban100 \
  --out results/myhost

python make_tables.py results/myhost --out RESULTS.md
```

With Manga109 present, drop `--datasets` for the full 50-cell matrix. With no
accelerator, use `--device cpu`. `--models srcnn,espcn --datasets Set5` cuts the
run to a few seconds while checking a setup.

## 4. The three arms

| arm | configuration |
|---|---|
| PyTorch eager | fp32 / NCHW, with no compile, no autocast, no graph, no channels-last and no TF32 |
| PyTorch + `torch.compile` | same precision and memory format Torch-ParaS chose, with `reduce-overhead` (PyTorch's CUDA-graph path) whenever Torch-ParaS took a graph |
| Torch-ParaS | `torch_paras.autotune` — autocast, channels-last, `torch.compile` and explicit CUDA-graph capture, selected per model after a numerical fidelity check |

Both optimized arms are compiled and graph-replayed, so the comparison between
them is like-for-like. All three run on the native `cuda`/`hip` device; the
`paras` device is exercised untimed, by the parity column.

Two ratios are reported: `speedup` against eager, which is deliberately
unoptimized and so mixes precision and compilation into the result, and
`speedup_vs_pytorch_optimized` against the `torch.compile` arm, which is the
like-for-like figure. `make_tables.py` also derives **vs PyTorch best**,
`min(eager, compiled)` per model, because tuning does not always help.

## 5. Output

`results/myhost/` holds `results.json` (`rows`, `model_summary`,
`model_evidence`, `environment`, `protocol`), `results.csv`,
`environment.json` and a `.sha256`.

| column | meaning |
|---|---|
| `pytorch_eager_ms` | eager arm |
| `pytorch_optimized_ms` | `torch.compile` arm |
| `torch_paras_optimized_ms` | Torch-ParaS arm |
| `speedup` | eager ÷ Torch-ParaS |
| `speedup_vs_pytorch_optimized` | compiled ÷ Torch-ParaS |
| `pytorch_optimized_status` | `compiled`, or why not |
| `fidelity_psnr_db` | agreement with the eager reference |
| `direct_paras_fp32_parity_max_abs` | the model run on the `paras` device, largest absolute difference from the eager reference — correctness, not latency |
| `*_peak_allocated_mib` | peak memory per arm |

Where `torch.compile` cannot run, `pytorch_optimized_status` records why and
those rows are marked; they are not a compile-versus-compile result.

## 6. Measurement notes

- Warm up first. The first model measured absorbs CUDA context and cuDNN handle
  creation into its own baseline, inflating that model's ratio.
- Quote medians over repeated processes; single runs vary by several percent.
- Check the machine is idle — `nvidia-smi`, `rocm-smi --showpids`.
- Models use deterministic random weights, so `fidelity_psnr_db` means
  agreement with the eager reference, not image quality against a trained
  checkpoint.
- One image per dataset is selected deterministically and its SHA-256 recorded
  in every row. A different distribution of the same dataset selects a
  different image and will not reproduce a published number exactly.
