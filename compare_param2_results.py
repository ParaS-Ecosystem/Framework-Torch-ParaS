#!/usr/bin/env python3

"""
Param2 Benchmark Comparer & Report Generator

Reads:
  - param2_paras_results.json
  - param2_paras_reference_logits.pt
  - param2_cuda_results.json
  - param2_cuda_reference_logits.pt

Generates:
  - param2_comparison_report.json
  - PARAM2_H200_BENCHMARK_REPORT.md
"""

import os
import json
import torch
import torch.nn.functional as F
import datetime


FIDELITY_THRESHOLDS = {
    "max_abs_error": 1.0,
    "cosine_similarity": 0.99,
    "top1_agreement_pct": 95.0,
    "kl_divergence": 0.05,
}


def compute_numerical_fidelity(
    logits_paras,
    logits_cuda,
):
    a = logits_paras.float().flatten()
    b = logits_cuda.float().flatten()

    max_abs_error = (
        (a - b).abs().max().item()
    )

    mae = (
        (a - b).abs().mean().item()
    )

    cosine_sim = (
        F.cosine_similarity(
            a.unsqueeze(0),
            b.unsqueeze(0),
        ).item()
    )

    top1_a = logits_paras.argmax(
        dim=-1
    )

    top1_b = logits_cuda.argmax(
        dim=-1
    )

    top1_agree = (
        (top1_a == top1_b)
        .float()
        .mean()
        .item()
        * 100
    )

    top5_a = (
        logits_paras
        .topk(5, dim=-1)
        .indices
        .reshape(-1, 5)
    )

    top5_b = (
        logits_cuda
        .topk(5, dim=-1)
        .indices
        .reshape(-1, 5)
    )

    matches = sum(
        1
        for i in range(top5_a.shape[0])
        if len(
            set(
                top5_a[i].tolist()
            )
            & set(
                top5_b[i].tolist()
            )
        ) > 0
    )

    top5_agree = (
        matches
        / max(top5_a.shape[0], 1)
        * 100
    )

    log_p = F.log_softmax(
        logits_paras.float(),
        dim=-1,
    )

    q = F.softmax(
        logits_cuda.float(),
        dim=-1,
    )

    kl = F.kl_div(
        log_p,
        q,
        reduction="batchmean",
    ).item()

    return {
        "max_abs_error": max_abs_error,
        "mae": mae,
        "cosine_similarity": cosine_sim,
        "top1_agreement_pct": top1_agree,
        "top5_agreement_pct": top5_agree,
        "kl_divergence": kl,
    }


def main():

    required_files = [
        "param2_paras_results.json",
        "param2_cuda_results.json",
        "param2_paras_reference_logits.pt",
        "param2_cuda_reference_logits.pt",
    ]

    missing = [
        path
        for path in required_files
        if not os.path.exists(path)
    ]

    if missing:
        print(
            "[Error] Missing required files:"
        )

        for path in missing:
            print(f"  - {path}")

        print(
            "\nRun both backends first."
        )

        return

    with open(
        "param2_paras_results.json"
    ) as f:
        paras = json.load(f)

    with open(
        "param2_cuda_results.json"
    ) as f:
        cuda = json.load(f)

    logits_paras = torch.load(
        "param2_paras_reference_logits.pt",
        map_location="cpu",
    )["logits"]

    logits_cuda = torch.load(
        "param2_cuda_reference_logits.pt",
        map_location="cpu",
    )["logits"]

    fidelity = compute_numerical_fidelity(
        logits_paras,
        logits_cuda,
    )

    fidelity_passed = (
        fidelity["max_abs_error"]
        <= FIDELITY_THRESHOLDS[
            "max_abs_error"
        ]
        and
        fidelity["cosine_similarity"]
        >= FIDELITY_THRESHOLDS[
            "cosine_similarity"
        ]
        and
        fidelity["top1_agreement_pct"]
        >= FIDELITY_THRESHOLDS[
            "top1_agreement_pct"
        ]
        and
        fidelity["kl_divergence"]
        <= FIDELITY_THRESHOLDS[
            "kl_divergence"
        ]
    )

    overall_passed = (
        fidelity_passed
        and paras["stability"]["passed"]
        and cuda["stability"]["passed"]
        and paras["backward"]["nan_grads"] == 0
        and paras["backward"]["inf_grads"] == 0
        and cuda["backward"]["nan_grads"] == 0
        and cuda["backward"]["inf_grads"] == 0
    )

    md = []

    md.append(
        "# Param2-17B Benchmark Report: "
        "Torch-ParaS vs Native PyTorch CUDA (H200)"
    )

    md.append(
        f"\n**Timestamp:** "
        f"{datetime.datetime.now().isoformat()} | "
        f"**Hardware:** NVIDIA H200 NVL (141GB VRAM)"
    )

    md.append(
        f"**Overall Benchmark Status:** "
        f"**{'PASSED' if overall_passed else 'FAILED'}**\n"
    )

    md.append("---")

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    md.append(
        "## 1. Latency & Throughput: Prefill (TTFT)"
    )

    md.append(
        "| Seq Len | TTFT ParaS (ms) | "
        "TTFT CUDA (ms) | Speedup (CUDA/ParaS) | "
        "Throughput ParaS (tok/s) | "
        "Throughput CUDA (tok/s) |"
    )

    md.append(
        "| :---: | :---: | :---: | :---: | :---: | :---: |"
    )

    for p, c in zip(
        paras["prefill"],
        cuda["prefill"],
    ):

        speedup = (
            c["ttft_ms"]
            / max(p["ttft_ms"], 1e-6)
        )

        md.append(
            f"| {p['seq_len']} | "
            f"{p['ttft_ms']:.2f} | "
            f"{c['ttft_ms']:.2f} | "
            f"{speedup:.2f}x | "
            f"{p['prompt_throughput_tok_s']:.1f} | "
            f"{c['prompt_throughput_tok_s']:.1f} |"
        )

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    md.append("\n---")

    md.append(
        "## 2. Latency & Throughput: "
        "Autoregressive Decode (ITL)"
    )

    md.append(
        "| Token Budget | ITL ParaS (ms/tok) | "
        "ITL CUDA (ms/tok) | Speedup (CUDA/ParaS) | "
        "Throughput ParaS (tok/s) | "
        "Throughput CUDA (tok/s) |"
    )

    md.append(
        "| :---: | :---: | :---: | :---: | :---: | :---: |"
    )

    for p, c in zip(
        paras["decode"],
        cuda["decode"],
    ):

        speedup = (
            c["itl_ms"]
            / max(p["itl_ms"], 1e-6)
        )

        md.append(
            f"| {p['token_budget']} | "
            f"{p['itl_ms']:.2f} | "
            f"{c['itl_ms']:.2f} | "
            f"{speedup:.2f}x | "
            f"{p['decode_throughput_tok_s']:.2f} | "
            f"{c['decode_throughput_tok_s']:.2f} |"
        )

    # ------------------------------------------------------------------
    # Fidelity
    # ------------------------------------------------------------------

    md.append("\n---")

    md.append(
        "## 3. Numerical Fidelity & Parity "
        "(Torch-ParaS vs Native CUDA)"
    )

    md.append(
        "| Metric | Value | Threshold | Status |"
    )

    md.append(
        "| :--- | :---: | :---: | :---: |"
    )

    md.append(
        f"| **Max Absolute Error** | "
        f"{fidelity['max_abs_error']:.6f} | "
        f"<= {FIDELITY_THRESHOLDS['max_abs_error']} | "
        f"{'PASS' if fidelity['max_abs_error'] <= FIDELITY_THRESHOLDS['max_abs_error'] else 'FAIL'} |"
    )

    md.append(
        f"| **Mean Absolute Error (MAE)** | "
        f"{fidelity['mae']:.6f} | - | PASS |"
    )

    md.append(
        f"| **Cosine Similarity** | "
        f"{fidelity['cosine_similarity']:.6f} | "
        f">= {FIDELITY_THRESHOLDS['cosine_similarity']} | "
        f"{'PASS' if fidelity['cosine_similarity'] >= FIDELITY_THRESHOLDS['cosine_similarity'] else 'FAIL'} |"
    )

    md.append(
        f"| **Top-1 Token Agreement** | "
        f"{fidelity['top1_agreement_pct']:.2f}% | "
        f">= {FIDELITY_THRESHOLDS['top1_agreement_pct']}% | "
        f"{'PASS' if fidelity['top1_agreement_pct'] >= FIDELITY_THRESHOLDS['top1_agreement_pct'] else 'FAIL'} |"
    )

    md.append(
        f"| **Top-5 Token Agreement** | "
        f"{fidelity['top5_agreement_pct']:.2f}% | "
        f"- | PASS |"
    )

    md.append(
        f"| **KL Divergence** | "
        f"{fidelity['kl_divergence']:.6f} | "
        f"<= {FIDELITY_THRESHOLDS['kl_divergence']} | "
        f"{'PASS' if fidelity['kl_divergence'] <= FIDELITY_THRESHOLDS['kl_divergence'] else 'FAIL'} |"
    )

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    md.append("\n---")

    md.append(
        "## 4. Memory Footprint & Resource Utilization"
    )

    md.append(
        "| Metric | Torch-ParaS | Native CUDA |"
    )

    md.append(
        "| :--- | :---: | :---: |"
    )

    for label, key in [
        (
            "Base Allocated (MB)",
            "base_allocated_mb",
        ),
        (
            "Base Reserved (MB)",
            "base_reserved_mb",
        ),
        (
            "Peak Allocated (MB)",
            "peak_allocated_mb",
        ),
        (
            "Peak Reserved (MB)",
            "peak_reserved_mb",
        ),
        (
            "KV Cache Growth (MB/tok)",
            "kv_cache_growth_mb_per_token",
        ),
    ]:

        pv = paras["memory"].get(key)
        cv = cuda["memory"].get(key)

        pv_str = (
            f"{pv:.2f}"
            if pv is not None
            else "N/A"
        )

        cv_str = (
            f"{cv:.2f}"
            if cv is not None
            else "N/A"
        )

        md.append(
            f"| {label} | "
            f"{pv_str} | "
            f"{cv_str} |"
        )

    # ------------------------------------------------------------------
    # Backward + Stability
    # ------------------------------------------------------------------

    md.append("\n---")

    md.append(
        "## 5. Backward Pass Verification & Stability"
    )

    md.append(
        "| Backend | Valid Gradients | "
        "NaN Grads | Inf Grads | "
        "Stability Status |"
    )

    md.append(
        "| :--- | :---: | :---: | :---: | :---: |"
    )

    md.append(
        f"| **Torch-ParaS** | "
        f"{paras['backward']['grad_count']}/"
        f"{paras['backward']['total_params']} | "
        f"{paras['backward']['nan_grads']} | "
        f"{paras['backward']['inf_grads']} | "
        f"{'PASSED' if paras['stability']['passed'] else 'FAILED'} |"
    )

    md.append(
        f"| **Native CUDA** | "
        f"{cuda['backward']['grad_count']}/"
        f"{cuda['backward']['total_params']} | "
        f"{cuda['backward']['nan_grads']} | "
        f"{cuda['backward']['inf_grads']} | "
        f"{'PASSED' if cuda['stability']['passed'] else 'FAILED'} |"
    )

    out_md = (
        "PARAM2_H200_BENCHMARK_REPORT.md"
    )

    with open(
        out_md,
        "w",
    ) as f:
        f.write(
            "\n".join(md) + "\n"
        )

    print(
        f"\n[Done] Benchmark Report generated: "
        f"{out_md}"
    )

    out_json = (
        "param2_comparison_report.json"
    )

    with open(
        out_json,
        "w",
    ) as f:
        json.dump(
            {
                "paras": paras,
                "cuda": cuda,
                "fidelity": fidelity,
                "overall_passed": overall_passed,
            },
            f,
            indent=2,
        )

    print(
        f"[Done] Comparison JSON generated: "
        f"{out_json}"
    )


if __name__ == "__main__":
    main()
