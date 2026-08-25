"""
Param2 Single-Backend Benchmark: Torch-ParaS or Native PyTorch CUDA
Process-Isolated Benchmark for PrivateUse1 vs CUDA comparison.
"""

import sys
import os
import gc
import time
import json
import argparse

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.insert(
    0,
    "/home/amdgpu02/parikshit/torch_paras/Framework-Torch-ParaS/python"
)

MODEL_PATH = "/home/amdgpu02/parikshit/models/Param2-17B-A2.4B-Thinking"
EXPECTED_VOCAB_SIZE = 128008

import torch
import torch.nn.functional as F
import torch_paras
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

QUICK_MODE = False

if QUICK_MODE:
    PREFILL_SEQ_LENGTHS = [16, 64]
    DECODE_TOKEN_BUDGETS = [8, 16]
    WARMUP_ITERS = 1
    MEASURE_ITERS = 2
    STABILITY_TURNS = 1
    STABILITY_BUDGET = 16
else:
    PREFILL_SEQ_LENGTHS = [16, 64, 128, 256, 512]
    DECODE_TOKEN_BUDGETS = [16, 32, 64, 128]
    WARMUP_ITERS = 2
    MEASURE_ITERS = 5
    STABILITY_TURNS = 3
    STABILITY_BUDGET = 64


STABILITY_PROMPTS = [
    "What is the capital of France?",
    "What is its approximate population?",
    "Name one famous landmark there.",
]

SEED = 42


BACKENDS = {
    "paras": {
        "key": "paras",
        "name": "Torch-ParaS",
        "device_str": "paras:1",
    },
    "cuda": {
        "key": "cuda",
        "name": "Native PyTorch CUDA",
        "device_str": "cuda:0",
    },
}


# ---------------------------------------------------------------------------
# DynamicLayer Compatibility Patch
# ---------------------------------------------------------------------------

def apply_dynamic_layer_patch():
    try:
        import transformers.cache_utils as cache_utils

        DynamicLayer = getattr(cache_utils, "DynamicLayer", None)

        if DynamicLayer is not None:

            def _safe_dynamic_layer_update(
                self,
                key_states,
                value_states,
                *args,
                **kwargs,
            ):
                if (
                    self.keys is None
                    or self.keys.numel() == 0
                    or self.keys.ndim != key_states.ndim
                ):
                    self.keys = key_states
                    self.values = value_states
                else:
                    self.keys = torch.cat(
                        [self.keys, key_states],
                        dim=-2,
                    )
                    self.values = torch.cat(
                        [self.values, value_states],
                        dim=-2,
                    )

                return self.keys, self.values

            DynamicLayer.update = _safe_dynamic_layer_update

            print("[Patch] Transformers DynamicLayer.update patched.")

    except Exception as e:
        print(f"[Patch] DynamicLayer patch notice: {e}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(device):
    if device.type == "cuda":
        torch.cuda.set_device(device)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    model = model.to(device)

    return tokenizer, model


def free_model(model, device):
    del model
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        return

    empty_cache_fn = getattr(
        torch_paras,
        "empty_cache",
        None,
    )

    if empty_cache_fn is not None:
        empty_cache_fn()


def make_sync_fn(device):
    if device.type == "cuda":

        def _sync():
            torch.cuda.synchronize(device)

        return _sync

    def _sync():
        try:
            torch_paras.synchronize(device)
        except TypeError:
            torch_paras.synchronize()

    return _sync


def reset_memory_stats(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        return

    reset_fn = getattr(
        torch_paras,
        "reset_peak_memory_stats",
        None,
    )

    if reset_fn is not None:
        try:
            reset_fn(device)
        except TypeError:
            reset_fn()


def get_memory_stats(device):
    if device.type == "cuda":
        return {
            "allocated_mb":
                torch.cuda.memory_allocated(device) / 1e6,
            "reserved_mb":
                torch.cuda.memory_reserved(device) / 1e6,
            "max_allocated_mb":
                torch.cuda.max_memory_allocated(device) / 1e6,
            "max_reserved_mb":
                torch.cuda.max_memory_reserved(device) / 1e6,
        }

    stats = {
        "allocated_mb": None,
        "reserved_mb": None,
        "max_allocated_mb": None,
        "max_reserved_mb": None,
    }

    fn_map = {
        "allocated_mb": "memory_allocated",
        "reserved_mb": "memory_reserved",
        "max_allocated_mb": "max_memory_allocated",
        "max_reserved_mb": "max_memory_reserved",
    }

    for key, fn_name in fn_map.items():
        fn = getattr(torch_paras, fn_name, None)

        if fn is None:
            continue

        try:
            value = fn(device)
        except TypeError:
            value = fn()

        stats[key] = value / 1e6

    return stats


def build_prompt_input(tokenizer, seq_len, device):
    filler = (
        "The quick brown fox jumps over the lazy dog "
        "and explores the neighboring forest. "
        * 40
    )

    encoded = tokenizer(
        filler,
        return_tensors="pt",
        truncation=True,
        max_length=seq_len,
    )

    input_ids = encoded["input_ids"]

    if input_ids.shape[1] < seq_len:
        pad_id = tokenizer.pad_token_id
        pad_len = seq_len - input_ids.shape[1]

        pad = torch.full(
            (1, pad_len),
            pad_id,
            dtype=input_ids.dtype,
        )

        input_ids = torch.cat(
            [input_ids, pad],
            dim=1,
        )

    attention_mask = torch.ones_like(input_ids)

    return (
        input_ids.to(device),
        attention_mask.to(device),
    )


def build_reference_input(tokenizer, device):
    messages = [
        {
            "role": "user",
            "content": "What is the capital of France?",
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)

    attention_mask = encoded.get("attention_mask")

    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    return input_ids, attention_mask


# ---------------------------------------------------------------------------
# Profiling Functions
# ---------------------------------------------------------------------------

def profile_prefill(
    model,
    tokenizer,
    device,
    sync_fn,
):
    results = []

    model.eval()

    for seq_len in PREFILL_SEQ_LENGTHS:

        print(
            f"  [prefill] starting seq_len={seq_len}...",
            flush=True,
        )

        input_ids, attention_mask = build_prompt_input(
            tokenizer,
            seq_len,
            device,
        )

        warmup_logits = None

        for w in range(WARMUP_ITERS):

            t0 = time.time()

            with torch.no_grad():
                warmup_logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits

            print(
                f"    warmup {w + 1}/{WARMUP_ITERS} "
                f"took {time.time() - t0:.2f}s",
                flush=True,
            )

        if EXPECTED_VOCAB_SIZE is not None:
            assert (
                warmup_logits.shape[-1]
                == EXPECTED_VOCAB_SIZE
            ), (
                f"Unexpected vocab size: "
                f"{warmup_logits.shape[-1]}"
            )

        assert not torch.isnan(warmup_logits).any(), (
            f"NaN logits at prefill S={seq_len}"
        )

        assert not torch.isinf(warmup_logits).any(), (
            f"Inf logits at prefill S={seq_len}"
        )

        times = []

        for _ in range(MEASURE_ITERS):

            sync_fn()

            t0 = time.perf_counter()

            with torch.no_grad():
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

            sync_fn()

            times.append(
                time.perf_counter() - t0
            )

        mean_time = sum(times) / len(times)

        std_time = (
            sum(
                (t - mean_time) ** 2
                for t in times
            )
            / len(times)
        ) ** 0.5

        results.append(
            {
                "seq_len": seq_len,
                "ttft_ms": mean_time * 1000,
                "ttft_std_ms": std_time * 1000,
                "prompt_throughput_tok_s":
                    seq_len / max(mean_time, 1e-6),
            }
        )

        print(
            f"  S={seq_len:4d} | "
            f"TTFT={mean_time * 1000:8.2f} ms "
            f"(+/-{std_time * 1000:.2f}) | "
            f"throughput="
            f"{seq_len / max(mean_time, 1e-6):9.1f} tok/s"
        )

    return results


def decode_run(
    model,
    input_ids,
    attention_mask,
    n_tokens,
    sync_fn,
    device,
    check_finite,
):
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )

    if check_finite:
        assert not torch.isnan(out.logits).any(), (
            "NaN logits in decode prefill step"
        )

        assert not torch.isinf(out.logits).any(), (
            "Inf logits in decode prefill step"
        )

    past = out.past_key_values

    cur_id = torch.argmax(
        out.logits[:, -1, :],
        dim=-1,
        keepdim=True,
    )

    cur_mask = torch.cat(
        [
            attention_mask,
            torch.ones_like(cur_id),
        ],
        dim=-1,
    )

    step_times = []

    for _ in range(n_tokens):

        sync_fn()

        t0 = time.perf_counter()

        with torch.no_grad():
            out = model(
                input_ids=cur_id,
                attention_mask=cur_mask,
                past_key_values=past,
                use_cache=True,
            )

        sync_fn()

        step_times.append(
            time.perf_counter() - t0
        )

        if check_finite:
            assert not torch.isnan(out.logits).any(), (
                "NaN logits during decode"
            )

            assert not torch.isinf(out.logits).any(), (
                "Inf logits during decode"
            )

        past = out.past_key_values

        cur_id = torch.argmax(
            out.logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

        cur_mask = torch.cat(
            [
                cur_mask,
                torch.ones_like(cur_id),
            ],
            dim=-1,
        )

    end_mem_mb = get_memory_stats(
        device
    )["allocated_mb"]

    return step_times, end_mem_mb


def profile_decode(
    model,
    tokenizer,
    device,
    sync_fn,
):
    results = []

    model.eval()

    input_ids, attention_mask = build_prompt_input(
        tokenizer,
        128,
        device,
    )

    for n in DECODE_TOKEN_BUDGETS:

        print(
            f"  [decode] starting token_budget={n}...",
            flush=True,
        )

        for w in range(WARMUP_ITERS):

            t0 = time.time()

            decode_run(
                model,
                input_ids,
                attention_mask,
                n,
                sync_fn,
                device,
                check_finite=True,
            )

            print(
                f"    warmup {w + 1}/{WARMUP_ITERS} "
                f"({n} tokens) took "
                f"{time.time() - t0:.2f}s",
                flush=True,
            )

        step_times, end_mem_mb = decode_run(
            model,
            input_ids,
            attention_mask,
            n,
            sync_fn,
            device,
            check_finite=False,
        )

        mean_step = (
            sum(step_times) / len(step_times)
        )

        throughput = (
            len(step_times)
            / max(sum(step_times), 1e-6)
        )

        results.append(
            {
                "token_budget": n,
                "itl_ms": mean_step * 1000,
                "decode_throughput_tok_s":
                    throughput,
                "allocated_at_budget_mb":
                    end_mem_mb,
            }
        )

        print(
            f"  N={n:4d} | "
            f"ITL={mean_step * 1000:7.2f} ms/tok | "
            f"throughput={throughput:7.2f} tok/s"
        )

    return results


def estimate_kv_cache_growth_mb_per_token(
    decode_results,
):
    valid = [
        r
        for r in decode_results
        if r["allocated_at_budget_mb"] is not None
    ]

    if len(valid) < 2:
        return None

    n = len(valid)

    xs = [
        r["token_budget"]
        for r in valid
    ]

    ys = [
        r["allocated_at_budget_mb"]
        for r in valid
    ]

    x_mean = sum(xs) / n
    y_mean = sum(ys) / n

    denom = sum(
        (x - x_mean) ** 2
        for x in xs
    )

    if denom == 0:
        return None

    numer = sum(
        (x - x_mean) * (y - y_mean)
        for x, y in zip(xs, ys)
    )

    return numer / denom


# ---------------------------------------------------------------------------
# Backward Verification
# ---------------------------------------------------------------------------

def run_backward_verification(
    model,
    tokenizer,
    device,
):
    model.train()

    if device.type == "cuda":
        torch.cuda.set_device(device)

    input_ids, attention_mask = build_reference_input(
        tokenizer,
        device,
    )

    labels = input_ids.clone()

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    loss = outputs.loss

    loss_val = loss.item()

    assert not torch.isnan(loss), (
        "Loss is NaN!"
    )

    assert not torch.isinf(loss), (
        "Loss is Inf!"
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    loss.backward()

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    total_params = 0
    grad_count = 0
    nan_grads = 0
    inf_grads = 0

    for _, param in model.named_parameters():

        total_params += 1

        if param.grad is not None:

            grad_count += 1

            if torch.isnan(param.grad).any():
                nan_grads += 1

            if torch.isinf(param.grad).any():
                inf_grads += 1

    model.zero_grad(set_to_none=True)

    assert grad_count > 0, (
        "No gradients were computed!"
    )

    assert nan_grads == 0, (
        f"Found {nan_grads} NaN gradients!"
    )

    assert inf_grads == 0, (
        f"Found {inf_grads} Inf gradients!"
    )

    return {
        "loss": loss_val,
        "total_params": total_params,
        "grad_count": grad_count,
        "nan_grads": nan_grads,
        "inf_grads": inf_grads,
    }


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------

def run_stability_test(
    model,
    tokenizer,
    device,
):
    model.eval()

    messages = []
    turn_results = []

    total_nan = 0
    total_inf = 0

    for i, user_msg in enumerate(
        STABILITY_PROMPTS[:STABILITY_TURNS]
    ):

        messages.append(
            {
                "role": "user",
                "content": user_msg,
            }
        )

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        encoded = tokenizer(
            prompt,
            return_tensors="pt",
        )

        input_ids = encoded[
            "input_ids"
        ].to(device)

        attention_mask = encoded.get(
            "attention_mask"
        )

        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        torch.manual_seed(
            SEED + i
        )

        t0 = time.time()

        with torch.no_grad():

            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=STABILITY_BUDGET,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        gen_time = (
            time.time() - t0
        )

        reply_ids = output_ids[
            0,
            input_ids.shape[1]:
        ]

        reply_text = tokenizer.decode(
            reply_ids,
            skip_special_tokens=True,
        )

        messages.append(
            {
                "role": "assistant",
                "content": reply_text,
            }
        )

        with torch.no_grad():
            hidden_out = model(
                input_ids=output_ids,
                output_hidden_states=True,
            )

        nan_layers = sum(
            int(torch.isnan(h).any().item())
            for h in hidden_out.hidden_states
        )

        inf_layers = sum(
            int(torch.isinf(h).any().item())
            for h in hidden_out.hidden_states
        )

        total_nan += nan_layers
        total_inf += inf_layers

        gen_speed = (
            len(reply_ids)
            / max(gen_time, 1e-5)
        )

        turn_results.append(
            {
                "turn": i + 1,
                "prompt": user_msg,
                "generated_tokens":
                    int(reply_ids.shape[0]),
                "latency_sec":
                    round(gen_time, 2),
                "speed_tok_s":
                    round(gen_speed, 2),
                "nan_layers": nan_layers,
                "inf_layers": inf_layers,
            }
        )

        print(
            f"  Turn {i + 1}: "
            f"generated {reply_ids.shape[0]} tokens "
            f"in {gen_time:.2f}s "
            f"({gen_speed:.2f} tok/s) | "
            f"NaN layers={nan_layers} | "
            f"Inf layers={inf_layers}",
            flush=True,
        )

    passed = (
        total_nan == 0
        and total_inf == 0
    )

    assert passed, (
        f"Stability test failed: "
        f"{total_nan} NaN layers, "
        f"{total_inf} Inf layers detected"
    )

    return {
        "turns": turn_results,
        "total_nan_layers": total_nan,
        "total_inf_layers": total_inf,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Single Backend Runner
# ---------------------------------------------------------------------------

def run_backend(backend):

    key = backend["key"]
    name = backend["name"]
    device_str = backend["device_str"]

    device = torch.device(device_str)

    print("\n" + "=" * 88)
    print(
        f"  BACKEND: {name} ({device_str})"
    )
    print("=" * 88)

    print(
        "\n[Load] Loading tokenizer and model..."
    )

    t0 = time.time()

    tokenizer, model = load_model(
        device
    )

    load_time = time.time() - t0

    print(
        f"Model ready on {device_str} "
        f"in {load_time:.1f}s"
    )

    sync_fn = make_sync_fn(device)

    reset_memory_stats(device)

    base_mem = get_memory_stats(
        device
    )

    print(
        "\n[Latency/Throughput] "
        "Profiling prefill (TTFT)..."
    )

    prefill_results = profile_prefill(
        model,
        tokenizer,
        device,
        sync_fn,
    )

    print(
        "\n[Latency/Throughput] "
        "Profiling decode (ITL)..."
    )

    decode_results = profile_decode(
        model,
        tokenizer,
        device,
        sync_fn,
    )

    peak_mem = get_memory_stats(
        device
    )

    kv_growth_mb_per_token = (
        estimate_kv_cache_growth_mb_per_token(
            decode_results
        )
    )

    print(
        "\n[Correctness] "
        "Capturing reference logits..."
    )

    ref_input_ids, ref_attention_mask = (
        build_reference_input(
            tokenizer,
            device,
        )
    )

    with torch.no_grad():

        ref_logits = model(
            input_ids=ref_input_ids,
            attention_mask=ref_attention_mask,
        ).logits.float().cpu()

    logits_path = (
        f"param2_{key}_reference_logits.pt"
    )

    torch.save(
        {
            "logits": ref_logits,
        },
        logits_path,
    )

    print(
        f"[Export] Reference logits written "
        f"to {logits_path}"
    )

    print(
        "\n[Stability] "
        "Running multi-turn conversational test..."
    )

    stability_result = run_stability_test(
        model,
        tokenizer,
        device,
    )

    print(
        "\n[Correctness] "
        "Running backward-pass gradient verification..."
    )

    backward_result = (
        run_backward_verification(
            model,
            tokenizer,
            device,
        )
    )

    print(
        f"Gradients: "
        f"{backward_result['grad_count']}/"
        f"{backward_result['total_params']} valid | "
        f"NaN={backward_result['nan_grads']} | "
        f"Inf={backward_result['inf_grads']}"
    )

    results = {
        "name": name,
        "device": device_str,
        "load_time_s": load_time,
        "memory": {
            "base_allocated_mb":
                base_mem["allocated_mb"],
            "base_reserved_mb":
                base_mem["reserved_mb"],
            "peak_allocated_mb":
                peak_mem["max_allocated_mb"],
            "peak_reserved_mb":
                peak_mem["max_reserved_mb"],
            "kv_cache_growth_mb_per_token":
                kv_growth_mb_per_token,
        },
        "prefill": prefill_results,
        "decode": decode_results,
        "stability": stability_result,
        "backward": backward_result,
        "reference_logits_path": logits_path,
        "config": {
            "prefill_seq_lengths":
                PREFILL_SEQ_LENGTHS,
            "decode_token_budgets":
                DECODE_TOKEN_BUDGETS,
            "warmup_iters":
                WARMUP_ITERS,
            "measure_iters":
                MEASURE_ITERS,
            "stability_turns":
                STABILITY_TURNS,
            "stability_budget":
                STABILITY_BUDGET,
            "seed":
                SEED,
        },
    }

    result_path = (
        f"param2_{key}_results.json"
    )

    with open(
        result_path,
        "w",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print(
        f"[Export] Results written to "
        f"{result_path}"
    )

    print(
        f"\n[Cleanup] Freeing {name} model..."
    )

    free_model(
        model,
        device,
    )

    return results


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Param2 Single-Backend "
            "Isolated Benchmark"
        )
    )

    parser.add_argument(
        "--backend",
        choices=[
            "paras",
            "cuda",
        ],
        required=True,
        help=(
            "Backend to benchmark "
            "in this process."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    args = parse_args()

    torch.manual_seed(SEED)

    apply_dynamic_layer_patch()

    print("=" * 88)
    print(
        "  Param2 Isolated Single-Backend Benchmark"
    )
    print("=" * 88)

    print(
        f"PyTorch version : {torch.__version__}"
    )

    print(
        f"Transformers    : "
        f"{transformers.__version__}"
    )

    print(
        f"Model           : {MODEL_PATH}"
    )

    print(
        f"Backend Target  : {args.backend}"
    )

    backend = BACKENDS[
        args.backend
    ]

    run_backend(backend)


if __name__ == "__main__":
    main()
