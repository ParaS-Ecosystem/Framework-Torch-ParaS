import argparse
import gc
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ------------------------------------------------------------------
# Transformers 5.x compatibility for Param2 trust_remote_code.
# Param2's modeling_param2moe.py still imports is_torch_fx_available,
# which was removed from newer Transformers.
# ------------------------------------------------------------------
import transformers.utils.import_utils as _hf_import_utils

if not hasattr(_hf_import_utils, "is_torch_fx_available"):
    def _is_torch_fx_available():
        return True

    _hf_import_utils.is_torch_fx_available = _is_torch_fx_available


DEFAULT_MODEL = "/storage/parikshit/models/Param2-17B-A2.4B-Thinking"


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--backend", choices=["paras", "cuda"], default="paras")
    p.add_argument("--device", default=None)

    p.add_argument(
        "--prompt",
        default="Explain in one short sentence why GPUs are useful for AI.",
    )

    p.add_argument("--max-new-tokens", type=int, default=2)

    p.add_argument(
        "--mode",
        choices=["auto", "resident", "stream"],
        default="auto",
        help="resident=whole model on GPU, stream=one transformer layer at a time",
    )

    return p.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------
    # Backend
    # ------------------------------------------------------------

    if args.backend == "paras":
        import torch_paras

        device = torch.device(args.device or "paras:1")

        def sync():
            torch_paras.synchronize(device)

        # ParaS convention:
        # paras:1 -> CUDA GPU 0
        # paras:2 -> CUDA GPU 1 ...
        cuda_index = max((device.index or 1) - 1, 0)

    else:
        device = torch.device(args.device or "cuda:0")

        def sync():
            torch.cuda.synchronize(device)

        cuda_index = device.index or 0

    print("=" * 80)
    print("Param2 REAL-WEIGHTS End-to-End Test")
    print("=" * 80)
    print("model   :", args.model)
    print("backend :", args.backend)
    print("device  :", device)
    print("dtype   :", torch.bfloat16)

    gpu_name = "unknown"
    vram_gb = 0.0

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(cuda_index)
        gpu_name = props.name
        vram_gb = props.total_memory / (1024 ** 3)

    print("GPU     :", gpu_name)
    print(f"VRAM    : {vram_gb:.1f} GB")

    # ------------------------------------------------------------
    # Automatically choose execution strategy
    # ------------------------------------------------------------

    if args.mode == "auto":
        # 17B BF16 needs >34 GB just for weights.
        execution_mode = "resident" if vram_gb >= 40 else "stream"
    else:
        execution_mode = args.mode

    print("mode    :", execution_mode)

    if execution_mode == "stream":
        print()
        print(
            "Low-VRAM mode: real Param2 weights will remain in CPU RAM and "
            "transformer layers will be streamed through the GPU one at a time."
        )
        print(
            "This is a CORRECTNESS/INTEGRATION test, not a performance benchmark."
        )

    # ------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------

    print()
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
    )

    # ------------------------------------------------------------
    # Real model weights
    # ------------------------------------------------------------

    print("Loading REAL Param2 weights into CPU memory...")

    load_start = time.perf_counter()

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    model.eval()

    load_seconds = time.perf_counter() - load_start

    print(f"Weights loaded in {load_seconds:.2f} s")

    hook_handles = []

    # ------------------------------------------------------------
    # GPU setup
    # ------------------------------------------------------------

    if execution_mode == "resident":

        print("Moving complete model to", device, "...")

        model.to(device)
        sync()

    else:
        # --------------------------------------------------------
        # Layer streaming
        #
        # Keep transformer blocks on CPU.
        # Move everything else:
        #   embeddings
        #   RoPE
        #   final norm
        #   LM head
        # to GPU permanently.
        #
        # Before each block:
        #   CPU -> ParaS GPU
        #
        # After each block:
        #   synchronize
        #   ParaS GPU -> CPU
        # --------------------------------------------------------

        core = getattr(model, "model", None)

        if core is None or not hasattr(core, "layers"):
            raise RuntimeError(
                "Cannot locate model.model.layers in Param2 model."
            )

        layers = core.layers

        print("Transformer layers:", len(layers))

        # Temporarily remove large transformer layers.
        # This lets model.to(device) move only embeddings / norm /
        # rotary embeddings / lm_head / other small components.
        core.layers = torch.nn.ModuleList()

        print("Moving non-transformer components to", device, "...")

        model.to(device)
        sync()

        # Put CPU transformer layers back.
        core.layers = layers

        print("Installing per-layer streaming hooks...")

        for layer_idx, layer in enumerate(core.layers):

            # Explicitly ensure layer starts on CPU.
            layer.to("cpu")

            def pre_hook(module, inputs, idx=layer_idx):
                print(
                    f"\r[Param2] loading layer {idx + 1:02d}/{len(core.layers):02d}",
                    end="",
                    flush=True,
                )

                module.to(device)

                return None

            def post_hook(module, inputs, output, idx=layer_idx):
                # Make sure GPU has finished using this layer's weights
                # before copying them back to CPU.
                sync()

                module.to("cpu")

                return output

            hook_handles.append(
                layer.register_forward_pre_hook(pre_hook)
            )

            hook_handles.append(
                layer.register_forward_hook(post_hook)
            )

        gc.collect()

        print()
        print("Layer streaming ready.")

    # ------------------------------------------------------------
    # Input
    # ------------------------------------------------------------

    encoded = tokenizer(
        args.prompt,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)

    attention_mask = encoded.get("attention_mask")

    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    print()
    print("=" * 80)
    print("REAL PARAM2 PREFILL")
    print("=" * 80)
    print("Prompt:", args.prompt)
    print("Prompt tokens:", input_ids.shape[-1])

    # ------------------------------------------------------------
    # PREFILL
    # ------------------------------------------------------------

    sync()

    t0 = time.perf_counter()

    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )

    sync()

    prefill_ms = (time.perf_counter() - t0) * 1000.0

    print()

    if out.past_key_values is None:
        raise RuntimeError(
            "Prefill completed but Param2 did not return KV cache."
        )

    last_logits = out.logits[:, -1, :]

    logits_cpu = last_logits.float().cpu()

    if not torch.isfinite(logits_cpu).all():
        raise RuntimeError(
            "Non-finite values found in Param2 prefill logits."
        )

    next_token = int(
        torch.argmax(logits_cpu, dim=-1).item()
    )

    generated = [next_token]
    past = out.past_key_values

    print()
    print("PASS: real Param2 prefill")
    print(f"Prefill wall time: {prefill_ms:.3f} ms")
    print("KV cache type:", type(past).__name__)
    print("First generated token id:", next_token)

    # ------------------------------------------------------------
    # AUTOREGRESSIVE DECODE
    # ------------------------------------------------------------

    decode_times = []

    prompt_length = input_ids.shape[-1]

    for step in range(1, args.max_new_tokens):

        print()
        print("=" * 80)
        print(f"REAL PARAM2 DECODE STEP {step}")
        print("=" * 80)

        token = torch.tensor(
            [[generated[-1]]],
            dtype=torch.long,
        ).to(device)

        total_length = prompt_length + len(generated)

        step_attention_mask = torch.ones(
            (1, total_length),
            dtype=torch.long,
            device=device,
        )

        sync()

        t0 = time.perf_counter()

        with torch.inference_mode():
            out = model(
                input_ids=token,
                attention_mask=step_attention_mask,
                past_key_values=past,
                use_cache=True,
            )

        sync()

        decode_ms = (
            time.perf_counter() - t0
        ) * 1000.0

        print()

        decode_times.append(decode_ms)

        past = out.past_key_values

        logits_cpu = (
            out.logits[:, -1, :]
            .float()
            .cpu()
        )

        if not torch.isfinite(logits_cpu).all():
            raise RuntimeError(
                f"Non-finite logits during decode step {step}."
            )

        next_token = int(
            torch.argmax(
                logits_cpu,
                dim=-1,
            ).item()
        )

        generated.append(next_token)

        print(
            f"PASS: decode step {step}, "
            f"wall time={decode_ms:.3f} ms"
        )

    # ------------------------------------------------------------
    # Final generated text
    # ------------------------------------------------------------

    generated_text = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    )

    print()
    print("=" * 80)
    print("FINAL RESULT")
    print("=" * 80)

    print("Generated token ids:", generated)

    print()
    print("Generated text:")
    print(generated_text)

    print()
    print(f"Prefill wall time : {prefill_ms:.3f} ms")

    if decode_times:
        avg_decode = sum(decode_times) / len(decode_times)

        print(
            f"Decode wall time  : "
            f"{avg_decode:.3f} ms/token"
        )

    if execution_mode == "stream":
        print()
        print(
            "NOTE: timings include CPU<->GPU layer transfers and are NOT "
            "representative of normal Param2 inference performance."
        )

    print()
    print(
        "PASS: REAL Param2 weights completed "
        "prefill + KV-cache decode through Torch-ParaS."
    )

    # Cleanup hooks.
    for handle in hook_handles:
        handle.remove()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print()
        print("=" * 80)
        print("PARAM2 E2E FAILED")
        print("=" * 80)
        raise
