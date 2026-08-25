#!/usr/bin/env python3
"""
Param2-17B Sanity Test on AMD MI300X (paras:1)
Tests:
  1. Prefill (Forward pass on prompt)
  2. KV Cache generation (Autoregressive decode step)
  3. Backward pass (Gradient computation and finite loss/grad checks)
"""

import sys
import os
import time
import gc

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.insert(0, "/home/amdgpu02/parikshit/torch_paras/Framework-Torch-ParaS/python")

MODEL_PATH = "/home/amdgpu02/parikshit/models/Param2-17B-A2.4B-Thinking"

import torch
import torch_paras
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    import transformers.cache_utils as cache_utils
    DynamicLayer = getattr(cache_utils, "DynamicLayer", None)
    if DynamicLayer is not None:
        def _safe_dynamic_layer_update(self, key_states, value_states, *args, **kwargs):
            if self.keys is None or self.keys.numel() == 0 or self.keys.ndim != key_states.ndim:
                self.keys = key_states
                self.values = value_states
            else:
                self.keys = torch.cat([self.keys, key_states], dim=-2)
                self.values = torch.cat([self.values, value_states], dim=-2)
            return self.keys, self.values
        DynamicLayer.update = _safe_dynamic_layer_update
        print("[Patch] Transformers DynamicLayer.update patched.")
except Exception as e:
    print(f"[Patch] DynamicLayer patch notice: {e}")


def test_param2_sanity():
    device_str = "paras:1"
    device = torch.device(device_str)

    print("=" * 80)
    print(f"Param2-17B Sanity Test on AMD MI300X ({device_str})")
    print(f"Model Path: {MODEL_PATH}")
    print("=" * 80)

    # 1. Load Tokenizer & Model
    print("\n>>> Loading Tokenizer & Model in bfloat16...")
    t0 = time.time()
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
    print(f"Model successfully loaded and placed on {device_str} in {time.time() - t0:.2f}s")

    prompt = "What is the capital of India?"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # --------------------------------------------------------------------------
    # Test 1: Prefill (Forward Pass)
    # --------------------------------------------------------------------------
    print("\n>>> [1/3] Testing Prefill (Forward pass)...")
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    torch_paras.synchronize(device)
    prefill_time = time.time() - t0

    logits = out.logits
    assert logits is not None, "Prefill failed: logits is None"
    assert not torch.isnan(logits).any(), "Prefill failed: logits contains NaN"
    assert not torch.isinf(logits).any(), "Prefill failed: logits contains Inf"
    print(f"  [PASS] Prefill: logits shape={tuple(logits.shape)}, time={prefill_time:.4f}s")

    # --------------------------------------------------------------------------
    # Test 2: KV Cache Generation (Autoregressive Decode)
    # --------------------------------------------------------------------------
    print("\n>>> [2/3] Testing KV Cache Generation (Autoregressive Decode)...")
    past_key_values = out.past_key_values
    cur_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    cur_mask = torch.cat([attention_mask, torch.ones_like(cur_id)], dim=-1)

    generated_tokens = [cur_id.item()]
    num_decode_tokens = 8

    t0 = time.time()
    for step in range(num_decode_tokens):
        with torch.no_grad():
            step_out = model(
                input_ids=cur_id,
                attention_mask=cur_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
        torch_paras.synchronize(device)
        past_key_values = step_out.past_key_values
        step_logits = step_out.logits
        assert not torch.isnan(step_logits).any(), f"Decode step {step}: logits contains NaN"
        assert not torch.isinf(step_logits).any(), f"Decode step {step}: logits contains Inf"

        cur_id = torch.argmax(step_logits[:, -1, :], dim=-1, keepdim=True)
        cur_mask = torch.cat([cur_mask, torch.ones_like(cur_id)], dim=-1)
        generated_tokens.append(cur_id.item())

    decode_time = time.time() - t0
    decoded_text = tokenizer.decode(generated_tokens)
    print(f"  [PASS] KV Cache Generation: generated {len(generated_tokens)} tokens in {decode_time:.4f}s")
    print(f"  Generated sample text: {repr(decoded_text)}")

    # --------------------------------------------------------------------------
    # Test 3: Backward Pass
    # --------------------------------------------------------------------------
    print("\n>>> [3/3] Testing Backward Pass...")
    model.train()
    labels = input_ids.clone()
    t0 = time.time()
    train_out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = train_out.loss

    assert loss is not None, "Backward test failed: loss is None"
    assert not torch.isnan(loss), "Backward test failed: loss is NaN"
    assert not torch.isinf(loss), "Backward test failed: loss is Inf"
    print(f"  Forward loss: {loss.item():.4f}")

    loss.backward()
    torch_paras.synchronize(device)
    bwd_time = time.time() - t0

    grad_count = 0
    nan_grads = 0
    inf_grads = 0
    total_params = 0

    for name, param in model.named_parameters():
        total_params += 1
        if param.grad is not None:
            grad_count += 1
            if torch.isnan(param.grad).any():
                nan_grads += 1
            if torch.isinf(param.grad).any():
                inf_grads += 1

    assert grad_count > 0, "Backward test failed: no gradients computed"
    assert nan_grads == 0, f"Backward test failed: {nan_grads} parameters have NaN gradients"
    assert inf_grads == 0, f"Backward test failed: {inf_grads} parameters have Inf gradients"

    print(f"  [PASS] Backward Pass: computed {grad_count}/{total_params} parameter gradients in {bwd_time:.4f}s")
    print(f"  NaN gradients: {nan_grads}, Inf gradients: {inf_grads}")

    print("\n" + "=" * 80)
    print("ALL PARAM2-17B SANITY CHECKS PASSED ON AMD MI300X (paras:1)!")
    print("=" * 80)


if __name__ == "__main__":
    test_param2_sanity()
