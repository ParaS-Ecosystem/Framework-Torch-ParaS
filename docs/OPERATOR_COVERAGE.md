# Operator Coverage

What runs natively on `paras`, what falls back to CPU, and what's tested.
This is hand-maintained until CI can generate it automatically (see
Roadmap Milestone 1/5) — update it in the same PR that adds or changes a
kernel.

## Summary

Counted from the `TORCH_LIBRARY_IMPL` blocks in `csrc/ops/*.cpp`
(commented-out registrations excluded).

| | Count |
|---|---|
| aten registrations, `PrivateUse1` | 234 |
| aten registrations, `AutogradPrivateUse1` | 0 |
| **Total aten kernel registrations** | **235** |
| Distinct aten operators (overloads collapsed, e.g. `sort` / `sort.stable`) | 182 |
| Custom ops, `paras::` namespace (`fused_ops.cpp`) | 8 |
| Custom ops, `torch_paras::` namespace (`rms_swiglu_ops.cpp`, also registered for CPU) | 1 |
| Custom op `paras::moe_experts` (`moe_ops.cpp`, also registered for CPU) | 1 |
| **Total native ops** | **245** |

CompositeImplicitAutograd operators (`linear`, `matmul`, `reshape`,
`contiguous`, `repeat_interleave`, `rms_norm`) are deliberately **not**
registered: a PrivateUse1 kernel for them would bypass autograd (and make
AOTAutograd trace them as leaves without a derivative). They decompose into
the kernels counted here. Operators that PyTorch gives a derivative formula of
their own (`_safe_softmax`, `silu_backward`,
`_scaled_dot_product_fused_attention_overrideable`) are safe to register and
are, as on CUDA.

Everything else goes through the boxed CPU fallback registered in
`tensor_ops.cpp`.

### Registrations per source file

| File | aten | custom | Total |
|---|---|---|---|
| `pointwise_ops.cpp` | 126 | — | 126 |
| `tensor_ops.cpp` | 28 | — | 28 |
| `indexing_ops.cpp` | 21 | — | 21 |
| `vision_ops.cpp` | 19 | — | 19 |
| `scatter_ops.cpp` | 12 | — | 12 |
| `creation_ops.cpp` | 7 | — | 7 |
| `loss_ops.cpp` | 7 | — | 7 |
| `random_ops.cpp` | 7 | — | 7 |
| `fused_ops.cpp` | — | 8 | 8 |
| `norm_ops.cpp` | 4 | — | 4 |
| `sdpa_ops.cpp` | 2 | — | 2 |
| `sort_ops.cpp` | 2 | — | 2 |
| `rms_swiglu_ops.cpp` | — | 1 | 1 |
| `moe_ops.cpp` | — | 1 | 1 |
| **Total** | **235** | **10** | **245** |

## Status Legend

- **Native** — implemented as a real ParaS kernel (`csrc/ops`), registered
  for the `paras` dispatch key.
- **Fallback** — not implemented natively; runs via the boxed CPU fallback
  (works, but with a host round-trip).
- **Partial** — some dtypes/shapes/modes are native, others fall back.
- **Tested** — a parity test exists comparing against native PyTorch CPU
  output (see `docs/TESTING.md` for which file).

## Coverage by Category

| Category | Kernels | Status | Tested | Test file | Notes |
|---|---|---|---|---|---|
| Tensor lifecycle & views (`empty*`, `_copy_from*`, `_to_copy`, `fill_`, `zero_`, `set_`, `resize_`, `view`, `as_strided`, `transpose`, `permute`, `expand`, `repeat*`, `split*`, `reshape`, `squeeze*`, `unsqueeze`, `slice`, `contiguous`; `chunk` and `narrow` decompose into `split` / `slice`, keeping their derivatives) | 30 | Native | Yes | `test_tensor_ops.py` | Includes cross-device and strided copies |
| Embedding (`embedding`, `embedding_dense_backward`) | 2 | Native | Forward only | `test_tensor_ops.py` | |
| Creation (`zeros`, `ones`, `full`, `*_like`, `arange`) | 8 | Native | Yes | `test_matmul_sort_creation_ops.py` | `arange.start_out` lives in `pointwise_ops.cpp` |
| Binary arithmetic (`add`, `sub`, `mul`, `div`, `maximum`, `minimum`, `addcmul`, `addcdiv`) | 11 | Native | Partial | `test_pointwise_ops.py` | `addcmul`/`addcdiv` untested |
| Unary math (`exp`, `log`, `sqrt`, `rsqrt`, `sin`, `cos`, `atan`, `neg`, `abs`, `sgn`, `ceil`, `round`, `reciprocal`, `pow`, `lerp`) | 16 | Native | Partial | `test_pointwise_ops.py` | `lerp` untested |
| Activations + backward (`relu`, `sigmoid`, `tanh`, `silu`, `gelu`, `hardtanh`, `hardswish`, `hardsigmoid`, `leaky_relu`, `threshold_backward`, `logit`, `log_sigmoid`, `clamp`, `clamp_min`) | 32 | Native | Partial | `test_pointwise_ops.py` | `hardswish` untested |
| Comparisons (`eq/ne/lt/gt/le/ge`, Tensor and Scalar) | 12 | Native | Yes | `test_pointwise_ops.py` | |
| Bitwise (`and`, `or`, `xor`, `not`) | 4 | Native | Yes | `test_pointwise_ops.py` | |
| Reductions (`sum`, `mean`, `prod`, `amax`, `amin`, `argmax`, `min`, `max`, `dot`, `any`, `all`, `linalg_vector_norm`) | 16 | Native | Yes | `test_pointwise_ops.py`, `test_device_blas_and_runtime.py` | Bool inputs supported; GPU rows of 256..16384 elements: one work-group per row, one launch; longer rows use a two-level (striped) reduction |
| Logical / predicates (`logical_not`, `isneginf`, `isposinf`, `isin`, `floor_divide`) | 7 | Native | Yes | `test_device_blas_and_runtime.py` | Needed by attention masks and `generate()` |
| `cat` | 2 | Native | Yes | `test_pointwise_ops.py` | |
| Softmax / log-softmax + backward | 8 | Native | Yes | `test_pointwise_ops.py` | |
| `cumsum` | 3 | Native | Yes | `test_pointwise_ops.py` | |
| Sort / top-k / `max.dim` / `min.dim` / `argsort` | 12 | Native | Yes | `test_pointwise_ops.py`, `test_matmul_sort_creation_ops.py`, `test_device_blas_and_runtime.py` | On the device, no length limit: rank sort (rows <= 8192) or bitonic network; exact keys, NaN largest, stable |
| Indexing (`gather`, `index_select`, `index_copy`, `index_add`, `index.Tensor`, `index_put`, `_index_put_impl_`, `nonzero`, `repeat_interleave.Tensor`, `where`, `triu`, `tril`) | 22 | Native | Yes | `test_indexing_ops.py`, `test_device_blas_and_runtime.py` | Accumulating `index_put_` covers every dtype (Half/BFloat16 accumulate in fp32) |
| Scatter / `scatter_add` / `masked_fill` | 11 | Native | Yes | `test_device_blas_and_runtime.py` | |
| Matrix multiply (`mm`, `bmm`, `addmm`) | 3 | Native | Yes | `test_matmul_sort_creation_ops.py`, `test_device_blas_and_runtime.py` | GPU: device GEMM through CLAP (fp32/fp64/fp16/bf16); up to 8 rows against a linear layer's weight (decode): a memory-bound ParaS kernel instead; `linear`/`matmul` decompose into these |
| Convolution (`convolution_overrideable` + backward) | 2 | Native | **No** | — | No parity test yet |
| Pooling (`_adaptive_avg_pool2d` ± backward, `avg_pool2d` ± backward, `max_pool2d_with_indices` ± backward) | 6 | Host-staged | Partial | `test_vision_ops.py` | `max_pool2d` decomposes into `max_pool2d_with_indices` (traceable by `torch.compile`) |
| Upsampling (`nearest2d`, `_nearest_exact2d`, `bilinear2d`, each ± backward) | 6 | Native | Partial | `test_vision_ops.py` | Only `upsample_nearest2d` forward tested |
| Batch / layer norm (`native_batch_norm`, `native_layer_norm`, each ± backward) | 4 | Native | **No** | — | No parity test in repo yet |
| RMS norm (`aten::rms_norm`) | — | Composite | Yes | `test_attention_norm_ops.py` | Decomposes into pow/mean/rsqrt/mul kernels; the `paras` compile backend replaces it with `paras::rms_norm` (one launch, bit-exact) |
| Losses (`mse_loss`, `binary_cross_entropy`, `nll_loss`, with backward) | 7 | Native | Partial | `test_pointwise_ops.py` | `nll_loss` covered only via log_softmax chain; MSE and BCE untested |
| RNG / dropout (`uniform_`, `normal_`, `bernoulli_`, `native_dropout` ± backward, `multinomial`) | 7 | Native | Yes | `test_random_ops.py` | Statistical checks, not exact-value |
| Scaled dot-product attention (`_fused_sdp_choice`, `_scaled_dot_product_fused_attention_overrideable`) | 2 | Native (inference) / Composite (training) | Yes | `test_attention_norm_ops.py` | Inference: fused path (decode, up to 8192 keys and head size 256: one kernel, grouped-query attention without repeating K/V; prefill on AMD: vendor GEMMs with fp32 scores around one row-softmax kernel), as accurate as the fp32 math path. Training, dropout, unusual masks and `sdpa_kernel(MATH)`: PyTorch's math decomposition |
| Multi-head attention (`_native_multi_head_attention` ± out, `_transform_bias_rescale_qkv`) | 3 | Native | **No** | — | No parity test yet |
| Custom fused: `paras::rms_norm`, `swiglu`, `rotate_half` + backward, `add_rms_norm`, `rope` | 8 | Native | Yes | `test_fused_ops.py`, `test_optimizer.py` | Call via `torch.ops.paras.*`; the `paras` compile backend inserts them (`docs/GRAPH_OPTIMIZER.md`). `add_rms_norm` and `rope` differentiate through the existing backward kernels |
| Custom fused: `torch_paras::swiglu` | 1 | Native (+CPU) | Yes | `test_attention_norm_ops.py` | Call via `torch.ops.torch_paras.swiglu` |
| MoE routed experts: `paras::moe_experts` | 1 | Native (+CPU) | Inference | `test_moe_ops.py` | DeepSeek-style `moe_infer` (SwiGLU experts, stacked weights) without a host read: on the device for up to 768 routes (graph capture: up to 4096), per-expert vendor GEMMs above, with eager's result. `torch_paras.optimizer.fuse_moe_experts(model)` substitutes it; `torch_paras.compile` does so by default |
| Anything not listed above | — | Fallback | N/A | — | Runs via boxed CPU fallback automatically |

## Known Gaps

- Untested native kernels: `masked_fill`, convolution, batch/layer
  norm, multi-head attention, MSE/BCE losses, `avg_pool2d`, `max_pool2d`,
  non-nearest upsampling, and all vision backward kernels.
- `swiglu` exists twice with different schemas: `paras::swiglu(gate, up)`
  in `fused_ops.cpp` and `torch_paras::swiglu(x, gate)` in
  `rms_swiglu_ops.cpp`. Consider consolidating.
- Convolution, pooling, upsampling and batch/layer norm still stage their
  operands through host memory; `torch_paras.runtime_stats()` counts those
  transfers, and strict mode (`PTSYCL_STRICT=1`) rejects any CPU fallback.
- Autograd (backward-pass) coverage isn't tracked separately from forward
  coverage yet.

## How to Update This Table

1. Adding a new kernel: update the Summary counts, the per-file table,
   and the category row (or add a new row).
2. Adding a parity test: fill in the Tested / Test file columns.
3. Finding an untested but implemented kernel: mark it clearly so it
   isn't mistaken for verified behavior.
4. If you're not sure whether something is native or fallback, check the
   dispatcher registration in `csrc/ops` rather than guessing from
   behavior alone (a fallback can still produce a correct result — it's
   just slower).
5. Recount with:
   ```bash
   cat csrc/ops/*.cpp | grep -v '^\s*//' | grep -c 'm\.impl('
   ```
   This includes the `torch_paras::swiglu` CPU registration, so
   subtract 1 for the native count.
6. To see what a model actually runs off-device, run it once with
   `torch_paras.reset_runtime_stats()` / `torch_paras.runtime_stats()`.
