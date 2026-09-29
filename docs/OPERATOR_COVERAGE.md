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
| aten registrations, `PrivateUse1` | 213 |
| aten registrations, `AutogradPrivateUse1` (`linear`, `max_pool2d`) | 2 |
| **Total aten kernel registrations** | **215** |
| Distinct aten operators (overloads collapsed, e.g. `sort` / `sort.stable`) | 170 |
| Custom ops, `paras::` namespace (`fused_ops.cpp`) | 6 |
| Custom ops, `torch_paras::` namespace (`rms_swiglu_ops.cpp`, also registered for CPU) | 1 |
| **Total native ops** | **222** |

Everything else goes through the boxed CPU fallback registered in
`tensor_ops.cpp`.

### Registrations per source file

| File | aten | custom | Total |
|---|---|---|---|
| `pointwise_ops.cpp` | 108 | — | 108 |
| `tensor_ops.cpp` | 34 | — | 34 |
| `vision_ops.cpp` | 20 | — | 20 |
| `indexing_ops.cpp` | 16 | — | 16 |
| `scatter_ops.cpp` | 8 | — | 8 |
| `creation_ops.cpp` | 7 | — | 7 |
| `loss_ops.cpp` | 7 | — | 7 |
| `random_ops.cpp` | 7 | — | 7 |
| `fused_ops.cpp` | — | 6 | 6 |
| `norm_ops.cpp` | 4 | — | 4 |
| `rms_swiglu_ops.cpp` | 1 | 1 | 2 |
| `sort_ops.cpp` | 2 | — | 2 |
| `attention_ops.cpp` | 1 | — | 1 |
| **Total** | **215** | **7** | **222** |

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
| Tensor lifecycle & views (`empty*`, `_copy_from*`, `_to_copy`, `fill_`, `zero_`, `set_`, `resize_`, `view`, `as_strided`, `transpose`, `permute`, `expand`, `repeat*`, `split*`, `chunk`, `reshape`, `squeeze*`, `unsqueeze`, `narrow`, `slice`, `contiguous`) | 32 | Native | Yes | `test_tensor_ops.py` | Includes cross-device and strided copies |
| Embedding (`embedding`, `embedding_dense_backward`) | 2 | Native | Forward only | `test_tensor_ops.py` | |
| Creation (`zeros`, `ones`, `full`, `*_like`, `arange`) | 8 | Native | Yes | `test_matmul_sort_creation_ops.py` | `arange.start_out` lives in `pointwise_ops.cpp` |
| Binary arithmetic (`add`, `sub`, `mul`, `div`, `maximum`, `minimum`, `addcmul`, `addcdiv`) | 11 | Native | Partial | `test_pointwise_ops.py` | `addcmul`/`addcdiv` untested |
| Unary math (`exp`, `log`, `sqrt`, `rsqrt`, `sin`, `cos`, `atan`, `neg`, `abs`, `sgn`, `ceil`, `round`, `reciprocal`, `pow`, `lerp`) | 16 | Native | Partial | `test_pointwise_ops.py` | `lerp` untested |
| Activations + backward (`relu`, `sigmoid`, `tanh`, `silu`, `gelu`, `hardtanh`, `hardswish`, `hardsigmoid`, `leaky_relu`, `threshold_backward`, `logit`, `log_sigmoid`, `clamp`, `clamp_min`) | 32 | Native | Partial | `test_pointwise_ops.py` | `hardswish` untested |
| Comparisons (`eq/ne/lt/gt/le/ge`, Tensor and Scalar) | 12 | Native | Yes | `test_pointwise_ops.py` | |
| Bitwise (`and`, `or`, `xor`, `not`) | 4 | Native | Yes | `test_pointwise_ops.py` | |
| Reductions (`sum`, `mean`, `prod`, `amax`, `amin`, `argmax`, `min`, `max`, `dot`) | 9 | Native | Yes | `test_pointwise_ops.py` | |
| `cat` | 2 | Native | Yes | `test_pointwise_ops.py` | |
| Softmax / log-softmax + backward | 8 | Native | Yes | `test_pointwise_ops.py` | |
| `cumsum` | 3 | Native | Yes | `test_pointwise_ops.py` | |
| Sort / top-k / `max.dim` / `min.dim` / `argsort` | 12 | Partial | Yes | `test_pointwise_ops.py`, `test_matmul_sort_creation_ops.py` | `sort`/`topk` rows longer than `PTSYCL_SORT_NAIVE_MAX` (4096) fall back to CPU |
| Indexing (`gather`, `index_select`, `index_copy`, `index.Tensor`, `index_put`, `where`, `triu`, `tril`) | 16 | Native | Yes | `test_indexing_ops.py` | |
| Scatter / `masked_fill` | 8 | Native | **No** | — | No parity test yet |
| Matrix multiply (`mm`, `bmm`, `addmm`, `linear`) | 4 | Native | Yes | `test_matmul_sort_creation_ops.py`, `test_vision_ops.py` | `linear` registered at `AutogradPrivateUse1` |
| Convolution (`convolution_overrideable` + backward) | 2 | Native | **No** | — | No parity test yet |
| Pooling (`_adaptive_avg_pool2d` ± backward, `avg_pool2d` ± backward, `max_pool2d`) | 5 | Native | Partial | `test_vision_ops.py` | Only adaptive avg pool tested; `max_pool2d` registered at `AutogradPrivateUse1` |
| Upsampling (`nearest2d`, `_nearest_exact2d`, `bilinear2d`, each ± backward) | 6 | Native | Partial | `test_vision_ops.py` | Only `upsample_nearest2d` forward tested |
| Batch / layer norm (`native_batch_norm`, `native_layer_norm`, each ± backward) | 4 | Native | **No** | — | No parity test in repo yet |
| RMS norm (`aten::rms_norm`) | 1 | Native | Yes | `test_attention_norm_ops.py` | |
| Losses (`mse_loss`, `binary_cross_entropy`, `nll_loss`, with backward) | 7 | Native | Partial | `test_pointwise_ops.py` | `nll_loss` covered only via log_softmax chain; MSE and BCE untested |
| RNG / dropout (`uniform_`, `normal_`, `bernoulli_`, `native_dropout` ± backward, `multinomial`) | 7 | Native | Yes | `test_random_ops.py` | Statistical checks, not exact-value |
| Scaled dot-product attention (`_scaled_dot_product_attention_math`) | 1 | Native | Yes | `test_attention_norm_ops.py` | Causal, masks, GQA covered |
| Multi-head attention (`_native_multi_head_attention` ± out, `_transform_bias_rescale_qkv`) | 3 | Native | **No** | — | No parity test yet |
| Custom fused: `paras::rms_norm`, `swiglu`, `rotate_half` + backward | 6 | Native | Yes | `test_fused_ops.py` | Call via `torch.ops.paras.*` |
| Custom fused: `torch_paras::swiglu` | 1 | Native (+CPU) | Yes | `test_attention_norm_ops.py` | Call via `torch.ops.torch_paras.swiglu` |
| Anything not listed above | — | Fallback | N/A | — | Runs via boxed CPU fallback automatically |

## Known Gaps

- Untested native kernels: scatter/`masked_fill`, convolution, batch/layer
  norm, multi-head attention, MSE/BCE losses, `avg_pool2d`, `max_pool2d`,
  non-nearest upsampling, and all vision backward kernels.
- `swiglu` exists twice with different schemas: `paras::swiglu(gate, up)`
  in `fused_ops.cpp` and `torch_paras::swiglu(x, gate)` in
  `rms_swiglu_ops.cpp`. Consider consolidating.
- No fallback-ratio measurement tooling yet for full models (tracked in
  `docs/MODEL_VALIDATION.md` and Roadmap Milestone 3).
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
