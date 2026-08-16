# Operator Coverage

What runs natively on `paras`, what falls back to CPU, and what's tested.
This is hand-maintained until CI can generate it automatically — update it
in the same PR that adds or changes a kernel.

## Status Legend

- **Native** — implemented as a real ParaS kernel (`csrc/ops`), registered
  for the `paras` dispatch key.
- **Fallback** — not implemented natively; runs via the boxed CPU fallback
  (works, but with a host round-trip).
- **Partial** — some dtypes/shapes/modes are native, others fall back.
- **Tested** — a parity test exists comparing against native PyTorch CPU
  output (see `docs/TESTING.md` for which file).

## Coverage by Category

Categories below are taken from the kernel groups listed in `README.md`.
Individual op names within each category should be filled in as they're
confirmed — this table currently tracks category-level status only.

| Category | Status | Tested | Test file | Notes |
|---|---|---|---|---|
| Tensor lifecycle (alloc, views, copies, resize, dtype conversion) | Native | Yes | `test_tensor_ops.py` | Includes cross-device and strided copies |
| Elementwise math / activations / comparisons | Native | Yes | `test_pointwise_ops.py` | |
| Reductions | Native | Yes | `test_pointwise_ops.py` | |
| Bitwise ops, `cat` | Native | Yes | `test_pointwise_ops.py` | |
| Matrix multiply (`mm`/`bmm`/`addmm`) | Native | Yes | `test_pointwise_ops.py` | |
| Convolution | Native | **No** | — | Kernel exists per README; no dedicated parity test yet |
| Pooling | Native | **No** | — | Kernel exists per README; no dedicated parity test yet |
| Upsampling | Native | **No** | — | Kernel exists per README; no dedicated parity test yet |
| Batch / layer norm | Native | **No** | — | Kernel exists per README; no dedicated parity test yet |
| Losses | Native | **No** | — | Kernel exists per README; no dedicated parity test yet |
| RNG (Philox counter-based) | Native | Yes | `test_random_ops.py` | Statistical checks, not exact-value |
| Dropout (train/eval, backward) | Native | Yes | `test_random_ops.py` | |
| Multi-head attention | Native | **No** | — | Kernel exists per README; no dedicated parity test yet |
| RMS norm, SwiGLU, rotate-half (fwd + bwd) | Native | Yes | `test_fused_ops.py` | |
| Indexing (`gather`, `index_select`, `index_copy_`, `index_put_`) | Native | Yes | `test_indexing_ops.py` | Includes accumulate mode |
| `where`, `triu`, `tril` | Native | Yes | `test_indexing_ops.py` | |
| Anything not listed above | Fallback | N/A | — | Runs via boxed CPU fallback automatically |

## Known Gaps

- No per-operator (as opposed to per-category) tracking yet — e.g. we
  don't currently distinguish `conv1d` vs `conv2d` vs `conv3d` coverage.
  Break this table out further as gaps are found.
- No fallback-ratio measurement tooling yet for full models.
- Autograd (backward-pass) coverage isn't tracked separately from forward
  coverage here yet — worth splitting into its own column once backward
  tests exist for more than dropout.

## How to Update This Table

1. Adding a new kernel: change its row from Fallback to Native (or add a
   new row if the category didn't exist).
2. Adding a parity test: fill in the Tested / Test file columns.
3. Finding an untested but implemented kernel: mark it clearly (as done
   above) so it isn't mistaken for verified behavior.
4. If you're not sure whether something is native or fallback, check the
   dispatcher registration in `csrc/ops` rather than guessing from
   behavior alone (a fallback can still produce a correct result — it's
   just slower).
