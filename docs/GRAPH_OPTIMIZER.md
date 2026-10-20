# Graph optimizer (`torch.compile(backend="paras")`)

`torch_paras.optimizer` is a TorchDynamo backend for paras devices. It works
for inference and training; a compiled forward computes exactly what eager
computes, and compiled gradients equal eager gradients up to rounding.

```python
import torch, torch_paras

model = model.to("paras:0")
fast = torch.compile(model, backend="paras")         # default options
fast = torch_paras.compile(model, strict=True)       # reject CPU-fallback ops
fast = torch_paras.compile(model, fusions=())        # capture without fusion
```

## Pipeline

For every graph TorchDynamo captures:

1. **Fusion passes** (`optimizer/passes.py`) run on the Dynamo graph, before
   autograd. A matched subgraph is replaced by one `paras::` kernel.
2. **AOTAutograd** traces functionalised ATen forward and backward graphs at
   compile time (the backward is traced immediately, not at the first
   backward call). Fused kernels have C++ autograd kernels
   (`csrc/ops/fused_ops.cpp`), so the backward of a fused RMSNorm is
   `paras::rms_norm_backward`; `optimizer/ops.py` gives tracing their fake
   kernels.
3. **Residency analysis** (`optimizer/residency.py`) classifies every ATen
   operator of each traced graph from the dispatcher tables:
   * `paras`: ParaS registers a PrivateUse1 kernel (or the op is a structured
     functional form whose out= variant has one);
   * `composite`: a PyTorch composite that decomposes into other operators,
     each dispatched (and counted, see below) when it runs;
   * `constant`: `lift_fresh` / `lift_fresh_copy`, AOTAutograd's copy of a
     tensor literal in the code (`torch.tensor(...)`);
   * `cpu`: no ParaS kernel; the op would run on the boxed CPU fallback.
   With `strict=True` a graph containing a `cpu` op raises
   `ParasCompileError`.
4. **Execution**: the fused Dynamo graph runs through the dispatcher and the
   autograd engine like eager code, on the same ParaS kernels eager mode
   uses. The traced ATen graphs are used for analysis only: running
   AOTAutograd's compiled function instead adds a runtime wrapper and a
   Python-level backward to every call, about 40 us per call -- for an MoE
   whose expert MLPs are compiled regions called from an eager expert loop
   (Param2-17B: ~1,200 calls per forward) that made compiled training 15 %
   slower than eager.
5. **Frames handed back to the interpreter.** Dynamo skips them from then on
   and they run as plain eager code -- which is what their graphs compute
   (`GraphReport.runs_eagerly` gives the reason).
   * *Small dynamic-shape graphs*: when Dynamo recompiles a frame with
     dynamic shapes, each later call evaluates guards over several cache
     entries and runs a SymInt prologue, about 10-20 us. A frame captured
     whole (no graph break) with fewer than `min_dynamic_graph_ops` tensor
     operators (default 8) cannot win that back; Dynamo is switched off for
     it and what it calls. Typical case: an MoE's expert MLP called once per
     expert with a different token count (Param2-17B prefill: ~1,200 Dynamo
     frame entries per forward).
   * *Frames Dynamo refuses to recompile*: at its recompile limit Dynamo
     keeps evaluating the frame's cached guards on every call without ever
     compiling a matching graph again. The backend learns of the refused
     compile through Dynamo's recompile hook and skips the frame from the next
     call of a paras graph on; the functions it calls keep their graphs. A
     frame that needs exactly the limit keeps its graphs. Typical cases: a
     decoder layer reading a growing KV cache without per-layer code objects
     (below), and a model's top-level forward called by `generate()` with
     ever new keyword sets. `eager_at_recompile_limit=False` keeps Dynamo's
     behaviour.

   `torch_paras.compile(model, min_dynamic_graph_ops=0,
   eager_at_recompile_limit=False)` compiles and keeps every graph.

## One Dynamo cache per layer

Dynamo keeps compiled graphs per code object, and the layers of a transformer
share one `forward`. Each layer specialises its graphs on its own Python
attributes -- `layer_idx`, and through it its own slot of the KV cache -- so
when the layers run as separate frames (a graph break inside the layer, e.g.
an MoE's host read) all their variants compete in one cache: lookups scan
dozens of entries and the frame soon reaches Dynamo's recompile limit.

`torch_paras.compile` therefore gives each instance of a repeated block its own
copy of the `forward` code object (`optimizer.specialize_repeated_blocks`; same
bytecode, bound as an instance attribute). A block qualifies when it is one of
two or more instances of a class in an `nn.ModuleList` / `nn.Sequential` and
the instances differ in a Python scalar attribute of the block or one of its
submodules (Param2-17B: `attention.layer_idx`). Blocks that differ only in their
weights, such as an MoE's experts, share graphs anyway and are left alone.
Each layer is then compiled on its own, after which a cached decode keeps every
layer compiled. The price is compilation time: Param2-17B's benchmark shape
mix (prefill at four shapes, generate() at batch 1 and 4) compiles in ~235 s
instead of ~30 s. A layer then
legitimately needs more graphs than Dynamo's default recompile limit (8)
allows -- uncached and cached prefill, decode, each first static and then
dynamic, per batch size; a `generate()` workload reaches ~12 -- so
specialising raises `torch._dynamo.config.recompile_limit` to at least 32
(process-wide; each layer keeps its own cache, so a lookup only scans that
layer's entries).

`specialize_blocks=False` turns this off; `optimizer.unspecialize(model)`
removes the per-instance copies (e.g. before pickling the module).
`torch.compile(model, backend="paras")` cannot do it: a Dynamo backend only
sees graphs.

For transformers' `generate()`, compile `forward` in place -- `generate()` on
the wrapper returned by `torch.compile` runs the original, uncompiled module:

```python
torch_paras.optimizer.specialize_repeated_blocks(model)
model.forward = torch.compile(model.forward, backend="paras")
model.generate(input_ids, max_new_tokens=128)
```

`torch_paras.optimizer.reports()` returns one `GraphReport` per captured
graph: the Python function it came from (`frame`, "name (file:line)"), the
fusions applied and the residency counts of its forward, backward or
inference graph.

## Fusions

| Pass | Matches | Becomes |
|---|---|---|
| `rms_norm` | `w * (x32 * rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)` with `x32 = x.to(float32)` (or `x` already fp32; `x*x` accepted for `pow(2)`) | `paras::rms_norm(x, w, eps)` |
| `swiglu` | `silu(gate) * up` (skipped where CLAP-DNN runs eager SiLU, see below) | `paras::swiglu(gate, up)` |
| `rotate_half` | `cat((-x[..., h:], x[..., :h]), -1)` with `h = x.shape[-1] // 2` | `paras::rotate_half(x)` |

Method, function and operator spellings are all recognised (`x.pow(2)`,
`torch.pow(x, 2)`, `x ** 2`). A pattern is rewritten only when the fused
kernel computes the same function, checked from Dynamo's FakeTensor
metadata: paras device, last-dimension reduction, weight shape `[H]` and the
same dtype as the input (the kernel reads the weight in the input dtype), no
broadcasting for SwiGLU, an even split for rotate-half, every
intermediate used only inside the pattern, and no operation that may write to
a tensor (in-place method, `out=`, module call) between the pattern's first
and last node -- the fused kernel reads its operands where the pattern ends.

In a build with CLAP-DNN (`PTSYCL_ENABLE_CLAP=ON`) eager SiLU may run in the
vendor DNN library, whose rounding the fused kernel does not reproduce. The
SwiGLU pass asks the runtime which dtype/device combinations those are
(`torch_paras._C.clap_dnn_activation_uses_vendor`) and leaves the pattern
unfused there: e.g. fp32 through MIOpen or oneDNN. bf16 on ROCm stays on the
ParaS kernel (MIOpen has no bf16 activations), so bf16 models are fused.

**Fused kernels are bit-exact with the eager decomposition** on the same
device: they perform the same rounding steps in the same order (RMSNorm
reuses the eager reduction, `reduce_outer`, with the same shape, so even the
summation order matches). Fusing therefore cannot change a model's result --
which matters for mixture-of-experts models, where a one-ulp difference can
flip a top-k routing decision. `tests/test_optimizer.py` asserts
`torch.equal` between fused and eager outputs.

The fused **backward** kernels (`paras::*_backward`) compute the same
gradients as autograd's decomposed backward but not bit for bit (they
accumulate in a different order), so compiled and eager training agree to
rounding (the tests check 1e-4) and an MoE's later training steps can route a
few tokens differently. With `fusions=()` compiled training reproduces eager
exactly.

## Execution telemetry and strict mode

Independently of the optimizer, the C++ runtime counts what leaves the ParaS
kernels:

```python
torch_paras.reset_runtime_stats()
model(x)
torch_paras.runtime_stats()
# Param2-17B forward: 20 copies of 512 B = the model's own tokens_per_expert.cpu()
# (64 int64 expert counts, one per MoE layer)
# {'cpu_fallbacks': {}, 'device_to_host_copies': 20, 'device_to_host_bytes': 10240,
#  'host_to_device_copies': 0, 'host_to_device_bytes': 0, 'scalar_reads': 0}
```

`cpu_fallbacks` lists operators that ran on the boxed CPU fallback; the copy
counters include transfers made inside operator implementations. With
`PTSYCL_STRICT=1` (or `torch_paras.set_strict(True)`) a CPU fallback raises.

## Limitations

* Without code generation the optimizer cannot make a captured graph faster
  than eager except through fusion; its gains come from fewer kernel
  launches.
* Models whose Python code branches on tensor values are split by TorchDynamo
  into several graphs. Param2's MoE dispatch (`moe_infer`) reads
  `tokens_per_expert.cpu().numpy()`, a graph break TorchDynamo reports as
  fundamental; those frames run eagerly (still on ParaS kernels). For such
  models eager mode is the faster path.

## Tests

`tests/test_optimizer.py`: fusion and non-fusion cases, bit-exactness,
inference and training parity on a LLaMA-style decoder block (AdamW, several
steps), dynamic sequence lengths, options, strict mode, and -- on GPU devices
-- zero CPU fallbacks and zero host transfers for a compiled training step.
