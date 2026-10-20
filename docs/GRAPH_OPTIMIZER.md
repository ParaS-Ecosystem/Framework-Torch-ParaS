# Graph optimizer (`torch.compile(backend="paras")`)

`torch_paras.optimizer` is a TorchDynamo backend for paras devices. It works
for inference and training and never changes what a model computes.

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
2. **AOTAutograd** produces functionalised ATen forward and backward graphs.
   Fused kernels carry registered backward formulas (`optimizer/ops.py`), so
   the backward of a fused RMSNorm is `paras::rms_norm_backward`.
3. **Residency analysis** (`optimizer/residency.py`) classifies every ATen
   operator of each graph from the dispatcher tables:
   * `paras`: ParaS registers a PrivateUse1 kernel (or the op is a structured
     functional form whose out= variant has one);
   * `composite`: a PyTorch composite that decomposes into other operators,
     each dispatched (and counted, see below) when it runs;
   * `cpu`: no ParaS kernel; the op would run on the boxed CPU fallback.
   With `strict=True` a graph containing a `cpu` op raises
   `ParasCompileError`.
4. The ATen graphs execute operator by operator on the ParaS kernels -- the
   same kernels eager mode uses.

`torch_paras.optimizer.reports()` returns one `GraphReport` per captured
graph: the fusions applied and the residency counts of its forward, backward
or inference graph.

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
