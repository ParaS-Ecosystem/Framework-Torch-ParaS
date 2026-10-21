# Graph capture and replay

Torch-ParaS can record the work a piece of code queues on a paras GPU device
and replay it with a single launch, like CUDA graphs. A replay skips the
per-operator host work -- Python, the PyTorch dispatcher, Torch-ParaS and the
kernel launch -- which dominates small-batch inference (a decode step of a
large model issues thousands of small kernels).

The mechanism lives in ParaS: record-and-replay command graphs
(`sycl_ext_oneapi_graph`, `sycl/graph.hpp`; on HIP and CUDA a stream capture
of the queue's stream). Vendor-library calls Torch-ParaS makes on the queue's
stream (rocBLAS / cuBLAS through CLAP) are recorded too. A ParaS without
command graphs builds fine; `torch_paras.graphs.is_supported()` then returns
False and `graphs=True` replays nothing.

## Direct use: `torch_paras.graphs`

```python
import torch, torch_paras
from torch_paras import graphs

g = graphs.Graph()
static_x = torch.zeros(8, 4096, device="paras")
with torch.no_grad():
    model(static_x)                          # warm up outside the capture
    with graphs.graph(g):
        static_y = model(static_x)           # recorded, not run
static_x.copy_(x)
g.replay()                                   # static_y now holds model(x)
```

Rules, as for CUDA graphs:

* A replay reads and writes the memory of the capture and uses its kernel
  arguments: inputs go in by copying into the captured tensors, results come
  out in the captured outputs, and the next replay overwrites them.
* Host decisions taken during the capture are fixed: shapes, Python control
  flow, random seeds (Torch-ParaS draws them on the host).
* Work that needs the host raises during the capture and the capture fails:
  `synchronize()`, `.item()`, copies to or from CPU tensors, and operators
  Torch-ParaS implements on the host (today: LayerNorm).
* Memory allocated during a capture comes from the graph's own pool and is
  never given to other tensors. Graphs that are replayed in capture order may
  share a pool (`graphs.new_pool()`, `graph(g, pool=...)`).

## In the compiler: `torch_paras.compile(model, graphs=True)`

With `graphs=True` the paras backend replays its inference graphs
(`optimizer/replay.py`):

* **Per input shape.** For each new set of input shapes a graph first runs
  normally (lazy initialisation happens outside any capture) and is captured
  when the shape comes again; up to 8 shapes are kept per graph. A replay
  copies the inputs into the capture's buffers -- parameters and tensors
  passed to `torch._dynamo.mark_static_address` are used in place, and so are
  the outputs of another capture -- copies inputs the graph mutates back, and
  returns copies of the outputs that leave the graph.
* **Dynamic shapes in pieces.** A graph with symbolic sizes runs normally for
  its first 3 calls while the backend records the values of those sizes. A
  size that changes between calls (a KV cache that grows every token) is
  volatile; the operators that involve no volatile size are replayed in
  pieces, the others -- appending to the cache, attention over it -- run
  normally. Sizes that are symbolic but steady (batch, query length) do not
  stop a piece: it is captured per shape.
* **Not replayed** (`GraphReport.replay` says why): graphs that need
  autograd, use random operators (attention without dropout is not random),
  return their inputs or views of them, change input metadata, have inputs on
  several devices or on the host, or fail to capture -- those run normally.
* `PTSYCL_GRAPHS_VERIFY=1` runs every replayed graph a second time normally
  and records outputs that differ (`optimizer.replay.mismatches`); a
  debugging aid.

For transformers' `generate()`, compile `forward` in place:

```python
from torch_paras import optimizer
optimizer.specialize_repeated_blocks(model)
model.forward = torch.compile(
    model.forward, backend=optimizer.make_backend(optimizer.Options(graphs=True)))
```

## Results (MI300X, batch 1, decode ms per token)

| Model, cache | eager | compiled | compiled + graphs |
|---|---|---|---|
| LLaMA architecture, 1.1B shapes, StaticCache | 16.4 | 11.7 | **10.4** |
| same, DynamicCache | 16.3 | 9.15 | 9.54 |
| Param2-17B (MoE), DynamicCache | 34.7 | 32.5 | 33.2 |

One process per configuration (`bench/dense_decode.py`) for the dense model;
Param2 alternates the three modes in one process (`bench/ab_graphs.py`).
Tokens equal eager for every replayed run.

* With a static cache the whole decode step of the dense model is one
  replay: the host cost per token is gone, and what remains is GPU time.
* With a growing cache each layer's attention splits the step into ~45
  pieces; their replays save about what feeding them costs.
* Param2's MoE layers read `tokens_per_expert` on the host
  (`.cpu().numpy()`), which splits every layer into separate graphs of ~29
  operators with attention in the middle, and its expert loop (~40 % of the
  operators of a token) runs eagerly. Replay cannot gain there without a
  graph-break-free MoE dispatch in the model.
* The replays leave GPU kernel time: per token of the dense model ParaS
  spends ~2.2x native ROCm's GPU time (attention kernel for decode, rocBLAS
  GEMM tile choice for M = 1, RMSNorm as three kernels). That is kernel work,
  not something graphs change.

## Limitations

* Captures need ParaS command graphs (CUDA or HIP; CUDA is compile-checked
  only). On the host CPU engine graphs are not available.
* Operators Torch-ParaS implements on the host (LayerNorm today) cannot be
  captured; graphs that use them run normally.
* Random operators are not replayed (their seeds are drawn on the host).
