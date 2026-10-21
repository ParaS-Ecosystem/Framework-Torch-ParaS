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

## Limitations

* Captures need ParaS command graphs (CUDA or HIP; CUDA is compile-checked
  only). On the host CPU engine graphs are not available.
* Operators Torch-ParaS implements on the host (LayerNorm today) cannot be
  captured; graphs that use them run normally.
* Random operators are not replayed (their seeds are drawn on the host).
