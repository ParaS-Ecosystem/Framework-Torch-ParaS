"""paras::moe_experts and fuse_moe_experts: the device-side expert dispatch
computes what a DeepSeek-style moe_infer computes, on the device path (few
routes) and the per-expert path (many routes), in graph capture and without
graph breaks under Dynamo."""

import unittest

import torch
from torch import nn
import torch_paras

from common import fast_kernels
from torch_paras import graphs
from torch_paras.optimizer import fuse_moe_experts


class _MLP(nn.Module):
    def __init__(self, hidden, inter):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class _MoE(nn.Module):
    """The routed part of DeepSeek-V2/V3 and Param2 MoE layers."""

    def __init__(self, hidden=64, inter=32, experts=8, k=2):
        super().__init__()
        self.num_experts_per_tok = k
        self.gate = nn.Linear(hidden, experts, bias=False)
        self.experts = nn.ModuleList(_MLP(hidden, inter) for _ in range(experts))

    def forward(self, x):
        scores = torch.sigmoid(self.gate(x).float())
        w, ids = scores.topk(self.num_experts_per_tok, dim=-1)
        w = w / w.sum(-1, keepdim=True)
        return self.moe_infer(x, ids, w)

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        tokens_per_expert = tokens_per_expert.cpu().numpy()
        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            outputs.append(self.experts[i](sorted_tokens[start_idx:end_idx]))
            start_idx = end_idx
        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        return (new_x.view(*topk_ids.shape, -1).type(topk_weight.dtype)
                .mul_(topk_weight.unsqueeze(dim=-1)).sum(dim=1).type(new_x.dtype))


class _Experts(nn.Module):
    """transformers' Experts layout and loop (Mixtral, Qwen2-MoE, DeepSeek-V3, ...)."""

    def __init__(self, hidden=64, inter=32, experts=8):
        super().__init__()
        self.num_experts = experts
        self.gate_up_proj = nn.Parameter(torch.randn(experts, 2 * inter, hidden) / 8)
        self.down_proj = nn.Parameter(torch.randn(experts, hidden, inter) / 6)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = nn.functional.one_hot(top_k_index, num_classes=self.num_experts + 1)[..., :self.num_experts]
            mask = mask.permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for e in hit:
            e = e[0]
            pos, tok = torch.where(mask[e])
            gate, up = nn.functional.linear(hidden_states[tok], self.gate_up_proj[e]).chunk(2, dim=-1)
            h = nn.functional.linear(self.act_fn(gate) * up, self.down_proj[e])
            final.index_add_(0, tok, (h * top_k_weights[tok, pos, None]).to(final.dtype))
        return final


def _model(device, dtype, **kw):
    torch.manual_seed(0)
    return _MoE(**kw).to(device, dtype).eval()


def _routing(tokens, experts, k, device):
    ids = torch.rand(tokens, experts).topk(k, dim=-1).indices.to(device)
    w = torch.rand(tokens, k).softmax(-1).to(device)
    return ids, w


def _rel(a, b):
    a, b = a.float().cpu(), b.float().cpu()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def test_moe_experts_matches_moe_infer(device):
    for dtype, tol in ((torch.float32, 1e-5), (torch.bfloat16, 1e-2)):
        m = _model(device, dtype)
        original = m.moe_infer
        assert fuse_moe_experts(m) == 1
        for tokens in (1, 3, 16, 400):
            ids, w = _routing(tokens, 8, 2, device)
            x = torch.randn(tokens, 64).to(device, dtype)
            with torch.no_grad():
                got, ref = m.moe_infer(x, ids, w), original(x, ids, w)
            assert got.dtype == dtype and got.shape == ref.shape
            err = _rel(got, ref)
            if tokens == 400:
                assert torch.equal(got.cpu(), ref.cpu()), f"{dtype} per-expert path differs ({err})"
            assert err < tol, f"{dtype} tokens={tokens}: relative error {err}"


def test_moe_experts_unused_and_repeated_experts(device):
    m = _model(device, torch.float32)
    original = m.moe_infer
    fuse_moe_experts(m)
    ids = torch.tensor([[3, 5], [3, 5], [5, 3], [0, 3]], device=device)
    w = torch.rand(4, 2).to(device)
    x = torch.randn(4, 64).to(device)
    with torch.no_grad():
        assert _rel(m.moe_infer(x, ids, w), original(x, ids, w)) < 1e-5


def test_fuse_moe_experts_keeps_parameters(device):
    m = _model(device, torch.float32)
    before = [e.up_proj.weight.detach().cpu().clone() for e in m.experts]
    params = [e.up_proj.weight for e in m.experts]
    fuse_moe_experts(m)
    assert [e.up_proj.weight for e in m.experts] == params
    for b, e in zip(before, m.experts):
        assert torch.equal(e.up_proj.weight.detach().cpu(), b)
    assert m.experts[1].up_proj.weight.data_ptr() == m._paras_moe_up[1].data_ptr()
    assert "_paras_moe_up" not in m.state_dict()
    assert fuse_moe_experts(m) == 0


def test_fuse_moe_experts_releases_former_weights(device):
    m = _model(device, torch.float32, hidden=256, inter=512, experts=16)
    allocated = torch_paras.memory_allocated(torch.device(device))
    fuse_moe_experts(m)
    assert torch_paras.memory_allocated(torch.device(device)) <= allocated
    cached = torch_paras.memory_reserved(torch.device(device)) - torch_paras.memory_allocated(torch.device(device))
    assert cached < 3 * 16 * 512 * 256 * 4, cached


def test_fuse_moe_experts_skips_other_layers(device):
    m = _model(device, torch.float32)
    m.experts[2].act_fn = nn.GELU()
    assert fuse_moe_experts(m) == 0
    m = _model(device, torch.float32)
    m.ep_size = 2
    assert fuse_moe_experts(m) == 0


def test_moe_experts_cpu_kernel(device):
    m = _model("cpu", torch.float32)
    original = m.moe_infer
    ids, w = _routing(5, 8, 2, "cpu")
    x = torch.randn(5, 64)
    with torch.no_grad():
        got = torch.ops.paras.moe_experts(
            x, ids, w, torch.stack([e.gate_proj.weight for e in m.experts]),
            torch.stack([e.up_proj.weight for e in m.experts]),
            torch.stack([e.down_proj.weight for e in m.experts]))
        assert torch.equal(got, original(x, ids, w))


def test_moe_experts_needs_no_grad(device):
    m = _model(device, torch.float32)
    fuse_moe_experts(m)
    ids, w = _routing(2, 8, 2, device)
    x = torch.randn(2, 64).to(device)
    w.requires_grad_()
    try:
        torch.ops.paras.moe_experts(x, ids, w, m._paras_moe_gate, m._paras_moe_up,
                                    m._paras_moe_down)
    except RuntimeError as e:
        assert "inference" in str(e)
    else:
        raise AssertionError("expected an error with weights that require grad")


def _device_path(device) -> bool:
    """paras::moe_experts' device path (no host read, capturable): GPUs with
    64-lane (CDNA) or 32-lane (NVIDIA) wavefronts, generic kernels not forced."""
    index = torch.device(device).index or 0
    caps = torch_paras.device_kernel_caps(index)
    return torch_paras.is_gpu_device(index) and not caps["generic"] and caps["wave_size"] in (32, 64)


def test_moe_experts_graph_capture(device):
    if not _device_path(device):
        raise unittest.SkipTest("the device path (capturable) needs 32- or 64-lane wavefronts")
    if not graphs.is_supported():
        raise unittest.SkipTest("ParaS without record-and-replay command graphs")
    m = _model(device, torch.bfloat16)
    original = m.moe_infer
    fuse_moe_experts(m)
    ids, w = _routing(4, 8, 2, device)
    x = torch.randn(4, 64).to(device, torch.bfloat16)
    with torch.no_grad():
        m.moe_infer(x, ids, w)
        g = graphs.Graph()
        with graphs.graph(g, device=device):
            out = m.moe_infer(x, ids, w)
        for _ in range(3):
            new_ids, new_w = _routing(4, 8, 2, device)
            new_x = torch.randn(4, 64).to(device, torch.bfloat16)
            ids.copy_(new_ids)
            w.copy_(new_w)
            x.copy_(new_x)
            g.replay()
            assert _rel(out, original(new_x, new_ids, new_w)) < 1e-2
    g.reset()


def test_fused_moe_compiles_without_graph_break(device):
    m = _model(device, torch.bfloat16)
    x = torch.randn(3, 64).to(device, torch.bfloat16)
    torch._dynamo.reset()
    before = torch._dynamo.explain(m)(x)
    assert before.graph_break_count > 0
    fuse_moe_experts(m)
    torch._dynamo.reset()
    after = torch._dynamo.explain(m)(x)
    assert after.graph_break_count == 0 and after.graph_count == 1, after
    torch._dynamo.reset()
    with torch.no_grad():
        ref = m(x)
        got = torch_paras.compile(m)(x)
    assert _rel(got, ref) < 1e-2
    torch._dynamo.reset()


def test_experts_module_matches_forward(device):
    for dtype, tol in ((torch.float32, 1e-5), (torch.bfloat16, 2e-2)):
        torch.manual_seed(0)
        m = _Experts().to(device, dtype)
        original = m.forward
        assert fuse_moe_experts(m) == 1
        for tokens in (1, 5, 400):
            ids, w = _routing(tokens, 8, 2, device)
            ids[0, 1] = 8
            x = torch.randn(tokens, 64).to(device, dtype)
            w = w.to(dtype)
            with torch.no_grad():
                err = _rel(m(x, ids, w), original(x, ids, w))
            assert err < tol, f"{dtype} tokens={tokens}: relative error {err}"


def test_experts_module_keeps_autograd(device):
    m = _Experts().to(device)
    fuse_moe_experts(m)
    ids, w = _routing(4, 8, 2, device)
    x = torch.randn(4, 64).to(device)
    m(x, ids, w).sum().backward()
    assert m.gate_up_proj.grad is not None and m.down_proj.grad is not None


class _W(nn.Module):
    def __init__(self, hidden, inter):
        super().__init__()
        self.w1 = nn.Linear(hidden, inter, bias=False)
        self.w2 = nn.Linear(inter, hidden, bias=False)
        self.w3 = nn.Linear(hidden, inter, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.w2(self.act_fn(self.w1(x)) * self.w3(x))


class _SparseMoeBlock(nn.Module):
    """transformers' Mixtral / Qwen2-MoE block before 5.0 (expert loop over torch.where)."""

    def __init__(self, hidden=64, inter=32, experts=8, k=2, shared=False, norm=True):
        super().__init__()
        self.top_k = k
        self.norm_topk_prob = norm
        self.gate = nn.Linear(hidden, experts, bias=False)
        self.experts = nn.ModuleList(_W(hidden, inter) for _ in range(experts))
        if shared:
            self.shared_expert = _MLP(hidden, 2 * inter)
            self.shared_expert_gate = nn.Linear(hidden, 1, bias=False)

    def forward(self, hidden_states):
        b, s, h = hidden_states.shape
        x = hidden_states.view(-1, h)
        router_logits = self.gate(x)
        w = torch.softmax(router_logits, dim=1, dtype=torch.float)
        w, sel = torch.topk(w, self.top_k, dim=-1)
        if self.norm_topk_prob:
            w /= w.sum(dim=-1, keepdim=True)
        w = w.to(x.dtype)
        out = torch.zeros_like(x)
        mask = nn.functional.one_hot(sel, num_classes=len(self.experts)).permute(2, 1, 0)
        for e in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero():
            idx, top_x = torch.where(mask[e].squeeze(0))
            y = self.experts[e](x[None, top_x].reshape(-1, h)) * w[top_x, idx, None]
            out.index_add_(0, top_x, y.to(x.dtype))
        if hasattr(self, "shared_expert"):
            out = out + torch.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
        return out.reshape(b, s, h), router_logits


def test_sparse_moe_block_matches_forward(device):
    for shared, norm in ((False, True), (True, False)):
        torch.manual_seed(0)
        m = _SparseMoeBlock(shared=shared, norm=norm).to(device).eval()
        original = m.forward
        assert fuse_moe_experts(m) == 1
        x = torch.randn(2, 5, 64).to(device)
        with torch.no_grad():
            (got, lg), (ref, lr) = m(x), original(x)
        assert torch.equal(lg.cpu(), lr.cpu())
        assert _rel(got, ref) < 1e-5, (shared, norm, _rel(got, ref))
        m.train()
        m(x)[0].sum().backward()
        assert m.experts[0].w1.weight.grad is not None


def test_transformers_moe_models_without_graph_break(device):
    try:
        import transformers as tr
    except ImportError:
        raise unittest.SkipTest("transformers is not installed")
    text = dict(vocab_size=500, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    makers = {
        "mixtral": lambda: tr.MixtralForCausalLM(tr.MixtralConfig(**text, num_local_experts=4, num_experts_per_tok=2)),
        "qwen2_moe": lambda: tr.Qwen2MoeForCausalLM(tr.Qwen2MoeConfig(
            **text, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=64,
            shared_expert_intermediate_size=128)),
    }
    if hasattr(tr, "DeepseekV3ForCausalLM"):
        makers["deepseek_v3"] = lambda: tr.DeepseekV3ForCausalLM(tr.DeepseekV3Config(
            vocab_size=500, hidden_size=128, intermediate_size=256, moe_intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4, n_routed_experts=4,
            num_experts_per_tok=2, n_group=2, topk_group=1, first_k_dense_replace=1, q_lora_rank=32,
            kv_lora_rank=32, qk_rope_head_dim=16, qk_nope_head_dim=16, v_head_dim=32,
            max_position_embeddings=128))
    ids = torch.randint(0, 500, (2, 12)).to(device)
    for name, make in makers.items():
        torch.manual_seed(0)
        m = make().to(device).eval()
        with torch.no_grad():
            ref = m(input_ids=ids, use_cache=False).logits
        assert fuse_moe_experts(m) >= 1, name
        torch._dynamo.reset()
        with torch.no_grad():
            got = m(input_ids=ids, use_cache=False).logits
            ex = torch._dynamo.explain(m)(input_ids=ids, use_cache=False)
        torch._dynamo.reset()
        assert _rel(got, ref) < 1e-4, (name, _rel(got, ref))
        assert ex.graph_break_count == 0, (name, [str(b.reason).splitlines()[0] for b in ex.break_reasons])


def test_moe_experts_grouped_gemm(device):
    if not fast_kernels(device):
        raise unittest.SkipTest("the grouped GEMM runs on AMD matrix cores")
    for dtype in (torch.bfloat16, torch.float16):
        m = _model(device, dtype, hidden=256, inter=128, experts=8, k=2)
        original = m.moe_infer
        fuse_moe_experts(m)
        for tokens, skew in ((400, False), (1000, False), (700, True)):
            ids, w = _routing(tokens, 8, 2, device)
            if skew:
                bias = torch.zeros(8)
                bias[5], bias[3] = 2.0, -float("inf")
                ids = (torch.rand(tokens, 8) + bias).topk(2, dim=-1).indices.to(device)
            x = torch.randn(tokens, 256).to(device, dtype)
            with torch.no_grad():
                torch_paras.reset_runtime_stats()
                got = m.moe_infer(x, ids, w)
                reads = torch_paras.runtime_stats().get("device_to_host_copies", 0)
                ref = original(x, ids, w)
            assert reads == 0, (dtype, tokens, "host reads", reads)
            err = _rel(got, ref)
            assert err < 1e-2, f"{dtype} tokens={tokens} skew={skew}: relative error {err}"


def test_moe_experts_grouped_gemm_graph_capture(device):
    if not fast_kernels(device):
        raise unittest.SkipTest("the grouped GEMM runs on AMD matrix cores")
    if not graphs.is_supported():
        raise unittest.SkipTest("ParaS without record-and-replay command graphs")
    m = _model(device, torch.bfloat16, hidden=256, inter=128, experts=8, k=2)
    original = m.moe_infer
    fuse_moe_experts(m)
    ids, w = _routing(600, 8, 2, device)
    x = torch.randn(600, 256).to(device, torch.bfloat16)
    with torch.no_grad():
        m.moe_infer(x, ids, w)
        g = graphs.Graph()
        with graphs.graph(g, device=device):
            out = m.moe_infer(x, ids, w)
        for _ in range(2):
            new_ids, new_w = _routing(600, 8, 2, device)
            new_x = torch.randn(600, 256).to(device, torch.bfloat16)
            ids.copy_(new_ids)
            w.copy_(new_w)
            x.copy_(new_x)
            g.replay()
            assert _rel(out, original(new_x, new_ids, new_w)) < 1e-2
    g.reset()
