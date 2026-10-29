import torch

from common import check_op


def _pair(device):
    a = torch.randn(4, 5)
    b = torch.randn(4, 5)
    return a, b, a.to(device), b.to(device)


def test_gather(device):
    a, _, da, _ = _pair(device)
    idx = torch.randint(0, 5, (4, 5))
    check_op("gather", lambda: torch.gather(a, 1, idx),
             lambda: torch.gather(da, 1, idx.to(device)))


def test_index_select(device):
    a, _, da, _ = _pair(device)
    idx = torch.randint(0, 4, (3,))
    check_op("index_select", lambda: torch.index_select(a, 0, idx),
             lambda: torch.index_select(da, 0, idx.to(device)))


def test_index_copy_(device):
    a, _, da, _ = _pair(device)
    idx = torch.randperm(4)[:2]
    src = torch.randn(2, 5)
    check_op("index_copy_",
             lambda: a.clone().index_copy_(0, idx, src),
             lambda: da.clone().index_copy_(0, idx.to(device), src.to(device)))


def test_index_put_(device):
    a, _, da, _ = _pair(device)
    idx = torch.randperm(4)[:3]
    values = torch.randn(3, 5)
    check_op("index_put_",
             lambda: a.clone().index_put_((idx,), values),
             lambda: da.clone().index_put_((idx.to(device),), values.to(device)))


def test_index_put_accumulate(device):
    a, _, da, _ = _pair(device)
    idx = torch.tensor([0, 1, 0, 2])
    values = torch.randn(4, 5)
    check_op("index_put_accumulate",
             lambda: a.clone().index_put_((idx,), values, accumulate=True),
             lambda: da.clone().index_put_((idx.to(device),), values.to(device),
                                           accumulate=True))


def test_index_put_accumulate_dtypes(device):
    for dtype in [torch.float32, torch.float64, torch.int32, torch.int64]:
        base = torch.zeros(10, dtype=dtype)
        idx = torch.tensor([1, 2, 1, 3, 1, 2, 1, 3, 1], dtype=torch.long)
        vals = torch.ones(len(idx), dtype=dtype)
        check_op(f"index_put_accumulate_{dtype}",
                 lambda: base.clone().index_put_((idx,), vals, accumulate=True),
                 lambda: base.clone().to(device).index_put_((idx.to(device),), vals.to(device),
                                                               accumulate=True))


def test_index_put_accumulate_stress(device):
    # Stress test: N duplicate indices all accumulating into index 0
    # N < 2^24 guarantees exact representation in float32 without rounding error
    N = 10000
    for dtype in [torch.int32, torch.int64, torch.float32, torch.float64]:
        target = torch.zeros(4, dtype=dtype, device=device)
        idx = torch.zeros(N, dtype=torch.long, device=device)
        vals = torch.ones(N, dtype=dtype, device=device)
        target.index_put_((idx,), vals, accumulate=True)
        accumulated = target[0].item()
        if accumulated != N:
            raise AssertionError(f"Atomic race lost updates: expected {N}, got {accumulated} for dtype {dtype}")


def test_index_put_accumulate_parity_random(device):
    # Parity against stock CPU PyTorch on duplicate-heavy indices with multi-dim and negative indices
    torch.manual_seed(42)
    base = torch.randn(8, 12, dtype=torch.float32)
    # Duplicate-heavy indices with negative index wrapping
    idx_row = torch.tensor([0, 2, -1, 2, 0, 5, -1, 2, 0, 5], dtype=torch.long)
    idx_col = torch.tensor([1, 3, -2, 3, 1, 4, -2, 3, 1, 4], dtype=torch.long)
    vals = torch.randn(len(idx_row), dtype=torch.float32)
    check_op("index_put_accumulate_parity_random",
             lambda: base.clone().index_put_((idx_row, idx_col), vals, accumulate=True),
             lambda: base.clone().to(device).index_put_((idx_row.to(device), idx_col.to(device)),
                                                        vals.to(device), accumulate=True))


def test_index_put_accumulate_token_histogram(device):
    # Token histogram regression test for int64
    torch.manual_seed(123)
    vocab_size = 256
    num_tokens = 5000
    tokens = torch.randint(0, vocab_size, (num_tokens,), dtype=torch.long)
    ones = torch.ones(num_tokens, dtype=torch.long)

    hist_ref = torch.zeros(vocab_size, dtype=torch.long)
    hist_ref.index_put_((tokens,), ones, accumulate=True)

    hist_dev = torch.zeros(vocab_size, dtype=torch.long, device=device)
    hist_dev.index_put_((tokens.to(device),), ones.to(device), accumulate=True)

    if not torch.equal(hist_dev.cpu(), hist_ref):
        raise AssertionError("Token histogram (int64) accumulation mismatch against CPU reference")


def test_index_put_accumulate_widened_dtypes(device):
    for dtype in [torch.half, torch.bfloat16, torch.int8]:
        t = torch.zeros(10, dtype=dtype)
        idx = torch.tensor([0, 1, 0, 9, 0], dtype=torch.long)
        vals = torch.tensor([1, 2, 3, 4, 5], dtype=dtype)
        ref = t.clone().index_put_((idx,), vals, accumulate=True)
        out = t.to(device).index_put_((idx.to(device),), vals.to(device), accumulate=True)
        if not torch.equal(out.cpu(), ref):
            raise AssertionError(f"index_put_ accumulate mismatch for {dtype}: {out.cpu()} vs {ref}")


def test_index_tensor(device):
    a, _, da, _ = _pair(device)
    idx = torch.randint(0, 4, (6,))
    check_op("index.Tensor", lambda: a[idx], lambda: da[idx.to(device)])


def test_where(device):
    a, b, da, db = _pair(device)
    cond = a > 0
    check_op("where", lambda: torch.where(cond, a, b),
             lambda: torch.where(cond.to(device), da, db))


def test_triu(device):
    a, _, da, _ = _pair(device)
    check_op("triu", lambda: torch.triu(a, 1), lambda: torch.triu(da, 1))


def test_tril(device):
    a, _, da, _ = _pair(device)
    check_op("tril", lambda: torch.tril(a, -1), lambda: torch.tril(da, -1))


def test_index_rows_dim0(device):
    g = torch.Generator().manual_seed(0)
    cases = [((48, 2048), torch.float32), ((48, 2048), torch.bfloat16),
             ((40, 3), torch.float32), ((40, 6), torch.float16),
             ((40, 3), torch.bool), ((40,), torch.int64), ((20, 3, 5), torch.int8),
             ((30, 4, 8), torch.float64)]
    for shape, dt in cases:
        x = (torch.randn(shape, generator=g) * 10).to(dt)
        idx = torch.randint(-shape[0], shape[0], (37,), generator=g)
        got = x.to(device)[idx.to(device)]
        assert torch.equal(got.cpu(), x[idx]), (shape, dt, "gather")
        perm = torch.randperm(shape[0], generator=g)[:17]
        perm[::3] -= shape[0]
        vals = (torch.randn((17,) + shape[1:], generator=g) * 10).to(dt)
        ref = x.clone()
        ref[perm] = vals
        dx = x.to(device)
        dx[perm.to(device)] = vals.to(device)
        assert torch.equal(dx.cpu(), ref), (shape, dt, "scatter")
    x = torch.randn(8, 6, generator=g)
    idx = torch.tensor([5, 0, -1, 2])
    assert torch.equal(x.to(device).t()[idx.to(device)].cpu(), x.t()[idx])
    assert torch.equal(x.to(device)[:, idx.to(device)].cpu(), x[:, idx])


def test_index_writes_skip_out_of_range_indices(device):
    import torch_paras
    i = lambda *xs: torch.tensor(xs, device=device)
    rows = torch.tensor([[4] * 8, [1] * 8], device=device)
    ones = lambda *shape: torch.ones(*shape, device=device)
    writes = {
        "scatter_": lambda t: t.scatter_(0, rows, ones(2, 8)),
        "scatter_value": lambda t: t.scatter_(0, rows, 5.0),
        "scatter_add_": lambda t: t.scatter_add_(0, rows, ones(2, 8)),
        "index_add_": lambda t: t.index_add_(0, i(4, 1), ones(2, 8)),
        "index_put_rows": lambda t: t.__setitem__(i(4, 1), ones(2, 8)),
        "index_put_points": lambda t: t.__setitem__((i(4, 1), i(0, 2)), ones(2)),
        "index_put_accumulate": lambda t: t.index_put_((i(4, 1), i(0, 2)), ones(2), accumulate=True),
    }
    for name, write in writes.items():
        store = torch.zeros(2, 4, 8, device=device)
        write(store[0])
        torch_paras.synchronize()
        got = store.cpu()
        assert torch.equal(got[1], torch.zeros(4, 8)), f"{name} wrote outside the tensor"
        assert got[0][1].abs().sum() > 0, f"{name} dropped the in-range index"
        assert torch.equal(got[0][[0, 2, 3]], torch.zeros(3, 8)), f"{name} wrote to another row"
