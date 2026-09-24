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


def test_index_put_accumulate_unsupported_dtypes(device):
    # Negative test: Half, BFloat16, int8 must raise RuntimeError
    for dtype in [torch.half, torch.bfloat16, torch.int8]:
        t = torch.zeros(10, dtype=dtype, device=device)
        idx = torch.tensor([0, 1, 0], dtype=torch.long, device=device)
        vals = torch.ones(3, dtype=dtype, device=device)
        raised = False
        try:
            t.index_put_((idx,), vals, accumulate=True)
        except RuntimeError:
            raised = True
        if not raised:
            raise AssertionError(f"Expected RuntimeError for unsupported dtype {dtype} in accumulate=True, but none raised")


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
