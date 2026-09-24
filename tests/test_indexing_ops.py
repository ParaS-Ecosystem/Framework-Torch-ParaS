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
