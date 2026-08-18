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
    idx = torch.randint(0, 4, (3,))
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


def test_index_put_accumulate_many_duplicates(device):
    # Large batch of indices with heavy duplication and a wide tail dim, so
    # that on a real GPU this spans many concurrent threads/blocks -- this
    # is the shape that exposes a racing (non-atomic) accumulate.
    rows, cols = 32, 256
    a = torch.zeros(rows, cols)
    da = a.to(device)
    idx = torch.randint(0, rows, (4096,))
    values = torch.randn(4096, cols)
    check_op(
        "index_put_accumulate_many_duplicates",
        lambda: a.clone().index_put_((idx,), values, accumulate=True),
        lambda: da.clone().index_put_((idx.to(device),), values.to(device),
                                      accumulate=True),
    )


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
