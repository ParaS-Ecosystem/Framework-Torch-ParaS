import torch

from common import check_op


def _expect_raises(fn, exc_type=RuntimeError):
    """Local substitute for pytest.raises so this file has no pytest
    dependency (tests/run_all.py is meant to run standalone)."""
    try:
        fn()
    except exc_type:
        return
    except Exception as e:
        raise AssertionError(
            f"expected {exc_type.__name__}, got {type(e).__name__}: {e}"
        ) from e
    else:
        raise AssertionError(
            f"expected {exc_type.__name__} but no exception was raised"
        )


# -----------------------------------------------------------------------------
# matmul / mm / bmm  (only meaningful once csrc/ops/matmul_ops.cpp is wired
# in -- see the warning at the top of that file. Skip cleanly if not.)
# -----------------------------------------------------------------------------

def test_mm(device):
    a = torch.randn(4, 6)
    b = torch.randn(6, 5)
    da, db = a.to(device), b.to(device)
    check_op("mm", lambda: torch.mm(a, b), lambda: torch.mm(da, db))


def test_bmm(device):
    a = torch.randn(3, 4, 6)
    b = torch.randn(3, 6, 5)
    da, db = a.to(device), b.to(device)
    check_op("bmm", lambda: torch.bmm(a, b), lambda: torch.bmm(da, db))


def test_matmul_1d_1d(device):
    a = torch.randn(8)
    b = torch.randn(8)
    da, db = a.to(device), b.to(device)
    check_op("matmul_1d_1d", lambda: torch.matmul(a, b), lambda: torch.matmul(da, db))


def test_matmul_1d_2d(device):
    a = torch.randn(6)
    b = torch.randn(6, 5)
    da, db = a.to(device), b.to(device)
    check_op("matmul_1d_2d", lambda: torch.matmul(a, b), lambda: torch.matmul(da, db))


def test_matmul_2d_1d(device):
    a = torch.randn(4, 6)
    b = torch.randn(6)
    da, db = a.to(device), b.to(device)
    check_op("matmul_2d_1d", lambda: torch.matmul(a, b), lambda: torch.matmul(da, db))


def test_matmul_2d_2d(device):
    a = torch.randn(4, 6)
    b = torch.randn(6, 5)
    da, db = a.to(device), b.to(device)
    check_op("matmul_2d_2d", lambda: torch.matmul(a, b), lambda: torch.matmul(da, db))


def test_matmul_batched(device):
    a = torch.randn(2, 3, 4, 6)
    b = torch.randn(2, 3, 6, 5)
    da, db = a.to(device), b.to(device)
    check_op("matmul_batched", lambda: torch.matmul(a, b), lambda: torch.matmul(da, db))


def test_matmul_broadcast_batch(device):
    # (2, 1, 4, 6) @ (3, 6, 5) broadcasts to (2, 3, 4, 5)
    a = torch.randn(2, 1, 4, 6)
    b = torch.randn(3, 6, 5)
    da, db = a.to(device), b.to(device)
    check_op("matmul_broadcast_batch",
             lambda: torch.matmul(a, b), lambda: torch.matmul(da, db))


def test_matmul_batched_vec(device):
    # (B, n, k) @ (k,) -> (B, n)
    a = torch.randn(3, 4, 6)
    b = torch.randn(6)
    da, db = a.to(device), b.to(device)
    check_op("matmul_batched_vec",
             lambda: torch.matmul(a, b), lambda: torch.matmul(da, db))


def test_mm_shape_mismatch_raises(device):
    a = torch.randn(4, 6).to(device)
    b = torch.randn(5, 5).to(device)
    _expect_raises(lambda: torch.mm(a, b))


# -----------------------------------------------------------------------------
# argsort
# -----------------------------------------------------------------------------

def test_argsort_1d(device):
    x = torch.randn(20)
    dx = x.to(device)
    check_op("argsort_1d", lambda: torch.argsort(x), lambda: torch.argsort(dx))


def test_argsort_last_dim(device):
    x = torch.randn(5, 12)
    dx = x.to(device)
    check_op("argsort_last_dim", lambda: torch.argsort(x, dim=-1),
              lambda: torch.argsort(dx, dim=-1))


def test_argsort_middle_dim(device):
    x = torch.randn(3, 10, 4)
    dx = x.to(device)
    check_op("argsort_middle_dim", lambda: torch.argsort(x, dim=1),
              lambda: torch.argsort(dx, dim=1))


def test_argsort_descending(device):
    x = torch.randn(6, 15)
    dx = x.to(device)
    check_op("argsort_descending",
              lambda: torch.argsort(x, dim=-1, descending=True),
              lambda: torch.argsort(dx, dim=-1, descending=True))


def test_argsort_with_ties(device):
    # Integer dtype guarantees ties -- exercises the stable-vs-not distinction.
    x = torch.randint(0, 4, (8, 20))
    dx = x.to(device)
    check_op("argsort_ties_stable",
              lambda: torch.argsort(x, dim=-1, stable=True),
              lambda: torch.argsort(dx, dim=-1, stable=True))


def test_argsort_matches_sorted_order(device):
    # Cross-check property (not just CPU-vs-device parity): gathering x
    # by its own argsort must be non-decreasing.
    x = torch.randn(50).to(device)
    idx = torch.argsort(x)
    gathered = x.cpu()[idx.cpu()]
    assert torch.all(gathered[1:] >= gathered[:-1])


# -----------------------------------------------------------------------------
# tensor creation wrappers
# -----------------------------------------------------------------------------

def test_zeros(device):
    check_op("zeros", lambda: torch.zeros(4, 5),
              lambda: torch.zeros(4, 5, device=device))


def test_ones(device):
    check_op("ones", lambda: torch.ones(4, 5),
              lambda: torch.ones(4, 5, device=device))


def test_full(device):
    check_op("full", lambda: torch.full((4, 5), 3.14),
              lambda: torch.full((4, 5), 3.14, device=device))


def test_zeros_like(device):
    x = torch.randn(4, 5)
    dx = x.to(device)
    check_op("zeros_like", lambda: torch.zeros_like(x), lambda: torch.zeros_like(dx))


def test_ones_like(device):
    x = torch.randn(4, 5)
    dx = x.to(device)
    check_op("ones_like", lambda: torch.ones_like(x), lambda: torch.ones_like(dx))


def test_full_like(device):
    x = torch.randn(4, 5)
    dx = x.to(device)
    check_op("full_like", lambda: torch.full_like(x, -1.0),
              lambda: torch.full_like(dx, -1.0))


def test_arange(device):
    check_op("arange", lambda: torch.arange(0, 10, 2),
              lambda: torch.arange(0, 10, 2, device=device))


def test_arange_float_step(device):
    check_op("arange_float_step", lambda: torch.arange(0.0, 2.0, 0.25),
              lambda: torch.arange(0.0, 2.0, 0.25, device=device))