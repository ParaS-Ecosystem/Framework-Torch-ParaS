import torch
import torch.nn.functional as F

from common import check_op, assert_close


def _pair(device):
    a = torch.randn(4, 5)
    b = torch.randn(4, 5)
    return a, b, a.to(device), b.to(device)


# --- unary ------------------------------------------------------------------

def test_relu_(device):
    a, _, da, _ = _pair(device)
    check_op("relu_", lambda: torch.relu_(a.clone()),
             lambda: torch.relu_(da.clone()))


def test_exp(device):
    a, _, da, _ = _pair(device)
    check_op("exp", lambda: torch.exp(a), lambda: torch.exp(da))


def test_log(device):
    a, _, da, _ = _pair(device)
    check_op("log", lambda: torch.log(torch.abs(a) + 1.0),
             lambda: torch.log(torch.abs(da) + 1.0))


def test_sqrt(device):
    a, _, da, _ = _pair(device)
    check_op("sqrt", lambda: torch.sqrt(torch.abs(a) + 1.0),
             lambda: torch.sqrt(torch.abs(da) + 1.0))


def test_sin(device):
    a, _, da, _ = _pair(device)
    check_op("sin", lambda: torch.sin(a), lambda: torch.sin(da))


def test_cos(device):
    a, _, da, _ = _pair(device)
    check_op("cos", lambda: torch.cos(a), lambda: torch.cos(da))


def test_sigmoid(device):
    a, _, da, _ = _pair(device)
    check_op("sigmoid", lambda: torch.sigmoid(a), lambda: torch.sigmoid(da))


def test_tanh(device):
    a, _, da, _ = _pair(device)
    check_op("tanh", lambda: torch.tanh(a), lambda: torch.tanh(da))


def test_abs(device):
    a, _, da, _ = _pair(device)
    check_op("abs", lambda: torch.abs(a), lambda: torch.abs(da))


def test_neg(device):
    a, _, da, _ = _pair(device)
    check_op("neg", lambda: torch.neg(a), lambda: torch.neg(da))


def test_reciprocal(device):
    a, _, da, _ = _pair(device)
    check_op("reciprocal", lambda: torch.reciprocal(a + 2.0),
             lambda: torch.reciprocal(da + 2.0))


def test_ceil(device):
    a, _, da, _ = _pair(device)
    check_op("ceil", lambda: torch.ceil(a), lambda: torch.ceil(da))


def test_round(device):
    a, _, da, _ = _pair(device)
    check_op("round", lambda: torch.round(a), lambda: torch.round(da))


def test_atan(device):
    a, _, da, _ = _pair(device)
    check_op("atan", lambda: torch.atan(a), lambda: torch.atan(da))


def test_gelu(device):
    a, _, da, _ = _pair(device)
    check_op("gelu", lambda: F.gelu(a), lambda: F.gelu(da))


def test_silu(device):
    a, _, da, _ = _pair(device)
    check_op("silu", lambda: F.silu(a), lambda: F.silu(da))


def test_log_sigmoid(device):
    a, _, da, _ = _pair(device)
    check_op("logsigmoid", lambda: F.logsigmoid(a), lambda: F.logsigmoid(da))


# --- binary -----------------------------------------------------------------

def test_add(device):
    a, b, da, db = _pair(device)
    check_op("add", lambda: a + b, lambda: da + db)


def test_sub(device):
    a, b, da, db = _pair(device)
    check_op("sub", lambda: a - b, lambda: da - db)


def test_mul(device):
    a, b, da, db = _pair(device)
    check_op("mul", lambda: a * b, lambda: da * db)


def test_div(device):
    a, b, da, db = _pair(device)
    check_op("div", lambda: (a + 3) / (b + 3), lambda: (da + 3) / (db + 3))


def test_maximum(device):
    a, b, da, db = _pair(device)
    check_op("maximum", lambda: torch.maximum(a, b),
             lambda: torch.maximum(da, db))


def test_minimum(device):
    a, b, da, db = _pair(device)
    check_op("minimum", lambda: torch.minimum(a, b),
             lambda: torch.minimum(da, db))


def test_pow(device):
    a, _, da, _ = _pair(device)
    check_op("pow", lambda: torch.pow(a, 2.0), lambda: torch.pow(da, 2.0))


def test_dot(device):
    a, b, da, db = _pair(device)
    check_op("dot", lambda: torch.dot(a.flatten(), b.flatten()),
             lambda: torch.dot(da.flatten(), db.flatten()))


def test_clamp(device):
    a, _, da, _ = _pair(device)
    check_op("clamp", lambda: torch.clamp(a, -1, 1),
             lambda: torch.clamp(da, -1, 1))


def test_clamp_min(device):
    a, _, da, _ = _pair(device)
    check_op("clamp_min", lambda: torch.clamp_min(a, 0.0),
             lambda: torch.clamp_min(da, 0.0))


# --- reductions ---------------------------------------------------------------

def test_mean(device):
    a, _, da, _ = _pair(device)
    check_op("mean", lambda: torch.mean(a), lambda: torch.mean(da))


def test_sum_dim(device):
    a, _, da, _ = _pair(device)
    check_op("sum.dim", lambda: torch.sum(a, dim=1),
             lambda: torch.sum(da, dim=1))


def test_amax(device):
    a, _, da, _ = _pair(device)
    check_op("amax", lambda: torch.amax(a, dim=1), lambda: torch.amax(da, dim=1))


def test_amin(device):
    a, _, da, _ = _pair(device)
    check_op("amin", lambda: torch.amin(a, dim=1), lambda: torch.amin(da, dim=1))


def test_argmax(device):
    a, _, da, _ = _pair(device)
    check_op("argmax", lambda: torch.argmax(a, dim=1),
             lambda: torch.argmax(da, dim=1))


# --- bitwise ------------------------------------------------------------------

def _bool_pair(device):
    a = torch.randint(0, 2, (4, 5), dtype=torch.bool)
    b = torch.randint(0, 2, (4, 5), dtype=torch.bool)
    return a, b, a.to(device), b.to(device)


def test_bitwise_and(device):
    a, b, da, db = _bool_pair(device)
    check_op("and", lambda: torch.bitwise_and(a, b),
             lambda: torch.bitwise_and(da, db))


def test_bitwise_or(device):
    a, b, da, db = _bool_pair(device)
    check_op("or", lambda: torch.bitwise_or(a, b),
             lambda: torch.bitwise_or(da, db))


def test_bitwise_xor(device):
    a, b, da, db = _bool_pair(device)
    check_op("xor", lambda: torch.bitwise_xor(a, b),
             lambda: torch.bitwise_xor(da, db))


def test_bitwise_not(device):
    a, _, da, _ = _bool_pair(device)
    check_op("not", lambda: torch.bitwise_not(a),
             lambda: torch.bitwise_not(da))


# --- softmax ------------------------------------------------------------------

def test_softmax_dim1(device):
    a, _, da, _ = _pair(device)
    check_op("softmax.dim1", lambda: torch.softmax(a, dim=1),
             lambda: torch.softmax(da, dim=1))


def test_softmax_dim0(device):
    a, _, da, _ = _pair(device)
    check_op("softmax.dim0", lambda: torch.softmax(a, dim=0),
             lambda: torch.softmax(da, dim=0))


def test_softmax_last_dim_3d(device):
    a = torch.randn(2, 3, 7)
    da = a.to(device)
    check_op("softmax.3d", lambda: torch.softmax(a, dim=-1),
             lambda: torch.softmax(da, dim=-1))


def test_softmax_negative_dim(device):
    a, _, da, _ = _pair(device)
    check_op("softmax.neg_dim", lambda: torch.softmax(a, dim=-1),
             lambda: torch.softmax(da, dim=-1))


def test_softmax_backward(device):
    a = torch.randn(4, 5, requires_grad=True)
    da = a.detach().to(device).requires_grad_(True)

    torch.softmax(a, dim=1).sum().backward()
    torch.softmax(da, dim=1).sum().backward()

    check_op("softmax.backward", lambda: a.grad, lambda: da.grad)


def test_softmax_backward_nontrivial_grad(device):
    a = torch.randn(3, 6, requires_grad=True)
    da = a.detach().to(device).requires_grad_(True)
    upstream = torch.randn(3, 6)

    torch.softmax(a, dim=1).backward(upstream)
    torch.softmax(da, dim=1).backward(upstream.to(device))

    check_op("softmax.backward_nontrivial", lambda: a.grad, lambda: da.grad)


# --- log_softmax --------------------------------------------------------------

def test_log_softmax_dim1(device):
    a, _, da, _ = _pair(device)
    check_op("log_softmax.dim1", lambda: torch.log_softmax(a, dim=1),
             lambda: torch.log_softmax(da, dim=1))


def test_log_softmax_last_dim_3d(device):
    a = torch.randn(2, 3, 7)
    da = a.to(device)
    check_op("log_softmax.3d", lambda: torch.log_softmax(a, dim=-1),
             lambda: torch.log_softmax(da, dim=-1))


def test_log_softmax_backward(device):
    a = torch.randn(4, 5, requires_grad=True)
    da = a.detach().to(device).requires_grad_(True)

    torch.log_softmax(a, dim=1).sum().backward()
    torch.log_softmax(da, dim=1).sum().backward()

    check_op("log_softmax.backward", lambda: a.grad, lambda: da.grad)


def test_log_softmax_backward_nontrivial_grad(device):
    a = torch.randn(3, 6, requires_grad=True)
    da = a.detach().to(device).requires_grad_(True)
    upstream = torch.randn(3, 6)

    torch.log_softmax(a, dim=1).backward(upstream)
    torch.log_softmax(da, dim=1).backward(upstream.to(device))

    check_op("log_softmax.backward_nontrivial", lambda: a.grad, lambda: da.grad)


def test_log_softmax_matches_cross_entropy(device):
    # This is the whole point of log_softmax: nn.CrossEntropyLoss calls it
    # directly rather than softmax()+log(), so verify the composed result
    # matches on-device too.
    x = torch.randn(4, 8, 6)
    target = torch.randint(0, 8, (4, 6))
    dx = x.to(device)
    dtarget = target.to(device)

    ref = torch.nn.functional.nll_loss(
        torch.log_softmax(x, dim=1), target)
    dev = torch.nn.functional.nll_loss(
        torch.log_softmax(dx, dim=1), dtarget)
    check_op("log_softmax.nll_loss_chain", lambda: ref, lambda: dev)


# --- cumsum -------------------------------------------------------------------

def test_cumsum_dim1(device):
    a, _, da, _ = _pair(device)
    check_op("cumsum.dim1", lambda: torch.cumsum(a, dim=1),
             lambda: torch.cumsum(da, dim=1))


def test_cumsum_dim0(device):
    a, _, da, _ = _pair(device)
    check_op("cumsum.dim0", lambda: torch.cumsum(a, dim=0),
             lambda: torch.cumsum(da, dim=0))


def test_cumsum_negative_dim(device):
    a = torch.randn(2, 3, 4)
    da = a.to(device)
    check_op("cumsum.neg_dim", lambda: torch.cumsum(a, dim=-1),
             lambda: torch.cumsum(da, dim=-1))


def test_cumsum_int64_exact(device):
    a = torch.tensor([2**62, 1, 1, 1, 1], dtype=torch.int64)
    da = a.to(device)
    check_op("cumsum.int64_exact", lambda: torch.cumsum(a, dim=0),
             lambda: torch.cumsum(da, dim=0))


def test_cumsum_bool_promotes_to_long(device):
    a = torch.tensor([True, False, True, True])
    da = a.to(device)
    dev_out = torch.cumsum(da, dim=0)
    assert dev_out.dtype == torch.int64
    check_op("cumsum.bool_promotion", lambda: torch.cumsum(a, dim=0),
             lambda: dev_out)


def test_cumsum_(device):
    a = torch.randn(4, 5)
    da = a.to(device)
    a.cumsum_(dim=1)
    da.cumsum_(dim=1)
    check_op("cumsum_", lambda: a, lambda: da)


# --- topk ---------------------------------------------------------------------

def test_topk_largest(device):
    a = torch.randn(3, 20)
    da = a.to(device)
    ref_v, ref_i = torch.topk(a, k=5, dim=1)
    dev_v, dev_i = torch.topk(da, k=5, dim=1)
    check_op("topk.largest.values", lambda: ref_v, lambda: dev_v)
    check_op("topk.largest.indices", lambda: ref_i, lambda: dev_i)


def test_topk_smallest(device):
    a = torch.randn(3, 20)
    da = a.to(device)
    ref_v, ref_i = torch.topk(a, k=4, dim=1, largest=False)
    dev_v, dev_i = torch.topk(da, k=4, dim=1, largest=False)
    check_op("topk.smallest.values", lambda: ref_v, lambda: dev_v)
    check_op("topk.smallest.indices", lambda: ref_i, lambda: dev_i)


def test_topk_dim0(device):
    a = torch.randn(10, 4)
    da = a.to(device)
    ref_v, ref_i = torch.topk(a, k=3, dim=0)
    dev_v, dev_i = torch.topk(da, k=3, dim=0)
    check_op("topk.dim0.values", lambda: ref_v, lambda: dev_v)
    check_op("topk.dim0.indices", lambda: ref_i, lambda: dev_i)


def test_topk_full_size(device):
    a = torch.randn(3, 20)
    da = a.to(device)
    ref_v, ref_i = torch.topk(a, k=20, dim=1)
    dev_v, dev_i = torch.topk(da, k=20, dim=1)
    check_op("topk.full.values", lambda: ref_v, lambda: dev_v)
    check_op("topk.full.indices", lambda: ref_i, lambda: dev_i)


def test_topk_values_only_with_ties(device):
    # PyTorch doesn't guarantee a particular tie-break index order for
    # duplicate values (only that values come back sorted), so we only
    # check the value sequence here, not exact indices.
    a = torch.tensor([[5.0, 3.0, 5.0, 5.0, 1.0, 3.0]])
    da = a.to(device)
    ref_v, _ = torch.topk(a, k=4, dim=1)
    dev_v, _ = torch.topk(da, k=4, dim=1)
    check_op("topk.ties.values", lambda: ref_v, lambda: dev_v)


# --- structure ----------------------------------------------------------------

def test_cat(device):
    a, b, da, db = _pair(device)
    check_op("cat", lambda: torch.cat([a, b], dim=0),
             lambda: torch.cat([da, db], dim=0))


def test_mm(device):
    a = torch.randn(8, 16)
    b = torch.randn(16, 4)
    check_op("mm", lambda: a @ b, lambda: a.to(device) @ b.to(device))


def test_sort_ascending(device):
    a = torch.randn(100, dtype=torch.float32)
    cpu_vals, cpu_idxs = torch.sort(a)
    dev_vals, dev_idxs = torch.sort(a.to(device))
    assert_close(dev_vals, cpu_vals, "sort ascending values")
    assert_close(dev_idxs, cpu_idxs, "sort ascending indices")


def test_sort_descending(device):
    a = torch.randn(100, dtype=torch.float32)
    cpu_vals, cpu_idxs = torch.sort(a, descending=True)
    dev_vals, dev_idxs = torch.sort(a.to(device), descending=True)
    assert_close(dev_vals, cpu_vals, "sort descending values")
    assert_close(dev_idxs, cpu_idxs, "sort descending indices")


def test_sort_dim0(device):
    a = torch.randn(5, 8, dtype=torch.float32)
    cpu_vals, cpu_idxs = torch.sort(a, dim=0)
    dev_vals, dev_idxs = torch.sort(a.to(device), dim=0)
    assert_close(dev_vals, cpu_vals, "sort dim0 values")
    assert_close(dev_idxs, cpu_idxs, "sort dim0 indices")


def test_sort_stable(device):
    a = torch.tensor([3.0, 1.0, 2.0, 1.0, 3.0])
    cpu_vals, cpu_idxs = torch.sort(a, stable=True)
    dev_vals, dev_idxs = torch.sort(a.to(device), stable=True)
    assert_close(dev_vals, cpu_vals, "sort stable values")
    assert_close(dev_idxs, cpu_idxs, "sort stable indices")




def test_bmm(device):
    a = torch.randn(3, 8, 16)
    b = torch.randn(3, 16, 4)
    check_op("bmm", lambda: torch.bmm(a, b),
             lambda: torch.bmm(a.to(device), b.to(device)))


def test_addmm(device):
    self = torch.randn(8, 4)
    a = torch.randn(8, 16)
    b = torch.randn(16, 4)
    check_op("addmm",
             lambda: torch.addmm(self, a, b, beta=0.5, alpha=2.0),
             lambda: torch.addmm(self.to(device), a.to(device), b.to(device),
                                 beta=0.5, alpha=2.0))


def test_linear_forward(device):
    x = torch.randn(2, 5, 16)
    w = torch.randn(4, 16)
    b = torch.randn(4)
    check_op("linear", lambda: torch.nn.functional.linear(x, w, b),
             lambda: torch.nn.functional.linear(x.to(device), w.to(device),
                                                b.to(device)))


def test_sort(device):
    a = torch.randn(4, 7)
    check_op("sort", lambda: torch.sort(a, dim=-1, descending=True),
             lambda: torch.sort(a.to(device), dim=-1, descending=True))


def test_argsort(device):
    a = torch.randn(4, 7)
    check_op("argsort", lambda: torch.argsort(a, dim=-1),
             lambda: torch.argsort(a.to(device), dim=-1))



