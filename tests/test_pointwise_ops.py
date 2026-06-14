import torch
import torch.nn.functional as F

from common import check_op


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


# --- structure ----------------------------------------------------------------

def test_cat(device):
    a, b, da, db = _pair(device)
    check_op("cat", lambda: torch.cat([a, b], dim=0),
             lambda: torch.cat([da, db], dim=0))


def test_mm(device):
    a = torch.randn(8, 16)
    b = torch.randn(16, 4)
    check_op("mm", lambda: a @ b, lambda: a.to(device) @ b.to(device))
