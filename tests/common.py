import os
import sys
import traceback

import torch

ATOL = 1e-4
RTOL = 1e-4


def target_device() -> str:
    return os.environ.get("TORCH_PARAS_DEVICE", "paras:0")


def assert_close(actual: torch.Tensor, expected: torch.Tensor, msg: str = ""):
    """Compares a paras tensor against a CPU reference tensor."""
    actual = actual.cpu()
    expected = expected.cpu()
    if actual.dtype != expected.dtype:
        raise AssertionError(
            f"{msg}: dtype mismatch {actual.dtype} vs {expected.dtype}")
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{msg}: shape mismatch {tuple(actual.shape)} vs {tuple(expected.shape)}")
    if actual.dtype.is_floating_point:
        ok = torch.allclose(actual, expected, atol=ATOL, rtol=RTOL, equal_nan=True)
    else:
        ok = torch.equal(actual, expected)
    if not ok:
        raise AssertionError(f"{msg}: values mismatch\nparas:\n{actual}\nref:\n{expected}")


def check_op(name: str, ref_fn, dev_fn):
    """Runs ``ref_fn`` (CPU reference) and ``dev_fn`` (paras) and compares."""
    ref = ref_fn()
    out = dev_fn()
    if isinstance(ref, tuple):
        for i, (r, o) in enumerate(zip(ref, out)):
            if torch.is_tensor(r):
                assert_close(o, r, f"{name}[{i}]")
    else:
        assert_close(out, ref, name)


def run_module(module, device: str) -> tuple[int, int]:
    """Runs every test_* function in ``module``; returns (passed, failed)."""
    tests = [(n, f) for n, f in vars(module).items()
             if n.startswith("test_") and callable(f)]
    passed = failed = 0
    for name, fn in tests:
        try:
            torch.manual_seed(0)
            fn(device)
            passed += 1
            print(f"  PASS  {name}")
        except Exception:
            failed += 1
            print(f"  FAIL  {name}")
            traceback.print_exc(limit=4, file=sys.stdout)
    return passed, failed
