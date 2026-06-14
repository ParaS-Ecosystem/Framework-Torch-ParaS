# Testing torch-paras

The suite lives in `tests/` and needs nothing beyond the built package
(no pytest). Every test compares the backend against native CPU PyTorch.

## Running

```bash
source scripts/env.sh

python tests/run_all.py                    # paras:0, the host CPU engine
python tests/run_all.py --device paras:1   # first GPU (cuda builds)
python tests/run_all.py --all-devices      # every enumerated device
```

The exit code is the number of failed tests, so it slots into CI as is.
`TORCH_PARAS_DEVICE` selects the device when you run a module directly.

## What is covered

- `test_tensor_ops.py` — allocation, dtypes, fill/zero, copies in all
  directions (including strided and cross-device), view/reshape/
  as_strided, masked_select, item, resize, storage sharing, a 1M element
  round trip.
- `test_pointwise_ops.py` — unary math, activations, binary ops with
  tensor and scalar operands, reductions, bitwise ops, cat, matmul.
- `test_random_ops.py` — uniform/bernoulli distributions, dropout in
  train/eval, dropout backward, RNG state advancement. These assert
  statistical properties, not exact values.

Each test function takes the device string as its one argument, so adding
a test means adding a `test_*` function to the right module. The harness
in `common.py` seeds torch before each test and prints a per-test
PASS/FAIL line with a short traceback on failure.


