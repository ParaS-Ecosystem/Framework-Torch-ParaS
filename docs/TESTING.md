# Testing torch-paras

The suite lives in `tests/` and needs nothing beyond the built package
(no pytest). Every test compares the backend against native CPU PyTorch.

## Running

```bash
source scripts/env.sh

python tests/hardware_check.py             # does the build match this machine?
python tests/run_all.py                    # paras:0, the host CPU engine
python tests/run_all.py --device paras:1   # first GPU (cuda or hip builds)
python tests/run_all.py --all-devices      # every enumerated device
```

The exit code is the number of failed tests, so it slots into CI as is.
`TORCH_PARAS_DEVICE` selects the device when you run a module directly.

Start with `hardware_check.py`. It prints the flavor, the enumerated
devices and what the vendor runtime reports, then multiplies a matrix on
every device. It is the fastest way to tell a build problem from an op
problem, and it recognises all three supported configurations: host CPU
alone, CPU plus NVIDIA GPUs, and CPU plus AMD GPUs. A `cuda` or `hip`
build on a machine with no GPU visible is a valid CPU-only setup — it
reports one device and passes.

## What is covered

- `test_tensor_ops.py` — allocation, dtypes, fill/zero, copies in all
  directions (including strided and cross-device), view/reshape/
  as_strided, masked_select, item, resize, storage sharing, shape-operation
  edge cases, a 1M element round trip, and repeated large parallel dispatch.
- `test_pointwise_ops.py` — unary math, activations, binary ops with
  tensor and scalar operands, reductions, bitwise ops, cat, matmul.
- `test_random_ops.py` — uniform/bernoulli distributions, dropout in
  train/eval, dropout backward, RNG state advancement. These assert
  statistical properties, not exact values.
- `test_vision_ops.py` — adaptive pooling, linear, and nearest-neighbor
  upsampling, ensuring the large vision translation unit is linked.

Each test function takes the device string as its one argument, so adding
a test means adding a `test_*` function to the right module. The harness
in `common.py` seeds torch before each test and prints a per-test
PASS/FAIL line with a short traceback on failure.

