#!/usr/bin/env python3
"""Runs the full torch_paras test suite.

    python tests/run_all.py                 # device from TORCH_PARAS_DEVICE (default paras:0)
    python tests/run_all.py --device paras:1
    python tests/run_all.py --all-devices   # every enumerated paras device

Exit code is the number of failed tests (0 = success).
"""

import argparse
import importlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "python"))

import torch  # noqa: E402
import torch_paras  # noqa: E402

from common import run_module, target_device  # noqa: E402

MODULES = [
    "test_tensor_ops",
    "test_pointwise_ops",
    "test_indexing_ops",
    "test_fused_ops",
    "test_random_ops",
    "test_vision_ops",
    "test_attention_norm_ops",
    "test_matmul_sort_creation_ops"
]


def run_device(device: str) -> tuple[int, int]:
    print(f"\n{'=' * 60}")
    print(f"Device {device}: {torch_paras.device_name(int(device.split(':')[1]))}")
    print(f"Backend flavor: {torch_paras.backend_name()}")
    print("=" * 60)
    total_pass = total_fail = 0
    for name in MODULES:
        print(f"\n[{name}]")
        mod = importlib.import_module(name)
        p, f = run_module(mod, device)
        total_pass += p
        total_fail += f
    return total_pass, total_fail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="paras device, e.g. paras:0")
    ap.add_argument("--all-devices", action="store_true")
    args = ap.parse_args()

    if args.all_devices:
        devices = [f"paras:{i}" for i in range(torch_paras.device_count())]
    else:
        devices = [args.device or target_device()]

    start = time.time()
    total_pass = total_fail = 0
    for dev in devices:
        p, f = run_device(dev)
        total_pass += p
        total_fail += f

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {total_pass} passed, {total_fail} failed "
          f"({time.time() - start:.1f}s, devices: {', '.join(devices)})")
    print("=" * 60)
    return total_fail


if __name__ == "__main__":
    sys.exit(main())
