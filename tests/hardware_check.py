#!/usr/bin/env python3
"""Checks that the built backend matches the hardware it is running on.

    python tests/hardware_check.py

Works on any of the three supported configurations -- host CPU only, CPU plus
NVIDIA GPUs, CPU plus AMD GPUs -- and reports which one it found. Every flavor
exposes paras:0 as the host CPU engine, so a cuda or hip build on a machine
with no GPU visible is a valid CPU-only setup rather than a failure.

Exit code is 0 when every check passes, 1 otherwise.
"""

import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "python"))

import torch  # noqa: E402
import torch_paras  # noqa: E402

failures = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    if not ok:
        failures.append(label)
    return ok


flavor = torch_paras.backend_name()
count = torch_paras.device_count()

print(f"backend flavor : {flavor}")
print(f"torch          : {torch.__version__} (cuda {torch.version.cuda}, hip {torch.version.hip})")
print(f"paras devices  : {count}")
for i in range(count):
    print(f"  paras:{i}  {torch_paras.device_name(i)}")

# What the vendor runtime says is present, independent of the paras backend.
if torch.version.hip:
    vendor, vendor_count = "amd", torch.cuda.device_count()
elif torch.version.cuda and torch.cuda.is_available():
    vendor, vendor_count = "nvidia", torch.cuda.device_count()
else:
    vendor, vendor_count = "none", 0

# A cpu-flavor extension on a GPU machine is a deliberate choice, not a fault:
# scripts/build.sh cpu produces exactly that. Only a GPU flavor that fails to
# see the GPUs it was built for is broken.
expected_flavor = {"nvidia": "paras-cuda", "amd": "paras-hip"}.get(vendor)
gpu_build = flavor in ("paras-cuda", "paras-hip")
configuration = f"cpu + {vendor}" if (vendor_count and gpu_build) else "cpu alone"
print(f"\nconfiguration  : {configuration}")
if vendor_count and not gpu_build:
    print(f"note           : {vendor_count} {vendor} GPU(s) present but this is a "
          f"{flavor} build; rebuild with scripts/build.sh auto to use them")

print("\nbackend:")
check("extension loaded", torch_paras.is_available())
check("paras:0 is the host CPU engine", count >= 1)
if gpu_build:
    check(f"flavor matches hardware ({vendor})", flavor == expected_flavor or vendor == "none",
          f"flavor={flavor}")
    check("every visible GPU is enumerated", count == vendor_count + 1,
          f"paras {count} vs vendor {vendor_count} + host")
else:
    check("cpu flavor exposes the host engine only", count == 1, f"count={count}")

print("\nper-device arithmetic:")
for i in range(count):
    dev = f"paras:{i}"
    try:
        a = torch.randn(256, 256)
        b = torch.randn(256, 256)
        got = (a.to(dev) @ b.to(dev)).cpu()
        err = (got - a @ b).abs().max().item()
        check(f"{dev} matmul", err < 1e-3, f"max|delta| = {err:.2e}")
    except Exception as exc:  # noqa: BLE001 - report, do not abort the sweep
        check(f"{dev} matmul", False, f"{type(exc).__name__}: {exc}")

print(f"\n{'-' * 52}")
if failures:
    print(f"FAILED ({len(failures)}): " + ", ".join(failures))
    raise SystemExit(1)
print(f"OK — {configuration}, {count} paras device(s), flavor {flavor}")
raise SystemExit(0)
