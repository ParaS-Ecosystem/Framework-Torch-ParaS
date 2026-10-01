#!/usr/bin/env python3
"""Queue regressions that also run with the reduced tensor_ops-only build.

    python tests/test_queue_smoke.py --expected-gpus 4

Each invocation starts a fresh interpreter before importing the source-tree
extension. This checks allocation, zeroing, nonzero fill, copies, device
ordering, and queue reuse from another thread. It is not the full operator
suite or proof that every operation executes on a GPU.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
EXTENSION = ROOT / "python" / "torch_paras" / "_C.so"


def stage(name, device=None):
    suffix = "" if device is None else f" paras:{device}"
    print(f"[queue smoke] {name}{suffix}", flush=True)


def digest(path):
    sha256 = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def run_worker(args):
    # The parent has not imported torch_paras. This process therefore checks
    # loading and registration without retaining an earlier loaded _C.so.
    sys.path.insert(0, str(ROOT / "python"))
    stage("fresh extension import")
    import torch
    import torch_paras

    loaded = Path(torch_paras._C.__file__).resolve()
    if loaded != EXTENSION.resolve():
        raise AssertionError(f"Expected extension {EXTENSION}, imported {loaded}")
    sha256 = digest(loaded)
    print(f"[queue smoke] extension={loaded} sha256={sha256}", flush=True)
    if sha256 != args.expected_extension_sha256:
        raise AssertionError("Extension changed between launch and fresh import")

    stage("device enumeration and GPU-first numbering")
    count = torch_paras.device_count()
    flags = [torch_paras.is_gpu_device(i) for i in range(count)]
    gpu_count = sum(flags)
    if flags != [True] * gpu_count + [False]:
        raise AssertionError(
            f"Expected GPUs at 0..N-1 and exactly one CPU at N; got {flags}"
        )
    if args.expected_gpus is not None and gpu_count != args.expected_gpus:
        raise AssertionError(
            f"Expected {args.expected_gpus} GPUs, enumerated {gpu_count}"
        )
    for index in range(count):
        kind = "GPU" if flags[index] else "CPU"
        print(
            f"[queue smoke] paras:{index} {kind}: "
            f"{torch_paras.device_name(index)}",
            flush=True,
        )

    # Do not let the generic CPU fallback silently satisfy these regressions.
    for op in (
        "aten::empty.memory_format",
        "aten::fill_.Scalar",
        "aten::zero_",
        "aten::_copy_from",
    ):
        if not torch._C._dispatch_has_kernel_for_dispatch_key(op, "PrivateUse1"):
            raise AssertionError(f"Required direct paras implementation missing: {op}")

    def check(actual, expected, label, index):
        if actual.device.type != "paras" or actual.device.index != index:
            raise AssertionError(f"{label}: wrong device {actual.device}")
        host = actual.cpu()
        if not torch.equal(host, expected):
            raise AssertionError(
                f"{label} on paras:{index}: values differ; "
                f"actual={host.flatten()[:8].tolist()}, "
                f"expected={expected.flatten()[:8].tolist()}"
            )

    def exercise(index, n, value):
        device = f"paras:{index}"
        for dtype in (torch.float32, torch.int64):
            stage(f"zeros size={n} dtype={dtype}", index)
            actual = torch.zeros(n, dtype=dtype, device=device)
            check(actual, torch.zeros(n, dtype=dtype), "zeros", index)

            stage(f"nonzero fill value={value} dtype={dtype}", index)
            actual.fill_(value)
            check(actual, torch.full((n,), value, dtype=dtype), "fill", index)

            stage(f"CPU -> paras -> CPU copy dtype={dtype}", index)
            expected = torch.arange(n, dtype=dtype)
            actual.copy_(expected)
            check(actual, expected, "copy round-trip", index)

            stage(f"zero after copy dtype={dtype}", index)
            actual.zero_()
            check(actual, torch.zeros(n, dtype=dtype), "zero after copy", index)

    # The CPU is deliberately included at its enumerated index N.
    for index in range(count):
        exercise(index, 257, index + 3)
        exercise(index, 4097, index + 11)

    stage(f"interleaved zero and nonzero fill on {gpu_count} GPUs")
    buffers = {
        index: torch.zeros(1025, device=f"paras:{index}")
        for index in range(gpu_count)
    }
    for turn in range(3):
        expected_values = {}
        for index in reversed(range(gpu_count)):
            value = 10 * (turn + 1) + index + 1
            stage(f"interleaved round={turn} fill value={value}", index)
            buffers[index].fill_(value)
            expected_values[index] = value
        for index in range(gpu_count):
            if (index + turn) % 2 == 0:
                stage(f"interleaved round={turn} zero", index)
                buffers[index].zero_()
                expected_values[index] = 0
        # Synchronize only after issuing operations across every GPU.
        torch_paras.synchronize()
        for index in range(gpu_count):
            check(
                buffers[index],
                torch.full((1025,), float(expected_values[index])),
                f"interleaved round {turn}",
                index,
            )
    buffers.clear()
    torch_paras.synchronize()

    # Reuse initialized queues from a second thread. Each task finishes before
    # the next starts, so this does not assume concurrent submission support.
    with ThreadPoolExecutor(max_workers=1) as executor:
        for turn in range(2):
            for index in range(count):
                stage(f"worker-thread queue reuse round={turn}", index)
                if executor.submit(torch_paras.device_count).result() != count:
                    raise AssertionError("Device count changed on repeated access")
                executor.submit(exercise, index, 513, 21 + index + turn).result()

    stage("synchronize and release cached allocations")
    torch_paras.synchronize()
    torch_paras.empty_cache()
    print(
        f"[queue smoke] PASS: {gpu_count} GPUs, CPU at paras:{gpu_count}",
        flush=True,
    )
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-gpus", type=int, default=None)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--expected-extension-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return run_worker(args)
    if not EXTENSION.is_file():
        raise FileNotFoundError(f"Build/install the extension first: {EXTENSION}")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--expected-extension-sha256",
        digest(EXTENSION),
    ]
    if args.expected_gpus is not None:
        command.extend(["--expected-gpus", str(args.expected_gpus)])
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode < 0:
        print(
            f"[queue smoke] child terminated by signal {-result.returncode}",
            file=sys.stderr,
        )
        return 128 - result.returncode
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
