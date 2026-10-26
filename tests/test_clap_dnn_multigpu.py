import ctypes
import os
import re
import subprocess
import sys
import unittest

import pytest
import torch
import torch.nn.functional as F
import torch_paras

from clap_result import RESULT_MARKER, assert_clap_result


def gpu_devices() -> list[str]:
    return [
        f"paras:{i}"
        for i in range(torch_paras.device_count())
        if torch_paras.is_gpu_device(i)
    ]


def cpu_devices() -> list[str]:
    return [
        f"paras:{i}"
        for i in range(torch_paras.device_count())
        if not torch_paras.is_gpu_device(i)
    ]


def _device_index(device: str) -> int:
    return int(device.split(":")[1]) if ":" in device else 0


def _skip(reason: str) -> None:
    raise unittest.SkipTest(reason)


def _gpu_dnn_vendor() -> tuple[str, int]:
    """(vendor library, maximum tensor rank) of the GPU DNN backend this build
    uses, read from the backend that actually runs a float32 sigmoid. Skips
    when the vendor runtime cannot be loaded (the ParaS kernel ran instead)."""
    torch_paras._C.clap_dnn_reset_stats()
    x = torch.ones(2, dtype=torch.float32, device=gpu_devices()[0])
    torch.sigmoid(x).cpu()
    for vendor, max_rank in (("MIOpen", 5), ("cuDNN", 8)):
        if torch_paras._C.clap_dnn_backend_call_count(vendor):
            return vendor, max_rank
    _skip("GPU DNN vendor runtime could not be loaded; ParaS kernels ran instead")


def _ensure_onednn_loaded() -> bool:
    try:
        ctypes.CDLL("libdnnl.so", mode=ctypes.RTLD_GLOBAL)
        return True
    except OSError:
        return False


def _run_interleaved_cached_tensors(
    name: str,
    reference_op,
    out_op,
) -> None:
    devices = gpu_devices()

    if len(devices) < 2:
        _skip("requires at least two ParaS GPU devices")
        return

    if not torch_paras._C.clap_dnn_enabled():
        _skip("CLAP-DNN is not enabled")
        return

    devices = devices[:4]

    reference = torch.linspace(
        -4,
        4,
        32,
        dtype=torch.float32,
    ).reshape(4, 8)

    expected = reference_op(reference)

    inputs = [
        reference.to(device)
        for device in devices
    ]

    outputs = [
        torch.empty_like(x)
        for x in inputs
    ]

    vendor = _gpu_dnn_vendor()[0]
    torch_paras._C.clap_dnn_reset_stats()
    calls = 3 * len(devices)
    on_paras = name == "silu" and vendor == "MIOpen"
    expected_calls = 0 if on_paras else calls

    for _ in range(3):
        for x, out in zip(
            reversed(inputs),
            reversed(outputs),
        ):
            out_op(x, out=out)

            actual = out.cpu()

            torch.testing.assert_close(
                actual,
                expected,
            )

    vendor_calls = torch_paras._C.clap_dnn_backend_call_count(vendor)
    fallbacks = torch_paras._C.clap_dnn_fallback_count()
    last_backend = torch_paras._C.clap_dnn_last_backend()

    assert vendor_calls == expected_calls, (
        f"{name}: expected {expected_calls} {vendor} calls, got {vendor_calls} "
        f"(last_backend={last_backend!r}, fallbacks={fallbacks})"
    )
    assert fallbacks == (calls if on_paras else 0)
    if not on_paras:
        assert vendor.lower() in last_backend.lower()


def test_clap_dnn_interleaved_cached_tensors(
    device: str = "paras:0",
) -> None:
    _run_interleaved_cached_tensors(
        "sigmoid",
        torch.sigmoid,
        torch.ops.aten.sigmoid.out,
    )
    _run_interleaved_cached_tensors(
        "silu",
        F.silu,
        torch.ops.aten.silu.out,
    )


def test_clap_dnn_cpu_dtypes_and_unsupported_fp16(
    device: str = "paras:0",
) -> None:
    if not torch_paras._C.clap_dnn_enabled():
        _skip("CLAP-DNN is not enabled")
        return

    cpus = cpu_devices()
    if not cpus:
        _skip("no ParaS CPU device available")
        return

    if not _ensure_onednn_loaded():
        _skip("oneDNN runtime (libdnnl.so) is not in LD_LIBRARY_PATH")
        return

    cpu_dev = cpus[0]
    saved_env = os.environ.get("CLAP_BACKEND")
    try:
        os.environ.pop("CLAP_BACKEND", None)
        ref = torch.linspace(-3.0, 3.0, 16, dtype=torch.float32).reshape(4, 4)

        for dtype, atol, rtol in [
            (torch.float32, 1e-4, 1e-4),
            (torch.bfloat16, 2e-2, 2e-2),
        ]:
            x_cpu = ref.to(dtype)
            x_dev = x_cpu.to(cpu_dev)

            for fn, ref_fn in [
                (torch.sigmoid, lambda t: torch.sigmoid(t.float()).to(t.dtype)),
                (F.silu, lambda t: F.silu(t.float()).to(t.dtype)),
            ]:
                torch_paras._C.clap_dnn_reset_stats()

                try:
                    out = fn(x_dev).cpu()
                except RuntimeError as exc:
                    if "dnnl_engine_create status=2" in str(exc):
                        _skip(
                            "oneDNN runtime is installed, but no usable CPU "
                            "engine is available on this machine"
                        )
                        return
                    raise

                torch.testing.assert_close(
                    out,
                    ref_fn(x_cpu),
                    atol=atol,
                    rtol=rtol,
                )
                assert torch_paras._C.clap_dnn_backend_call_count("oneDNN") == 1
                assert torch_paras._C.clap_dnn_fallback_count() == 0

        x_fp16 = ref.to(torch.float16).to(cpu_dev)
        out_fp16 = torch.empty_like(x_fp16)

        fp16_calls = [
            lambda: torch.sigmoid(x_fp16),
            lambda: x_fp16.clone().sigmoid_(),
            lambda: torch.ops.aten.sigmoid.out(x_fp16, out=out_fp16),
            lambda: F.silu(x_fp16),
            lambda: F.silu(x_fp16.clone(), inplace=True),
            lambda: torch.ops.aten.silu.out(x_fp16, out=out_fp16),
        ]

        expected_err = (
            "CLAP_DNN oneDNN: no native implementation for "
            "dnnl_eltwise_forward_primitive_desc_create with the requested datatype/layout"
        )

        x16 = ref.to(torch.float16).float()
        wants = [torch.sigmoid(x16).to(torch.float16)] * 3 + [F.silu(x16).to(torch.float16)] * 3
        for call, want in zip(fp16_calls, wants):
            try:
                out = call()
            except RuntimeError as exc:
                assert re.search(expected_err, str(exc)), exc
                continue
            assert out.dtype == torch.float16
            torch.testing.assert_close(out.cpu(), want, atol=2e-3, rtol=2e-3)
    finally:
        if saved_env is not None:
            os.environ["CLAP_BACKEND"] = saved_env
        else:
            os.environ.pop("CLAP_BACKEND", None)


def test_clap_dnn_input_limits_and_fallbacks(
    device: str = "paras:0",
) -> None:
    if not torch_paras._C.clap_dnn_enabled():
        _skip("CLAP-DNN is not enabled")
        return

    is_gpu = torch_paras.is_gpu_device(_device_index(device))
    if not is_gpu and not _ensure_onednn_loaded():
        _skip("oneDNN runtime (libdnnl.so) is not in LD_LIBRARY_PATH")
        return

    saved_env = os.environ.get("CLAP_BACKEND")
    try:
        os.environ.pop("CLAP_BACKEND", None)

        scalar_cpu = torch.tensor(1.25, dtype=torch.float32)
        torch_paras._C.clap_dnn_reset_stats()
        scalar_out = torch.sigmoid(scalar_cpu.to(device)).cpu()
        torch.testing.assert_close(scalar_out, torch.sigmoid(scalar_cpu))
        assert torch_paras._C.clap_dnn_call_count() == 0
        assert torch_paras._C.clap_dnn_fallback_count() == 1

        empty_cpu = torch.empty((0, 4), dtype=torch.float32)
        torch_paras._C.clap_dnn_reset_stats()
        empty_out = F.silu(empty_cpu.to(device)).cpu()
        assert empty_out.shape == (0, 4)
        assert torch_paras._C.clap_dnn_call_count() == 0
        assert torch_paras._C.clap_dnn_fallback_count() == 1

        f64_cpu = torch.linspace(-2.0, 2.0, 8, dtype=torch.float64)
        torch_paras._C.clap_dnn_reset_stats()
        f64_out = F.silu(f64_cpu.to(device)).cpu()
        torch.testing.assert_close(f64_out, F.silu(f64_cpu))
        assert torch_paras._C.clap_dnn_call_count() == 0
        assert torch_paras._C.clap_dnn_fallback_count() == 1

        base_cpu = torch.linspace(-3.0, 3.0, 24, dtype=torch.float32).reshape(4, 6)
        strided_dev = base_cpu.to(device).t()
        torch_paras._C.clap_dnn_reset_stats()
        strided_out = torch.sigmoid(strided_dev).cpu()
        torch.testing.assert_close(strided_out, torch.sigmoid(base_cpu.t()))
        assert torch_paras._C.clap_dnn_call_count() == 0
        assert torch_paras._C.clap_dnn_fallback_count() == 1

        gpus = gpu_devices()
        if gpus:
            gpu_dev = device if is_gpu else gpus[0]
            x_9d = torch.ones((2,) * 9, dtype=torch.float32, device=gpu_dev)
            vendor, max_rank = _gpu_dnn_vendor()
            with pytest.raises(
                RuntimeError,
                match=f"CLAP_DNN {vendor}: tensors must have rank 1..{max_rank}",
            ):
                _ = torch.sigmoid(x_9d).cpu()
    finally:
        if saved_env is not None:
            os.environ["CLAP_BACKEND"] = saved_env
        else:
            os.environ.pop("CLAP_BACKEND", None)


def test_clap_dnn_bfloat16_activation_on_gpu(device: str = "paras:0") -> None:
    if not torch_paras._C.clap_dnn_enabled() or not torch_paras.is_gpu_device(_device_index(device)):
        _skip("needs CLAP-DNN on a ParaS GPU device")
        return
    vendor = _gpu_dnn_vendor()[0]
    x = torch.linspace(-3.0, 3.0, 64, dtype=torch.bfloat16)
    for fn in (torch.sigmoid, F.silu):
        torch_paras._C.clap_dnn_reset_stats()
        out = fn(x.to(device)).cpu()
        torch.testing.assert_close(out, fn(x.float()).to(torch.bfloat16))
        if vendor == "MIOpen":
            assert torch_paras._C.clap_dnn_backend_call_count("MIOpen") == 0


def test_clap_dnn_backend_override_validation(
    device: str = "paras:0",
) -> None:
    if not torch_paras._C.clap_dnn_enabled():
        _skip("CLAP-DNN is not enabled")
        return

    is_gpu = torch_paras.is_gpu_device(_device_index(device))
    if not is_gpu and not _ensure_onednn_loaded():
        _skip("oneDNN runtime (libdnnl.so) is not in LD_LIBRARY_PATH")
        return

    vendor = _gpu_dnn_vendor()[0] if is_gpu else None
    gpu_env, gpu_cli = ("ROCM", "-rocm") if vendor == "MIOpen" else ("CUDA", "-cuda")
    matching_env = gpu_env if is_gpu else "CPU"
    mismatched_env = "CPU" if is_gpu else "CUDA"
    expected_vendor = vendor if is_gpu else "oneDNN"

    x_cpu = torch.linspace(-2.0, 2.0, 16, dtype=torch.float32).reshape(4, 4)
    x_dev = x_cpu.to(device)

    saved_env = os.environ.get("CLAP_BACKEND")
    try:
        os.environ["CLAP_BACKEND"] = matching_env
        torch_paras._C.clap_dnn_reset_stats()
        out = torch.sigmoid(x_dev).cpu()
        torch.testing.assert_close(out, torch.sigmoid(x_cpu))
        assert torch_paras._C.clap_dnn_backend_call_count(expected_vendor) == 1
        assert torch_paras._C.clap_dnn_fallback_count() == 0

        os.environ["CLAP_BACKEND"] = mismatched_env
        torch_paras._C.clap_dnn_reset_stats()
        out_fb = torch.sigmoid(x_dev).cpu()
        torch.testing.assert_close(out_fb, torch.sigmoid(x_cpu))
        assert torch_paras._C.clap_dnn_call_count() == 0
        assert torch_paras._C.clap_dnn_fallback_count() == 1
    finally:
        if saved_env is not None:
            os.environ["CLAP_BACKEND"] = saved_env
        else:
            os.environ.pop("CLAP_BACKEND", None)

    mismatched_cli = "-cpu" if is_gpu else "-cuda"
    matching_cli = gpu_cli if is_gpu else "-cpu"

    code = (
        "import json, torch, torch_paras; "
        f"x = torch.ones((2, 4), dtype=torch.float32, device={device!r}); "
        "torch_paras._C.clap_dnn_reset_stats(); "
        "y = torch.sigmoid(x).cpu(); "
        f"print({RESULT_MARKER!r} + json.dumps({{"
        "'calls': torch_paras._C.clap_dnn_call_count(), "
        "'fallbacks': torch_paras._C.clap_dnn_fallback_count()}), flush=True)"
    )

    env = dict(os.environ)
    env.pop("CLAP_BACKEND", None)

    res_mismatch = subprocess.run(
        [sys.executable, "-c", code, mismatched_cli],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert_clap_result(res_mismatch, 0, 1)

    res_match = subprocess.run(
        [sys.executable, "-c", code, matching_cli],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert_clap_result(res_match, 1, 0)
