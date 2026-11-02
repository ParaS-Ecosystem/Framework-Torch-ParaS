import pytest
import torch
import torch_paras


def gpu_devices() -> list[str]:
    return [
        f"paras:{i}"
        for i in range(torch_paras.device_count())
        if torch_paras.is_gpu_device(i)
    ]


def test_clap_blas_gemm_after_other_gpu_activity() -> None:
    devices = gpu_devices()
    if len(devices) < 2:
        pytest.skip("requires at least two ParaS GPU devices")

    gemm_device = devices[0]
    other_device = devices[1]

    a_cpu = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        dtype=torch.float32,
    )
    b_cpu = torch.tensor(
        [[5.0, 6.0], [7.0, 8.0]],
        dtype=torch.float32,
    )
    expected = torch.mm(a_cpu, b_cpu)

    a = a_cpu.to(gemm_device)
    b = b_cpu.to(gemm_device)
    out = torch.empty(
        (2, 2),
        dtype=torch.float32,
        device=gemm_device,
    )

    # Leave another GPU as the current native CUDA device.
    # The following GEMM must select the device owning `out`
    # without requiring an explicit synchronize().
    other_gpu_tensor = torch.arange(
        16,
        dtype=torch.float32,
        device=other_device,
    )
    assert other_gpu_tensor.device == torch.device(other_device)

    torch.mm(a, b, out=out)

    actual = out.cpu()

    torch.testing.assert_close(
        actual,
        expected,
        rtol=0,
        atol=0,
    )
