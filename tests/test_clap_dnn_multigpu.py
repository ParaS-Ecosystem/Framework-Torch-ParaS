import pytest
import torch
import torch.nn.functional as F
import torch_paras


def gpu_devices():
    return [
        f"paras:{i}"
        for i in range(torch_paras.device_count())
        if torch_paras.is_gpu_device(i)
    ]


@pytest.mark.parametrize(
    "name,reference_op,out_op",
    [
        (
            "sigmoid",
            torch.sigmoid,
            torch.ops.aten.sigmoid.out,
        ),
        (
            "silu",
            F.silu,
            torch.ops.aten.silu.out,
        ),
    ],
)
def test_clap_dnn_interleaved_cached_tensors(
    name,
    reference_op,
    out_op,
):
    devices = gpu_devices()

    if len(devices) < 2:
        pytest.skip("requires at least two ParaS GPU devices")

    # Enough GPUs to reproduce device switching without making the
    # regression unnecessarily expensive.
    devices = devices[:4]

    reference = torch.linspace(
        -4,
        4,
        32,
        dtype=torch.float32,
    ).reshape(4, 8)

    expected = reference_op(reference)

    # Preallocate tensors across all GPUs first. This deliberately leaves
    # the calling thread associated with the last allocated CUDA device.
    inputs = [
        reference.to(device)
        for device in devices
    ]

    outputs = [
        torch.empty_like(x)
        for x in inputs
    ]

    # Execute in reverse order so cached tensors/outputs are repeatedly
    # consumed from a device different from the thread's previous device.
    for _ in range(3):
        for x, out in zip(
            reversed(inputs),
            reversed(outputs),
        ):
            out_op(x, out=out)

            # CPU copy makes completion explicit before comparison.
            actual = out.cpu()

            torch.testing.assert_close(
                actual,
                expected,
            )
