import torch

from common import assert_close


def test_empty(device):
    a = torch.empty((2, 3), device=device, dtype=torch.float32)
    assert a.shape == (2, 3)
    assert a.device.type == "paras"
    assert a.dtype == torch.float32


def test_empty_dtypes(device):
    for dtype in (torch.float32, torch.float64, torch.int32,
                  torch.int64, torch.uint8, torch.bool):
        a = torch.empty((3, 4), device=device, dtype=dtype)
        assert a.dtype == dtype, dtype


def test_fill_(device):
    a = torch.empty((2, 3), device=device)
    a.fill_(5.5)
    assert_close(a, torch.full((2, 3), 5.5), "fill_")


def test_zero_(device):
    a = torch.empty((2, 3), device=device)
    a.zero_()
    assert_close(a, torch.zeros((2, 3)), "zero_")


def test_copy_cpu_to_device(device):
    cpu = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    dev = cpu.to(device)
    assert_close(dev, cpu, "cpu->paras")


def test_copy_device_to_cpu(device):
    cpu = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    assert_close(cpu.to(device).cpu(), cpu, "paras->cpu")


def test_copy_device_to_device(device):
    src = torch.arange(12, dtype=torch.float32).reshape(3, 4).to(device)
    dst = torch.empty_like(src)
    dst.copy_(src)
    assert_close(dst, src.cpu(), "paras->paras")


def test_copy_strided(device):
    cpu = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    dev = cpu.to(device)
    assert_close(dev.t().contiguous(), cpu.t().contiguous(), "transpose copy")
    assert_close(dev[:, 1:5].contiguous(), cpu[:, 1:5].contiguous(), "slice copy")


def test_view(device):
    x = torch.arange(16, device=device, dtype=torch.float32)
    y = x.view(4, 4)
    assert y.shape == (4, 4)
    assert_close(y, torch.arange(16, dtype=torch.float32).view(4, 4), "view")


def test_reshape(device):
    x = torch.arange(16, device=device, dtype=torch.float32).view(4, 4)
    r = x.reshape(2, 8)
    assert_close(r, torch.arange(16, dtype=torch.float32).reshape(2, 8), "reshape")


def test_as_strided(device):
    base = torch.arange(10, device=device, dtype=torch.float32)
    s = torch.as_strided(base, size=(3,), stride=(2,))
    assert_close(s, torch.tensor([0.0, 2.0, 4.0]), "as_strided")


def test_masked_select(device):
    m = torch.tensor([1, 2, 3, 4, 5], device=device)
    mask = torch.tensor([True, False, True, False, True], device=device)
    assert_close(torch.masked_select(m, mask),
                 torch.tensor([1, 3, 5]), "masked_select")


def test_item(device):
    assert torch.tensor([42.0], device=device).item() == 42.0


def test_resize_(device):
    z = torch.empty((2, 2), device=device)
    z.resize_(4, 4)
    assert z.shape == (4, 4)


def test_empty_strided(device):
    es = torch.empty_strided(size=(2, 3), stride=(3, 1),
                             device=device, dtype=torch.float32)
    assert es.shape == (2, 3)
    assert es.stride() == (3, 1)


def test_shallow_copy_shares_storage(device):
    src = torch.arange(6, device=device)
    dst = src.view(2, 3)
    assert dst.storage().data_ptr() == src.storage().data_ptr()


def test_large_tensor_roundtrip(device):
    cpu = torch.randn(1_000_000)
    assert_close(cpu.to(device).cpu(), cpu, "1M roundtrip")


def test_cross_device_copy(device):
    import torch_paras
    n = torch_paras.device_count()
    if n < 2:
        return  # single-device build, nothing to cross to
    other = f"paras:{(int(device.split(':')[1]) + 1) % n}"
    cpu = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    moved = cpu.to(device).to(other)
    assert moved.device.type == "paras"
    assert_close(moved, cpu, "cross-device copy")
