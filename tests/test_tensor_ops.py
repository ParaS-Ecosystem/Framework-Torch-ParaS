from concurrent.futures import ThreadPoolExecutor

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


def test_embedding(device):
    weight = torch.randn(10, 4, dtype=torch.float32)
    indices = torch.tensor([1, 2, 2, 5, 0, 9])
    dev_out = torch.nn.functional.embedding(
        indices.to(device), weight.to(device))
    cpu_out = torch.nn.functional.embedding(indices, weight)
    assert_close(dev_out, cpu_out, "embedding")


def test_embedding_multi_dim_indices(device):
    weight = torch.randn(8, 3, dtype=torch.float32)
    indices = torch.tensor([[0, 1], [2, 3], [4, 5]])
    dev_out = torch.nn.functional.embedding(
        indices.to(device), weight.to(device))
    cpu_out = torch.nn.functional.embedding(indices, weight)
    assert_close(dev_out, cpu_out, "embedding (multi-dim indices)")


def test_embedding_backward(device):
    indices = torch.tensor([1, 2, 2, 5, 1, 0, 9])

    w_cpu = torch.randn(10, 4, dtype=torch.float32, requires_grad=True)
    torch.nn.functional.embedding(indices, w_cpu).sum().backward()

    w_dev = w_cpu.detach().to(device).requires_grad_(True)
    torch.nn.functional.embedding(indices.to(device), w_dev).sum().backward()

    assert_close(w_dev.grad, w_cpu.grad, "embedding backward")


def test_embedding_padding_idx_and_freq_scaling(device):
    indices = torch.tensor([1, 2, 2, 5, 1, 0, 9])

    w_cpu = torch.randn(10, 4, dtype=torch.float32, requires_grad=True)
    torch.nn.functional.embedding(
        indices, w_cpu, padding_idx=0, scale_grad_by_freq=True
    ).sum().backward()

    w_dev = w_cpu.detach().to(device).requires_grad_(True)
    torch.nn.functional.embedding(
        indices.to(device), w_dev, padding_idx=0, scale_grad_by_freq=True
    ).sum().backward()

    assert_close(w_dev.grad, w_cpu.grad,
                 "embedding backward (padding_idx + scale_grad_by_freq)")


def test_masked_fill__scalar(device):
    a = torch.randn(4, 5, dtype=torch.float32)
    mask = torch.rand(4, 5) > 0.5
    dev_a = a.to(device)
    dev_mask = mask.to(device)
    a.masked_fill_(mask, -1.5)
    dev_a.masked_fill_(dev_mask, -1.5)
    assert_close(dev_a, a, "masked_fill_ (scalar)")


def test_masked_fill__broadcast_mask(device):
    a = torch.randn(4, 5, dtype=torch.float32)
    mask = torch.rand(5) > 0.5  # broadcasts along dim 0
    dev_a = a.to(device)
    dev_mask = mask.to(device)
    a.masked_fill_(mask, 2.25)
    dev_a.masked_fill_(dev_mask, 2.25)
    assert_close(dev_a, a, "masked_fill_ (broadcast mask)")


def test_masked_fill__tensor_value(device):
    a = torch.randn(3, 4, dtype=torch.float32)
    mask = torch.rand(3, 4) > 0.5
    value = torch.tensor(3.14)
    dev_a = a.to(device)
    dev_mask = mask.to(device)
    a.masked_fill_(mask, value)
    dev_a.masked_fill_(dev_mask, value.to(device))
    assert_close(dev_a, a, "masked_fill_ (tensor value)")


def test_scatter_src(device):
    a = torch.randn(5, 6, dtype=torch.float32)
    index = torch.stack([torch.randperm(5)[:3] for _ in range(6)], dim=1)
    src = torch.randn(3, 6, dtype=torch.float32)
    cpu_out = torch.scatter(a, 0, index, src)
    dev_out = torch.scatter(a.to(device), 0, index.to(device), src.to(device))
    assert_close(dev_out, cpu_out, "scatter (src)")


def test_scatter_value(device):
    a = torch.randn(5, 6, dtype=torch.float32)
    index = torch.randint(0, 6, (5, 3))
    cpu_out = torch.scatter(a, 1, index, 7.0)
    dev_out = torch.scatter(a.to(device), 1, index.to(device), 7.0)
    assert_close(dev_out, cpu_out, "scatter (value)")


def test_scatter__in_place(device):
    a = torch.randn(5, 6, dtype=torch.float32)
    index = torch.randint(0, 6, (5, 3))
    a_cpu = a.clone()
    a_dev = a.to(device)
    a_cpu.scatter_(1, index, 7.0)
    a_dev.scatter_(1, index.to(device), 7.0)
    assert_close(a_dev, a_cpu, "scatter_ (in-place)")


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


def test_transpose(device):
    x = torch.randn(3, 4, 5, dtype=torch.float32)
    dev_x = x.to(device)
    cpu_out = torch.transpose(x, 0, 2)
    dev_out = torch.transpose(dev_x, 0, 2)
    assert dev_out.shape == cpu_out.shape
    assert dev_out.stride() == cpu_out.stride()
    assert_close(dev_out, cpu_out, "transpose")


def test_permute(device):
    x = torch.randn(2, 3, 4, dtype=torch.float32)
    dev_x = x.to(device)
    cpu_out = torch.permute(x, (2, 0, 1))
    dev_out = torch.permute(dev_x, (2, 0, 1))
    assert dev_out.shape == cpu_out.shape
    assert dev_out.stride() == cpu_out.stride()
    assert_close(dev_out, cpu_out, "permute")


def test_expand(device):
    x = torch.randn(1, 4, dtype=torch.float32)
    dev_x = x.to(device)
    cpu_out = x.expand(3, 4)
    dev_out = dev_x.expand(3, 4)
    assert dev_out.shape == cpu_out.shape
    assert dev_out.stride() == cpu_out.stride()
    assert_close(dev_out, cpu_out, "expand")


def test_repeat(device):
    x = torch.randn(2, 3, dtype=torch.float32)
    dev_x = x.to(device)
    cpu_out = x.repeat(2, 4)
    dev_out = dev_x.repeat(2, 4)
    assert dev_out.shape == cpu_out.shape
    assert_close(dev_out, cpu_out, "repeat")


def test_repeat_interleave_int(device):
    # Test 8 -> 32 KV head expansion pattern for GQA/MHA
    kv_heads = torch.randn(2, 8, 16, dtype=torch.float32)
    dev_kv = kv_heads.to(device)
    cpu_out = torch.repeat_interleave(kv_heads, 4, dim=1)
    dev_out = torch.repeat_interleave(dev_kv, 4, dim=1)
    assert dev_out.shape == (2, 32, 16)
    assert_close(dev_out, cpu_out, "repeat_interleave (8 -> 32 heads)")


def test_repeat_interleave_tensor(device):
    x = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32)
    repeats = torch.tensor([2, 3, 1], dtype=torch.int64)
    dev_x = x.to(device)
    dev_repeats = repeats.to(device)
    cpu_out = torch.repeat_interleave(x, repeats)
    dev_out = torch.repeat_interleave(dev_x, dev_repeats)
    assert_close(dev_out, cpu_out, "repeat_interleave (tensor repeats)")


def test_split(device):
    x = torch.arange(10, dtype=torch.float32)
    dev_x = x.to(device)
    cpu_splits = torch.split(x, 3)
    dev_splits = torch.split(dev_x, 3)
    assert len(dev_splits) == len(cpu_splits)
    for d, c in zip(dev_splits, cpu_splits):
        assert_close(d, c, "split")

    cpu_splits_sizes = torch.split(x, [2, 5, 3])
    dev_splits_sizes = torch.split(dev_x, [2, 5, 3])
    assert len(dev_splits_sizes) == len(cpu_splits_sizes)
    for d, c in zip(dev_splits_sizes, cpu_splits_sizes):
        assert_close(d, c, "split_with_sizes")


def test_chunk(device):
    x = torch.arange(11, dtype=torch.float32)
    dev_x = x.to(device)
    cpu_chunks = torch.chunk(x, 3)
    dev_chunks = torch.chunk(dev_x, 3)
    assert len(dev_chunks) == len(cpu_chunks)
    for d, c in zip(dev_chunks, cpu_chunks):
        assert_close(d, c, "chunk")


def test_contiguous(device):
    x = torch.randn(4, 5, dtype=torch.float32).t()
    dev_x = x.to(device)
    assert not dev_x.is_contiguous()
    dev_contig = dev_x.contiguous()
    assert dev_contig.is_contiguous()
    assert_close(dev_contig, x.contiguous(), "contiguous")


def test_squeeze(device):
    x = torch.randn(1, 3, 1, 4, dtype=torch.float32)
    dev_x = x.to(device)
    assert_close(torch.squeeze(dev_x), torch.squeeze(x), "squeeze all")
    assert_close(torch.squeeze(dev_x, 0), torch.squeeze(x, 0), "squeeze dim")
    assert_close(torch.squeeze(dev_x, (0, 2)), torch.squeeze(x, (0, 2)), "squeeze dims")


def test_unsqueeze(device):
    x = torch.randn(3, 4, dtype=torch.float32)
    dev_x = x.to(device)
    assert_close(torch.unsqueeze(dev_x, 0), torch.unsqueeze(x, 0), "unsqueeze dim 0")
    assert_close(torch.unsqueeze(dev_x, 1), torch.unsqueeze(x, 1), "unsqueeze dim 1")
    assert_close(torch.unsqueeze(dev_x, 2), torch.unsqueeze(x, 2), "unsqueeze dim 2")


def test_narrow(device):
    x = torch.randn(5, 6, dtype=torch.float32)
    dev_x = x.to(device)
    cpu_out = torch.narrow(x, 1, 2, 3)
    dev_out = torch.narrow(dev_x, 1, 2, 3)
    assert_close(dev_out, cpu_out, "narrow")


def test_shape_edge_cases(device):
    empty = torch.empty((0, 3), dtype=torch.float32)
    dev_empty = empty.to(device)
    assert_close(dev_empty.expand(2, 0, 3), empty.expand(2, 0, 3),
                 "expand empty")

    split_cpu = torch.empty((0,), dtype=torch.float32)
    split_dev = split_cpu.to(device)
    cpu_parts = torch.split(split_cpu, 4)
    dev_parts = torch.split(split_dev, 4)
    assert len(dev_parts) == len(cpu_parts) == 1
    assert_close(dev_parts[0], cpu_parts[0], "split empty")

    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    dev_x = x.to(device)
    assert_close(dev_x.permute(2, 0, 1).reshape(4, 6),
                 x.permute(2, 0, 1).reshape(4, 6),
                 "reshape noncontiguous")


def test_large_parallel_dispatch_stress(device):
    x = torch.linspace(-2.0, 2.0, 1_000_000, dtype=torch.float32)
    y = torch.linspace(1.0, 3.0, 1_000_000, dtype=torch.float32)
    dev_x = x.to(device)
    dev_y = y.to(device)
    out = None
    for _ in range(100):
        out = torch.relu(dev_x + dev_y)
    assert out is not None
    assert_close(out, torch.relu(x + y), "large repeated parallel dispatch")


def test_concurrent_parallel_dispatch(device):
    def work(value):
        x = torch.full((250_000,), float(value), device=device)
        for _ in range(20):
            x = torch.relu(x + 1.0)
        return float(x[0])

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(work, range(4)))
    assert results == [20.0, 21.0, 22.0, 23.0]


def test_slice_basic(device):
    a = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    cpu_out = a[1:3, 2:5]
    dev_out = a.to(device)[1:3, 2:5]
    assert_close(dev_out, cpu_out, "slice basic")


def test_slice_step(device):
    a = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    cpu_out = a[::2, 1::3]
    dev_out = a.to(device)[::2, 1::3]
    assert_close(dev_out, cpu_out, "slice step")


def test_slice_negative(device):
    a = torch.randn(5, 8, dtype=torch.float32)
    cpu_out = a[-3:, :-2]
    dev_out = a.to(device)[-3:, :-2]
    assert_close(dev_out, cpu_out, "slice negative")


def test_to_dtype(device):
    a = torch.randn(4, 5, dtype=torch.float32).to(device)
    b = a.to(dtype=torch.float64)
    assert b.dtype == torch.float64 and b.device.type == 'paras'
    assert_close(b, a.cpu().to(torch.float64), "to dtype")


def test_to_device_roundtrip(device):
    a = torch.randn(3, 4, dtype=torch.float32).to(device)
    b = a.cpu()
    c = b.to(device)
    assert_close(c, a, "to device roundtrip")
