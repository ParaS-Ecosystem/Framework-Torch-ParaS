"""Streams and events of paras devices: work on a side stream, ordering
between streams with events (no host wait), timing, record_stream, graph
capture on a side stream."""

import unittest

import torch
import torch_paras
from torch_paras import graphs

from common import assert_close


def _gpu(device):
    if not torch_paras.is_gpu_device(torch.device(device).index or 0):
        raise unittest.SkipTest("streams of a GPU paras device")


def _work(x, n=30):
    for _ in range(n):
        x = torch.sin(x) * 1.0001 + 0.001
    return x


def test_side_stream_and_events(device):
    _gpu(device)
    torch.manual_seed(0)
    a = torch.randn(1 << 20)
    x = a.to(device)
    s = torch_paras.Stream(device)
    assert s.device == torch.device(device) and s != torch_paras.default_stream(device)
    done = torch_paras.Event()
    s.wait_stream(torch_paras.default_stream(device))
    with torch_paras.stream(s):
        assert torch_paras.current_stream(device) == s
        y = _work(x)
        done.record(s)
    assert torch_paras.current_stream(device) == torch_paras.default_stream(device)
    torch_paras.current_stream(device).wait_event(done)
    z = y * 2
    assert_close(z, _work(a) * 2, "side stream")


def test_wait_stream_both_ways(device):
    _gpu(device)
    torch.manual_seed(0)
    a = torch.randn(1 << 18)
    s = torch_paras.Stream(device)
    x = a.to(device)
    produced = torch_paras.Event()
    produced.record()
    with torch_paras.stream(s):
        torch_paras.current_stream(device).wait_event(produced)
        y = _work(x, 10)
        x.record_stream(s)
        y.record_stream(torch_paras.default_stream(device))
    torch_paras.current_stream(device).wait_stream(s)
    assert_close(y, _work(a, 10), "wait_stream")


def test_concurrent_streams(device):
    _gpu(device)
    torch.manual_seed(0)
    inputs = [torch.randn(1 << 18) for _ in range(4)]
    streams = [torch_paras.Stream(device) for _ in inputs]
    outs = []
    for s, a in zip(streams, inputs):
        with torch_paras.stream(s):
            outs.append(_work(a.to(device), 10))
    for s in streams:
        torch_paras.current_stream(device).wait_stream(s)
    for o, a in zip(outs, inputs):
        assert_close(o, _work(a, 10), "concurrent")


def test_event_timing_query_synchronize(device):
    _gpu(device)
    x = torch.randn(1 << 22, device=device)
    start, end = torch_paras.Event(enable_timing=True), torch_paras.Event(enable_timing=True)
    start.record()
    _work(x, 20)
    end.record()
    end.synchronize()
    assert end.query()
    ms = start.elapsed_time(end)
    assert 0 < ms < 10_000, ms
    s = torch_paras.Stream(device)
    s.synchronize()
    assert s.query()
    torch_paras.synchronize()


def test_graph_capture_on_side_stream(device):
    _gpu(device)
    if not graphs.is_supported():
        raise unittest.SkipTest("ParaS without record-and-replay command graphs")
    x = torch.randn(4096, device=device)
    s = torch_paras.Stream(device)
    with torch.no_grad(), torch_paras.stream(s):
        y = _work(x, 3)
        g = graphs.Graph()
        with graphs.graph(g, device=device):
            y = _work(x, 3)
        x.copy_(torch.randn(4096, device=device))
        g.replay()
        s.synchronize()
        assert_close(y, _work(x, 3), "graph on side stream")
    g.reset()


def test_streams_on_host_device(device):
    host = torch.device("paras", torch_paras.device_count() - 1)
    if torch_paras.is_gpu_device(host.index):
        raise unittest.SkipTest("no host paras device")
    s = torch_paras.Stream(host)
    with torch_paras.stream(s):
        y = torch.arange(10.0, device=host) * 3
    e = torch_paras.Event(device=host)
    e.record(s)
    e.synchronize()
    assert e.query()
    assert_close(y, torch.arange(10.0) * 3, "host stream")


def test_many_new_streams_reuse_queues(device):
    streams = [torch.Stream(device=device) for _ in range(200)]
    x = torch.ones(1000, device=device)
    for s in streams[::37]:
        with s:
            x.mul_(1.0)
    torch_paras.synchronize()
    assert torch.equal(x.cpu(), torch.ones(1000))
