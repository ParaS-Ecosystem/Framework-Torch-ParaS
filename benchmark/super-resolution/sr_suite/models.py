"""
// -----------------------------------------------------------------------------
// Copyright (c) 2026 Centre for Development of Advanced Computing (C-DAC)
//
// This file is part of Torch_ParaS, a component of the ParaS Ecosystem
//
// This library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License (LGPL)
// version 3 as published by the Free Software Foundation.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this library. If not, see <https://www.gnu.org/licenses/>.
// -----------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelInfo:
    name: str
    scale: int
    expects_preupsampled: bool
    architecture: str
    profile: str


class SRModelBase(nn.Module):
    expects_preupsampled: bool = False
    architecture_name: str = "unknown"

    def __init__(self, scale: int) -> None:
        super().__init__()
        self.scale = int(scale)


class SRCNN(SRModelBase):

    expects_preupsampled = True
    architecture_name = "SRCNN"

    def __init__(self, scale: int = 4, channels1: int = 64, channels2: int = 32) -> None:
        super().__init__(scale)
        self.conv1 = nn.Conv2d(3, channels1, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(channels1, channels2, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(channels2, 3, kernel_size=5, padding=2)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.conv1(x), inplace=False)
        x = F.relu(self.conv2(x), inplace=False)
        return self.conv3(x)


class FSRCNN(SRModelBase):
    architecture_name = "FSRCNN"

    def __init__(self, scale: int = 4, d: int = 56, s: int = 12, m: int = 4) -> None:
        super().__init__(scale)
        layers = [nn.Conv2d(3, d, 5, padding=2), nn.PReLU(d), nn.Conv2d(d, s, 1), nn.PReLU(s)]
        for _ in range(m):
            layers.extend([nn.Conv2d(s, s, 3, padding=1), nn.PReLU(s)])
        layers.extend([nn.Conv2d(s, d, 1), nn.PReLU(d)])
        self.feature = nn.Sequential(*layers)
        self.deconv = nn.ConvTranspose2d(
            d,
            3,
            kernel_size=9,
            stride=scale,
            padding=4,
            output_padding=scale - 1,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.deconv(self.feature(x))


class ESPCN(SRModelBase):
    architecture_name = "ESPCN"

    def __init__(self, scale: int = 4, features1: int = 64, features2: int = 32) -> None:
        super().__init__(scale)
        self.conv1 = nn.Conv2d(3, features1, 5, padding=2)
        self.conv2 = nn.Conv2d(features1, features2, 3, padding=1)
        self.conv3 = nn.Conv2d(features2, 3 * scale * scale, 3, padding=1)
        self.shuffle = nn.PixelShuffle(scale)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        return self.shuffle(self.conv3(x))


class VDSR(SRModelBase):
    expects_preupsampled = True
    architecture_name = "VDSR"

    def __init__(self, scale: int = 4, depth: int = 20, features: int = 64) -> None:
        super().__init__(scale)
        if depth < 3:
            raise ValueError("VDSR depth must be at least 3")
        layers = [nn.Conv2d(3, features, 3, padding=1), nn.ReLU(inplace=False)]
        for _ in range(depth - 2):
            layers.extend([nn.Conv2d(features, features, 3, padding=1), nn.ReLU(inplace=False)])
        layers.append(nn.Conv2d(features, 3, 3, padding=1))
        self.body = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.body(x)


class ResidualBlockNoBN(nn.Module):
    def __init__(self, channels: int, res_scale: float = 1.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.res_scale = float(res_scale)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.conv2(F.relu(self.conv1(x), inplace=False))
        return x + residual * self.res_scale


def make_pixelshuffle_upsampler(channels: int, scale: int) -> nn.Sequential:
    if scale in (2, 4, 8):
        layers = []
        for _ in range(int(math.log2(scale))):
            layers.extend([nn.Conv2d(channels, channels * 4, 3, padding=1), nn.PixelShuffle(2)])
        return nn.Sequential(*layers)
    if scale == 3:
        return nn.Sequential(nn.Conv2d(channels, channels * 9, 3, padding=1), nn.PixelShuffle(3))
    raise ValueError(f"Unsupported scale: {scale}; use 2, 3, 4, or 8")


class EDSR(SRModelBase):
    architecture_name = "EDSR"

    def __init__(
        self,
        scale: int = 4,
        n_resblocks: int = 16,
        n_feats: int = 64,
        res_scale: float = 0.1,
    ) -> None:
        super().__init__(scale)
        self.head = nn.Conv2d(3, n_feats, 3, padding=1)
        self.body = nn.Sequential(*[ResidualBlockNoBN(n_feats, res_scale) for _ in range(n_resblocks)])
        self.body_tail = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        self.up = make_pixelshuffle_upsampler(n_feats, scale)
        self.tail = nn.Conv2d(n_feats, 3, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        head = self.head(x)
        body = self.body_tail(self.body(head)) + head
        return self.tail(self.up(body))


class SRResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        y = self.prelu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return x + y


class SRResNet(SRModelBase):
    architecture_name = "SRResNet"

    def __init__(self, scale: int = 4, n_blocks: int = 16, n_feats: int = 64) -> None:
        super().__init__(scale)
        self.head = nn.Conv2d(3, n_feats, 9, padding=4)
        self.prelu = nn.PReLU(n_feats)
        self.body = nn.Sequential(*[SRResBlock(n_feats) for _ in range(n_blocks)])
        self.body_conv = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        self.body_bn = nn.BatchNorm2d(n_feats)
        self.up = make_pixelshuffle_upsampler(n_feats, scale)
        self.tail = nn.Conv2d(n_feats, 3, 9, padding=4)

    def forward(self, x: Tensor) -> Tensor:
        x0 = self.prelu(self.head(x))
        body = self.body_bn(self.body_conv(self.body(x0))) + x0
        return self.tail(self.up(body))


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.net(self.pool(x))


class RCAB(nn.Module):
    def __init__(self, channels: int, reduction: int, res_scale: float = 1.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.ca = ChannelAttention(channels, reduction)
        self.res_scale = res_scale

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv2(F.relu(self.conv1(x), inplace=False))
        y = self.ca(y)
        return x + y * self.res_scale


class ResidualGroup(nn.Module):
    def __init__(self, channels: int, n_blocks: int, reduction: int) -> None:
        super().__init__()
        self.body = nn.Sequential(*[RCAB(channels, reduction) for _ in range(n_blocks)])
        self.tail = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.tail(self.body(x))


class RCAN(SRModelBase):
    architecture_name = "RCAN"

    def __init__(
        self,
        scale: int = 4,
        n_groups: int = 5,
        n_blocks: int = 10,
        n_feats: int = 64,
        reduction: int = 16,
    ) -> None:
        super().__init__(scale)
        self.head = nn.Conv2d(3, n_feats, 3, padding=1)
        self.groups = nn.Sequential(
            *[ResidualGroup(n_feats, n_blocks, reduction) for _ in range(n_groups)]
        )
        self.body_tail = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        self.up = make_pixelshuffle_upsampler(n_feats, scale)
        self.tail = nn.Conv2d(n_feats, 3, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        head = self.head(x)
        body = self.body_tail(self.groups(head)) + head
        return self.tail(self.up(body))


class DenseLayer(nn.Module):
    def __init__(self, in_channels: int, growth: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, growth, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([x, F.relu(self.conv(x), inplace=False)], dim=1)


class RDBlock(nn.Module):
    def __init__(self, base_channels: int, growth: int, n_layers: int) -> None:
        super().__init__()
        channels = base_channels
        layers = []
        for _ in range(n_layers):
            layers.append(DenseLayer(channels, growth))
            channels += growth
        self.layers = nn.Sequential(*layers)
        self.local_fusion = nn.Conv2d(channels, base_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.local_fusion(self.layers(x))


class RDN(SRModelBase):
    architecture_name = "RDN"

    def __init__(
        self,
        scale: int = 4,
        n_blocks: int = 10,
        n_layers: int = 6,
        growth: int = 32,
        n_feats: int = 64,
    ) -> None:
        super().__init__(scale)
        self.sfe1 = nn.Conv2d(3, n_feats, 3, padding=1)
        self.sfe2 = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        self.rdbs = nn.ModuleList([RDBlock(n_feats, growth, n_layers) for _ in range(n_blocks)])
        self.gff1 = nn.Conv2d(n_blocks * n_feats, n_feats, 1)
        self.gff2 = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        self.up = make_pixelshuffle_upsampler(n_feats, scale)
        self.tail = nn.Conv2d(n_feats, 3, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        f1 = self.sfe1(x)
        x = self.sfe2(f1)
        local = []
        for block in self.rdbs:
            x = block(x)
            local.append(x)
        x = self.gff2(self.gff1(torch.cat(local, dim=1))) + f1
        return self.tail(self.up(x))


class ResidualDenseBlock5C(nn.Module):
    def __init__(self, channels: int, growth: int) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(channels, growth, 3, padding=1)
        self.c2 = nn.Conv2d(channels + growth, growth, 3, padding=1)
        self.c3 = nn.Conv2d(channels + 2 * growth, growth, 3, padding=1)
        self.c4 = nn.Conv2d(channels + 3 * growth, growth, 3, padding=1)
        self.c5 = nn.Conv2d(channels + 4 * growth, channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x1 = F.leaky_relu(self.c1(x), negative_slope=0.2, inplace=False)
        x2 = F.leaky_relu(self.c2(torch.cat([x, x1], dim=1)), 0.2, inplace=False)
        x3 = F.leaky_relu(self.c3(torch.cat([x, x1, x2], dim=1)), 0.2, inplace=False)
        x4 = F.leaky_relu(self.c4(torch.cat([x, x1, x2, x3], dim=1)), 0.2, inplace=False)
        x5 = self.c5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x + 0.2 * x5


class RRDB(nn.Module):
    def __init__(self, channels: int, growth: int) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock5C(channels, growth)
        self.rdb2 = ResidualDenseBlock5C(channels, growth)
        self.rdb3 = ResidualDenseBlock5C(channels, growth)

    def forward(self, x: Tensor) -> Tensor:
        return x + 0.2 * self.rdb3(self.rdb2(self.rdb1(x)))


class RRDBNet(SRModelBase):

    architecture_name = "ESRGAN-RRDBNet"

    def __init__(
        self,
        scale: int = 4,
        n_blocks: int = 12,
        n_feats: int = 64,
        growth: int = 32,
    ) -> None:
        super().__init__(scale)
        if scale not in (2, 4):
            raise ValueError("RRDBNet implementation supports scale 2 or 4")
        self.conv_first = nn.Conv2d(3, n_feats, 3, padding=1)
        self.trunk = nn.Sequential(*[RRDB(n_feats, growth) for _ in range(n_blocks)])
        self.trunk_conv = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        self.upconvs = nn.ModuleList(
            [nn.Conv2d(n_feats, n_feats, 3, padding=1) for _ in range(int(math.log2(scale)))]
        )
        self.hr_conv = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        self.last_conv = nn.Conv2d(n_feats, 3, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        feat = self.conv_first(x)
        feat = feat + self.trunk_conv(self.trunk(feat))
        for conv in self.upconvs:
            feat = F.interpolate(feat, scale_factor=2, mode="nearest")
            feat = F.leaky_relu(conv(feat), 0.2, inplace=False)
        return self.last_conv(F.leaky_relu(self.hr_conv(feat), 0.2, inplace=False))


def window_partition(x: Tensor, window_size: int) -> Tensor:
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, c)


def window_reverse(windows: Tensor, window_size: int, h: int, w: int, b: int) -> Tensor:
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)


class MLP(nn.Module):
    def __init__(self, dim: int, ratio: float = 2.0) -> None:
        super().__init__()
        hidden = int(dim * ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class WindowAttention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("SwinIR embedding dimension must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attention = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention = torch.softmax(attention, dim=-1)
        x = torch.matmul(attention, v).transpose(1, 2).reshape(b, n, c)
        return self.proj(x)


class SwinBlock(nn.Module):
    def __init__(self, dim: int, heads: int, window_size: int, shift: bool) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2 if shift else 0
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, ratio=2.0)

    def forward(self, x: Tensor) -> Tensor:
        b, h, w, c = x.shape
        shortcut = x
        x = self.norm1(x)
        if self.shift_size:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        windows = window_partition(x, self.window_size)
        windows = self.attn(windows)
        x = window_reverse(windows, self.window_size, h, w, b)
        if self.shift_size:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class SwinIR(SRModelBase):

    architecture_name = "SwinIR-family"

    def __init__(
        self,
        scale: int = 4,
        dim: int = 96,
        depths: Tuple[int, ...] = (4, 4, 4, 4),
        heads: int = 6,
        window_size: int = 8,
    ) -> None:
        super().__init__(scale)
        self.window_size = window_size
        self.conv_first = nn.Conv2d(3, dim, 3, padding=1)
        blocks = []
        index = 0
        for depth in depths:
            for _ in range(depth):
                blocks.append(SwinBlock(dim, heads, window_size, shift=(index % 2 == 1)))
                index += 1
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(dim)
        self.conv_after_body = nn.Conv2d(dim, dim, 3, padding=1)
        self.up = make_pixelshuffle_upsampler(dim, scale)
        self.conv_last = nn.Conv2d(dim, 3, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        b, _, original_h, original_w = x.shape
        ws = self.window_size
        pad_h = (ws - original_h % ws) % ws
        pad_w = (ws - original_w % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
        feat = self.conv_first(x)
        residual = feat
        feat = feat.permute(0, 2, 3, 1).contiguous()
        for block in self.blocks:
            feat = block(feat)
        feat = self.norm(feat).permute(0, 3, 1, 2).contiguous()
        feat = self.conv_after_body(feat) + residual
        out = self.conv_last(self.up(feat))
        return out[:, :, : original_h * self.scale, : original_w * self.scale]


MODEL_NAMES = (
    "srcnn",
    "fsrcnn",
    "espcn",
    "vdsr",
    "edsr",
    "srresnet",
    "rcan",
    "rdn",
    "esrgan",
    "swinir",
)


SMOKE_CONFIG: Dict[str, Dict[str, object]] = {
    "srcnn": {"channels1": 16, "channels2": 8},
    "fsrcnn": {"d": 16, "s": 8, "m": 2},
    "espcn": {"features1": 16, "features2": 8},
    "vdsr": {"depth": 5, "features": 16},
    "edsr": {"n_resblocks": 3, "n_feats": 16, "res_scale": 0.1},
    "srresnet": {"n_blocks": 3, "n_feats": 16},
    "rcan": {"n_groups": 2, "n_blocks": 2, "n_feats": 16, "reduction": 4},
    "rdn": {"n_blocks": 3, "n_layers": 3, "growth": 8, "n_feats": 16},
    "esrgan": {"n_blocks": 2, "n_feats": 16, "growth": 8},
    "swinir": {"dim": 24, "depths": (2, 2), "heads": 3, "window_size": 4},
}


BENCHMARK_CONFIG: Dict[str, Dict[str, object]] = {
    "srcnn": {"channels1": 64, "channels2": 32},
    "fsrcnn": {"d": 56, "s": 12, "m": 4},
    "espcn": {"features1": 64, "features2": 32},
    "vdsr": {"depth": 20, "features": 64},
    "edsr": {"n_resblocks": 16, "n_feats": 64, "res_scale": 0.1},
    "srresnet": {"n_blocks": 16, "n_feats": 64},
    "rcan": {"n_groups": 5, "n_blocks": 10, "n_feats": 64, "reduction": 16},
    "rdn": {"n_blocks": 10, "n_layers": 6, "growth": 32, "n_feats": 64},
    "esrgan": {"n_blocks": 12, "n_feats": 64, "growth": 32},
    "swinir": {"dim": 96, "depths": (4, 4, 4, 4), "heads": 6, "window_size": 8},
}


FACTORIES: Dict[str, Callable[..., SRModelBase]] = {
    "srcnn": SRCNN,
    "fsrcnn": FSRCNN,
    "espcn": ESPCN,
    "vdsr": VDSR,
    "edsr": EDSR,
    "srresnet": SRResNet,
    "rcan": RCAN,
    "rdn": RDN,
    "esrgan": RRDBNet,
    "swinir": SwinIR,
}


def build_model(
    name: str,
    scale: int,
    profile: str = "smoke",
    overrides: Dict[str, object] | None = None,
) -> SRModelBase:
    name = name.lower()
    if name not in FACTORIES:
        raise KeyError(f"Unknown model {name!r}. Available: {', '.join(MODEL_NAMES)}")
    if profile not in ("smoke", "benchmark"):
        raise ValueError("profile must be 'smoke' or 'benchmark'")
    config = dict(SMOKE_CONFIG[name] if profile == "smoke" else BENCHMARK_CONFIG[name])
    if overrides:
        config.update(overrides)
    model = FACTORIES[name](scale=scale, **config)
    return model


def model_info(model: SRModelBase, name: str, profile: str) -> ModelInfo:
    return ModelInfo(
        name=name,
        scale=model.scale,
        expects_preupsampled=bool(model.expects_preupsampled),
        architecture=model.architecture_name,
        profile=profile,
    )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
