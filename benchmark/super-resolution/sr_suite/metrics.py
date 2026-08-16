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

import torch
from torch import Tensor
import torch.nn.functional as F


def _as_nchw(x: Tensor) -> Tensor:
    if x.ndim == 3:
        return x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError(f"Expected CHW or NCHW tensor, got {tuple(x.shape)}")
    return x


def align_images(sr: Tensor, hr: Tensor) -> tuple[Tensor, Tensor]:
    sr = _as_nchw(sr).to(dtype=torch.float32, device="cpu")
    hr = _as_nchw(hr).to(dtype=torch.float32, device="cpu")
    height = min(sr.shape[-2], hr.shape[-2])
    width = min(sr.shape[-1], hr.shape[-1])
    return sr[..., :height, :width], hr[..., :height, :width]


def crop_border(x: Tensor, border: int) -> Tensor:
    if border <= 0:
        return x
    if x.shape[-2] <= 2 * border or x.shape[-1] <= 2 * border:
        return x
    return x[..., border:-border, border:-border]


def rgb_to_y(x: Tensor) -> Tensor:
    x = _as_nchw(x)
    if x.shape[1] == 1:
        return x
    if x.shape[1] != 3:
        raise ValueError("RGB-to-Y conversion expects 3 channels")

    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def psnr(sr: Tensor, hr: Tensor, border: int = 0, y_channel: bool = True) -> float:
    sr, hr = align_images(sr, hr)
    if y_channel:
        sr, hr = rgb_to_y(sr), rgb_to_y(hr)
    sr, hr = crop_border(sr, border), crop_border(hr, border)
    mse = torch.mean((sr - hr) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def _gaussian_kernel(window_size: int, sigma: float, dtype: torch.dtype) -> Tensor:
    coords = torch.arange(window_size, dtype=dtype) - window_size // 2
    kernel = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    kernel /= kernel.sum()
    kernel2d = torch.outer(kernel, kernel)
    return kernel2d.view(1, 1, window_size, window_size)


def ssim(sr: Tensor, hr: Tensor, border: int = 0, y_channel: bool = True) -> float:
    sr, hr = align_images(sr, hr)
    if y_channel:
        sr, hr = rgb_to_y(sr), rgb_to_y(hr)
    sr, hr = crop_border(sr, border), crop_border(hr, border)

    min_side = min(sr.shape[-2:])
    if min_side < 3:
        return float("nan")
    window_size = min(11, min_side)
    if window_size % 2 == 0:
        window_size -= 1
    kernel = _gaussian_kernel(window_size, 1.5, sr.dtype)
    padding = 0

    mu_x = F.conv2d(sr, kernel, padding=padding)
    mu_y = F.conv2d(hr, kernel, padding=padding)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x2 = F.conv2d(sr * sr, kernel, padding=padding) - mu_x2
    sigma_y2 = F.conv2d(hr * hr, kernel, padding=padding) - mu_y2
    sigma_xy = F.conv2d(sr * hr, kernel, padding=padding) - mu_xy

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    return (numerator / denominator).mean().item()


def parity(cpu: Tensor, device_output: Tensor) -> tuple[float, float]:
    cpu, device_output = align_images(cpu, device_output)
    difference = (cpu - device_output).abs()
    return difference.max().item(), difference.mean().item()
