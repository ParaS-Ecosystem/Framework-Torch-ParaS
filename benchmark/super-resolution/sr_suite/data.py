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

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch
from PIL import Image
from torch import Tensor


DEFAULT_DATASET_NAMES = (
    "Set5",
    "Set14",
    "BSD100",
    "Urban100",
    "Manga109",
    "DIV2K",
    "General100",
    "T91",
    "Flickr2K",
    "Historical",
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class ImageSample:
    dataset: str
    image_path: Path
    image_id: str


def load_dataset_config(path: Path | None) -> dict[str, dict[str, object]]:
    if path is None:
        return {name: {} for name in DEFAULT_DATASET_NAMES}
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Dataset config must be a JSON object")
    return payload


def _default_search_roots(root: Path, dataset: str) -> list[Path]:
    base = root / dataset
    candidates = [
        base / "HR",
        base / "GT",
        base / "ground_truth",
        base / f"{dataset}_HR",
        base / "DIV2K_valid_HR",
        base / "DIV2K_train_HR",
        base,
    ]
    return [candidate for candidate in candidates if candidate.exists()]


def discover_images(root: Path, dataset: str, config: dict[str, object]) -> list[Path]:
    paths: list[Path] = []
    globs = config.get("hr_globs", [])
    if globs:
        if not isinstance(globs, list):
            raise ValueError(f"{dataset}.hr_globs must be a list")
        for pattern in globs:
            paths.extend(root.glob(str(pattern)))
    else:
        for search_root in _default_search_roots(root, dataset):
            paths.extend(path for path in search_root.rglob("*") if path.is_file())

    unique = sorted({path.resolve() for path in paths if path.suffix.lower() in IMAGE_EXTENSIONS})
    return unique


def iter_samples(
    root: Path,
    dataset_names: Iterable[str],
    config: dict[str, dict[str, object]],
    max_images: int | None,
) -> Iterator[ImageSample]:
    for dataset in dataset_names:
        images = discover_images(root, dataset, config.get(dataset, {}))
        if max_images is not None and max_images >= 0:
            images = images[:max_images]
        for path in images:
            yield ImageSample(dataset=dataset, image_path=path, image_id=path.stem)


def pil_to_tensor(image: Image.Image) -> Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def tensor_to_pil(tensor: Tensor) -> Image.Image:
    tensor = tensor.detach().to("cpu", dtype=torch.float32).clamp(0.0, 1.0)
    if tensor.ndim == 4:
        tensor = tensor[0]
    array = (tensor.permute(1, 2, 0).contiguous().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _resize(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    resampling = getattr(Image, "Resampling", Image)
    return image.resize(size, resample=resampling.BICUBIC)


def prepare_pair(
    path: Path,
    scale: int,
    max_hr_side: int | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    hr = Image.open(path).convert("RGB")

    if max_hr_side and max(hr.size) > max_hr_side:
        ratio = max_hr_side / max(hr.size)
        resized = (max(scale, int(hr.width * ratio)), max(scale, int(hr.height * ratio)))
        hr = _resize(hr, resized)

    width = max(scale, hr.width - (hr.width % scale))
    height = max(scale, hr.height - (hr.height % scale))
    hr = hr.crop((0, 0, min(width, hr.width), min(height, hr.height)))
    width, height = hr.size
    if width < scale or height < scale:
        raise ValueError(f"Image is too small for x{scale}: {path} ({width}x{height})")

    lr = _resize(hr, (width // scale, height // scale))
    bicubic = _resize(lr, (width, height))
    return pil_to_tensor(lr), pil_to_tensor(hr), pil_to_tensor(bicubic)
