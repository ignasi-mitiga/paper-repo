"""Memory-mapped pair dataset in the authors' on-disk layout.

Each split is two float32 ``.npy`` files: ``hr`` shaped (N, H, W) and ``lr`` shaped (N, h, w),
one frame per time step at the same time stamps. This is the layout of the Kaggle files
released with the paper, of ``make-synthetic`` and of ``download-era5``.

A sample for target time ``t`` is the (F+1, H, W) tensor of the F low-res frames at
``t + offset`` for each offset in ``frame_offsets`` (default -2, -1, 0, +1, i.e. t-6h, t-3h, t0,
t+3h in the paper's 3-hourly data), each divided by ``lr_max`` and bilinearly upsampled to the
high-res grid, followed by the high-res frame at ``t`` divided by ``hr_max``. The upsampling is
the authors' pre-emptive bilinear upsampling (``cv2.INTER_LINEAR``, half-pixel centres).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def upsample_frames(frames: torch.Tensor, hr_shape: tuple[int, int]) -> torch.Tensor:
    """(F, h, w) or (B, F, h, w) -> same leading dims at ``hr_shape``, bilinear."""
    squeeze = frames.ndim == 3
    if squeeze:
        frames = frames[None]
    out = F.interpolate(frames, size=tuple(hr_shape), mode="bilinear", align_corners=False)
    return out[0] if squeeze else out


def bilinear_baseline(batch: torch.Tensor, frame_offsets: list[int]) -> torch.Tensor:
    """The paper's baseline: the upsampled low-res frame at t0, shaped (B, 1, H, W)."""
    index = list(frame_offsets).index(0)
    return batch[:, index : index + 1]


class MemmapPairDataset(Dataset):
    def __init__(
        self,
        hr_path: str | Path,
        lr_path: str | Path,
        frame_offsets: list[int],
        hr_max: float,
        lr_max: float,
        hr_shape: tuple[int, int] | None = None,
    ):
        self.hr = np.load(hr_path, mmap_mode="r")
        self.lr = np.load(lr_path, mmap_mode="r")
        if self.hr.ndim != 3 or self.lr.ndim != 3:
            raise ValueError("expected (N, H, W) arrays")
        if self.hr.shape[0] != self.lr.shape[0]:
            raise ValueError("high-res and low-res files must have the same number of frames")
        self.frame_offsets = list(frame_offsets)
        self.hr_max = float(hr_max)
        self.lr_max = float(lr_max)
        self.hr_shape = tuple(hr_shape) if hr_shape is not None else tuple(self.hr.shape[1:])
        self.first = -min(self.frame_offsets)
        self.last = self.hr.shape[0] - max(self.frame_offsets)  # exclusive
        if self.last <= self.first:
            raise ValueError("not enough frames for the requested offsets")

    def __len__(self) -> int:
        return self.last - self.first

    def anchor(self, index: int) -> int:
        """Time index of the target frame for dataset index ``index``."""
        return self.first + index

    def __getitem__(self, index: int) -> torch.Tensor:
        t = self.anchor(index)
        lr = np.stack([self.lr[t + o] for o in self.frame_offsets]).astype(np.float32)
        cond = upsample_frames(torch.from_numpy(lr / self.lr_max), self.hr_shape)
        target = torch.from_numpy(np.asarray(self.hr[t], dtype=np.float32) / self.hr_max)
        return torch.cat([cond, target[None]], dim=0)


def field_max(path: str | Path, chunk: int = 256) -> float:
    arr = np.load(path, mmap_mode="r")
    best = 0.0
    for start in range(0, arr.shape[0], chunk):
        best = max(best, float(np.max(arr[start : start + chunk])))
    return best


def field_mean_var(path: str | Path, scale: float, chunk: int = 256) -> tuple[float, float]:
    """Mean and variance of ``array / scale`` in one streaming pass."""
    arr = np.load(path, mmap_mode="r")
    total = 0.0
    total_sq = 0.0
    count = 0
    for start in range(0, arr.shape[0], chunk):
        block = np.asarray(arr[start : start + chunk], dtype=np.float64) / scale
        total += float(block.sum())
        total_sq += float((block**2).sum())
        count += block.size
    mean = total / count
    return mean, max(total_sq / count - mean**2, 0.0)
