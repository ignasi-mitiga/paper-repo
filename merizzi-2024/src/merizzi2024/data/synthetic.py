"""Synthetic stand-in for the ERA5 -> CERRA pair.

The high-res sequence is a wind-speed-like field that drifts and evolves smoothly in time
(several octaves of spatially smooth noise, each advected with its own velocity), modulated by
a static fine-scale "terrain" pattern and a land/sea mask, so that there is genuine
sub-grid detail to recover. The low-res sequence is the paper's degradation model
(Eq. 3: blur, then downsample): a Gaussian blur followed by area averaging onto the low-res
grid. Files are written in the authors' ``.npy`` layout so the same dataset class reads them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from merizzi2024.config import ChannelStats, Config, save_config
from merizzi2024.data.memmap import field_max, field_mean_var


def _smooth_noise(rng: np.random.Generator, shape: tuple[int, ...], sigma: tuple[float, ...],
                  mode: tuple[str, ...]) -> np.ndarray:
    noise = ndimage.gaussian_filter(rng.standard_normal(shape, dtype=np.float32), sigma, mode=mode)
    return noise / (noise.std() + 1e-8)


def generate_wind_sequence(num_frames: int, hr_shape: tuple[int, int], rng: np.random.Generator,
                           octaves: int = 3, temporal_sigma: float = 3.0,
                           drift: float = 0.6) -> np.ndarray:
    """(T, H, W) float32 non-negative field with coherent drift and static fine detail."""
    h, w = hr_shape
    yy = np.arange(h)[:, None]
    field = np.zeros((num_frames, h, w), dtype=np.float32)
    for k in range(octaves):
        spatial_sigma = max(h, w) / (6 * 2**k)
        base = _smooth_noise(rng, (num_frames, h, w), (temporal_sigma, spatial_sigma, spatial_sigma),
                             ("reflect", "wrap", "wrap"))
        angle = rng.uniform(0, 2 * np.pi)
        velocity = drift * (k + 1) * np.array([np.sin(angle), np.cos(angle)])
        amplitude = 1.0 / 2**k
        for t in range(num_frames):
            spectrum = np.fft.fft2(base[t])
            shifted = ndimage.fourier_shift(spectrum, shift=velocity * t)
            field[t] += amplitude * np.real(np.fft.ifft2(shifted)).astype(np.float32)
    field /= field.reshape(num_frames, -1).std(axis=1).mean() + 1e-8

    # static sub-grid detail: rough terrain over land, smoother and windier over sea
    land = _smooth_noise(rng, (h, w), (h / 6, w / 6), ("wrap", "wrap")) > 0.0
    terrain = _smooth_noise(rng, (h, w), (1.0, 1.0), ("wrap", "wrap"))
    coast = ndimage.gaussian_filter(land.astype(np.float32), 2.0, mode="wrap")
    modulation = np.where(land, 0.75 + 0.3 * terrain, 1.2 + 0.1 * terrain) + 0.3 * (coast - land)
    speed = (8.0 + 4.0 * field) * modulation[None].astype(np.float32)
    speed += (0.4 * np.sin(2 * np.pi * yy / h) * (1.0 - land))[None]  # gentle latitude gradient
    return np.clip(speed, 0.0, None).astype(np.float32)


def degrade(hr: np.ndarray, lr_shape: tuple[int, int], blur_sigma: float) -> np.ndarray:
    """Gaussian blur then area-average each (H, W) frame onto ``lr_shape``."""
    blurred = ndimage.gaussian_filter(hr, (0.0, blur_sigma, blur_sigma),
                                      mode=("nearest", "wrap", "wrap"))
    lr = F.adaptive_avg_pool2d(torch.from_numpy(blurred)[:, None], tuple(lr_shape))[:, 0]
    return lr.numpy().astype(np.float32)


def make_synthetic_dataset(cfg: Config, write_config: bool = True) -> dict[str, dict[str, Path]]:
    """Write train/test pairs for every split in ``cfg.data`` and fill in normalisation stats."""
    root = Path(cfg.data.root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.synthetic.seed)
    written: dict[str, dict[str, Path]] = {}
    for split in cfg.data.splits:
        num = cfg.synthetic.num_train if split == cfg.data.train_split else cfg.synthetic.num_test
        hr = generate_wind_sequence(num, cfg.data.hr_shape, rng)
        lr = degrade(hr, cfg.data.lr_shape, cfg.synthetic.blur_sigma)
        hr_path, lr_path = cfg.data.split_paths(split)
        np.save(hr_path, hr)
        np.save(lr_path, lr)
        written[split] = {"hr": hr_path, "lr": lr_path}
    fill_normalisation(cfg)
    if write_config and cfg.path is not None:
        save_config(cfg)
    return written


def fill_normalisation(cfg: Config) -> None:
    """Training-split maxima (the authors' setup.py constants) and standardisation stats."""
    hr_path, lr_path = cfg.data.split_paths(cfg.data.train_split)
    cfg.data.hr_max = field_max(hr_path)
    cfg.data.lr_max = field_max(lr_path)
    hr_mean, hr_var = field_mean_var(hr_path, cfg.data.hr_max)
    lr_mean, lr_var = field_mean_var(lr_path, cfg.data.lr_max)
    cfg.data.stats = ChannelStats(lr_mean=lr_mean, lr_var=lr_var, hr_mean=hr_mean, hr_var=hr_var)
