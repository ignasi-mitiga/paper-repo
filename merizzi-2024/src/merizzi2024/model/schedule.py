"""Noise schedules mapping a diffusion time t in [0, 1] to (noise_rate, signal_rate).

The released code uses the cosine-angle schedule of the Keras DDIM example: the diffusion
time is mapped linearly onto an angle between acos(max_signal_rate) and acos(min_signal_rate),
and signal_rate = cos(angle), noise_rate = sin(angle), so that their squares always sum to 1.

The paper text instead describes "a linear schedule" of noise rates (e.g. 1, 0.8, 0.6, 0.4, 0.2
for five steps). ``linear`` reproduces that reading while keeping the same end points as the
cosine schedule, so the signal rate never reaches exactly zero (which would make the x0
estimate in the sampler divide by zero).
"""

from __future__ import annotations

import math

import torch


def cosine_angle_schedule(
    t: torch.Tensor, min_signal_rate: float, max_signal_rate: float
) -> tuple[torch.Tensor, torch.Tensor]:
    start_angle = math.acos(max_signal_rate)
    end_angle = math.acos(min_signal_rate)
    angles = start_angle + t * (end_angle - start_angle)
    return torch.sin(angles), torch.cos(angles)


def linear_noise_schedule(
    t: torch.Tensor, min_signal_rate: float, max_signal_rate: float
) -> tuple[torch.Tensor, torch.Tensor]:
    noise_start = math.sqrt(1.0 - max_signal_rate**2)
    noise_end = math.sqrt(1.0 - min_signal_rate**2)
    noise_rates = noise_start + t * (noise_end - noise_start)
    signal_rates = torch.sqrt(1.0 - noise_rates**2)
    return noise_rates, signal_rates


class NoiseSchedule:
    """Callable schedule: ``noise_rates, signal_rates = schedule(diffusion_times)``."""

    KINDS = {"cosine": cosine_angle_schedule, "linear": linear_noise_schedule}

    def __init__(self, kind: str = "cosine", min_signal_rate: float = 0.015,
                 max_signal_rate: float = 0.95):
        if kind not in self.KINDS:
            raise ValueError(f"unknown schedule {kind!r}, expected one of {sorted(self.KINDS)}")
        if not 0.0 < min_signal_rate < max_signal_rate < 1.0:
            raise ValueError("need 0 < min_signal_rate < max_signal_rate < 1")
        self.kind = kind
        self.min_signal_rate = min_signal_rate
        self.max_signal_rate = max_signal_rate

    def __call__(self, diffusion_times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.KINDS[self.kind](diffusion_times, self.min_signal_rate, self.max_signal_rate)
