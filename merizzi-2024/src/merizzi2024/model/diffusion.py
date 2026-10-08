"""Conditional DDIM for sequence-to-one super-resolution (the authors' ``DiffusionModel``).

A sample is a tensor of ``num_frames + 1`` channels in [0, 1] units (each field divided by
its training maximum): the bilinearly pre-upsampled low-res frames followed by the high-res
target. As in the released code, the channels are standardised per field before the noise is
mixed in, the network predicts the noise, and the x0 estimate at each sampling step is
``(x_t - noise_rate * eps) / signal_rate``. Sampling alternates denoising and re-noising at the
next (lower) noise rate for ``diffusion_steps`` steps; the ensemble repeats it with fresh noise
and averages the clipped, de-standardised outputs.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn

from merizzi2024.config import ChannelStats, ModelConfig
from merizzi2024.model.schedule import NoiseSchedule
from merizzi2024.model.unet import UNet


class DiffusionSR(nn.Module):
    def __init__(self, model_cfg: ModelConfig, num_frames: int = 4, standardize: bool = True,
                 stats: ChannelStats | None = None):
        super().__init__()
        self.cfg = model_cfg
        self.num_frames = num_frames
        self.standardize = standardize
        self.network = UNet(
            in_channels=num_frames + 1,
            out_channels=1,
            widths=tuple(model_cfg.widths),
            block_depth=model_cfg.block_depth,
            embedding_dims=model_cfg.embedding_dims,
            embedding_max_frequency=model_cfg.embedding_max_frequency,
            bottleneck_attention=model_cfg.bottleneck_attention,
        )
        self.ema_network = copy.deepcopy(self.network).requires_grad_(False)
        self.schedule = NoiseSchedule(
            model_cfg.schedule, model_cfg.min_signal_rate, model_cfg.max_signal_rate
        )
        channels = num_frames + 1
        self.register_buffer("channel_mean", torch.zeros(1, channels, 1, 1))
        self.register_buffer("channel_std", torch.ones(1, channels, 1, 1))
        if stats is not None:
            self.set_stats(stats)

    # -- normalisation -------------------------------------------------------------------
    def set_stats(self, stats: ChannelStats) -> None:
        """Per-field standardisation statistics (computed by the data preparation step)."""
        means = [stats.lr_mean] * self.num_frames + [stats.hr_mean]
        stds = [stats.lr_var**0.5] * self.num_frames + [stats.hr_var**0.5]
        self.channel_mean.copy_(torch.tensor(means).view(1, -1, 1, 1))
        self.channel_std.copy_(torch.tensor(stds).view(1, -1, 1, 1))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.standardize:
            return x
        return (x - self.channel_mean) / self.channel_std

    def denormalize_target(self, x: torch.Tensor) -> torch.Tensor:
        """Standardised target channel -> [0, 1] units, clipped like the released code."""
        if self.standardize:
            x = x * self.channel_std[:, -1:] + self.channel_mean[:, -1:]
        return x.clamp(0.0, 1.0)

    # -- core ----------------------------------------------------------------------------
    def denoise(self, noisy_images: torch.Tensor, noise_rates: torch.Tensor,
                signal_rates: torch.Tensor, use_ema: bool) -> tuple[torch.Tensor, torch.Tensor]:
        network = self.ema_network if use_ema else self.network
        pred_noises = network(noisy_images, noise_rates**2)
        pred_images = (noisy_images[:, -1:] - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    def training_losses(self, batch: torch.Tensor, generator: torch.Generator | None = None,
                        use_ema: bool = False) -> dict[str, torch.Tensor]:
        """``batch``: (B, F+1, H, W) in [0, 1] units. Returns noise (training) and image losses."""
        images = self.normalize(batch)
        cond, target = images[:, :-1], images[:, -1:]
        noises = torch.randn(target.shape, generator=generator, device=batch.device,
                             dtype=batch.dtype)
        diffusion_times = torch.rand((batch.shape[0], 1, 1, 1), generator=generator,
                                     device=batch.device, dtype=batch.dtype)
        noise_rates, signal_rates = self.schedule(diffusion_times)
        noisy_target = signal_rates * target + noise_rates * noises
        pred_noises, pred_images = self.denoise(
            torch.cat([cond, noisy_target], dim=1), noise_rates, signal_rates, use_ema=use_ema
        )
        return {
            "noise_loss": F.l1_loss(pred_noises, noises),
            "image_loss": F.l1_loss(pred_images, target),
        }

    @torch.no_grad()
    def reverse_diffusion(self, cond: torch.Tensor, initial_noise: torch.Tensor,
                          diffusion_steps: int, use_ema: bool = True) -> torch.Tensor:
        """Algorithm 3 of the paper. ``cond`` is standardised; returns the standardised x0."""
        b = cond.shape[0]
        step_size = 1.0 / diffusion_steps
        next_noisy = initial_noise
        pred_images = initial_noise
        for step in range(diffusion_steps):
            noisy = next_noisy
            diffusion_times = torch.full((b, 1, 1, 1), 1.0 - step * step_size,
                                         device=cond.device, dtype=cond.dtype)
            noise_rates, signal_rates = self.schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                torch.cat([cond, noisy], dim=1), noise_rates, signal_rates, use_ema
            )
            next_noise_rates, next_signal_rates = self.schedule(diffusion_times - step_size)
            next_noisy = next_signal_rates * pred_images + next_noise_rates * pred_noises
        return pred_images

    @torch.no_grad()
    def sample(self, cond_raw: torch.Tensor, diffusion_steps: int,
               generator: torch.Generator | None = None, use_ema: bool = True) -> torch.Tensor:
        """``cond_raw``: (B, F, H, W) upsampled low-res frames in [0, 1] units -> (B, 1, H, W)."""
        if cond_raw.shape[1] != self.num_frames:
            raise ValueError(f"expected {self.num_frames} conditioning frames")
        cond = self.normalize(torch.cat([cond_raw, torch.zeros_like(cond_raw[:, :1])], 1))[:, :-1]
        noise = torch.randn((cond.shape[0], 1, *cond.shape[2:]), generator=generator,
                            device=cond.device, dtype=cond.dtype)
        x0 = self.reverse_diffusion(cond, noise, diffusion_steps, use_ema)
        return self.denormalize_target(x0)

    @torch.no_grad()
    def sample_ensemble(self, cond_raw: torch.Tensor, diffusion_steps: int, ensemble_size: int,
                        generator: torch.Generator | None = None, use_ema: bool = True
                        ) -> torch.Tensor:
        """Mean of ``ensemble_size`` independent samples (the paper's ensemble diffusion)."""
        total = None
        for _ in range(ensemble_size):
            out = self.sample(cond_raw, diffusion_steps, generator, use_ema)
            total = out if total is None else total + out
        return total / ensemble_size

    # -- EMA -----------------------------------------------------------------------------
    @torch.no_grad()
    def update_ema(self, decay: float | None = None) -> None:
        decay = self.cfg.ema_decay if decay is None else decay
        for p_ema, p in zip(self.ema_network.parameters(), self.network.parameters(), strict=True):
            p_ema.lerp_(p, 1.0 - decay)
        for b_ema, b in zip(self.ema_network.buffers(), self.network.buffers(), strict=True):
            b_ema.copy_(b)

    @torch.no_grad()
    def copy_weights_to_ema(self) -> None:
        self.ema_network.load_state_dict(self.network.state_dict())
