"""The denoising residual U-Net, ported layer by layer from the authors' ``denoising_unet.py``.

Layout for the paper configuration (widths 64/128/256/384, block depth 3):

* 1x1 conv of the 5 input channels (4 upsampled low-res frames + the noisy target) to 64
  channels, concatenated with a sinusoidal embedding of the noise variance tiled over the image;
* three down stages (widths 64, 128, 256), each ``block_depth`` residual blocks followed by
  2x2 average pooling, every block output kept as a skip connection;
* a bottleneck of ``block_depth`` residual blocks at width 384 that, in the released code,
  carry an additive gated-attention branch (``ResidualBlockWithAttention``); the paper text
  does not mention it, so ``bottleneck_attention=False`` gives the plain variant;
* three up stages (widths 256, 128, 64): bilinear 2x upsampling, then ``block_depth`` times
  "concatenate the matching skip, residual block";
* a zero-initialised 1x1 conv to the single output channel (the predicted noise).

Residual blocks are LayerNorm over channels (Keras ``LayerNormalization(axis=-1)``, eps 1e-3),
3x3 conv + SiLU, 3x3 conv, plus a 1x1 projection of the input when the width changes.
Feature maps are channels-first; the spatial size must be divisible by 8.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm2d(nn.Module):
    """Normalises each pixel over its channels, like Keras LayerNormalization(axis=-1)."""

    def __init__(self, channels: int, eps: float = 1e-3):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) * torch.rsqrt(var + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class SinusoidalEmbedding(nn.Module):
    """Embeds the scalar noise variance into sin/cos features at log-spaced frequencies."""

    def __init__(self, dims: int, min_frequency: float = 1.0, max_frequency: float = 1000.0):
        super().__init__()
        if dims % 2:
            raise ValueError("embedding_dims must be even")
        frequencies = torch.exp(
            torch.linspace(math.log(min_frequency), math.log(max_frequency), dims // 2)
        )
        self.register_buffer("angular_speeds", 2.0 * math.pi * frequencies, persistent=False)
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 1, 1) -> (B, dims, 1, 1)
        angles = self.angular_speeds[None, :, None, None] * x
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, width: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, width, 1) if in_channels != width else nn.Identity()
        self.norm = LayerNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, width, 3, padding=1)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)

    def _branch(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.silu(self.conv1(h))
        return self.conv2(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._branch(x) + self.proj(x)


class GatedAttentionResidualBlock(ResidualBlock):
    """``ResidualBlockWithAttention``: the residual acts as a gate on the conv branch."""

    def __init__(self, in_channels: int, width: int):
        super().__init__(in_channels, width)
        self.theta = nn.Conv2d(width, width, 1)
        self.phi = nn.Conv2d(width, width, 1)
        self.psi = nn.Conv2d(width, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        h = self._branch(x)
        gate = torch.sigmoid(self.psi(F.relu(self.theta(h) + self.phi(residual))))
        return h * gate + residual


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, width: int, block_depth: int):
        super().__init__()
        blocks = []
        for i in range(block_depth):
            blocks.append(ResidualBlock(in_channels if i == 0 else width, width))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
            skips.append(x)
        return F.avg_pool2d(x, 2)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, width: int, block_depth: int):
        super().__init__()
        blocks = []
        for i in range(block_depth):
            blocks.append(ResidualBlock((in_channels if i == 0 else width) + skip_channels, width))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        for block in self.blocks:
            x = torch.cat([x, skips.pop()], dim=1)
            x = block(x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 1,
        widths: tuple[int, ...] = (64, 128, 256, 384),
        block_depth: int = 3,
        embedding_dims: int = 64,
        embedding_max_frequency: float = 1000.0,
        bottleneck_attention: bool = True,
    ):
        super().__init__()
        if len(widths) < 2:
            raise ValueError("need at least one down/up stage plus a bottleneck width")
        self.widths = tuple(widths)
        self.num_stages = len(widths) - 1
        self.embedding = SinusoidalEmbedding(embedding_dims, max_frequency=embedding_max_frequency)
        self.stem = nn.Conv2d(in_channels, widths[0], 1)

        channels = widths[0] + embedding_dims
        self.down_blocks = nn.ModuleList()
        for width in widths[:-1]:
            self.down_blocks.append(DownBlock(channels, width, block_depth))
            channels = width

        bottleneck_cls = GatedAttentionResidualBlock if bottleneck_attention else ResidualBlock
        bottleneck = []
        for i in range(block_depth):
            bottleneck.append(bottleneck_cls(channels if i == 0 else widths[-1], widths[-1]))
        self.bottleneck = nn.ModuleList(bottleneck)
        channels = widths[-1]

        self.up_blocks = nn.ModuleList()
        for width in reversed(widths[:-1]):
            self.up_blocks.append(UpBlock(channels, width, width, block_depth))
            channels = width

        self.head = nn.Conv2d(channels, out_channels, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, images: torch.Tensor, noise_variances: torch.Tensor) -> torch.Tensor:
        """images: (B, C, H, W); noise_variances: (B, 1, 1, 1) -> predicted noise (B, 1, H, W)."""
        b, _, h, w = images.shape
        divisor = 2**self.num_stages
        if h % divisor or w % divisor:
            raise ValueError(f"spatial size {(h, w)} must be divisible by {divisor}")
        e = self.embedding(noise_variances.reshape(b, 1, 1, 1).to(images.dtype))
        e = e.expand(-1, -1, h, w)
        x = torch.cat([self.stem(images), e], dim=1)
        skips: list[torch.Tensor] = []
        for block in self.down_blocks:
            x = block(x, skips)
        for block in self.bottleneck:
            x = block(x)
        for block in self.up_blocks:
            x = block(x, skips)
        return self.head(x)
