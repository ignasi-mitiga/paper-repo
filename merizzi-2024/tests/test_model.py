import dataclasses

import pytest
import torch

from merizzi2024.model.diffusion import DiffusionSR
from merizzi2024.model.unet import UNet


def _count(net):
    return sum(p.numel() for p in net.parameters())


def test_paper_size_unet_builds_and_matches_reference_counts(paper_cfg):
    m = paper_cfg.model
    released = UNet(widths=tuple(m.widths), block_depth=m.block_depth,
                    embedding_dims=m.embedding_dims, bottleneck_attention=True)
    plain = UNet(widths=tuple(m.widths), block_depth=m.block_depth,
                 embedding_dims=m.embedding_dims, bottleneck_attention=False)
    # The released Keras denoiser (gated-attention bottleneck) and the paper-text variant.
    assert _count(released) == 20_905_668
    assert _count(plain) == 20_017_473
    x = torch.randn(1, 5, 32, 64)  # non-square, divisible by 8
    with torch.no_grad():
        out = released(x, torch.rand(1, 1, 1, 1))
    assert out.shape == (1, 1, 32, 64)
    assert float(out.abs().max()) == 0.0  # zero-initialised head


def test_unet_rejects_sizes_not_divisible_by_eight():
    net = UNet(widths=(8, 8, 8, 8), block_depth=1, embedding_dims=4)
    with pytest.raises(ValueError):
        net(torch.randn(1, 5, 12, 16), torch.rand(1, 1, 1, 1))


@pytest.mark.parametrize("schedule", ["cosine", "linear"])
@pytest.mark.parametrize("attention", [True, False])
@pytest.mark.parametrize("standardize", [True, False])
def test_small_variants_train_step_and_sampling(small_cfg, schedule, attention, standardize):
    model_cfg = dataclasses.replace(small_cfg.model, schedule=schedule,
                                    bottleneck_attention=attention)
    model = DiffusionSR(model_cfg, num_frames=4, standardize=standardize,
                        stats=small_cfg.data.stats)
    batch = torch.rand(2, 5, 64, 64)
    losses = model.training_losses(batch, generator=torch.Generator().manual_seed(0))
    assert losses["noise_loss"].requires_grad and torch.isfinite(losses["noise_loss"])
    single = model.sample(batch[:, :4], diffusion_steps=2)
    ensemble = model.sample_ensemble(batch[:, :4], diffusion_steps=2, ensemble_size=3)
    assert single.shape == ensemble.shape == (2, 1, 64, 64)
    assert float(single.min()) >= 0.0 and float(single.max()) <= 1.0


def test_ema_moves_towards_online_weights(small_cfg):
    model = DiffusionSR(small_cfg.model, num_frames=4, standardize=False)
    with torch.no_grad():
        for p in model.network.parameters():
            p.add_(1.0)
    before = [p.clone() for p in model.ema_network.parameters()]
    model.update_ema(decay=0.5)
    for b, e, p in zip(before, model.ema_network.parameters(), model.network.parameters(),
                       strict=True):
        assert torch.allclose(e, (b + p) / 2)
