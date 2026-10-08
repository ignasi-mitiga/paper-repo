"""Training tests on the small synthetic configuration."""

import dataclasses

import pytest
import torch
from torch.utils.data import DataLoader

from merizzi2024.data.memmap import bilinear_baseline
from merizzi2024.evaluate import evaluate
from merizzi2024.metrics import batch_metrics
from merizzi2024.training import build_dataset, build_model, train


def _loss_curve(cfg, steps: int, lr: float = 1e-3, seed: int = 0) -> list[float]:
    torch.manual_seed(seed)
    model = build_model(cfg)
    loader = DataLoader(build_dataset(cfg, cfg.data.train_split), batch_size=cfg.train.batch_size,
                        shuffle=True, drop_last=True, generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.AdamW(model.network.parameters(), lr=lr, weight_decay=1e-5)
    losses: list[float] = []
    gen = torch.Generator().manual_seed(seed)
    while len(losses) < steps:
        for batch in loader:
            loss = model.training_losses(batch, generator=gen)["noise_loss"]
            opt.zero_grad()
            loss.backward()
            opt.step()
            model.update_ema()
            losses.append(float(loss))
            if len(losses) >= steps:
                break
    return losses


def test_noise_loss_decreases(small_cfg):
    losses = _loss_curve(small_cfg, steps=60)
    first, last = sum(losses[:10]) / 10, sum(losses[-10:]) / 10
    assert last < 0.7 * first, (first, last)


def test_lightning_train_and_evaluate_end_to_end(small_cfg, tmp_path):
    cfg = dataclasses.replace(small_cfg, train=dataclasses.replace(
        small_cfg.train, max_steps=6, val_every_n_steps=3, val_batches=1, num_workers=0))
    ckpt = train(cfg, out_dir=tmp_path / "run", enable_progress_bar=False)
    assert ckpt.exists()
    results = evaluate(cfg, ckpt, max_batches=1, diffusion_steps=2, ensemble_size=2,
                       batch_size=4, out_dir=tmp_path / "eval", device="cpu")
    for method in ("bilinear", "single", "ensemble"):
        assert set(results["test"][method]) == {"mse", "psnr", "ssim", "batches"}
    assert (tmp_path / "eval" / "samples_test.png").exists()


@pytest.mark.slow
def test_ensemble_diffusion_beats_bilinear(small_cfg):
    """The paper's claim in miniature: after training, the sampled field is closer to the
    high-res target than bilinear upsampling of the low-res input."""
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(small_cfg).to(device)
    train_loader = DataLoader(build_dataset(small_cfg, "train"), batch_size=8, shuffle=True,
                              drop_last=True, generator=torch.Generator().manual_seed(0))
    opt = torch.optim.AdamW(model.network.parameters(), lr=1e-3, weight_decay=1e-5)
    step = 0
    while step < 1200:
        for batch in train_loader:
            loss = model.training_losses(batch.to(device))["noise_loss"]
            opt.zero_grad()
            loss.backward()
            opt.step()
            model.update_ema(decay=0.99)
            step += 1
            if step >= 1200:
                break
    test_batch = torch.stack([build_dataset(small_cfg, "test")[i] for i in range(16)]).to(device)
    target = test_batch[:, -1:]
    gen = torch.Generator(device=device).manual_seed(0)
    ensemble = model.sample_ensemble(test_batch[:, :4], diffusion_steps=5, ensemble_size=8,
                                     generator=gen)
    baseline = bilinear_baseline(test_batch, small_cfg.data.frame_offsets)
    mse_model = batch_metrics(ensemble, target)["mse"]
    mse_bilinear = batch_metrics(baseline, target)["mse"]
    assert mse_model < mse_bilinear, (mse_model, mse_bilinear)
