"""Evaluation in the authors' protocol: walk a test split sequentially in full batches and
average per-batch MSE / PSNR / SSIM for bilinear upsampling, single diffusion and ensemble
diffusion (``ensemble_size`` samples of ``diffusion_steps`` steps, averaged)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from merizzi2024.config import Config
from merizzi2024.data.memmap import bilinear_baseline
from merizzi2024.metrics import MetricAccumulator, batch_metrics
from merizzi2024.model.diffusion import DiffusionSR
from merizzi2024.training import DiffusionLightningModule, build_dataset

METHODS = ("bilinear", "single", "ensemble")


def load_model(cfg: Config, checkpoint: str | Path, device: torch.device) -> DiffusionSR:
    module = DiffusionLightningModule.from_checkpoint(checkpoint, cfg)
    return module.model.to(device).eval()


@torch.no_grad()
def predict_batch(model: DiffusionSR, batch: torch.Tensor, cfg: Config, methods: tuple[str, ...],
                  diffusion_steps: int, ensemble_size: int, generator: torch.Generator | None,
                  amp: bool) -> dict[str, torch.Tensor]:
    cond = batch[:, :-1]
    out: dict[str, torch.Tensor] = {}
    device_type = "cuda" if batch.is_cuda else "cpu"
    with torch.autocast(device_type=device_type, dtype=torch.float16,
                        enabled=amp and batch.is_cuda):
        if "bilinear" in methods:
            out["bilinear"] = bilinear_baseline(batch, cfg.data.frame_offsets)
        if "single" in methods:
            out["single"] = model.sample(cond, diffusion_steps, generator)
        if "ensemble" in methods:
            out["ensemble"] = model.sample_ensemble(cond, diffusion_steps, ensemble_size, generator)
    return {k: v.float() for k, v in out.items()}


def evaluate(cfg: Config, checkpoint: str | Path, splits: list[str] | None = None,
             methods: tuple[str, ...] = METHODS, max_batches: int | None = None,
             diffusion_steps: int | None = None, ensemble_size: int | None = None,
             batch_size: int | None = None, out_dir: str | Path | None = None,
             save_images: bool = True, seed: int = 0, amp: bool = True,
             device: str | None = None) -> dict[str, dict[str, dict[str, float]]]:
    device_ = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(cfg, checkpoint, device_)
    diffusion_steps = diffusion_steps or cfg.sample.diffusion_steps
    ensemble_size = ensemble_size or cfg.sample.ensemble_size
    batch_size = batch_size or cfg.sample.batch_size
    splits = splits or cfg.data.test_splits
    out_dir = Path(out_dir) if out_dir is not None else Path(cfg.train.out_dir) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, dict[str, float]]] = {}
    for split in splits:
        dataset = build_dataset(cfg, split)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        generator = torch.Generator(device=device_).manual_seed(seed)
        accumulators = {m: MetricAccumulator() for m in methods}
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            batch = batch.to(device_, non_blocking=True)
            preds = predict_batch(model, batch, cfg, methods, diffusion_steps, ensemble_size,
                                  generator, amp)
            target = batch[:, -1:]
            for m, pred in preds.items():
                accumulators[m].update(batch_metrics(pred, target))
            if i == 0 and save_images:
                save_sample_figure(preds, target, out_dir / f"samples_{split}.png", split)
        results[split] = {m: acc.result() for m, acc in accumulators.items()}
    return results


def save_sample_figure(preds: dict[str, torch.Tensor], target: torch.Tensor, path: Path,
                       title: str, rows: int = 4) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = list(preds) + ["target"]
    tensors = dict(preds, target=target)
    rows = min(rows, target.shape[0])
    fig, axes = plt.subplots(rows, len(columns), figsize=(3.2 * len(columns), 2.6 * rows),
                             squeeze=False)
    vmax = float(target.max())
    for r in range(rows):
        for c, name in enumerate(columns):
            ax = axes[r, c]
            ax.imshow(tensors[name][r, 0].cpu().numpy(), cmap="viridis", vmin=0, vmax=vmax)
            ax.set_axis_off()
            if r == 0:
                ax.set_title(name)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def format_table(results: dict[str, dict[str, dict[str, float]]]) -> str:
    lines = ["| split | method | batches | MSE | PSNR | SSIM |", "|---|---|---|---|---|---|"]
    for split, per_method in results.items():
        for method, m in per_method.items():
            lines.append(f"| {split} | {method} | {int(m['batches'])} | {m['mse']:.3e} | "
                         f"{m['psnr']:.2f} | {m['ssim']:.3f} |")
    return "\n".join(lines)


def save_results(results: dict, path: str | Path) -> None:
    import json

    with open(path, "w") as fh:
        json.dump(results, fh, indent=2)
    np.set_printoptions(precision=4)
