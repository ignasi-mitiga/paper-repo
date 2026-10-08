"""Lightning wrapper around :class:`DiffusionSR`: the authors' custom train/test steps.

Training follows Algorithm 1 of the paper: random batches of windows from the training years,
noise-prediction MAE as the loss, AdamW, learning rate and weight decay decayed geometrically
from their initial to their final values over ``max_steps`` (the paper: 1e-4 -> 1e-5 and
1e-5 -> 1e-6), and an exponential moving average of the weights that is used for sampling.
Validation reports the losses with the EMA network and a quick single-sample reconstruction
of a few test batches against the bilinear baseline.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, Subset

from merizzi2024.config import Config, _build, _to_plain
from merizzi2024.data.memmap import MemmapPairDataset, bilinear_baseline
from merizzi2024.metrics import batch_metrics
from merizzi2024.model.diffusion import DiffusionSR


def build_model(cfg: Config) -> DiffusionSR:
    if cfg.data.hr_max is None or cfg.data.lr_max is None:
        raise ValueError("data.hr_max/lr_max are unset: run make-synthetic or download-era5 first")
    if cfg.data.standardize and cfg.data.stats is None:
        raise ValueError("data.standardize is on but data.stats is unset: rerun the prepare step")
    return DiffusionSR(cfg.model, num_frames=cfg.data.num_frames,
                       standardize=cfg.data.standardize, stats=cfg.data.stats)


def build_dataset(cfg: Config, split: str) -> MemmapPairDataset:
    hr_path, lr_path = cfg.data.split_paths(split)
    return MemmapPairDataset(hr_path, lr_path, cfg.data.frame_offsets, cfg.data.hr_max,
                             cfg.data.lr_max, cfg.data.hr_shape)


class PairDataModule(L.LightningDataModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: str | None = None) -> None:
        self.train_set = build_dataset(self.cfg, self.cfg.data.train_split)
        val_split = self.cfg.data.test_splits[0]
        full_val = build_dataset(self.cfg, val_split)
        n = min(len(full_val), self.cfg.train.val_batches * self.cfg.train.batch_size)
        self.val_set = Subset(full_val, list(range(n)))

    def train_dataloader(self) -> DataLoader:
        workers = self.cfg.train.num_workers
        return DataLoader(self.train_set, batch_size=self.cfg.train.batch_size, shuffle=True,
                          drop_last=True, num_workers=workers, persistent_workers=workers > 0,
                          pin_memory=torch.cuda.is_available())

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, batch_size=self.cfg.train.batch_size, shuffle=False,
                          num_workers=0)


class DiffusionLightningModule(L.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters({"config": _to_plain(cfg)})
        self.model = build_model(cfg)

    @classmethod
    def from_checkpoint(cls, path: str | Path, cfg: Config | None = None,
                        map_location: str | torch.device | None = "cpu",
                        ) -> DiffusionLightningModule:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        if cfg is None:
            cfg = _build(Config, ckpt["hyper_parameters"]["config"])
        module = cls(cfg)
        module.load_state_dict(ckpt["state_dict"])
        return module

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        losses = self.model.training_losses(batch)
        self.log("train/noise_loss", losses["noise_loss"], prog_bar=True, on_step=True)
        self.log("train/image_loss", losses["image_loss"], on_step=True)
        return losses["noise_loss"]

    def on_train_batch_start(self, batch: torch.Tensor, batch_idx: int) -> None:
        frac = min(self.global_step / max(self.cfg.train.max_steps, 1), 1.0)
        tc = self.cfg.train
        lr = tc.lr * math.exp(frac * math.log(tc.lr_final / tc.lr))
        wd = tc.weight_decay * math.exp(frac * math.log(tc.weight_decay_final / tc.weight_decay))
        for group in self.optimizers().param_groups:
            group["lr"] = lr
            group["weight_decay"] = wd
        self.log("train/lr", lr, on_step=True)

    def on_train_batch_end(self, outputs, batch: torch.Tensor, batch_idx: int) -> None:
        self.model.update_ema()

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        losses = self.model.training_losses(batch, use_ema=True)
        self.log("val/noise_loss", losses["noise_loss"], prog_bar=True)
        self.log("val/image_loss", losses["image_loss"])
        cond, target = batch[:, :-1], batch[:, -1:]
        sample = self.model.sample(cond, self.cfg.sample.diffusion_steps)
        for name, pred in (("single", sample),
                           ("bilinear", bilinear_baseline(batch, self.cfg.data.frame_offsets))):
            m = batch_metrics(pred, target)
            for k, v in m.items():
                self.log(f"val/{name}_{k}", v, prog_bar=(k == "mse"))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        tc = self.cfg.train
        return torch.optim.AdamW(self.model.network.parameters(), lr=tc.lr,
                                 weight_decay=tc.weight_decay)


def train(cfg: Config, max_steps: int | None = None, max_time: str | None = None,
          out_dir: str | Path | None = None, resume: str | Path | None = None,
          enable_progress_bar: bool = True) -> Path:
    """Train and return the path of the last checkpoint."""
    if max_steps is not None:
        cfg = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, max_steps=max_steps))
    out_dir = Path(out_dir) if out_dir is not None else Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    L.seed_everything(cfg.train.seed, workers=True)
    module = DiffusionLightningModule(cfg)
    datamodule = PairDataModule(cfg)
    checkpoint = ModelCheckpoint(dirpath=out_dir / "checkpoints", filename="step{step:07d}",
                                 auto_insert_metric_name=False, save_last=True, save_top_k=2,
                                 monitor="val/single_mse", mode="min",
                                 every_n_train_steps=cfg.train.val_every_n_steps)
    trainer = L.Trainer(
        max_steps=cfg.train.max_steps,
        max_time=max_time,
        precision=cfg.train.precision,
        accelerator="auto",
        devices=1,
        val_check_interval=cfg.train.val_every_n_steps,
        check_val_every_n_epoch=None,
        limit_val_batches=cfg.train.val_batches,
        log_every_n_steps=10,
        callbacks=[checkpoint],
        logger=CSVLogger(str(out_dir), name="logs"),
        default_root_dir=str(out_dir),
        enable_progress_bar=enable_progress_bar,
        enable_model_summary=False,
    )
    trainer.fit(module, datamodule=datamodule, ckpt_path=str(resume) if resume else None)
    last = out_dir / "checkpoints" / "last.ckpt"
    if not last.exists():  # max_steps smaller than the checkpoint interval
        trainer.save_checkpoint(last)
    return last
