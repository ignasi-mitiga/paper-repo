"""MSE, PSNR and SSIM exactly as the authors compute them (``utils.py``).

All three are computed on the [0, 1]-normalised fields (``data_range=1``). MSE is the mean over
the whole batch; PSNR and SSIM are computed per image with scikit-image's defaults (7x7 uniform
window for SSIM) and then averaged over the batch.
"""

from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().numpy()
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 4:  # (B, 1, H, W)
        x = x[:, 0]
    return x


def batch_metrics(pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray
                  ) -> dict[str, float]:
    pred, target = _to_numpy(pred), _to_numpy(target)
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {target.shape}")
    mse = float(np.mean((target - pred) ** 2))
    psnr = float(np.mean([peak_signal_noise_ratio(t, p, data_range=1.0)
                          for t, p in zip(target, pred, strict=True)]))
    ssim = float(np.mean([structural_similarity(t, p, data_range=1.0)
                          for t, p in zip(target, pred, strict=True)]))
    return {"mse": mse, "psnr": psnr, "ssim": ssim}


class MetricAccumulator:
    """Averages per-batch metrics over batches, like the authors' ``experiment`` loops."""

    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.count = 0

    def update(self, metrics: dict[str, float]) -> None:
        for k, v in metrics.items():
            self.sums[k] = self.sums.get(k, 0.0) + v
        self.count += 1

    def result(self) -> dict[str, float]:
        out = {k: v / max(self.count, 1) for k, v in self.sums.items()}
        out["batches"] = self.count
        return out
