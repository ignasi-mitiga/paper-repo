import numpy as np
import torch

from merizzi2024.data.memmap import MemmapPairDataset, bilinear_baseline, upsample_frames
from merizzi2024.data.synthetic import degrade, generate_wind_sequence


def test_synthetic_files_have_authors_layout(small_cfg):
    hr_path, lr_path = small_cfg.data.split_paths("train")
    hr = np.load(hr_path, mmap_mode="r")
    lr = np.load(lr_path, mmap_mode="r")
    assert hr.dtype == np.float32 and lr.dtype == np.float32
    assert hr.shape == (96, 64, 64) and lr.shape == (96, 13, 13)
    assert float(hr.min()) >= 0.0
    assert small_cfg.data.hr_max == float(hr.max())
    assert small_cfg.data.stats is not None and small_cfg.data.stats.hr_var > 0


def test_degradation_is_blur_then_area_average():
    rng = np.random.default_rng(0)
    hr = generate_wind_sequence(6, (64, 64), rng)
    lr = degrade(hr, (13, 13), blur_sigma=1.5)
    assert lr.shape == (6, 13, 13)
    # area averaging preserves the mean closely; blurring removes fine detail
    assert abs(lr.mean() - hr.mean()) < 0.05 * hr.mean()
    assert lr.std() < hr.std()


def test_sequence_is_temporally_coherent():
    rng = np.random.default_rng(1)
    hr = generate_wind_sequence(12, (32, 32), rng)
    neighbour = np.mean((hr[1:] - hr[:-1]) ** 2)
    far = np.mean((hr[6:] - hr[:-6]) ** 2)
    assert neighbour < far


def test_window_indexing_matches_authors_generator(small_cfg):
    hr_path, lr_path = small_cfg.data.split_paths("test")
    ds = MemmapPairDataset(hr_path, lr_path, [-2, -1, 0, 1], small_cfg.data.hr_max,
                           small_cfg.data.lr_max, (64, 64))
    lr = np.load(lr_path, mmap_mode="r")
    hr = np.load(hr_path, mmap_mode="r")
    assert len(ds) == 40 - 3
    sample = ds[5]
    t = ds.anchor(5)
    assert t == 7
    assert sample.shape == (5, 64, 64)
    expected_cond = upsample_frames(torch.from_numpy(np.stack(lr[t - 2 : t + 2]) / ds.lr_max),
                                    (64, 64))
    assert torch.allclose(sample[:4], expected_cond)
    assert torch.allclose(sample[4], torch.from_numpy(hr[t] / ds.hr_max))
    assert torch.allclose(bilinear_baseline(sample[None], [-2, -1, 0, 1])[0, 0], sample[2])
