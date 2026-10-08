# merizzi-2024: ERA5 to CERRA wind-speed super-resolution via diffusion, in PyTorch

A clean PyTorch reimplementation of the model introduced in

> Merizzi F., Asperti A., Colamonaco S. (2024). *Wind speed super-resolution and validation:
> from ERA5 to CERRA via diffusion models.* Neural Computing and Applications 36, 21899–21921.
> https://doi.org/10.1007/s00521-024-10139-9

The paper downscales 10 m wind speed from the ERA5 reanalysis (0.25°, 52×52 over Italy) to the
CERRA regional reanalysis (0.05°, 256×256) with a conditional DDIM: a residual U-Net denoiser
conditioned on four bilinearly pre-upsampled low-res frames (t−6h, t−3h, t0, t+3h), sampled
for 5 steps, 15 times, and averaged ("ensemble diffusion"). This project ports that model and
its training and evaluation protocol from the authors' TensorFlow/Keras code, keeps the
paper-text variants as config switches, and ships two datasets that need no credentials:

* a **synthetic** blur-and-downsample stand-in used by the tests (`make-synthetic`), and
* a **real** example, ERA5 10 m wind speed from the public WeatherBench 2 bucket
  (`download-era5`): 128×64 (2.8°) → 512×256 (0.7°), global, 6-hourly, train 2010–2019,
  test 2009 and 2020.

The authors' own Kaggle `.npy` files use the same on-disk layout and drop into the same
loader. Everything else, including every place where the released code and the paper text
disagree, is written up in [docs/notes.md](docs/notes.md).

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[era5,dev]"      # torch, lightning, scikit-image, xarray/zarr/gcsfs, pytest, ruff
```

A CUDA build of PyTorch is picked up automatically if one is available for your platform.

## Quick start (synthetic data, a couple of minutes on a laptop)

```bash
merizzi2024 make-synthetic --config configs/small.yaml   # writes data/synthetic-small/*.npy
merizzi2024 train --config configs/small.yaml            # 300 steps, runs/small/
merizzi2024 evaluate --config configs/small.yaml --checkpoint runs/small/checkpoints/last.ckpt
```

`evaluate` prints MSE, PSNR and SSIM for bilinear upsampling, single diffusion and ensemble
diffusion, and saves a side-by-side figure under `runs/<name>/eval/`.

## Real data (ERA5 from WeatherBench 2, about 10 GB)

```bash
merizzi2024 download-era5 --config configs/paper.yaml    # anonymous GCS reads, writes data/era5-wb2/
merizzi2024 train --config configs/paper.yaml --max-time 00:01:00:00   # or the full 110k steps
merizzi2024 evaluate --config configs/paper.yaml --checkpoint runs/paper/checkpoints/last.ckpt \
    --max-batches 10                                     # drop --max-batches for the full years
```

The prepare commands write the training-split maxima and standardisation statistics back
into the YAML, where the authors hard-code them in `setup.py`.

## Tests

```bash
pytest            # shape, schedule, data and a short "loss goes down" training test
pytest -m slow    # trains the small config and asserts ensemble diffusion beats bilinear
```

## Layout

```
configs/         small.yaml (tests, synthetic) and paper.yaml (paper-size, ERA5 example)
src/merizzi2024/ model/ (unet, schedule, diffusion), data/ (memmap, synthetic, era5_wb2),
                 training.py (Lightning), evaluate.py, metrics.py, cli.py, config.py
tests/
docs/            notes.md (paper summary, architecture, discrepancies, data), the paper PDF
```

License: MIT (this code). The paper is CC BY 4.0; ERA5 data is provided by the Copernicus
Climate Change Service and redistributed by WeatherBench 2.
