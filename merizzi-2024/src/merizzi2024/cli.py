"""Command line interface: ``merizzi2024 <command> --config configs/<name>.yaml``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from merizzi2024.config import load_config


def cmd_make_synthetic(args: argparse.Namespace) -> None:
    from merizzi2024.data.synthetic import make_synthetic_dataset

    cfg = load_config(args.config)
    if args.root:
        cfg.data.root = args.root
    if args.num_train:
        cfg.synthetic.num_train = args.num_train
    if args.num_test:
        cfg.synthetic.num_test = args.num_test
    written = make_synthetic_dataset(cfg, write_config=not args.no_write_config)
    for split, files in written.items():
        print(f"{split}: hr={files['hr']} lr={files['lr']}")
    print(f"hr_max={cfg.data.hr_max:.4f} lr_max={cfg.data.lr_max:.4f}")


def cmd_prepare(args: argparse.Namespace) -> None:
    """Compute the normalisation entries for data files that already exist (e.g. Kaggle)."""
    from merizzi2024.config import write_normalisation
    from merizzi2024.data.synthetic import fill_normalisation

    cfg = load_config(args.config)
    for split in cfg.data.splits:
        for path in cfg.data.split_paths(split):
            if not path.exists():
                raise SystemExit(f"missing data file: {path}")
    fill_normalisation(cfg)
    print(f"hr_max={cfg.data.hr_max:.6f} lr_max={cfg.data.lr_max:.6f} stats={cfg.data.stats}")
    if not args.no_write_config:
        print(f"written into {write_normalisation(cfg)}")


def cmd_download_era5(args: argparse.Namespace) -> None:
    from merizzi2024.data.era5_wb2 import download_era5

    cfg = load_config(args.config)
    download_era5(cfg, splits=args.splits, overwrite=args.overwrite)


def cmd_train(args: argparse.Namespace) -> None:
    from merizzi2024.training import train

    cfg = load_config(args.config)
    last = train(cfg, max_steps=args.max_steps, max_time=args.max_time, out_dir=args.out_dir,
                 resume=args.resume, enable_progress_bar=not args.no_progress)
    print(f"last checkpoint: {last}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    from merizzi2024.evaluate import evaluate, format_table, save_results

    cfg = load_config(args.config)
    results = evaluate(cfg, args.checkpoint, splits=args.splits, methods=tuple(args.methods),
                       max_batches=args.max_batches, diffusion_steps=args.steps,
                       ensemble_size=args.ensemble_size, batch_size=args.batch_size,
                       out_dir=args.out_dir, save_images=not args.no_images, seed=args.seed,
                       amp=not args.no_amp)
    print(format_table(results))
    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg.train.out_dir) / "eval"
    save_results(results, out_dir / "results.json")
    print(f"written {out_dir / 'results.json'}")


def cmd_summary(args: argparse.Namespace) -> None:
    from merizzi2024.model.unet import UNet

    cfg = load_config(args.config)
    net = UNet(in_channels=cfg.data.num_frames + 1, widths=tuple(cfg.model.widths),
               block_depth=cfg.model.block_depth, embedding_dims=cfg.model.embedding_dims,
               embedding_max_frequency=cfg.model.embedding_max_frequency,
               bottleneck_attention=cfg.model.bottleneck_attention)
    info = {
        "config": cfg.name,
        "hr_shape": list(cfg.data.hr_shape),
        "lr_shape": list(cfg.data.lr_shape),
        "frame_offsets": cfg.data.frame_offsets,
        "schedule": cfg.model.schedule,
        "bottleneck_attention": cfg.model.bottleneck_attention,
        "denoiser_parameters": sum(p.numel() for p in net.parameters()),
        "hr_max": cfg.data.hr_max,
        "lr_max": cfg.data.lr_max,
    }
    print(json.dumps(info, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="merizzi2024", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("make-synthetic", help="write synthetic blur-downscaling data")
    p.add_argument("--config", required=True)
    p.add_argument("--root", help="override data.root")
    p.add_argument("--num-train", type=int)
    p.add_argument("--num-test", type=int)
    p.add_argument("--no-write-config", action="store_true",
                   help="do not write the normalisation stats back into the YAML")
    p.set_defaults(func=cmd_make_synthetic)

    p = sub.add_parser("prepare", help="compute normalisation stats for existing .npy files")
    p.add_argument("--config", required=True)
    p.add_argument("--no-write-config", action="store_true")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("download-era5", help="fetch ERA5 wind speed from WeatherBench 2 (GCS)")
    p.add_argument("--config", required=True)
    p.add_argument("--splits", nargs="*")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_download_era5)

    p = sub.add_parser("train", help="train the diffusion model")
    p.add_argument("--config", required=True)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--max-time", help="Lightning max_time, e.g. 00:01:00:00 for one hour")
    p.add_argument("--out-dir")
    p.add_argument("--resume", help="checkpoint to resume from")
    p.add_argument("--no-progress", action="store_true")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="bilinear / single / ensemble metrics on test splits")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--splits", nargs="*")
    p.add_argument("--methods", nargs="*", default=["bilinear", "single", "ensemble"])
    p.add_argument("--max-batches", type=int)
    p.add_argument("--steps", type=int, help="diffusion steps (default: config)")
    p.add_argument("--ensemble-size", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--out-dir")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-images", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("summary", help="print the configuration and parameter count")
    p.add_argument("--config", required=True)
    p.set_defaults(func=cmd_summary)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
