import shutil
from pathlib import Path

from merizzi2024.cli import main
from merizzi2024.config import ChannelStats, load_config, write_normalisation

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_write_normalisation_keeps_comments(tmp_path):
    path = tmp_path / "cfg.yaml"
    shutil.copy(CONFIGS / "kaggle-italy.yaml", path)
    cfg = load_config(path)
    assert cfg.data.hr_max is None and cfg.data.stats is None
    cfg.data.hr_max, cfg.data.lr_max = 31.347172, 26.298004
    cfg.data.stats = ChannelStats(lr_mean=0.2, lr_var=0.01, hr_mean=0.15, hr_var=0.008)
    write_normalisation(cfg)
    text = path.read_text()
    assert "# Table 7" in text and "# t-6h, t-3h, t0, t+3h" in text  # comments survive
    reloaded = load_config(path)
    assert reloaded.data.hr_max == 31.347172 and reloaded.data.lr_max == 26.298004
    assert reloaded.data.stats == cfg.data.stats
    # a second write replaces the nested block instead of duplicating it
    cfg.data.stats = ChannelStats(lr_mean=0.3, lr_var=0.02, hr_mean=0.1, hr_var=0.009)
    write_normalisation(cfg)
    assert path.read_text().count("lr_mean:") == 1
    assert load_config(path).data.stats.lr_mean == 0.3
    assert load_config(path).model.widths == [64, 128, 256, 384]


def test_prepare_command_fills_stats_for_existing_files(small_cfg, tmp_path):
    path = tmp_path / "small.yaml"
    text = (CONFIGS / "small.yaml").read_text().replace("root: data/synthetic-small",
                                                        f"root: {small_cfg.data.root}")
    path.write_text(text)
    main(["prepare", "--config", str(path)])
    cfg = load_config(path)
    assert cfg.data.hr_max == small_cfg.data.hr_max
    assert cfg.data.stats == small_cfg.data.stats
