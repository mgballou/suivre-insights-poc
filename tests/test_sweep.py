import numpy as np
import pandas as pd
from insights.config import SimConfig
from insights.sweep import is_hit, cell_metrics, run_sweep


def test_is_hit_true_when_trigger_top3_and_above_noise():
    df = pd.DataFrame({
        "tag": ["tag_0", "tag_1", "tag_2", "tag_3"],
        "lift": [3.0, 0.2, 0.1, 0.0],
        "d": [1.0, 0.1, 0.0, 0.0], "n_exposed": [10] * 4, "n_occurrences": [5] * 4,
    })
    assert is_hit(df, true_idx=(0,), k=3) is True


def test_is_hit_false_when_trigger_buried():
    df = pd.DataFrame({
        "tag": ["tag_1", "tag_2", "tag_3", "tag_0"],
        "lift": [3.0, 2.0, 1.0, 0.05],
        "d": [1, 1, 1, 0], "n_exposed": [10] * 4, "n_occurrences": [5] * 4,
    })
    assert is_hit(df, true_idx=(0,), k=3) is False


def test_strong_clean_signal_has_high_hit_rate():
    c = SimConfig(days=180, n_tags=5, true_trigger_idx=(0,), effect_points=3.0,
                  flare_phi=0.0, flare_sd=0.0, confounder_strength=0.0,
                  confounding_path=False, cooccur_pairs=(), noise_sd=0.5)
    m = cell_metrics(c, n_datasets=40)
    assert m["hit_rate"] > 0.8


def test_run_sweep_shape():
    base = SimConfig(n_tags=4, true_trigger_idx=(0,), cooccur_pairs=(),
                     confounding_path=False)
    df = run_sweep(base, days_grid=[60, 120], effect_grid=[1.0, 2.0], n_datasets=15)
    assert len(df) == 4
    assert {"days", "effect_points", "hit_rate", "fp_rate"} <= set(df.columns)
