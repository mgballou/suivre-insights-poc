from dataclasses import replace

import numpy as np
import pandas as pd

from insights.config import SimConfig
from insights.generate import generate_logs
from insights.lag_lift import rank_suspects


def is_hit(suspects: pd.DataFrame, true_idx: tuple[int, ...], k: int = 3,
           noise_pct: float = 95.0) -> bool:
    true_tags = {f"tag_{i}" for i in true_idx}
    top = suspects.head(k)["tag"].tolist()
    if not any(t in true_tags for t in top):
        return False
    noise = suspects[~suspects["tag"].isin(true_tags)]["lift"].dropna()
    band = np.percentile(noise, noise_pct) if len(noise) else -np.inf
    best_true = suspects[suspects["tag"].isin(true_tags)]["lift"].max()
    return bool(best_true > band)


def cell_metrics(config: SimConfig, n_datasets: int = 300, k: int = 3) -> dict:
    hits = fps = 0
    damage = 0
    # a confounding-path tag that is NOT a true trigger = pure spurious candidate.
    spurious = [i for i in config.confounding_tag_idx
                if config.confounding_path and i not in config.true_trigger_idx]
    true_tags = {f"tag_{i}" for i in config.true_trigger_idx}
    for s in range(n_datasets):
        rng = np.random.default_rng(config.seed + s)
        df = generate_logs(config, rng)
        inten = df["intensity"].to_numpy()
        tags = np.column_stack([df[f"tag_{i}"].to_numpy() for i in range(config.n_tags)])
        suspects = rank_suspects(inten, tags, config.n_window)
        if is_hit(suspects, config.true_trigger_idx, k=k):
            hits += 1
        top = set(suspects.head(k)["tag"])
        if top - true_tags:                       # any non-true tag in the top-k
            fps += 1
        if spurious and top & {f"tag_{i}" for i in spurious}:
            damage += 1
    return {
        "hit_rate": hits / n_datasets,
        "fp_rate": fps / n_datasets,
        "confounding_damage": (damage / n_datasets) if spurious else float("nan"),
    }


def run_sweep(base: SimConfig, days_grid, effect_grid, n_datasets: int = 300) -> pd.DataFrame:
    rows = []
    for d in days_grid:
        for e in effect_grid:
            cfg = replace(base, days=d, effect_points=e)
            m = cell_metrics(cfg, n_datasets=n_datasets)
            rows.append({"days": d, "effect_points": e,
                         "hit_rate": m["hit_rate"], "fp_rate": m["fp_rate"]})
    return pd.DataFrame(rows)
