from dataclasses import replace

import numpy as np

from insights.config import SimConfig
from insights.generate import generate_logs
from insights.lag_lift import rank_suspects
from insights.sweep import noise_band


def alert_precision(config: SimConfig, days: int, alert_threshold: float,
                     n_datasets: int = 300) -> dict:
    """Model a conservative "soft alert": a tag fires when its marginal lift
    both clears `alert_threshold` and exceeds the 95th-pct noise band of
    that dataset's non-true tags (same noise-band pattern as `sweep.is_hit`).
    """
    cfg = replace(config, days=days)
    true_tags = {f"tag_{i}" for i in cfg.true_trigger_idx}

    n_alerts = 0
    true_alerts = 0
    datasets_with_alert = 0

    for s in range(n_datasets):
        rng = np.random.default_rng(cfg.seed + s)
        df = generate_logs(cfg, rng)
        inten = df["intensity"].to_numpy()
        tags = np.column_stack([df[f"tag_{i}"].to_numpy() for i in range(cfg.n_tags)])
        suspects = rank_suspects(inten, tags, cfg.n_window)

        band = noise_band(suspects, true_tags, 95.0)

        fired = suspects[(suspects["lift"] >= alert_threshold) & (suspects["lift"] > band)]
        if len(fired) > 0:
            datasets_with_alert += 1
            n_alerts += len(fired)
            true_alerts += int(fired["tag"].isin(true_tags).sum())

    precision = (true_alerts / n_alerts) if n_alerts > 0 else float("nan")
    return {
        "fire_rate": datasets_with_alert / n_datasets,
        "precision": precision,
        "n_alerts": int(n_alerts),
        "true_alerts": int(true_alerts),
    }
