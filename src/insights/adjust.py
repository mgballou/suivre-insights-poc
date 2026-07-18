import numpy as np
import pandas as pd

from insights.config import SimConfig
from insights.generate import generate_logs
from insights.lag_lift import exposed_mask


def _quantile_bins(confounder: np.ndarray, n_strata: int) -> np.ndarray:
    """Bin `confounder` into up to `n_strata` quantile groups.

    Degenerate inputs (fewer unique values than bins, or a constant
    confounder) fall back to pandas' `duplicates="drop"` behaviour, which
    may yield fewer usable bins, or -- for a fully constant confounder --
    an all-NaN result meaning no stratification is possible at all.
    """
    try:
        bins = pd.qcut(pd.Series(confounder), n_strata, labels=False, duplicates="drop")
    except ValueError:
        bins = pd.Series(np.full(len(confounder), np.nan))
    return bins.to_numpy()


def stress_adjusted_lift(intensity: np.ndarray, tag_presence: np.ndarray,
                          confounder: np.ndarray, n_window: int, n_strata: int = 3) -> dict:
    intensity = np.asarray(intensity, dtype=float)
    tag_presence = np.asarray(tag_presence, dtype=bool)
    confounder = np.asarray(confounder, dtype=float)

    exposed = exposed_mask(tag_presence, n_window)
    baseline = ~exposed

    bins = _quantile_bins(confounder, n_strata)
    valid = ~pd.isna(bins)

    lifts, weights = [], []
    for b in np.unique(bins[valid]):
        in_stratum = valid & (bins == b)
        exp_s = intensity[in_stratum & exposed]
        base_s = intensity[in_stratum & baseline]
        if len(exp_s) == 0 or len(base_s) == 0:
            continue
        lifts.append(float(exp_s.mean() - base_s.mean()))
        weights.append(len(exp_s))

    if not weights:
        return {"lift": np.nan, "n_exposed": 0, "n_strata_used": 0}

    weights_arr = np.array(weights, dtype=float)
    lift = float(np.average(lifts, weights=weights_arr))
    return {
        "lift": lift,
        "n_exposed": int(weights_arr.sum()),
        "n_strata_used": len(weights),
    }


def rank_adjusted(intensity: np.ndarray, tag_matrix: np.ndarray, confounder: np.ndarray,
                   n_window: int, n_strata: int = 3) -> pd.DataFrame:
    rows = []
    for i in range(tag_matrix.shape[1]):
        r = stress_adjusted_lift(intensity, tag_matrix[:, i], confounder, n_window, n_strata)
        rows.append({"tag": f"tag_{i}", "lift": r["lift"], "n_exposed": r["n_exposed"]})
    df = pd.DataFrame(rows)[["tag", "lift", "n_exposed"]]
    return df.sort_values("lift", ascending=False, na_position="last").reset_index(drop=True)


def adjusted_damage(config: SimConfig, n_datasets: int = 300, k: int = 3,
                     n_strata: int = 3) -> float:
    spurious = [i for i in config.confounding_tag_idx
                if config.confounding_path and i not in config.true_trigger_idx]
    if not spurious:
        return float("nan")

    spurious_tags = {f"tag_{i}" for i in spurious}
    damage = 0
    for s in range(n_datasets):
        rng = np.random.default_rng(config.seed + s)
        df = generate_logs(config, rng)
        inten = df["intensity"].to_numpy()
        tags = np.column_stack([df[f"tag_{i}"].to_numpy() for i in range(config.n_tags)])
        confounder = df["stress"].to_numpy()
        ranked = rank_adjusted(inten, tags, confounder, config.n_window, n_strata)
        top = set(ranked.head(k)["tag"])
        if top & spurious_tags:
            damage += 1
    return damage / n_datasets
