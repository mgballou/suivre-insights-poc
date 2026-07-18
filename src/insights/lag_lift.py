import numpy as np
import pandas as pd


def exposed_mask(tag_presence: np.ndarray, n_window: int) -> np.ndarray:
    d = len(tag_presence)
    mask = np.zeros(d, dtype=bool)
    for t in np.flatnonzero(tag_presence):
        mask[t : min(t + n_window + 1, d)] = True
    return mask


def _pooled_sd(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    num = a.var(ddof=1) * (na - 1) + b.var(ddof=1) * (nb - 1)
    return float(np.sqrt(num / (na + nb - 2)))


def _lift_from_masks(intensity, exposed, baseline, n_occurrences) -> dict:
    if exposed.sum() == 0 or baseline.sum() == 0:
        return {"lift": np.nan, "d": np.nan, "n_exposed": int(exposed.sum()),
                "n_occurrences": int(n_occurrences), "exposed_mean": np.nan,
                "baseline_mean": np.nan}
    e, b = intensity[exposed], intensity[baseline]
    sd = _pooled_sd(e, b)
    lift = float(e.mean() - b.mean())
    return {"lift": lift, "d": (lift / sd) if sd > 0 else 0.0,
            "n_exposed": int(exposed.sum()), "n_occurrences": int(n_occurrences),
            "exposed_mean": float(e.mean()), "baseline_mean": float(b.mean())}


def lift_for_tag(intensity: np.ndarray, tag_presence: np.ndarray, n_window: int) -> dict:
    exp = exposed_mask(tag_presence, n_window)
    return _lift_from_masks(intensity, exp, ~exp, tag_presence.sum())


def rank_suspects(intensity: np.ndarray, tag_matrix: np.ndarray, n_window: int) -> pd.DataFrame:
    rows = []
    for i in range(tag_matrix.shape[1]):
        r = lift_for_tag(intensity, tag_matrix[:, i], n_window)
        r["tag"] = f"tag_{i}"
        rows.append(r)
    df = pd.DataFrame(rows)[["tag", "lift", "d", "n_exposed", "n_occurrences"]]
    return df.sort_values("lift", ascending=False, na_position="last").reset_index(drop=True)


def lag_profile(intensity: np.ndarray, tag_presence: np.ndarray, max_lag: int = 7) -> np.ndarray:
    prof = np.full(max_lag + 1, np.nan)
    occ = np.flatnonzero(tag_presence)
    d = len(intensity)
    baseline = intensity[~exposed_mask(tag_presence, max_lag)]
    if len(baseline) == 0:
        return prof
    bmean = baseline.mean()
    for j in range(max_lag + 1):
        hit = occ[occ + j < d] + j
        if len(hit):
            prof[j] = intensity[hit].mean() - bmean
    return prof


def stratified_lift(intensity: np.ndarray, tag_a: np.ndarray, tag_b: np.ndarray,
                    n_window: int) -> dict:
    exp_a = exposed_mask(tag_a, n_window)
    exp_b = exposed_mask(tag_b, n_window)
    exposed = exp_a & ~exp_b            # A-exposed but not B-exposed
    baseline = ~exp_a & ~exp_b          # clean of both
    return _lift_from_masks(intensity, exposed, baseline, tag_a.sum())
