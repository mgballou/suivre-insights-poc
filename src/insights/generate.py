import numpy as np
import pandas as pd

from insights.config import SimConfig


def _ar1(n: int, phi: float, sd: float, rng: np.random.Generator) -> np.ndarray:
    x = np.zeros(n)
    innov = rng.normal(0.0, sd, size=n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + innov[t]
    return x


def _tag_matrix(config: SimConfig, stress: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    d, k = config.days, config.n_tags
    tags = np.zeros((d, k), dtype=bool)

    # co-occurrence: a shared latent factor drives each cluster together.
    latent = {}
    for a, b in config.cooccur_pairs:
        latent[(a, b)] = rng.random(d) < 0.5

    for i in range(k):
        p = np.full(d, config.base_rates[i])
        # confounding path: stress raises the odds of comfort-food tags.
        if config.confounding_path and i in config.confounding_tag_idx:
            p = np.clip(p + config.confounder_strength * 0.05 * np.clip(stress, 0, None), 0, 1)
        draw = rng.random(d) < p
        tags[:, i] = draw

    # apply co-occurrence: when the latent factor is on, force both members on together.
    for (a, b), on in latent.items():
        both = on & (rng.random(config.days) < config.cooccur_strength)
        tags[both, a] = True
        tags[both, b] = True
    return tags


def _kernel_contribution(tag_col: np.ndarray, kernel: np.ndarray, beta: float) -> np.ndarray:
    d = len(tag_col)
    out = np.zeros(d)
    occ = np.flatnonzero(tag_col)
    for t in occ:
        for j, w in enumerate(kernel):
            if t + j < d:
                out[t + j] += beta * w
    return out


def generate_logs(config: SimConfig, rng: np.random.Generator) -> pd.DataFrame:
    d = config.days
    sleep = _ar1(d, config.sleep_phi, 1.0, rng)
    stress = _ar1(d, config.stress_phi, 1.0, rng)
    flare = _ar1(d, config.flare_phi, config.flare_sd, rng)

    tags = _tag_matrix(config, stress, rng)
    kernel = np.asarray(config.kernel, dtype=float)

    latent = np.full(d, config.baseline_intensity, dtype=float)
    latent += config.confounder_strength * np.clip(sleep, 0, None)
    latent += config.confounder_strength * np.clip(stress, 0, None)
    latent += flare
    for i in config.true_trigger_idx:
        latent += _kernel_contribution(tags[:, i], kernel, config.effect_points)
    latent += rng.normal(0.0, config.noise_sd, size=d)

    intensity = np.clip(np.round(latent), 0, 10).astype(int)
    observed = rng.random(d) >= config.missingness

    data = {"intensity": intensity, "sleep": sleep, "stress": stress, "observed": observed}
    for i in range(config.n_tags):
        data[f"tag_{i}"] = tags[:, i]
    return pd.DataFrame(data)
