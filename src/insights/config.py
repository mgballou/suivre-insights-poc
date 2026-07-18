from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class SimConfig:
    days: int = 90
    n_tags: int = 8
    true_trigger_idx: tuple[int, ...] = (0, 2)
    base_rates: tuple[float, ...] = (0.35, 0.30, 0.15, 0.25, 0.10, 0.20, 0.12, 0.18)
    # index 0 = "dairy", 1 = "sugar": a latent dessert factor co-fires them.
    cooccur_pairs: tuple[tuple[int, int], ...] = ((0, 1),)
    cooccur_strength: float = 0.7
    effect_points: float = 2.0
    # lag kernel over days 0..7; peaks day 1-2, tails to ~a week.
    kernel: tuple[float, ...] = (0.2, 0.6, 0.8, 0.6, 0.4, 0.25, 0.15, 0.1)
    flare_phi: float = 0.6
    flare_sd: float = 1.0
    sleep_phi: float = 0.5
    stress_phi: float = 0.5
    confounder_strength: float = 1.0
    confounding_path: bool = True
    confounding_tag_idx: tuple[int, ...] = (0, 1)
    baseline_intensity: float = 2.0
    noise_sd: float = 0.8
    missingness: float = 0.0
    n_window: int = 2
    seed: int = 0

    def rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)
