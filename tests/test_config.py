import numpy as np
from insights.config import SimConfig


def test_defaults_are_consistent():
    c = SimConfig()
    assert len(c.base_rates) == c.n_tags
    assert all(0 <= i < c.n_tags for i in c.true_trigger_idx)
    assert len(c.kernel) >= 1


def test_rng_is_deterministic():
    a = SimConfig(seed=7).rng().random(5)
    b = SimConfig(seed=7).rng().random(5)
    assert np.array_equal(a, b)
