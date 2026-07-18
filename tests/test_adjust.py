import numpy as np
from insights.config import SimConfig
from insights.adjust import stress_adjusted_lift, rank_adjusted, adjusted_damage


def test_stress_adjusted_lift_collapses_purely_confounded_tag():
    # 3 confounder "regimes" (blocks) of 100 days each. Tag presence probability
    # rises steeply with the regime (0.1 / 0.5 / 0.9), and intensity is driven
    # only by the regime (+3 pts/regime) plus noise -- the tag itself has zero
    # real effect. Marginally the tag looks like a strong trigger because its
    # presence is concentrated in high-intensity regimes; within a regime
    # (near-constant confounder) tag presence carries no information.
    rng = np.random.default_rng(1)
    block = np.repeat([0, 1, 2], 100)
    confounder = block.astype(float)
    p = np.select([block == 0, block == 1, block == 2], [0.1, 0.5, 0.9]).astype(float)
    tag_presence = rng.random(300) < p
    intensity = 2.0 + 3.0 * block + rng.normal(0, 0.5, size=300)

    marginal = intensity[tag_presence].mean() - intensity[~tag_presence].mean()
    assert marginal > 1.0

    r = stress_adjusted_lift(intensity, tag_presence, confounder, n_window=0, n_strata=3)
    assert abs(r["lift"]) < 0.5
    assert r["n_strata_used"] > 1
    assert r["n_exposed"] > 0


def test_stress_adjusted_lift_preserves_genuine_effect():
    # Same confounded tag as above, but the tag now ALSO carries a genuine
    # +2.0 effect on top of the confound. Marginal lift over-states the
    # effect (confound inflates it); adjusted lift should recover ~2.0.
    rng = np.random.default_rng(1)
    block = np.repeat([0, 1, 2], 100)
    confounder = block.astype(float)
    p = np.select([block == 0, block == 1, block == 2], [0.1, 0.5, 0.9]).astype(float)
    tag_presence = rng.random(300) < p
    base_intensity = 2.0 + 3.0 * block + rng.normal(0, 0.5, size=300)
    intensity = base_intensity + 2.0 * tag_presence

    marginal = intensity[tag_presence].mean() - intensity[~tag_presence].mean()
    r = stress_adjusted_lift(intensity, tag_presence, confounder, n_window=0, n_strata=3)

    assert marginal > r["lift"] + 1.0   # marginal is inflated relative to adjusted
    assert abs(r["lift"] - 2.0) < 0.5


def test_stress_adjusted_lift_returns_nan_when_no_strata_usable():
    # Constant confounder -> cannot form any quantile strata.
    intensity = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    tag_presence = np.array([True, False, True, False, True, False])
    confounder = np.zeros(6)
    r = stress_adjusted_lift(intensity, tag_presence, confounder, n_window=0, n_strata=3)
    assert np.isnan(r["lift"])
    assert r["n_strata_used"] == 0
    assert r["n_exposed"] == 0


def test_rank_adjusted_shape_and_order():
    rng = np.random.default_rng(2)
    d = 120
    confounder = rng.normal(size=d)
    tags = rng.random((d, 3)) < 0.3
    intensity = 2.0 + 3.0 * tags[:, 0].astype(float) + rng.normal(0, 0.5, size=d)

    df = rank_adjusted(intensity, tags, confounder, n_window=0, n_strata=3)
    assert list(df.columns) == ["tag", "lift", "n_exposed"]
    assert len(df) == 3
    assert df.iloc[0]["tag"] == "tag_0"
    assert df.iloc[0]["lift"] > df.iloc[1]["lift"]


def test_adjusted_damage_returns_float_in_unit_interval():
    c = SimConfig()  # default confounded config: tag_1 is confounding-but-not-true
    damage = adjusted_damage(c, n_datasets=30, k=3, n_strata=3)
    assert isinstance(damage, float)
    assert 0.0 <= damage <= 1.0
