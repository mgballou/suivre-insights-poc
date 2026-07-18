import numpy as np
from insights.config import SimConfig
from insights.generate import generate_logs


def _clean(**kw):
    # a clean world: one strong trigger, no confounding, no stickiness, no noise.
    return SimConfig(
        days=400, n_tags=3, true_trigger_idx=(0,), base_rates=(0.3, 0.3, 0.3),
        cooccur_pairs=(), effect_points=3.0, kernel=(0.0, 1.0, 0.0),
        flare_phi=0.0, flare_sd=0.0, confounder_strength=0.0,
        confounding_path=False, noise_sd=0.0, baseline_intensity=1.0, **kw,
    )


def test_shape_and_columns():
    c = SimConfig()
    df = generate_logs(c, c.rng())
    assert len(df) == c.days
    assert df["intensity"].between(0, 10).all()
    assert df["intensity"].dtype.kind in "iu"
    for i in range(c.n_tags):
        assert f"tag_{i}" in df.columns


def test_planted_trigger_raises_next_day_intensity():
    c = _clean()
    df = generate_logs(c, c.rng())
    trig = df["tag_0"].to_numpy()
    inten = df["intensity"].to_numpy()
    after = inten[1:][trig[:-1]]          # day after a trigger
    other = inten[1:][~trig[:-1]]         # day after a non-trigger
    assert after.mean() - other.mean() > 1.5


def test_noise_tag_has_no_effect():
    c = _clean()
    df = generate_logs(c, c.rng())
    tag2 = df["tag_2"].to_numpy()
    inten = df["intensity"].to_numpy()
    after = inten[1:][tag2[:-1]]
    other = inten[1:][~tag2[:-1]]
    assert abs(after.mean() - other.mean()) < 0.5


def test_determinism():
    c = SimConfig(seed=3)
    a = generate_logs(c, c.rng())
    b = generate_logs(c, c.rng())
    assert a.equals(b)
