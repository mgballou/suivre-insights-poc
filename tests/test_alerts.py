import math
from insights.config import SimConfig
from insights.alerts import alert_precision, top_hint_precision


def _strong_clean_config():
    # Mirrors test_sweep.test_strong_clean_signal_has_high_hit_rate: a single
    # strong, unconfounded true trigger plus pure-noise tags. Noise tags'
    # lifts never approach the true trigger's, giving a clean threshold gap.
    return SimConfig(days=180, n_tags=5, true_trigger_idx=(0,), effect_points=3.0,
                      flare_phi=0.0, flare_sd=0.0, confounder_strength=0.0,
                      confounding_path=False, cooccur_pairs=(), noise_sd=0.5)


def test_alert_precision_is_perfect_for_strong_trigger_at_high_threshold():
    c = _strong_clean_config()
    r = alert_precision(c, days=c.days, alert_threshold=1.5, n_datasets=60)
    assert r["fire_rate"] > 0.0
    assert r["n_alerts"] > 0
    assert r["precision"] == 1.0


def test_alert_precision_nan_when_no_alerts_fire():
    c = _strong_clean_config()
    r = alert_precision(c, days=c.days, alert_threshold=1000.0, n_datasets=20)
    assert r["n_alerts"] == 0
    assert r["true_alerts"] == 0
    assert r["fire_rate"] == 0.0
    assert math.isnan(r["precision"])


def test_alert_precision_keys():
    c = _strong_clean_config()
    r = alert_precision(c, days=c.days, alert_threshold=1.5, n_datasets=10)
    assert {"fire_rate", "precision", "n_alerts", "true_alerts"} <= set(r.keys())


def _no_true_trigger_config():
    # No true trigger at all: every tag is pure noise, so the #1-ranked
    # tag can never be a genuine trigger. Precision must therefore be
    # either NaN (nothing ever surfaced) or exactly 0.0 (something
    # surfaced but it was noise) -- deterministic regardless of which
    # specific tag happens to win the #1 slot on any given draw.
    return SimConfig(days=90, n_tags=5, true_trigger_idx=(), effect_points=0.0,
                      flare_phi=0.0, flare_sd=0.0, confounder_strength=0.0,
                      confounding_path=False, cooccur_pairs=(), noise_sd=1.0)


def test_top_hint_precision_is_perfect_for_strong_clean_trigger():
    c = _strong_clean_config()
    r = top_hint_precision(c, days=180, n_datasets=30)
    assert r["precision"] == 1.0
    assert r["coverage"] > 0.0


def test_top_hint_precision_cannot_exceed_zero_with_no_true_trigger():
    c = _no_true_trigger_config()
    r = top_hint_precision(c, days=c.days, n_datasets=40, require_noise_band=True)
    assert math.isnan(r["precision"]) or r["precision"] == 0.0


def test_top_hint_precision_is_deterministic():
    c = _strong_clean_config()
    r1 = top_hint_precision(c, days=180, n_datasets=20)
    r2 = top_hint_precision(c, days=180, n_datasets=20)
    assert r1 == r2 or (
        math.isnan(r1["precision"]) and math.isnan(r2["precision"])
        and r1["coverage"] == r2["coverage"] and r1["n_surfaced"] == r2["n_surfaced"]
    )
