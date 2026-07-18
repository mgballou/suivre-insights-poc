import math
from insights.config import SimConfig
from insights.alerts import alert_precision


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
