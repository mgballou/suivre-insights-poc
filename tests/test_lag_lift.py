import numpy as np
from insights.lag_lift import (
    exposed_mask, lift_for_tag, rank_suspects, lag_profile, stratified_lift,
)


def test_exposed_mask_unions_overlapping_windows():
    pres = np.array([True, False, False, True, False, False])
    # window 2: day0 -> {0,1,2}; day3 -> {3,4,5}
    assert exposed_mask(pres, 2).tolist() == [True, True, True, True, True, True]


def test_exposed_mask_dedupes_close_occurrences():
    pres = np.array([True, True, False, False])
    assert exposed_mask(pres, 1).tolist() == [True, True, True, False]


def test_lift_is_mean_difference():
    inten = np.array([0, 5, 0, 0, 0, 0])
    pres = np.array([True, False, False, False, False, False])  # exposed {0,1} at window 1; baseline {2,3,4,5} all 0
    r = lift_for_tag(inten, pres, 1)
    assert r["n_exposed"] == 2
    assert r["n_occurrences"] == 1
    assert np.isclose(r["exposed_mean"], 2.5)
    assert np.isclose(r["baseline_mean"], 0.0)
    assert np.isclose(r["lift"], 2.5)


def test_rank_orders_by_lift_desc():
    inten = np.array([0, 6, 0, 6, 0, 6, 0, 0])
    tags = np.zeros((8, 2), dtype=bool)
    tags[[0, 2, 4], 0] = True   # strong tag precedes the 6s
    tags[[7], 1] = True         # weak/noise tag
    df = rank_suspects(inten, tags, 1)
    assert df.iloc[0]["tag"] == "tag_0"
    assert df.iloc[0]["lift"] > df.iloc[1]["lift"]


def test_lag_profile_peaks_at_true_lag():
    inten = np.zeros(50)
    pres = np.zeros(50, dtype=bool)
    pres[::10] = True
    inten[np.flatnonzero(pres) + 2] = 8   # effect lands exactly 2 days later
    prof = lag_profile(inten, pres, max_lag=5)
    assert int(np.argmax(prof)) == 2


def test_stratified_lift_recovers_a_only_effect():
    # A raises same-day intensity by 5; B is inert but co-occurs with A on day 0.
    # Stratifying to A-present/B-absent days (2, 4) must recover A's true +5 effect.
    inten = np.array([5, 0, 5, 0, 5, 0, 0, 0, 0, 0])
    a = np.array([True, False, True, False, True, False, False, False, False, False])
    b = np.array([True, False, False, False, False, False, False, False, False, False])
    strat = stratified_lift(inten, a, b, 0)  # window 0: exposed = presence day itself
    assert strat["n_exposed"] == 2          # days 2 and 4 (day 0 dropped: B present)
    assert np.isclose(strat["lift"], 5.0)
