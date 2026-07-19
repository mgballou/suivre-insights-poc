# suivre-insights-poc

A disposable-but-persisted **proof-of-concept** that stress-tests Suivre's north-star insight — the **lag-lift** food↔flare correlation (decision D11) — on realistic synthetic data, *before* building it (SUI-36). The value is the findings; the code is kept because it's inspectable and a rehearsal for the real `ComputeCorrelations`.

## What it does

Generates synthetic single-user daily logs — condition intensity (0–10) + trigger-tag presence — with planted lagged tag→intensity effects, AR(1) "sticky" flares, sleep/stress confounders (including a true stress→comfort-food path), a multi-day smeared lag kernel, and deliberately co-occurring tags. It then computes the lag-lift ranking, sweeps the parameter space, and measures how reliably a real trigger surfaces vs. how often an innocent co-occurring food is falsely flagged.

**Nothing here is real health data or clinical guidance.** The synthetic model is grounded only loosely in published inflammatory-trigger patterns to make the parameters realistic.

## Layout

```
src/insights/
  config.py     # SimConfig — every knob, one seed
  generate.py   # synthetic daily-log generator
  lag_lift.py   # lift math: windowing, lift, ranking, lag profile, stratified lift
  sweep.py      # Monte Carlo sweep, detection hit-criterion, damage metrics
  adjust.py     # confounder (stress) stratified lift + adjusted damage
  alerts.py     # soft-alert precision + single-hint precision
notebooks/01_lag_lift_spike.ipynb   # the narrative: model → math → sweep → verdict
outputs/        # committed figures + example tables
tests/          # pytest (28 tests over the math + metrics)
```

## Run it

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/pip install -r requirements.txt
.venv/bin/pytest -q                      # 28 passed
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1800 notebooks/01_lag_lift_spike.ipynb
```

Everything is seeded — figures regenerate byte-stably.

## Headline verdict — **ADJUST**

The insight is real for moderate/strong triggers given ~90 days of data, but fragile at small `n` and against confounding:

- Detectable (hit-rate ≥0.8) for effects ≥1.5 pts at ~75–90 days; weaker effects need many months.
- A single small-`n` ranking is noisy — pure-noise tags can out-rank real ones; only the aggregate is trustworthy.
- A zero-effect food that **co-occurs with a real trigger** is flagged ~61% of the time, and this does **not** wash out with more data. Stress-adjustment fixes the *estimate* but not the *ranking* — co-occurrence, not lifestyle, is the dominant confounder.
- A soft "have you noticed…?" nudge only clears a trust bar (precision ≥0.7) from **~90 days**; at 30 days precision peaks at ~0.58 even for strong triggers.

Full write-up and the guidance it feeds into SUI-21 / SUI-22 live in the Suivre repo: `docs/2026-07-18-lag-lift-spike-findings.md`.
