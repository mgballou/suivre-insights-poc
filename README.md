# suivre-insights-poc

A simulation study that tested the central claim behind [Suivre](https://github.com/mgballou/suivre)
before any of it was built.

Suivre is a personal food-and-symptom journal. Its one interesting claim is that a **lag-lift**
statistic — mean condition intensity on the days *after* a food was eaten, against days it was not —
can surface a real dietary trigger out of one person's journal. This repository asks the only
question worth asking about that claim: at the amount of data a single person can realistically
produce, how often does it find the true trigger, and how often does it blame an innocent food?

The answer changed the product. It is written up in
[**the findings note**](https://github.com/mgballou/suivre/blob/main/docs/2026-07-18-lag-lift-spike-findings.md)
in the main repository; this repository is where every number in it comes from.

## The verdict: adjust, don't abandon

The insight is real for moderate and strong triggers given roughly three months of data, and
fragile everywhere else.

- **Detection needs time.** A trigger worth ≥1.5 intensity points is reliably surfaced (hit rate
  ≥0.8) at about 75–90 days of logging. Weaker effects need many months.
- **One ranking is noise.** At personal scale, pure-noise tags routinely out-rank real ones in a
  single draw. Only the aggregate over many draws is trustworthy — which is exactly what a user
  never sees.
- **Co-occurrence is the real enemy.** A food with *zero* effect that travels with a real trigger —
  dessert is dairy and sugar together — is flagged about 61% of the time, and that does **not**
  wash out with more data. Adjusting for sleep and stress fixes the estimate but not the ranking.
- **A soft nudge needs 90 days.** "Have you noticed…?" only clears a precision bar of 0.7 from
  about 90 days. At 30 days precision tops out near 0.58 even for strong triggers.

So: build the insight, but as an uncertainty-forward, data-gated, descriptive nudge — never a
confident trigger list.

## Why synthetic data

You cannot measure a detection rate against real journals, because with real journals nobody knows
the truth. Here the answer is planted in advance and then hidden under everything that makes real
logs hard to read:

- lagged tag → intensity effects, smeared over several days by a kernel rather than landing on one
- AR(1) "sticky" flares, so yesterday's bad day predicts today's
- sleep and stress confounders, including a genuine stress → comfort-food path
- tags that deliberately co-fire, so an innocent food shadows a guilty one

**None of this is real health data, and none of it is medical guidance.** The parameters are
loosely grounded in published inflammatory-trigger patterns only so the simulation is not absurd.

## Layout

```
src/insights/
  config.py     # SimConfig — every knob, one seed
  generate.py   # the synthetic daily-log generator
  lag_lift.py   # the statistic: windowing, lift, ranking, lag profile, stratified lift
  sweep.py      # Monte Carlo sweep, detection hit criterion, damage metrics
  adjust.py     # stress-stratified lift and adjusted damage
  alerts.py     # soft-alert precision, single-hint precision
notebooks/01_lag_lift_spike.ipynb   # the narrative: model → statistic → sweep → verdict
outputs/        # committed figures and example tables
tests/          # 28 pytest cases over the math and the metrics
```

The notebook is the argument; `src/` is the machinery it drives, kept as pure seeded functions so
the tests can pin the math down separately from the story.

## Run it

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/pip install -r requirements.txt
.venv/bin/pytest -q                      # 28 passed
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1800 notebooks/01_lag_lift_spike.ipynb
```

Every draw is seeded and every dependency pinned, so a rerun reproduces the same figures and the
same numbers. `requirements.txt` is a full freeze rather than a list of direct dependencies, for
that reason; the direct three are in `pyproject.toml`.

## License

MIT — see [`LICENSE`](LICENSE).
