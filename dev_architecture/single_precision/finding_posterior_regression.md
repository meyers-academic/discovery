# Finding — f32 posterior regression on NG15 m2a/CURN (sets "the bar")

**Question (ADR 0003):** the discriminator showed the f32 logL has only a ~f32-eps *relative*
error (~0.75 abs ΔlogL at NG15 scale). Does that move the *posterior* a sampler sees? "The bar"
should be set from posteriors, not absolute logL error.

**Experiment** (`posterior_regression.py`): NG15 **m2a / CURN** model (per-pulsar red noise + one
common uncorrelated process, NO Hellings–Downs), full **67 pulsars / 674683 TOA**, sampled in
**float32** via the metamath kernel with **timing-model projection ON**, NUTS 200 warmup + 200
samples, `max_tree_depth=6`. Compared to the production **f64 `NG15yr-m2a-chain.feather`** (~100k
samples) by 1-D Jensen–Shannon divergence per parameter (136 params). Bar: JS excess < 1e-2.

## The trap, and the control

Raw JS f32-vs-chain looked alarming: median **0.062**, max 0.14, "everything over 1e-2." But that
is a **finite-sample artifact** — 200 samples histogrammed into 60 bins (~3/bin) against a 100k
chain gives large JS *even for two identical distributions*. The control proves it:

| quantity | median | max |
|---|---|---|
| JS f32 (200) vs f64 chain | 6.2e-2 | 1.4e-1 |
| **JS floor** = f64 chain 200-subsample vs same chain | 5.9e-2 | — |
| **f32 EXCESS over floor** | **+3.8e-3** | +7.0e-2 |

So the ~0.06 is the sampling-noise floor; the f32 *excess* over a perfect f64 sampler with the
same 200 draws is **+3.8e-3 (median) — under the 1e-2 bar.**

## Proper null test

For each parameter, build the null JS distribution {JS(200-subsample of f64 chain, full chain)}
and ask whether the f32 JS exceeds it. Under the null (f32≡f64) ~5% exceed the 95th percentile by
chance.

- **median z-score of f32 JS vs the null = +0.33** → the typical parameter is dead-centre in the
  null; f32 is sampling-noise-indistinguishable from f64.
- tail: **12/136** params above the 95th percentile (≈7 expected); 10.3% above z=1.64 (5%
  expected); one outlier at **z=6.87** (`J2010-1323_red_noise_gamma`, a single per-pulsar red-noise
  index).
- robust moments (sample-efficient at n=200): standardised mean shift |Δμ|/σ median **0.056**
  (max 0.386), std ratio median **0.992**. The science parameters (common process) agree to
  **0.09σ** (`crn_gamma`) and **0.04σ** (`crn_log10_A`).

## Verdict

**Nothing is crazy.** The f32 CURN posterior matches the f64 m2a chain to within the sampling
noise of a 200-draw chain — median deviation essentially zero — with at most a **faint marginal
excess** on a handful of parameters (≈2× the expected tail count, one z≈7 outlier) that only a
longer chain could resolve. The 0.75 abs logL error does **not** detectably move the m2a posterior.
**The bar (median JS excess < 1e-2) is met**, confirming ADR 0003's premise: f32 (projection + Half
A) already gives usable posteriors on this dataset.

## Caveats / scope

- **CURN only (no HD).** Hellings–Downs, evidence/Bayes factors, and NUTS at tight spatial
  correlations are the regimes ADR 0003 flags as more logL-sensitive — not tested here.
- **200 samples** ⇒ the floor is high; we can only *bound* the f32 excess (≲ a few ×1e-3 median,
  faint tail), not measure it precisely. A longer chain would tighten the bound and settle whether
  the z≈7 outlier is real.
- Reference+delta (`woodbury_refdelta`, Half B) is therefore **not required for m2a-scale CURN**;
  it remains the lever for the flagged harder regimes (>NG15, HD, evidence). Its single-level test
  rung is built and verified (f32 ~100–10⁴× tighter than direct), ready to extend to the fused path
  if/when a regime needs it.
