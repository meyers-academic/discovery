# Finding — the discriminator table for reference+delta (single-level)

The `finding_projection_discriminator.md` table (f32-vs-f64 abs logL error vs array size),
now with a **`woodbury_refdelta`** column beside the direct `woodbury` baseline.

**Scope.** `woodbury_refdelta` is the single-level (per-pulsar) graph; the fused cross-pulsar
(HD) path is not built yet. For a model with **no Hellings–Downs coupling the array likelihood
factorises** (logL = Σ_i logL_i), so summing per-pulsar single-level results is the exact array
logL. Per-pulsar model: **white + ECORR + intrinsic red-noise GP** (timing omitted — projection
handles it, shown neutral in the discriminator). Test point θ drawn from the NG15 **m3a** chain;
reference Φ_ref = the per-pulsar red-noise prior at the **chain median** (frozen f64), per ADR 0001.

`harness_refdelta_table.py`, max abs error over 5 draws:

| Npsr | Ntoa | \|logL\| | woodbury_f32 | refdelta_f32 | gain | f64 check |
|---|---|---|---|---|---|---|
| 3  | 35907  | 5.0e5 | 2.52e-1 | 6.64e-4 | **379×** | 6e-10 |
| 6  | 82467  | 1.07e6 | 2.52e-1 | 6.61e-4 | **381×** | 5e-10 |
| 12 | 98876  | 1.27e6 | 2.53e-1 | 6.49e-4 | **389×** | 5e-10 |
| 45 | 480094 | 6.0e6 | 2.58e-1 | 4.05e-2 | 6.4× | 9e-10 |
| 67 | 674683 | 8.5e6 | 2.58e-1 | 4.05e-2 | 6.4× | 2e-9 |

## Reading it

1. **f64 exact at every scale** (`f64 check` = |refdelta_f64 − woodbury_f64| ~ 1e-9). The
   decomposition is algebraically the same logL; refdelta only changes *how* it is summed.

2. **Direct woodbury sits at ~0.25 abs error**, roughly flat in n (the max over draws is set by
   the worst single-pulsar contribution, which does not grow as more pulsars are added). This is
   the Half-A-combined path — the floor reference+delta targets.

3. **refdelta wins ~380× when the sample is near the reference** (n ≤ 12 here: 6.5e-4 vs 0.25).

4. **The n=45/67 figure (4.05e-2, 6.4×) is ONE outlier pulsar, not systematic growth.** Per-pulsar
   diagnosis of the worst draw: total error 0.0406, of which pulsar idx 31 contributes 0.0405 and
   all 44 others ≈ 0. That pulsar's sampled red noise sits far from its median reference — the
   "large move from Φ_ref" regime where the single-level test (`test_woodbury_refdelta.py`,
   `da=2.0`) already showed refdelta relaxing toward the baseline, because the f32 error scales with
   the *size* of the increment. Typical draws stay at ~3–5e-3 (still ~50–90× better than woodbury);
   the max is just whichever pulsar is deepest in its tail on that draw. (45 and 67 report the
   identical 4.05e-2 because the worst pulsar is within the first 45.)

## Takeaways

- reference+delta delivers the expected order(s)-of-magnitude f32 accuracy gain, exactly and
  verifiably (f64-identical), with the win largest near the reference.
- Its accuracy degrades gracefully with distance from Φ_ref: a single pulsar far in its tail caps
  the worst-case at ~4e-2 (still better than the 0.25 baseline). Mitigations if ever needed:
  refresh/re-centre Φ_ref periodically, or accept it — recall the **posterior** regression
  (`finding_posterior_regression.md`) showed even the 0.25 baseline gives correct m2a posteriors,
  so 4e-2 is comfortably sub-threshold.
- HD is the natural next table, and needs the **fused nested refdelta** (§4 two-perturbation,
  `research_note_nested_increment.md`) — the per-pulsar-sum factorisation used here does not apply
  once the ORF couples pulsars.
