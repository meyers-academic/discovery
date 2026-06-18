# Finding — reference+delta on the real CURN+IRN array graph (batched single level)

`finding_refdelta_table.md` measured reference+delta by summing the **scalar**
`woodbury_refdelta` per pulsar in a Python loop. This finding drives the **actual
production array graphs** a no-Hellings-Downs `ArrayLikelihood` uses:
`vectorwoodbury` (direct, Half-A combined) vs the new `vectorwoodbury_refdelta`
(Half-B), on the same NG15 data and m3a chain draws.

**Model.** `recipes.intrinsic_plus_crn` — per-pulsar intrinsic red noise + a common
(CURN) spectrum on one shared basis via `make_combined_crn` (name `red_noise`,
`crn_prefix='gw_'` so params match the chain). No HD, so the commongp routes through
`VectorWoodburyKernel` → `vectorwoodbury` (single level, batched over pulsars). Fixed
white noise (folds to f64 pins). Test point θ from the NG15 **m3a** chain; reference
Φ_ref = the commongp prior at the **chain median**, frozen f64 (ADR 0001).

`harness_vectorwoodbury_refdelta.py`, max abs error over 5 draws:

| Npsr | Ntoa | \|logL\| | vectorwoodbury_f32 | refdelta_f32 | gain | f64 check |
|---|---|---|---|---|---|---|
| 3  | 35907  | 4.7e5 | 0.0156  | 7.8e-4 | **20×**  | 5.8e-11 |
| 6  | 82467  | 1.0e6 | 0.0259  | 1.9e-4 | **135×** | 1.2e-10 |
| 12 | 98876  | 1.2e6 | 0.0392  | 1.9e-4 | **203×** | 2.3e-10 |
| 45 | 480094 | 5.7e6 | 0.081   | 9.3e-4 | **87×**  | 9.3e-10 |
| 67 | 674683 | 8.0e6 | 0.0515  | 9.3e-4 | **55×**  | 9.3e-10 |

## Reading it

1. **f64-exact at every scale** (`f64 check` = |refdelta_f64 − vectorwoodbury_f64| ~ 1e-10).
   The batched decomposition is algebraically the same logL; refdelta only changes how
   it is summed.

2. **refdelta f32 error is ~1e-4–1e-3 and roughly flat in array size**, while the direct
   `vectorwoodbury` drifts up to ~0.05–0.08 by 45–67 pulsars → **20–200× tighter** on the
   real production graph.

3. **The direct-path numbers here (~0.02–0.08) are smaller than the old table's ~0.25.**
   That is the path, not a contradiction: the old table summed *scalar* `woodbury` per
   pulsar (Python f64 sum of f32-`μ`-rounded per-pulsar logLs, worst-pulsar dominated),
   whereas this is the batched array graph with one f64 combine over the whole array sum
   (`vectorwoodbury`). The batched graph is what production actually evaluates, so this is
   the more faithful baseline.

## Takeaways

- `vectorwoodbury_refdelta` delivers the expected order(s)-of-magnitude f32 accuracy gain
  on the **real CURN+IRN array graph**, exactly and verifiably (f64-identical), at NG15
  scale up to 67 pulsars.
- This is the no-HD (commongp-only) regime: per-pulsar IRN, CURN, IRN+CURN. The
  Hellings-Downs (globalgp) case needs the **fused two-level** refdelta
  (`vectorwoodburyjointsolve_refdelta` + `globalwoodbury_fused_refdelta`), of which the
  inner half (rung 1) is built and unit-tested; the outer half (rung 2) is next.
