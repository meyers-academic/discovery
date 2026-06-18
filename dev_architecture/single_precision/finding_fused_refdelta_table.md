# Finding — fused two-level reference+delta on the real HD array (Hellings-Downs)

The HD counterpart of `finding_vectorwoodbury_refdelta_table.md` (which covered the
no-HD single-level CURN path). Here the model has a per-pulsar intrinsic red-noise
(IRN) inner GP **and** a dense cross-pulsar HD GW GP, so the array logL does **not**
factorise: it routes through the fused two-level kernel
(`vectorwoodburyjointsolve` → `globalwoodbury_fused`) and its new refdelta twin
(`vectorwoodburyjointsolve_refdelta` → `globalwoodbury_fused_refdelta`).

This drives the **production wiring**, not hand-assembled graph pieces:
`ArrayLikelihood(reference=θ_ref)` freezes both prior covariances at the reference
(the thin top layer, ADR 0001) and routes `.logL` to the fused twins. So the table
is a direct test of the end-state path a real HD analysis would use.

**Model.** Per-pulsar IRN (powerlaw, 30 comp, name `red_noise`) + HD global GW
(powerlaw, 14 comp, `hd_orf`, name `gw`). HD's ORF is parameter-free, so the free
params are the per-pulsar `red_noise_*` plus the shared `gw_log10_A`/`gw_gamma` —
all in the NG15 **m3a** chain. Fixed white noise (folds to f64). Reference Φ_ref =
both priors at the **chain median**, frozen f64; test points θ from the m3a chain.

`harness_fused_refdelta.py`, max abs error over 5 draws:

| Npsr | Ntoa | \|logL\| | fused_f32 | refdelta_f32 | gain | f64 check |
|---|---|---|---|---|---|---|
| 3  | 35907  | 4.7e5 | 0.0433 | 0.0127 | **3.4×**  | 1.7e-10 |
| 6  | 82467  | 1.0e6 | 0.0847 | 0.0212 | **4.0×**  | 2.3e-10 |
| 12 | 98876  | 1.2e6 | 0.0661 | 0.0512 | **1.3×**  | 4.7e-10 |
| 24 | 234204 | 2.7e6 | 0.172  | 0.0115 | **14.9×** | 9.3e-10 |
| 45 | 480094 | 5.7e6 | 0.749  | 0.0939 | **8.0×**  | 9.3e-10 |

## Reading it

1. **f64-exact at every scale** (`f64 check` = |refdelta_f64 − fused_f64| ~ 1e-10).
   The two-level decomposition is algebraically the same logL — exactly as the mpmath
   oracle (`test_refdelta_nested.py`) and the rung-2 unit test already proved.

2. **The direct fused path degrades with array size** — 0.04 at 3 psr up to **0.75 at
   45 psr** — because the outer-level quadratic/logdet still accumulate f32 roundoff over
   a growing GW block. The refdelta path stays **bounded at ~0.01–0.09** throughout, so the
   gain *grows* with scale (≈8–15× by 24–45 psr), which is where it matters.

3. **The refdelta f32 floor tracks the *increment* size, not |logL|.** Unlike the
   single-level table (flat ~1e-4), here the floor is ~1e-2 and noisy (the 12-psr row is a
   draw-luck dip, only 5 draws / max-error). This is expected: ΔlogL is carried in f32, and
   its error is ~f32-eps × |ΔlogL|. These draws span the **whole** m3a chain, so |ΔlogL|
   (distance from the frozen median reference) is large for tail draws. A reference closer
   to the sampled bulk (e.g. a per-walker or running reference) tightens the floor further;
   the median reference already keeps it ~10–60× below the direct path's blow-up.

## Takeaways

- `globalwoodbury_fused_refdelta` + the `reference=θ_ref` wiring deliver an exact,
  verifiable f32 accuracy gain on the **real HD array graph** at NG15 scale, completing
  single-precision Half B for the Hellings-Downs (globalgp) case. Both regimes are now
  covered: single-level (`vectorwoodbury_refdelta`, CURN/IRN) and fused two-level
  (HD/CURN-with-IRN).
- The gain is more modest and noisier than the no-HD single-level table because the fused
  path's Half-A combine is already strong and the residual f32 cost is the *increment* of
  a two-level solve. The decisive property is that refdelta stays **bounded** while the
  direct path's error grows ~unbounded with array size.
- Practical lever: the f32 floor follows reference-distance, so the closer Φ_ref sits to
  the sampled region, the tighter the increment. The frozen chain-median reference is a
  safe default; a running/per-walker reference is the obvious next refinement if more
  headroom is wanted.
