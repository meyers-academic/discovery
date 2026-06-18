# Handoff — reference+delta (Piece 2 "Half B"), for the next session

"What's done / decide next / watch out" sheet for the **single-precision** work, now on the
**reference+delta** track. Supersedes `HANDOFF_projection.md` (projection is done and *ruled out*
as a precision lever — see below). Authoritative design: `CONTEXT.md`, ADRs 0001–0004,
`research_note_on_split_with_reference.md` (single-level §2–3), `research_note_nested_increment.md`
(fused §4). This file is NOT committed.

> **Audience:** Patrick is a physicist, not a software engineer. Plain language; math/physics
> framing. "push" = fork `origin` (meyers-academic), `git push origin HEAD:metamatrix-meyers`,
> NEVER bare push. Don't commit/push without explicit go-ahead.

## Where we are (the arc)

The f32 likelihood has a floor: `logL = −½[ytNmy − FtNmy·μ + logdet]` subtracts two ~1e6 numbers
to get an O(1) result, so the f32 ulp (~0.25 at NG15 scale) survives. Three things settled this
session:

1. **Timing-model projection (ADR 0004) is NEITHER a precision NOR a conditioning lever** — it is
   *neutral* on this path. Discriminator on realistic m3a draws: every (svd, project) config is
   finite and sits at the same ~f32-eps relative floor; the broad-prior NaNs/heavy-tails that once
   looked like signal were `sample_uniform` artifacts. Stays opt-in/off. (`finding_projection_discriminator.md`.)
2. **The "bar" is met for m2a/CURN.** f32-vs-f64 *posterior* regression (67 psr, projection on,
   NUTS 200+200 depth6) vs the f64 m2a chain: median JS excess over the sampling-noise floor =
   **+3.8e-3 (< 1e-2)**; median z-score of f32 JS vs the f64-vs-f64 null = **+0.33** (dead-centre);
   science params agree to 0.04–0.09σ. **f32 + projection + Half A already gives correct m2a
   posteriors.** Faint tail excess (one z≈7 outlier) only a longer chain could resolve.
   (`finding_posterior_regression.md`.) → refdelta is **not required for m2a-CURN**; it is the lever
   for the ADR-0003 harder regimes (HD, evidence, >NG15, tight-correlation NUTS) — none yet tested.
3. **reference+delta works, exactly and with a big f32 win** (single-level). See below.

So the science conclusion so far: **don't block production on Half B for m2a.** Half B is built to
the single-level stage and is the right tool when a *harder* regime (HD first) needs it.

## What's BUILT this session (UNCOMMITTED, on top of HEAD `cef8a0a`)

- `src/discovery/metamath.py` (modified): **`woodbury_refdelta(g, y, Nsolve, F, Pinv, Pinv_ref)`**
  graph, right after `woodbury_proj`. Single-level reference+delta. `Pinv_ref` is the frozen
  reference Φ_ref⁻¹ leaf (constant → folds to an f64 constant, ADR 0001). Math (companion note §2–3):
  ```
  v = FtNmy, G = FtNmF        (fixed under fixed WN -> fold)
  C = Phi^-1 + G,  u = C^-1 v,  w = Phi^-1 u   (and _ref at Phi_ref)
  dPhi = Phi - Phi_ref                          (benign covariance-space increment)
  dQ    = w^T dPhi w_ref                         (f64-accumulated via combine_f64)
  dLdet = slogdet(I + S0^-1 dPhi G),  S0 = I + Phi_ref G
  logL  = logL_ref(f64 const) + 1/2 dQ - 1/2 dLdet     (final add in f64)
  ```
- `tests/single_precision/test_woodbury_refdelta.py` (new, **4 green**): f64-exact vs `woodbury`
  (~1e-11, any Φ/Φ_ref); reference-independence; f32 ≥10× tighter than direct; f32 dtype contract
  (GP-block cho_factor stays f32, logL_ref/dQ/final-combine f64). 145 single-prec+metamatrix all pass.
- `dev_architecture/single_precision/` (new):
  - `harness_proj_discriminator.py` — the (svd × project) 2×2 table on chain draws (finding #1).
  - `posterior_regression.py` — m2a/CURN f32 NUTS vs f64 chain + JS-vs-floor compare (finding #2).
    Output `posterior_regression_f32.feather`. Re-summarise anytime: `python posterior_regression.py compare`.
  - `harness_refdelta_table.py` — the refdelta discriminator table (finding #3).
  - `_proto_refdelta.py` — numpy sanity prototype (kept; documents the math + one oracle-bug story).
  - `finding_projection_discriminator.md`, `finding_posterior_regression.md`, `finding_refdelta_table.md`.

## The refdelta table (THE headline numbers, single-level / factorised CURN-IRN)

`harness_refdelta_table.py`, max abs logL err over 5 m3a draws, per-pulsar Φ_ref = chain median:

| Npsr | \|logL\| | woodbury f32 | refdelta f32 | gain | f64 check |
|---|---|---|---|---|---|
| 3  | 5.0e5 | 0.252 | 6.6e-4 | 379× | 6e-10 |
| 6  | 1.1e6 | 0.252 | 6.6e-4 | 381× | 5e-10 |
| 12 | 1.3e6 | 0.253 | 6.5e-4 | 389× | 5e-10 |
| 45 | 6.0e6 | 0.258 | 4.05e-2 | 6.4× | 9e-10 |
| 67 | 8.5e6 | 0.258 | 4.05e-2 | 6.4× | 2e-9 |

f64 check (= |refdelta_f64 − woodbury_f64|) ~1e-9 everywhere → exact. The n=45/67 4e-2 is **one
outlier pulsar** sampled far from its median reference (worst-draw total 0.0406, pulsar idx31 =
0.0405, other 44 ≈ 0) — the "large move from Φ_ref" regime (matches the test's `da=2.0`); f32 error
scales with increment SIZE. Typical draws ~3–5e-3 (50–90× win). woodbury baseline flat ~0.25.

## DO NEXT — the fused nested refdelta (for the HD table / the real array path)

The single-level graph + the factorised CURN table are the warm-up. The real target is the
**fused, nested** path where HD couples pulsars (no factorisation). Math is **derived & f64-verified**:
`research_note_nested_increment.md` §4 (the §2 inner resolvent increment, batched per pulsar,
propagated through the linear projection, fed as the Δv/ΔG of the §7 two-perturbation outer
increment). Build order from that note + ADR 0002 (NO flattening — keep the per-pulsar batching):

1. **Inner per-pulsar increment** at `vectorwoodburyjointsolve` (takes `Φ_ref,irn` block-diagonal).
2. **Outer two-perturbation increment** at `globalwoodbury_fused` (takes `Φ_ref,gw`).
3. **`θ_ref → {Φ_ref}` top layer** (thin, at the likelihood boundary — kernel never sees params,
   ADR 0001). Source θ_ref from the chain **median** (the "good spot"; works well, and refdelta
   degrades gracefully with distance from it). Hand-supply path also lives here.
4. **HD discriminator table** (extend `harness_refdelta_table.py` to the fused path) + an **HD
   posterior regression** (`makemodel_hd` + projection vs the **m3a** chain) — the sensitivity test
   ADR 0003 actually cares about. Reuse `posterior_regression.py`'s JS-vs-floor machinery.

ADR 0003 opt-in still holds: no reference supplied → today's Half-A path, byte-identical. Add a
"no-reference graph identical to today" structural test when wiring into a kernel/likelihood.

## Gotchas learned this session (so you don't re-derive)

- **Test on CHAIN draws, not `sample_uniform`.** Broad priors wander into extreme (efac~10) regions
  that NaN the f32 Cholesky — artifacts, not the effect under study. Chains: `data/NG15yr-m2a-chain.feather`
  (CURN, `gw_*` = common process) / `m3a` (HD). Model param names match chain cols if you name the
  red noise `red_noise` and (CURN) pass `common=['crn_log10_A','crn_gamma']`; the m2a/m3a common
  process is `gw_*` in the chain vs `crn_*` in `makemodel_curn` — map at compare. ([[feedback-test-on-chain-draws]])
- **JS at 200 samples is dominated by the sampling-noise floor.** ALWAYS compare against the floor
  = JS(f64-chain N-subsample vs full chain), and flag on EXCESS, not raw JS. `compare()` does this.
- **Nested `_working(dtype)` context managers COLLIDE** on the global kernels/working config (the
  inner `finally` resets to matrix). Compute all f64 reference quantities in one f64 pass; no nesting.
- **In metamath mode `ds.PulsarLikelihood`/`ds.ArrayLikelihood` are the rebinds** done by
  `ds.config(kernels='metamath')` — that's how you get the metamath path (and projection routing in
  `likelihood_metamath.py`). The `WoodburyKernel_varP` class is the MATRIX backend — if you see it,
  metamath isn't active.
- **Single-pulsar kernel leaves:** `psl.N` = outer red-noise `WoodburyKernel`; `.N` = inner
  white+ECORR solve provider; `.F` = red-noise Fourier basis; `.P` = red-noise prior NoiseMatrix
  (`.P.make_inv` → (Φ⁻¹, logdetΦ); `.P.N(params)` → forward Φ vector, used to build Φ_ref).
- **refdelta + 1e40 timing prior is unsafe in f32** (the `inv(Phi^-1)` step: the 1e-40 block
  underflows → inf). So the refdelta table omits timing (projection handles it, neutral). A future
  production "projection + refdelta" path needs a small composite graph (`woodbury_proj` front-end +
  refdelta increment on the kept GP block).
- The fused 67-psr f32 NUTS / graph compile is SLOW (many minutes) — budget for it; run in background.

## TASK LIST

- [x] Discriminator on chain draws → projection ruled out (neutral)
- [x] Posterior regression (m2a/CURN) → bar met (+3.8e-3 < 1e-2)
- [x] `woodbury_refdelta` single-level graph + test rung (4 green)
- [x] refdelta discriminator table (factorised CURN-IRN)
- [ ] Fused nested refdelta: inner (`vectorwoodburyjointsolve`) + outer (`globalwoodbury_fused`)
- [ ] `θ_ref → {Φ_ref}` top layer from chain median
- [ ] HD discriminator table + HD (m3a) posterior regression
- [ ] Wire into kernel/likelihood (opt-in, ADR 0003) + "no-reference identical to today" test

## Provenance / loose ends

- Nothing committed this session. HEAD is `cef8a0a` (projection). All Half-B work is uncommitted on
  top: `metamath.py` (woodbury_refdelta), `test_woodbury_refdelta.py`, 4 dev scripts, 3 finding docs,
  `posterior_regression_f32.feather`.
- Memory updated: `project_single_precision_stage2.md` (+ MEMORY.md), new `feedback_test_on_chain_draws.md`.
- Open science question: the z≈7 posterior-regression outlier (`J2010-1323_red_noise_gamma`) and the
  faint tail excess — a longer m2a chain would settle whether they are real f32 effects or MC noise.
