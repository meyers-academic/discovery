# Finding — the projection discriminator (closes the Session-3 open thread)

**Question (from `HANDOFF_projection.md`):** Session 3 showed timing-model projection does
*not* improve the float32 logL error on the standard array harness, because that harness
builds the timing model with `svd=True` (orthonormal design `M`), under which the 1e40 timing
prior lands as a harmless `1e-40` block. The open thread: with **`svd=False`** (raw timing
design, many orders of magnitude of dynamic range) the baseline 1e40 Woodbury *should* break in
f32 and projection *should* hold — making projection a **conditioning** tool worth keeping for
hard / raw-basis datasets even if it is not a **precision** tool. Test it.

**Experiment:** `harness_proj_discriminator.py`. The production fused array model
(per-pulsar white+ECORR+timing, commongp red noise, globalgp HD = the NG15 m3a topology) over
a 2×2 sweep of `(svd, project)`, f32-vs-f64 abs logL error. **Parameter points are drawn from
the NG15 m3a posterior chain** (`data/NG15yr-m3a-chain.feather`; model param names match the
chain columns exactly) — i.e. physically-realistic draws, NOT broad-prior `sample_uniform`
draws (which wander into extreme efac~10 regions that NaN the f32 Cholesky for reasons unrelated
to the timing treatment). Reports finite-fraction and a basis-dependence cross-check.

## Result (realistic m3a posterior draws) — projection is NEUTRAL; the f32 floor is real

`abs_err(f32)` = max over 10 draws; all cells **finite 10/10**. `|logL|` ~ 1e6 (n6) → 5.9e6 (n45):

| svd | project | n=6 | n=24 | n=45 |
|-----|---------|-----|------|------|
| True  | True  | 1.1e-1 | 2.0e-1 | 9.3e-1 |
| True  | False | 8.5e-2 | 1.7e-1 | 7.5e-1 |
| False | True  | 6.2e-2 | 1.9e-1 | 5.1e-1 |
| False | False | 9.1e-2 | 1.8e-1 | 3.5e-1 |

1. **On physical parameters every config stays finite (10/10).** The NaNs seen with prior
   draws (below) were broad-prior artifacts — extreme white noise overflowing the f32 GP-block
   Cholesky — NOT anything about the timing treatment.

2. **Projection is neutral.** All four configs sit within ~2–3× of each other at every scale,
   all at the **f32-epsilon relative floor** (abs_err / |logL| ≈ 6e-8–1.3e-7 ≈ `float32` eps).
   The raw-design baseline (`svd=False, project=False`) does NOT break — the `1e40`→`1e-40`
   prior block flushes to zero in f32 regardless of the basis of `M`. Projection wins nothing
   and (within noise) costs nothing on realistic params.

3. **The f32 floor is real, physical, and scales with `|logL|`.** abs_err grows roughly
   linearly with the magnitude of logL (0.09 → 0.18 → 0.75 across n=6/24/45) while staying
   ~f32-eps *relative*. So it is not a sampler/prior artifact: it is the genuine f32-component
   error of the GP-block (commongp + HD globalgp) solve — exactly what **Half B / reference+delta**
   removes. Whether ~0.75 absolute ΔlogL at NG15 scale matters for sampling is the still-unset
   "bar" question, but it is **identical with and without projection.**

### Prior-draw run (for contrast — has artifacts, do not over-read)
With `sample_uniform` draws the picture is noisier: `svd=False, project=True` showed a one-draw
**2.76** spike at n=24 and several configs lost a draw to NaN (9/10). Both effects **vanish on
m3a draws**, so they were prior-draw conditioning artifacts, not real weaknesses of projection
on the raw design. (Kept here only to explain why the realistic draws matter.)

## Correctness cross-check (a bonus, strongly positive)

Projection's f64 logL shifts between `svd=True` and `svd=False` by **161.3 (n=6) / 626 (n=24)
/ 1153 (n=45)** — *and the 1e40 Woodbury shifts by the identical amount* at every scale (e.g.
1152.7 vs 1153.0 at n=45). This is correct: the flat improper prior is not
reparametrization-invariant, so the timing
Jacobian `logdet(AᵀA)=logdet(MᵀK⁻¹M)` genuinely depends on the basis of `M`. The point is that
**projection reproduces the 1e40-Woodbury's basis dependence exactly** — independent numerical
confirmation that `woodbury_proj` computes the same marginal likelihood (up to the dropped
`0.5·m_tm·log σ²` constant) as the production path, now across two different `M` bases.

## Conclusion

Timing-model projection (ADR 0004) is **mathematically correct and verified** (the basis-
dependence cross-check confirms it reproduces the production likelihood exactly), but on the
metamath f32 path it is **neither a precision tool (Session 3) nor a conditioning rescue
(this session)**. On realistic NG15 m3a posterior draws it is simply **neutral** — every
config is finite and sits at the same f32-eps relative floor, with or without projection or
SVD. It stays **opt-in, off by default** (ADR 0003); nothing about the default path changes.
(It would only earn its keep on a backend/dataset forced to a raw, ill-conditioned `M`, which
is not this one.)

**The f32 floor is the genuine sampled-GP (commongp + HD globalgp) f32-component error → the
next real work is Half B / reference+delta** (`research_note_nested_increment.md`, ADR 0003),
the only thing that touches the quantity actually limiting the array path. The broad-prior NaNs
are a separate, known sampler-conditioning artifact (extreme white noise), not a precision bug.
