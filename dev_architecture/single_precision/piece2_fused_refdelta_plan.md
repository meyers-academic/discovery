# Implementation plan — fused two-level reference+delta (HD / CURN-with-IRN)

> **Audience:** Patrick is a physicist, not a software engineer. Plain language;
> math/physics framing; explain any jargon.

Piece 2 "Half B", the **fused two-level** reference+delta. The single-level
`woodbury_refdelta` (committed, `metamath.py:170`) and its harness table
(`finding_refdelta_table.md`) only cover models with **no cross-pulsar coupling**,
where the array likelihood factorises `logL = Σ_i logL_i` and you can just sum
per-pulsar single-level results. This plan builds the version that the coupled
levels need.

## What this covers (and why one build does both)

The fused path (`vectorwoodburyjointsolve` → `globalwoodbury_fused`) is a **two-level
nested Woodbury**: an inner per-pulsar intrinsic-red-noise (IRN) GP, then an outer
global GP. Two regimes route through it:

- **IRN + CURN** — outer prior block-diagonal per pulsar (no spatial correlation).
- **IRN + HD** — outer prior dense (Hellings–Downs ORF couples all pulsars).

These are the *same graph*; the only difference is the shape/structure of the outer
prior Φ. So building the fused refdelta once buys both. (Pure CURN-only, with no IRN
— a single-level vectorised refdelta — is **deliberately skipped**: not needed.)

The increment math is already derived and **machine-verified in f64** against an
mpmath (60-digit) oracle: `research_note_nested_increment.md` (~1e-16 at every move
size) + `tests/single_precision/test_refdelta_nested.py`. This plan is the *graph
wiring* of that verified math — no new math.

## Where the references attach (file: `likelihood_metamath.py`, NOT `likelihood.py`)

The fused kernel is built at `likelihood_metamath.py:701`:
```python
self.vsm = metamath.VectorWoodburyKernel(Ns, commongp.F, commongp.Phi)   # inner
self.gsm = metamath.GlobalWoodburyKernel(self.vsm, self.globalgp.Fs, self.globalgp.Phi)
```
So inside the fused kernel the two live priors are:

| level | live prior leaf | reference needed | shape (= conform to live Φ) |
|---|---|---|---|
| **inner** (per-pulsar IRN) | `self.Ns.P` = `commongp.Phi` | **Φ_ref,in** | **batched per-pulsar** (diagonal for power-law IRN) — NOT one dense block |
| **outer** (CURN/HD) | `self.P` = `globalgp.Phi` | **Φ_ref,gw** | **whatever `globalgp.Phi` is** — dense (HD) / block (CURN) |

## API (decided)

- **Single `reference=θ_ref` params dict** at the `ArrayLikelihood` /
  `GlobalLikelihood` boundary. One central point (ML / Gibbs-bootstrapped /
  chain-median) covers both levels — each level's `Phi` reads its own params.
- **The Φ reference is built at that boundary** (the "thin top layer"): evaluate
  `commongp.Phi.make_inv` and `globalgp.Phi.make_inv` at θ_ref **once, in f64**,
  freeze each result into a constant leaf (the `metamath.NoiseMatrix(jnp.asarray(...))`
  pattern the harness already uses).
- **Hard guardrail (ADR 0001):** θ_ref is consumed *entirely* in the top layer. The
  kernel and graph see **only the frozen Φ_ref constant leaves** — never a params
  dict, never a reference parameter threaded downstream. If a params dict reaches a
  kernel signature / graph node, that is the bug.
- **Opt-in (ADR 0003):** `reference=None` → today's `vectorwoodburyjointsolve` /
  `globalwoodbury_fused`, byte-identical. References present → the refdelta twins.
- **Fixed components self-cancel:** freezing the *whole* level's Φ at θ_ref is safe —
  a non-sampled component (e.g. ECORR) has φ = φ_ref ⇒ Δφ = 0 ⇒ no increment
  contribution. No need to surgically select sampled sub-components.

Data flow:
```
ArrayLikelihood(psls, commongp=…, globalgp=…, reference=θ_ref)
   │  (thin top layer, evaluated ONCE in f64)
   ├─ Pinv_ref_inner = freeze( commongp.Phi.make_inv (θ_ref) )   # batched per-pulsar
   └─ Pinv_ref_outer = freeze( globalgp.Phi.make_inv (θ_ref) )   # dense (HD) / block (CURN)
   ▼
GlobalWoodburyKernel(vsm, globalgp.Fs, globalgp.Phi, Pinv_ref_inner, Pinv_ref_outer)
   make_kernelproduct → references present?
   ├─ no  → vectorwoodburyjointsolve          → globalwoodbury_fused           (today)
   └─ yes → vectorwoodburyjointsolve_refdelta → globalwoodbury_fused_refdelta
```

## Build order (each rung validated against the f64 path before the next)

**Rung 1 — `vectorwoodburyjointsolve_refdelta`** (inner half, note §2–3).
Mirror today's `vectorwoodburyjointsolve`, but off the reference emit the *projected
increments* (Δã_i, Δb̃_i, ΔG̃_i) and the inner logdet increment, all relative to
Φ_ref,in. Batched per-pulsar resolvent identity `Δμ = −C⁻¹ ΔD μ_ref` (note §2),
`ΔD` formed per mode as `−Δφ/(φ φ_ref)` (no inverse-difference cancellation).
Batching preserved (inner stays per-pulsar).
*Test:* inner increments reproduce the trusted single-level `woodbury_refdelta` per
pulsar.

**Rung 2 — `globalwoodbury_fused_refdelta`** (outer half, note §4 two-perturbation +
§5 assembly). Outer combined perturbation `ΔK_out = ΔD_gw + ΔG̃`; quadratic +
merged-logdet increments; `logL_ref` (folds to f64) + ΔlogL (f32); final scalar add
in f64.
*Test:* matches the f64 fused path and the mpmath oracle in `test_refdelta_nested.py`.

**Rung 3 — opt-in routing + plumbing + scale test.**
`GlobalWoodburyKernel` gains optional `Pinv_ref_inner/outer`; `make_kernelproduct`
routes to the twins only when both present. Top layer in `ArrayLikelihood`
(`likelihood_metamath.py`) builds the frozen references from `reference=θ_ref`.
Tests: **"no-reference graph identical to today"** structural guard; HD-scale
f32-vs-f64 table (the HD analogue of `finding_refdelta_table.md`,
before≈1.6 / after target ~1e-3).

## House rules in force
- metamath methods **return graphs**; precision/reference intent is graph intent set
  where the kernel math is written — never a `func()`/materialise call inside kernel
  methods.
- Work in **`likelihood_metamath.py`** for the array/fused path. `likelihood.py` is
  the legacy matrix path — do not edit it.
- Tests on **chain draws** (m3a), fixed WN; `sample_uniform` NaNs the f32 Cholesky.
- Don't commit/push without explicit go-ahead; push = fork `origin` only.
