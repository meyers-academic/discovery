# Handoff — timing-model projection (Phase 1), for the next session

"What's done / decide next / watch out" sheet for the **timing-model projection** work.
Supersedes `HANDOFF_piece2.md` for everything projection-related (that file covered the
reference+delta plan, which is now correctly sequenced as Phase 2, *after* projection).
Authoritative design: `docs/adr/0004-timing-model-projection.md` (math is LaTeX-in-markdown
inside that ADR), `CONTEXT.md`, ADRs 0001–0003. This file is NOT committed.

> **Audience:** Patrick is a physicist, not a software engineer. Plain language; math/physics
> framing. "push" = fork `origin` (meyers-academic), `git push origin HEAD:metamatrix-meyers`,
> NEVER bare push. Don't commit/push without explicit go-ahead.

## The big reframe this session

Patrick's old branch `origin/whitening_and_safe_solves` (Rutger's work, ~4 months ago) is
**prior art** for exactly this. It established empirically:
- The **timing-model projection** was *the* main fix for float32. Once it worked, the
  `regularize_FtNmF` eigenvalue hack was no longer needed.
- **Projection makes the NG15 dataset work.** (The 67-psr / 675k-TOA harness ≈ NG15 scale,
  where the current metamath f32 path leaves ΔlogL = 1.62.)
- The **reference+delta split is still needed, but only for datasets LARGER than NG15.**

So the plan was reordered: **Phase 1 = port the projection to metamath (this work). Phase 2 =
reference+delta (the already-verified nested-increment math, for >NG15).** They are
complementary: projection kills the *static* timing-prior (flat-prior) cancellation;
reference+delta kills the *dynamic* sampled-GP cancellation.

**Smoking gun:** the timing model is a `ConstantGP` with prior variance `constant = 1.0e40`
(`signals.py:165,191`), which **overflows float32** (max ≈ 3.4e38). In `woodbury` it lands as
a `1e-40` block inside `cho_factor(Pinv + FtNmF)` — the worst conditioning offender.

## Decisions locked this session

1. **Projection is opt-in via an explicit `makegp_timing(..., project=True)` flag** (NOT
   auto-detect, NOT a global config). Off by default → existing models byte-identical
   (ADR 0003 no-surprises stance). DONE (see below).
2. **Projection = the exact flat-prior ($\sigma_\varepsilon^2\to\infty$) marginalization.**
   Math in ADR 0004. Correction to an earlier wrong claim: projection is **NOT** marginal-
   logL-only. Because the projector $Q$ is linear, it works for **clogL / decentered**
   (sample the Fourier coefficients on the projected $r_\perp, B_\perp$) too. And the **timing
   corrections are recoverable** — they're exactly the GLS regression coefficients the
   projection already computes (`coeffr`); back-substitute $\hat\varepsilon$ from the
   projection pieces. So the conditional path is not lost; keeping the un-projected Woodbury
   as the conditional path is an implementation choice, not a math limitation.
3. **Whiten FIRST** (project the $K^{-1/2}$-scaled design, then sum squares). Forming
   $F^TK^{-1}F$ and subtracting re-introduces large-minus-large cancellation in the
   low-frequency Fourier modes that overlap the timing polynomials — catastrophic in f32.

## What's DONE this session (UNCOMMITTED, on top of committed HEAD `e3f8f05`)

1. **ADR 0004** `docs/adr/0004-timing-model-projection.md` — full math as **LaTeX-in-markdown**
   (`$$...$$`, MathJax-style — Patrick explicitly did NOT want a standalone `.tex`; an earlier
   `.tex`/`.pdf` was written then deleted). Covers model, whitening closed form, projection =
   flat-prior limit (with derivation), clogL compatibility, corrections recoverable.
2. **`project=True` flag** on `makegp_timing`/`makegp_improper` (`signals.py`). Sets
   `gp.project`; default False. Verified on/off.
3. **Whitening in graph form — DONE + validated to machine precision.**
   - `utils.py`: `smwhiten_ind`, `vsmwhiten_ind`, `smwhiten_ind_correct`,
     `vsmwhiten_ind_correct` — port of the branch's `SM_whiten_*`. Apply $W=K^{-1/2}$ for
     $K=\mathrm{diag}(N)+F_{ec}\,\mathrm{diag}(P)\,F_{ec}^T$ via the per-epoch factor
     $\alpha_k=((1+u_k)^{-1/2}-1)/u_k$ (safe limit $-1/2$); $\log\!1p$ logdet.
   - `metamath.py`: `smwhiten` graph emitter (mirrors `smsolve`; `_sm_whiten_apply` does the
     ndim 1d/2d dispatch). **Returns a Graph (OrderedDict of nodes), not a func** — verified
     identical type to `smsolve`. `mm.func` only at the boundary.
   - Validated through `mm.func`: 1d quad $Wy\cdot Wy = y^TK^{-1}y$ exact; logdet 3.5e-15; 2d
     **Gram $(WM)^T(WM)=M^TK^{-1}M$ to 2.5e-16** (the property projection rides on).
   - Imports clean; `tests/metamatrix/test_sm.py` still green (7 passed).

## DO NEXT — write `woodbury_proj` (task #3). Design already derived:

Mirror `woodbury` (`metamath.py:57`). `M` = whitened-against timing basis, `F` = Fourier
basis kept. Pass the whitening in as a sub-graph leaf `Nwhiten` (same pattern as `Nsolve`):

```python
@mm.graph
def woodbury_proj(g, y, Nwhiten, M, F, Pinv):
    yw, lN = Nwhiten(y).split()          # W y, logdet K (use lN once)
    A,  _  = Nwhiten(M).split()          # whitened timing basis
    B,  _  = Nwhiten(F).split()          # whitened Fourier basis

    AtA       = g.dot(A, A)              # A^T A   (m_tm x m_tm), well-conditioned
    cfA, ldA  = g.cho_factor(AtA)        # ldA = logdet(M^T K^-1 M)  <-- timing Jacobian
    coeffy    = g.cho_solve(cfA, g.dot(A, yw)); r_perp = yw - A @ coeffy
    coeffB    = g.cho_solve(cfA, g.dot(A, B));  B_perp = B  - A @ coeffB

    FtNmF = g.dot(B_perp, B_perp)
    FtNmy = g.dot(B_perp, r_perp)
    ytNmy = g.pin_f64(g.dot(r_perp, r_perp))   # pin like woodbury does
    Pm, lP = Pinv
    cf, lS = g.cho_factor(Pm + FtNmF)
    mu = g.cho_solve(cf, FtNmy)
    g.pin_f64(lN); g.pin_f64(lP)         # mirror woodbury's pins
    logp = g.combine_logp_f64(ytNmy, g.dot(FtNmy, mu), [lN, ldA, lP, lS])
```

Notes / gotchas baked in above:
- `cho_factor` returns `(cf, logdet)`; unpack with `cf, ld = g.cho_factor(X)`. So the timing
  Jacobian `logdet(A^T A)` comes for free — **no separate SVD-logdet precompute needed** (the
  branch's `_get_timing_svd` was only for conditioning; column-normalized M is fine).
- `Sym` supports `@` (matmul, see `woodbury` line 81 `Nmy - NmF @ mu`), `+ - *`, `.dot`,
  `.split`, `__getitem__`. `g.dot(A,B)` = `A.T @ B`.
- The two `cho_solve(cfA, ...)` reuse the same `cfA` → one timing factorization.
- Whitening sub-graph: for **SM/ECORR noise** use `smwhiten(None, N, Uind, P)` (validated).
  For **pure diagonal** noise there is no `smwhiten` path yet → either add a tiny diagonal
  whitening emitter (`y/sqrt(N)`, logdet `sum log N`) OR make the first rung use a model WITH
  ECORR (NG15 has ECORR anyway, and it reuses the validated `smwhiten`). Recommend the latter
  for the first rung.

### f64 equivalence check (the test rung, task #5)
In float64, `woodbury_proj` logL should equal the existing `woodbury` logL (built with the
combined `[M | F]` basis and the `1e40` timing prior) **plus** the dropped flat-prior constant:

```
logL_proj  ==  logL_woodbury  +  0.5 * m_tm * log(1e40)
```

(The $\sigma_\varepsilon^2$ term enters `logdet\Sigma` as `m_tm * log(1e40)`, with the `-0.5`
prefactor; everything else — quadratic, n·log2π, logdet K — matches.) Confirm to f64 tol on a
single-pulsar timing+ECORR+1-Fourier-GP model, then on the f32 harness confirm **1.62 → small**
at NG15 scale.

## DO NEXT — route in `CompoundGP` (task #4). Patrick: "wire it in so we can see if it
## helped with the 67-pulsar test." This is THE next action.

Current line numbers (re-verified Session 2):
- `class CompoundGP` at `metamath.py:912`. `gplist` = the GPs in the PulsarLikelihood.
- `CompoundGP.F` property at **`:967`** — concatenates every gp's columns (the combined
  Woodbury basis). The timing GP's columns are in here today.
- `CompoundGP.Phi` property at **`:1062`** → returns a `NoiseMatrix` whose `combined_inv`
  (**`:1082`**) block-diagonalizes each gp's `Phi_inv`. The timing block is `1e-40·I`
  (the `1e40` prior inverted) — that's the f32 offender we want to stop forming.
- `class WoodburyKernel` at **`:651`**: `make_kernelproduct(y)` (`:676`) is the marginal
  logL seam → `woodbury(y, self.N.make_solve, self.F, self.P.make_inv)`. This is where a
  `woodbury_proj(y, self.N.make_whiten, M, F_rest, Pinv_rest)` swap goes.

**OPEN THREAD (was mid-search when session cleared):** find where the `WoodburyKernel` (or
`VectorWoodburyKernel`/`GlobalWoodburyKernel`) is actually *instantiated from a CompoundGP* —
i.e. who passes `CompoundGP.F` and `CompoundGP.Phi` into the kernel. Likely in `likelihood.py`
(grep `WoodburyKernel(` / `VectorWoodburyKernel(` and `.Phi`/`.F` there). That instantiation
site is where you detect "one gp in gplist has `project=True`", split its columns out as `M`,
and build the combined `F`/`Pinv` from the *rest*. The 67-psr harness uses the **array/fused**
path (`vectorwoodburyjointsolve`+`globalwoodbury_fused`), NOT the single-pulsar `woodbury` —
so routing must reach the VECTOR kernel, not just `WoodburyKernel`. Plan the single-pulsar
`woodbury_proj` swap first (test rung exists), THEN propagate to the vector/fused path for the
harness. The harness lives at `dev_architecture/single_precision/harness_f32_array.py`.

**Scope reminder:** marginal logL path ONLY. `make_conditional` / `make_kernelsolve` /
`make_coefficientproduct` keep the un-projected `woodbury` so timing corrections stay
recoverable (until the §2 back-substitution is wired). The `project=True` flag (signals.py)
is the routing switch; off by default → byte-identical (ADR 0003).

## TASK LIST (keep)

- [x] #1 Write ADR 0004: timing-projection math + clogL compatibility — DONE
- [x] #2 Add `project=True` flag to `makegp_timing`/`makegp_improper` — DONE
- [x] #3 Write `woodbury_proj` graph in metamath.py — DONE (Session 2)
- [x] #4 Route marked timing GP to projection — DONE (Session 3; in **likelihood_metamath.py**)
- [x] #5 Test rung: f64 equivalence + array f64 + f32 67-psr measured — DONE (see finding below)
- [x] #6 Build Sherman-Morrison whitening in graph form — DONE + validated
- [x] #7 Diagonal whitening emitter `dwhiten` (PTAs without ECORR) — DONE

## SESSION 3 — wired in + measured. KEY FINDING: projection does NOT fix the f32 floor.

**What was built (committed):**
1. `woodbury_proj` got a **`solve` output** (parallel to `woodbury`'s) — the TOA-space
   timing-projected inverse operator `A = W(P_perp − P_perp B(Pⁱⁿᵛ+BᵀB)⁻¹BᵀP_perp)W`.
   Verified: `A·x` matches the ordinary `woodbury` solve at the 1e40 limit to **8e-16**
   (1d and 2d). This is the make_solve the fused array path consumes.
2. **`WoodburyProjKernel(N, M, F, P)`** in metamath.py — `make_solve` (projected) +
   `make_kernelproduct` (projected marginal logL); `make_sample` raises (flat prior).
3. **Routing in `likelihood_metamath.py`** (NOT `likelihood.py`!). cgps with
   `getattr(g,'project',False)` → `M` = their basis; the rest stay Woodbury blocks;
   build `WoodburyProjKernel(noise, M, kept.F, kept.Phi)`. Default path byte-identical.

**CRITICAL GOTCHA (cost a wrong-file edit):** in metamath mode `ds.PulsarLikelihood`
resolves to **`discovery.likelihood_metamath.PulsarLikelihood`**, a *separate class* — NOT
`discovery.likelihood.PulsarLikelihood`. The `WoodburyKernel(` grep hits in `likelihood.py`
are the legacy/matrix path and never run under metamath. Always route the metamath path in
`likelihood_metamath.py`. (Verify with `type(psl).__module__`.)

**f64 correctness — all pass:** single-pulsar `logL(project=True) − logL(False)` ==
`0.5·m_tm·log(1e40)` to ~1e-12; the **full 67-psr-topology fused array** (commongp + globalgp
HD) `logL(True) − logL(False)` == `Σ_psr 0.5·m_tmᵢ·log(1e40)` to 8e-9 rel. So the projection
propagates correctly through the whole array via the per-pulsar `make_solve`.

**f32 result — projection does NOT improve the floor on the SVD harness:**
abs logL error (f32 vs f64), same param point, OFF vs ON:
`n=12: 9.6e-4 → 3.7e-2` · `n=24: 1.1e-1 → 1.7e-1` · `n=40: 5.6e-1 → 8.6e-1` ·
`n=67: 3.1e-1 → 1.4e-1`. Comparable-to-slightly-WORSE at small n, marginally better at 67,
all within noise. **Why:** (1) the harness uses `svd=True` timing → orthonormal M, so the
1e40 prior never actually ill-conditions `cho_factor(Pⁱⁿᵛ+FtNmF)` (the 1e-40 block flushes to
0 harmlessly). (2) The residual ~0.1–0.8 f32 error is the **GP-block (commongp+HD globalgp)
cancellation** — `mu` from the f32 Cholesky — which projection doesn't touch; that's
**reference+delta (Half B)** territory. (3) Projection *adds* f32 arithmetic to the GP-block
inputs (whiten-twice + projected-residual subtraction → `B_perp`), so it can mildly *worsen*
the f32 floor.

**Reframing the handoff's premise:** "projection is THE main fix / makes NG15 work" was about
**conditioning** (Rutger no longer needing `regularize_FtNmF`) — it rescues *hard / raw-basis*
cases — NOT the f32 precision floor this effort chases. With `svd=True` the baseline already
handles the timing model; nothing to win.

**Open next step (the discriminator):** run the same comparison with **`svd=False`** (raw
timing design, huge dynamic range) where the baseline 1e40 Woodbury should genuinely break in
f32 and projection should hold. That tells us if projection is a *conditioning* tool (keep for
raw-basis / harder datasets) vs a *precision* tool (it isn't, on this harness). Either way the
f32 floor on the standard SVD harness is **Half B / reference+delta**, not projection.

## What's DONE in Session 2 (UNCOMMITTED, on top of the Session-1 uncommitted work)

1. **`woodbury_proj` graph** (`metamath.py`, right after `woodbury`). Exactly the
   design block above. Output node is `logp` (last node → what `mm.func` returns).
2. **Diagonal whitening `dwhiten`** + `_diag_whiten_apply` (`metamath.py`, after
   `smwhiten`): `W = diag(N)^{-1/2}`, returns `(y/sqrt(N), sum log N)`. The non-ECORR
   counterpart of `smwhiten` — Patrick: "there are PTAs that do not use ECORR."
3. **`make_whiten` property** on both `NoiseMatrixSM` (→ `smwhiten`) and `NoiseMatrix`
   (→ `dwhiten`), parallel to `make_solve`. This is the `Nwhiten` leaf `woodbury_proj`
   takes — so routing (#4) just calls `N.make_whiten` like it calls `N.make_solve`.
4. **Test rung** `tests/single_precision/test_woodbury_proj.py` (5 tests, all green):
   - f64 equivalence `logL_proj == logL_woodbury(1e40) + 0.5·m_tm·log(1e40)` for BOTH
     ECORR (`smwhiten`) and diagonal (`dwhiten`) noise. (Derivation confirmed: under
     discovery's no-2π convention and equal dimension count, the ONLY difference is the
     prior-logdet block `m_tm·log σ²`. No 2π, no sign subtlety — verified numerically.)
   - the offset is data-independent (same gap for a second unrelated `y`).
   - f32: stays finite + close; **white-box dtype contract** asserted.
5. **Precision finding (important for #4):** pinning `ytNmy` pulls its ancestors to f64 —
   and in projection the projected residual `r_perp` depends on the timing factorization
   `cho_factor(MᵀK⁻¹M)`, so **that small timing-Gram Cholesky runs in f64** (cheap; m_tm³;
   and it's the cancellation-prone projection step, so f64 there is a *feature*). The
   **expensive GP-block `cho_factor(Pⁱⁿᵛ+FₜₚₑᵣₚᵀK⁻¹Fₚₑᵣₚ)` stays f32** — the speed win is
   intact. `dtype_map` confirmed: 2 Choleskys → timing-Gram f64, GP-block f32; ytNmy/lN/lP
   pinned f64; final combine f64. (Note: graphs with all-constant leaves FOLD to a single
   constant — to see the dtype map / exercise the real f32 path you need a param-dependent
   leaf, e.g. an `efac`-scaled `getN`. The test does this.)

Suites after Session 2: `tests/single_precision` + `test_sm` = 100 passed; metamatrix
parity = 48 passed. All additions are purely additive; nothing committed.

## Things learned this session (so you don't re-derive)

- **`@mm.graph` calling convention:** the decorator injects the builder `g`, so call factories
  WITHOUT a leading arg: `smwhiten(y, N, Uind, P)` (not `smwhiten(None, ...)`). EXCEPTION: the
  kernels pass `y=None` deliberately to build a *sub-graph with a free input leaf* — that's
  how `make_solve` (and our future `make_whiten`) returns a graph that gets applied later via
  `Sym.__call__` (`Nsolve(y)` → `Apply`).
- **Evaluate a graph:** `mm.func(graph)(params={})` (concrete-array leaves fold via
  `fold_constants`; pass param-dependent leaves as zero-arg... no — as `getX(params)` with
  `.params`). See `tests/metamatrix/test_sm.py` for the canonical pattern.
- **`cho_factor` node returns `(cf, logdet)`**; `matrix_inv(Φ)` returns `(diag(1/Φ), Σlog|Φ|)`;
  a metamath graph IS an `OrderedDict` of nodes (not callable).
- **`make_uind(U)` trips under jax x64 if `U` is float** (`jnp.max(jnp.sum(U))+1` used as a
  shape). Pass integer `U`, or build Uind with numpy in standalone scripts. (Possible latent
  bug in the real path under x64 — worth a look, not urgent.)
- Default kernel backend is `matrix`; graph precision only runs under
  `ds.config(kernels='metamath')`. `utils.config(working=float32)` is the precision knob.
- The harness (`harness_f32_array.py`) uses fixed WN on purpose (pins fold to f64 constants).

## Provenance / loose ends

- Branch `origin/whitening_and_safe_solves`: `_projection_products`(_np/_jax), `SM_whiten_*`,
  `WhiteningMixin_*`, the `phi0_params` likelihood split (= reference+delta prior art).
- A LaTeX manuscript Patrick uploaded in an earlier chat (NOT in this repo) may state the same
  projection result — reconcile if re-supplied.
- Memory updated: `project_single_precision_stage2.md` (+ MEMORY.md pointer).
