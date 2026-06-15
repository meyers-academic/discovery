# Exit plan: deleting `matrix.py` and `likelihood.py`

_Goal: reach a state where the metamath graph path is the **only** path.
`matrix.py` and `likelihood.py` are gone; `likelihood_metamath.py` is promoted
to `likelihood.py`; `_kernel_switch.py` is deleted; `signals.py` constructs
metamath kernels directly; the parity suite has certified equivalence on every
production path before anything is removed._

## Definition of done
- [ ] `src/discovery/matrix.py` does not exist.
- [ ] `src/discovery/likelihood.py` is the (former) metamath implementation.
- [ ] `src/discovery/_kernel_switch.py` does not exist.
- [ ] `discovery.config(kernels=...)` is gone (only one path remains) — or kept
      as a no-op shim for one release with a deprecation warning.
- [ ] `signals.py` / `deterministic.py` contain no `matrix.*` references.
- [ ] The existing parity tests pass with the surviving path, and covered every
      kernel constructor in production/example use before deletion.

---

## Evaluation of your three suggestions

### (1) Start with a copied `signals_metamath.py` — *partially agree; reframe it*
A wholesale copy duplicates ~1000 lines of basis/PSD/ORF/selection code that is
**identical** between paths and has nothing to do with kernels. The only lines
that differ are the handful of `matrix.X(...)` *constructor* calls. A full,
long-lived copy means every future `signals.py` fix has to land twice — exactly
the combinatorial-maintenance trap the whole refactor is trying to escape.

Two things make the copy less necessary than it feels:
- The `_kernel_switch` shim **already** drives the metamath path through the
  *existing, unmodified* `signals.py`. So we don't need a second signals file to
  *validate* metamath — we need one only to *retire the shim*.
- The end-state survivor is the metamath path, so the real move is "make
  `signals.py` emit metamath kernels and delete the matrix branch," not "keep a
  matrix-flavored signals.py forever."

**Recommendation.** Don't keep a permanent parallel file. Two acceptable shapes,
in preference order:

- **Preferred — explicit kernel factory (no copy).** Replace the spooky
  `setattr`-based monkeypatch with an explicit indirection both paths share:
  a small `kernels` namespace (e.g. `_kernels.py`) exposing `WoodburyKernel`,
  `NoiseMatrix*`, `NoiseMatrixSM`, `CompoundGP`, `CompoundDelay`, … that resolves
  to matrix *or* metamath per `config`. `signals.py` calls `K.WoodburyKernel(...)`
  instead of `matrix.WoodburyKernel(...)`. One `signals.py`, no duplication, no
  module-attribute mutation. When the matrix path is deleted, the factory
  collapses to metamath-only and can be inlined away.

- **Acceptable — strangler copy, but time-boxed.** If you prefer hard isolation
  while migrating, copy `signals.py → signals_metamath.py`, migrate *its*
  constructor calls to metamath directly (no shim), parity-test it as a new
  route, then `mv signals_metamath.py signals.py` and delete the original. The
  rule: it is a transient, it ends in a rename-over, it is never a permanent
  second file.

Either way the principle holds: **one signals.py at the end**, and the
constructor calls are the only thing that changes.

### (2) Move `matrix.config` + `kernel_helpers.py` into a shared `utils.py` — *strongly agree*
This is the highest-value, lowest-risk first step, and the code proves the need:
`kernel_helpers.__getattr__` currently proxies `jnp`/`jsp`/`jnparray`/… **back
from `matrix.py`** to "break the kernel_helpers ↔ matrix cycle." That cycle
exists only because the numerical-backend config physically lives in
`matrix.py`. Extracting it removes the cycle outright and gives both paths a
neutral substrate to stand on — a prerequisite for deleting `matrix.py` at all.

What should move out of `matrix.py` (lines 13–113):
- `config(**kwargs)` + the module-level backend globals (`jnp, jsp, jnparray,
  jnpzeros, intarray, jnpkey, jnpsplit, jnpnormal, matrix_factor, matrix_solve,
  matrix_norm, partial, SM_algorithm, regularize_FtNmF`) and the auto-call
  `config(backend='jax', factor='cholesky')`.
- `rngkey`.
- `cgsolve`, `make_logdet_estimator` and their `dense_funm_sym_eigh` /
  `integrand_funm_sym_logdet` helpers (the optional matfree/jaxopt block).

Then fold `kernel_helpers.py` (marker classes + SM index helpers) into the same
module and delete its `__getattr__` proxy.

**One naming nuance.** A single `utils.py` works, but "utils" tends to become a
junk drawer. The contents are really *two* concerns: (a) numerical backend
config + linear-algebra utilities, (b) GP/Kernel type markers. Consider either a
single intent-named module (`backend.py` or `core.py`) or two small ones. Not a
blocker — your `utils.py` is fine; just naming it for what it is will age
better. `matrix.config` then re-exports from it for back-compat during the
transition, and `discovery.config(kernels=...)` is unaffected (different switch).

### (3) Keep the parity tests; reuse the existing assertions on the new path — *agree*
This is exactly the harness's design payoff: the same assertions run through
multiple **routes** in `_routes.py`, so a migrated-signals path is covered by
the existing `test_pulsar/global/array/sm` files with **no new test bodies** —
you add a route, not tests. Agreed we don't need to test *all* of `signals.py`.

One addition to make it a real deletion gate: before removing `matrix.py`, do a
**coverage sweep** — enumerate every `matrix.*` / kernel constructor that
production code and the example notebooks actually call, and confirm each is
exercised by at least one parity route. The `_PATCHES` table is a good starting
inventory of the kernel types in play (Woodbury, NoiseMatrix 1D/2D, SM/ecorr,
Vector, CompoundGP, CompoundDelay). Anything used but unrouted blocks deletion.

---

## The plan (phased; each phase is independently shippable and reversible)

### Phase 0 — Extract the shared substrate (your #2)
1. Create the shared module (`utils.py`, or `backend.py` + keep `kernel_helpers`).
2. Move `config`/globals/`rngkey`/`cgsolve`/`make_logdet_estimator` there;
   `matrix.py` imports them back for now.
3. Fold `kernel_helpers` in (or leave it, but point its numerics at the new
   module and delete the `__getattr__` matrix-proxy).
4. Point `metamath.py` / `likelihood_metamath.py` at the shared module for
   `jnp`/`jsp`/`make_logdet_estimator`/`cgsolve` instead of `matrix.*`.
- **Gate:** full test suite green on *both* `kernels='matrix'` and
  `kernels='metamath'`. No behavior change intended.

### Phase 1 — Migrate `signals.py` off `matrix.*` (your #1, refined)

`signals.py`'s `matrix.*` references fall into **three categories**, and only
one goes through the kernel factory:

1. **Backend/numeric utilities** — `matrix.jnparray`, `matrix.jnp`,
   `matrix.jsp`, `matrix.partial` (the majority of hits). Not kernels: after
   Phase 0 these come from the shared backend module, **not** the factory.
2. **GP marker classes** — `ConstantGP`, `VariableGP`, `GlobalVariableGP`,
   `ExtSignal`. Move to the shared module with the other markers.
3. **Kernel constructors** — the `NoiseMatrix*` family. These go through the
   factory.

Category 3 itself splits by **role**, which dictates the migration style:

- **Role B — `NoiseMatrix1D/2D/12D` as a GP *prior* `Φ`** (wrapped in a GP
  marker). Used by `makegp_ecorr*`, `makegp_improper`, `makegp_fourier`,
  `makecommongp_fourier`, `makegp_fourier_variance`, `makegp_fourier_allpsr`,
  `makeglobalgp_fourier`. All share the uniform `GP(NoiseMatrixXD(prior), F)`
  shape. **Migration = mechanical in-place name swap** `matrix.X → K.X` (these
  are already monkeypatched today, so the factory does the same job, honestly).
- **Role A — `NoiseMatrix*` as the actual `N` (measurement-noise) kernel,
  returned directly.** Only `makenoise_measurement_simple` and
  `makenoise_measurement`. This is the gnarliest, most kernel-*selection*-heavy
  code in the file — `makenoise_measurement` chooses `NoiseMatrixSM` vs
  `NoiseMatrix1D` on whether ecorr is present, crossed with a `_novar`/`_var`
  axis (precomputed array vs `getnoise` closure). **Migration = real
  restructuring:** the metamath classes take an array *or* a callable, so the
  `_novar`/`_var` class dispatch collapses (4 terminal returns → 2:
  `NoiseMatrixSM(...)` / `NoiseMatrix1D(...)`). The ecorr-vs-not `if` stays.

**Phase 1a — extract & strangle `measurement_noise.py` (Role A).**
Pull the white-noise layer into its own module:
`residuals`, `selection_backend_flags`, `makenoise_measurement_simple`,
`makenoise_measurement`, `quantize` (epoch helper for the ecorr/SM path).
Strangler-duplicate it (see definition below), migrate *its* `NoiseMatrixSM`/
`NoiseMatrix1D` construction to metamath directly, collapse the `_novar`/`_var`
dispatch, parity-test against `test_sm.py` + `test_pulsar.py`, then rename over.
Isolating the SM/ecorr branch — where matrix and metamath diverge most — is the
real payoff of the split.
_Note: this does **not** remove `NoiseMatrix` from the rest of `signals.py`;
the Role-B builders still construct `NoiseMatrix1D/2D/12D` as priors, so the
factory still needs those entries. The split buys isolation of the hard path,
not elimination._

**Phase 1b — factory swap for the rest (Roles 1, 2, B).**
1. Add `_kernels.py` (`K`) exposing the kernel constructors, resolving per
   `config`. **Build its inventory from actual `signals.py`/`deterministic.py`
   usage, not from `_PATCHES`** — the patch table is missing at least
   `NoiseMatrix2D_novar` (`signals.py:507`) and `VectorNoiseMatrix12D_var`
   (`signals.py:461`), which are used but currently unpatched (so they don't run
   through metamath today — see Phase 3).
2. Repoint Role-B constructor calls `matrix.X → K.X`; repoint markers/utilities
   to the shared module.
3. Add a parity route that uses the factory in metamath mode; keep the legacy
   route (factory in matrix mode) as the oracle.
- **Gate (1a+1b):** existing parity tests pass via the new modules/route.
  Behavior identical. `signals.py`/`deterministic.py` have no `matrix.*` left.

> **"Strangler" / "time-boxed strangler".** A migration pattern: build the new
> version alongside the old, move callers over, then delete the old. "Strangle"
> = the new copy grows around and replaces the original. "Time-boxed" = the
> duplicate is **temporary** — it ends in a `mv new.py old.py` + delete, never a
> permanently-maintained second file. Used here only for `measurement_noise.py`.

> **"Shim".** The shim is `_kernel_switch.py` — a thin compatibility layer
> inserted between `signals.py` (written to call `matrix.X`) and the metamath
> classes, translating one to the other by overwriting attributes on the
> `matrix` module at runtime. Phase 1b replaces this spooky mutation with the
> explicit `_kernels.py` factory; the shim is deleted in Phase 5.

### Phase 2 — Rehome `likelihood_metamath.py`'s remaining `matrix.*` deps
- Move/needs homes (in the shared module or `metamath.py`):
  `ConstantGP`/`VariableGP` markers (isinstance dispatch), `CompoundGlobalGP`,
  and confirm `make_logdet_estimator`/`cgsolve`/linalg aliases now come from
  Phase 0's module.
- **Gate:** `likelihood_metamath.py` has zero `from . import matrix` /
  `matrix.` references.

### Phase 3 — Parity coverage gate (your #3, hardened)
1. Coverage sweep over **all of these (production scope, confirmed):**
   - `examples/*.ipynb` (incl. `likelihood_example`, `numpyro_example`)
   - `docs/tutorials/*.ipynb`, **including the new `cw_extsignal_example.ipynb`**
   - package source call sites
   Enumerate every kernel constructor each uses; map each to a parity route.
2. Fill known/likely gaps — these are already flagged as used-but-unrouted or
   divergence-prone:
   - `NoiseMatrix2D_novar` (`signals.py:507`, `makegp_fourier_variance` constant
     branch) — used, **not in `_PATCHES`**.
   - `VectorNoiseMatrix12D_var` (`signals.py:461`, `makecommongp_fourier` vector
     branch) — used, **not in `_PATCHES`**.
   - ecorr → Sherman–Morrison path (`makenoise_measurement`, Phase 1a) — highest
     divergence risk.
   - CW / ExtSignal cross-terms; global & vector Woodbury.
3. Decide tolerance policy for the metamath-vs-matrix numeric comparison and
   document it in `_comparison.py`.
- **Gate:** documented checklist of "constructor → covering test" with no holes.
  Any constructor used by the scope in (1) but lacking a route is a **hard
  blocker**, not a TODO.

**Phase 3 outcome (done) — see `phase3_coverage.md`.** Sweep complete. Closed
gaps by adding parity rows: `fftcov_2d` (`NoiseMatrix2D_var`) and `delay`
(`CompoundDelay`); `VectorNoiseMatrix12D_var` was already closed in Phase 1.
`cw_extsignal_example.ipynb` verified in both modes (~1e-9 agreement). Two
carry-overs remain, both pinned for Phase 4:
  - **`NoiseMatrix2D_novar`** — kernel maps, but the all-constant 2D GP
    likelihood path is unsupported in metamath (`CompoundGP._build_mixed_logprior`
    needs `gp.index`). Pinned by strict `xfail`
    `test_pulsar.py::test_logL[fourier_variance_fixed]`.
  - **`CompoundGlobalGP`** — factory fallthrough to `matrix.py`; no metamath
    port; untested globalgp-as-list edge case.

### Phase 4 — Close the carry-overs (no deletion)

Make the metamath path feature-complete so nothing real depends on the matrix
path's unique behavior — **without removing anything yet**. Both paths stay
present and parity-tested; this is the checkpoint where others can exercise the
metamath path on their own workflows before we commit to deletion in Phase 5.

1. **All-constant 2D GP prior.** Port the path so metamath handles a constant
   (index-less) 2D GP prior (`metamath.CompoundGP._build_mixed_logprior`
   currently requires `gp.index`). Then the strict `xfail` on
   `test_pulsar.py::test_logL[fourier_variance_fixed]` flips to pass — remove
   the marker.
2. **`CompoundGlobalGP`.** Relocate it out of `matrix.py` (into the `_kernels`
   factory + a metamath implementation) so it builds metamath kernels in
   metamath mode; add a parity route for the globalgp-as-list form.
3. Any other behavior gaps surfaced while others test the metamath path.
- **Gate:** full suite green with **no `xfail` carry-overs remaining**; the
  metamath path matches the matrix oracle on every routed model. `matrix.py`
  and `likelihood.py` are untouched and still serve as the oracle.

> **Checkpoint for external testing.** After Phase 4, the metamath path is
> intended to be complete. Pause here for collaborators to validate it against
> real analyses before Phase 5 removes the fallback.

### Phase 5 — Delete

Only once Phase 4 is signed off and the metamath path has been externally
exercised.
1. Remove the matrix branch from `_kernels.py` (factory collapses to metamath);
   delete `_kernel_switch.py`.
2. Delete `matrix.py`. Delete the matrix-mode parity routes/oracle. Retire the
   original measurement-noise oracle if still distinct.
3. `git mv likelihood_metamath.py likelihood.py`; delete the old `likelihood.py`.
4. Fix `__init__.py`: drop `from .matrix import *`; either remove
   `config(kernels=...)` or keep it as a deprecation no-op for one release.
5. Update `examples/` and `docs/tutorials/` notebooks that import/select paths.
- **Gate:** full suite green; example notebooks run; no `matrix`/
  `likelihood_metamath` references remain.

---

## Sequencing notes & risks
- **Phase 0 before everything.** Until the backend config leaves `matrix.py`,
  deleting `matrix.py` is impossible by construction (everything reaches in for
  `jnp`). It's also the cheapest, most mechanical phase.
- **Keep the oracle until Phase 3 passes.** Phases 0–2 must preserve the
  matrix path as the comparison oracle; do not delete anything in those phases.
- **Biggest risk is silent numeric drift** in less-exercised kernels (ecorr
  Sherman–Morrison, global/vector Woodbury, CW cross-terms). The coverage sweep
  in Phase 3 is the guard; treat an unrouted-but-used constructor as a hard
  blocker, not a TODO.
- **Notebooks count as production.** They're the most likely place a kernel
  constructor is used that the unit tests miss — include them in the sweep.
- Each phase leaves the tree shippable and green, so the work can pause between
  phases without leaving a half-migrated state.
