# Plan: mixed-Phi `CompoundGP` and the external-prior path

## 1. Overall goal

The metamatrix branch exists to **delete `matrix.py`**. Every kernel/GP
operation that real models hit must have a metamath equivalent built from
the metamatrix DSL, and the parity suite under `tests/metamatrix/` must
show that the metamath implementation is numerically equivalent to the
matrix.py oracle on every model topology a user can build. Once that is
done — and after a small mechanical rewrite of `signals.py` and
`likelihood.py` to construct metamath objects directly — `matrix.py` (and
the `_patch.py` monkeypatch) get deleted.

Today the parity suite reports **35 passing / 1 xfailed**. The single
remaining xfail is the only topology blocking deletion from a coverage
standpoint:

```
tests/metamatrix/test_array.py::test_clogL_new_features[decenter+common_rn+global_hd]
```

i.e. an `ArrayLikelihood` with **both** a `commongp` (per-pulsar
diagonal Phi) **and** a `globalgp` (HD-correlated dense Phi). Closing
this gap is the immediate next step in the migration; see
`metamatrix_architecture.md` ("Current state and next steps").

## 2. How this fits the overall goal

The xfail is two distinct problems wearing the same name:

1. **`mh.CompoundGP` cannot represent a mixed-Phi compound.** It tries to
   `concat` per-GP `Phi.N` arrays. When one GP's Phi is shape
   `(npsr, 2*n_rn)` (per-pulsar diagonal red noise) and the other is
   `(npsr*2*n_gw, npsr*2*n_gw)` (HD-correlated global GP), the concat
   blows up before any kernel is even constructed. This is the error the
   `cw_extsignal_example` notebook hits.

2. **`mh.VectorWoodburyKernel.make_kernelproduct_gpcomponent` has no
   graph path for an externally-provided prior.** When the compound GP
   cannot collapse into a single `P.make_inv` (the mixed-Phi case
   above), the prior on the GP coefficients has to be supplied as a
   separate callable. `matrix.VectorCompoundGP` does this by setting
   `multigp.prior = priorfunc` and `multigp.Phi = None`; the matrix.py
   `VectorWoodburyKernel_varP.make_kernelproduct_gpcomponent` then takes
   the `hasattr(self, 'prior')` branch. Our recent rewrite raises
   `NotImplementedError` on that branch — intentionally, so we close
   it as a graph leaf rather than reintroducing a closure path.

Closing both pieces:

- **Unblocks the notebook** (`cw_extsignal_example.ipynb`) and any
  user-built model with `commongp + globalgp(hd_orf)`.
- **Flips the last parity xfail** to passing → 36/36.
- **Removes the last metamath capability gap** before `signals.py` /
  `likelihood.py` can be rewritten to construct mh objects directly and
  matrix.py can be deleted.

So this isn't a side feature: it's the last piece of mh coverage that
the migration plan in `metamatrix_architecture.md` ("Migration plan: how
matrix.py goes away") is waiting on.

## 3. Plan of attack

Approach the two problems in order; each is a self-contained step with a
parity oracle waiting.

### Step 1 — diagnose mixed-Phi at `CompoundGP` construction time

`mh.CompoundGP.__init__` (metamath.py:729-758) already classifies its
inputs to decide between "list-of-dicts index" (vector commongp /
commongp + globalgp) and "cumulative-offset flat dict" (single
PulsarLikelihood). Extend that classification with a third decision:

- **uniform-Phi compound**: every `gp.Phi.N` has compatible shape →
  concat works → today's behavior (single `Phi` property,
  `make_kernelproduct_gpcomponent` uses the `P.make_inv` path).
- **mixed-Phi compound**: at least one `gp.Phi.N` shape doesn't match
  the others → cannot collapse into a single `Phi`. Set a flag
  (`self._mixed_phi = True`) and **do not** build the `.Phi` property at
  all — leave it absent (or `None`).

The shape check is local: compare `gp.Phi.N.shape` across `self.gplist`
at `__init__` time. (Calls to `getN(params)` aren't required for the
shape decision; the `.shape` of the constant `Phi.N` array, or of the
graph leaf's `.shape` attribute if it carries one, is enough. We can
fall back to a single `getN({})` evaluation if needed — at construction
time this is fine; we're not inside a graph yet.)

### Step 2 — build a per-GP prior log-density as a graph factory

For a mixed-Phi compound, the joint prior decomposes as a **sum of
per-GP priors**, each over its slice of the coefficient vector:

```
log p(c | hyperparams) = Σ_g  log p_g(c[g_slice] | hyperparams_g)
```

For a GP whose Phi is per-pulsar diagonal (red noise, CURN-style), the
contribution is `-0.5 (c_g · Phi_g^{-1} · c_g) - 0.5 sum(log Phi_g)`
batched over pulsars. For the HD globalgp, Phi is dense `(npsr*k,
npsr*k)` and the contribution is `-0.5 c_g^T Phi_g^{-1} c_g - 0.5
logdet(Phi_g)` on the flattened coefficient vector.

Express this as a new `@mm.graph`:

```python
@mm.graph
def compoundgp_prior(g, c, per_gp_invs, slices):
    # c : (npsr, k_total)   — full per-pulsar coefficient matrix
    # per_gp_invs : list of GraphLeafs, each returning (Pm_g, ldP_g)
    # slices     : (per-gp coefficient layout, resolved at build time)
    ...
    g.named(logpr_total, 'logpr')
```

The slicing pattern (which columns of `c` belong to which GP) is fixed
at graph-construction time and is the same information that
`CompoundGP.index` already encodes. Each GP contributes its own
`Pinv_g` graph leaf (re-using `noiseinv` / each GP's own `Phi_inv` if
one was supplied), so folding bakes whatever pieces are constant per
GP.

This is the "right way" version of `matrix.VectorCompoundGP.priorfunc`.

> *If needed:* matrix.VectorCompoundGP at this branch (in
> `meyers_fork_metamatrix/discovery/src/discovery/matrix.py`) gives the
> concrete arithmetic for both the diagonal-per-pulsar and the dense-HD
> contributions — useful as a reference for the einsum shapes. Do not
> port the *structure* (it's a closure factory); only consult it to
> confirm the math.

### Step 3 — expose the prior on `mh.CompoundGP`

In the mixed-Phi branch, `CompoundGP` should expose a `.prior` graph
factory: a function `prior(c_for_prior) -> Graph` that, given a
graph-symbol `c_for_prior`, returns a single-output graph computing
`logpr`. Equivalently, the GP can carry a precomputed graph dict
already wired against an `ArgLeaf` named `c`; `VectorWoodburyKernel`
then passes that graph as a `GraphLeaf` and `Apply`s it with the live
`c_for_prior` symbol from inside `vectorgpcomponent`.

Concretely:

```python
class CompoundGP:
    ...
    # set only in the mixed-Phi branch:
    self.prior = compoundgp_prior(None, per_gp_invs=..., slices=...)
```

Then `make_kernelproduct_gpcomponent` reads `self.prior` once and
treats it like any other GraphLeaf, just as it already does for
`P.make_inv` in the uniform-Phi case.

### Step 4 — branch `vectorgpcomponent` on prior vs Pinv

`vectorgpcomponent` (metamath.py:234) currently always computes
`logpr = -0.5 c^T Pm c - 0.5 sum(ldP)` from a `Pinv` leaf. Generalize
the prior subexpression so it accepts either shape:

- **Pinv path** (today's behavior): `Pinv` returns `(Pm, ldP)`; graph
  computes the einsum.
- **Prior-graph path**: a `prior_graph` GraphLeaf taking `c_for_prior`
  and returning `logpr` directly. Graph applies it.

These are structurally different — pick one of:

- **(a)** Keep `vectorgpcomponent` parameterized on a single `prior`
  leaf with a uniform contract: *return scalar `logpr` given
  `c_for_prior` and `params`*. The uniform-Phi case wraps its
  `(Pm, ldP)` computation as a tiny graph leaf that internally does the
  einsum. Cleanest from the graph's perspective; one code path.
- **(b)** Build two variants of `vectorgpcomponent`, or branch
  internally at graph-build time on whether `Pinv` or `prior_graph` was
  supplied. More duplication; less risk of regressing the uniform path.

**Recommend (a)** — it matches the architecture-doc principle ("collapse
const/var into one graph, don't translate matrix.py"). The uniform-Phi
wrapping is a one-line `@mm.graph` that calls `noiseinv` and computes
the einsum, then exposes a `'logpr'` output. The mixed-Phi version is
the `compoundgp_prior` graph from Step 2. Both look identical to
`vectorgpcomponent` from the outside.

### Step 5 — lift the `NotImplementedError` and reach parity

With `mh.CompoundGP` producing a `prior` graph in the mixed case, and
`VectorWoodburyKernel.make_kernelproduct_gpcomponent` plumbing it
through `vectorgpcomponent` instead of `self.P.make_inv`, the parity
xfail row should pass with no changes to the test file. Steps:

1. Remove the `xfail` marker on
   `tests/metamatrix/test_array.py::_decenter_common_rn_global_hd`.
2. Run `pytest tests/metamatrix/` → expect 36/0/0.
3. Re-run the `cw_extsignal_example.ipynb` notebook end-to-end under
   the `metamatrix_patch()` context.
4. Verify the inspected graph (`mm.print_graph(model.clogL.graph)`)
   shows the prior contribution as a single named sub-graph rather than
   a closure-evaluated black box. This is the smell-test that the
   architecture-doc rules were followed.

### Step 6 (optional, after parity) — exercise the new `prior` branch in tests

Add a targeted unit test that builds `mh.CompoundGP` with a
deliberately-mixed Phi pair (one per-pulsar-diagonal GP, one dense
HD-correlated GP), checks that `.prior` is set, calls
`mm.func(prior_graph)` with a synthetic `c`, and compares against an
analytic per-GP sum. This is independent of `VectorWoodburyKernel` and
makes future regressions in the `prior` math easier to localize.

## Out of scope (deliberately)

- Rewriting `signals.py` to construct metamath objects directly. That is
  the *next* migration step after parity is at 36/36 (see
  `metamatrix_architecture.md`'s migration plan).
- Touching `cglogL` / Lanczos. That remains the open Tier-3 question and
  is independent of this change.
- The matrix.py-side `VectorWoodburyKernel_varP.make_kernelproduct_gpcomponent`
  oracle. It stays as the parity oracle until the row goes green; then
  it gets deleted with the rest of matrix.py.

---

## Status: completed

The plan above is implemented. Parity suite: **36 passed / 0 xfailed**.
The `cw_extsignal_example` notebook end-to-end commongp+HD-globalgp+CW
model also runs under `metamatrix_patch()` (verified at 3 pulsars in
the parity test and at 67 pulsars in an interactive driver).

### What landed

1. **`metamath.gaussian_coefficient_logprior`** (`@mm.graph`). General
   Gaussian log-prior on per-pulsar coefficients given a single batched
   `Pinv` leaf. Replaces the inline einsum that used to live in
   `vectorgpcomponent`. Carries explicit shape documentation
   (`c_for_prior` (npsr, k); `Pm` (npsr, k, k); `ldP` (npsr,)).

2. **`metamath.vectorgpcomponent` refactor** (Step 4 option (a) of the
   plan). Replaced the `Pinv` argument with a unified `prior` GraphLeaf
   that takes `c_for_prior` and returns scalar `logpr`. One code path
   serves both the single-Pinv compound (via
   `gaussian_coefficient_logprior` wrapping `P.make_inv`) and the
   mixed-Phi compound (via the per-GP-sum graph below).

3. **`metamath.CompoundGP` mixed-Phi handling**:
   - `__init__` now detects mixed Phi by dispatching on `NoiseMatrix1D`
     vs `NoiseMatrix2D` across `gplist` (sets `self._mixed_phi`).
   - When mixed, sets `self.prior` to a graph built by the new static
     `_build_mixed_logprior(gplist)` — uses `GraphBuilder` directly to
     slice `c_for_prior` by per-GP widths (read from `gp.index`) and
     sum per-GP contributions:
       * `NoiseMatrix1D` gp (per-pulsar diagonal):
         `-0.5 Σ c²/Φ - 0.5 Σ log|Φ|`.
       * `NoiseMatrix2D` gp (e.g. HD-correlated globalgp):
         `-0.5 cᵀ Φ⁻¹ c - 0.5 logdet(Φ)`, with `c` flattened row-major
         (matches the per-pulsar block ordering encoded by
         `gp.index`).
   - The `Phi` property returns `None` in the mixed case (matching
     `matrix.VectorCompoundGP`); `make_kernelproduct_gpcomponent` then
     reads `self.prior` instead of calling `self.P.make_inv`.

4. **`VectorWoodburyKernel.make_kernelproduct_gpcomponent`**. Removed
   the `NotImplementedError` on the `prior` branch. Reads
   `self.prior` if set (mixed-Phi), otherwise wraps `self.P.make_inv`
   in `gaussian_coefficient_logprior`. Both look the same to
   `vectorgpcomponent`.

5. **Parity test row.** `tests/metamatrix/test_array.py`'s
   `decenter+common_rn+global_hd` row had its `xfail` marker removed
   and now passes. **Total: 36 passed, 0 xfailed.**

### What is NOT done (deferred, in rough priority order)

The remaining gaps that block `matrix.py` deletion or that are
worth tightening before/after. Pulled from a fresh read of
`metamatrix_architecture.md` and the architecture-doc handoff
section, plus things noticed while implementing the above.

#### Tier-3 / hard blockers for `matrix.py` deletion

1. **`cglogL` / CG-MDL / Lanczos logdet path.** Entirely matrix.py-side;
   no metamath equivalent yet. Lanczos is iterative and doesn't fit
   the static graph DSL cleanly — open design question is whether to
   carve out a callable exception (like `make_sample`) or introduce a
   new graph construct. **Largest remaining Tier-3 deliverable.**

2. **`GlobalLikelihood.conditional` graph rewrite.** Today it works via
   `make_kernelsolve` (a callable wrapping a graph) plus a Python
   closure over `Pinv + block_diag(FtNmF)`. Functionally correct but
   it's a closure on top of a graph rather than one composite graph,
   so folding can't see across the seam. Eventually subsume into a
   single graph the way the other kernel methods are.

#### Coverage / correctness polish

3. **Targeted unit test for the new mixed-Phi prior branch.**
   Plan Step 6. Build `mh.CompoundGP` with a deliberately-mixed Phi
   pair (one per-pulsar-diagonal GP, one dense HD-correlated GP),
   confirm `.prior` is set, call `mm.func(prior_graph)` with a
   synthetic `c`, and compare against an analytic per-GP sum.
   Independent of `VectorWoodburyKernel`; localizes future regressions
   in the prior math.

4. **Inspect the folded mixed-Phi graph.** Step 5.4 of the original
   plan — run `mm.print_graph(model.clogL.graph)` and verify the
   prior contribution is a single named sub-graph rather than a
   closure-evaluated black box. Smell-test that the
   architecture-doc rules (collapse const/var into one graph, no
   closure factories) were followed.

5. **`regularize_FtNmF` symmetrization** in
   `make_kernelproduct_gpcomponent`. Not carried into the mh version.
   matrix.py defaults this `False` for f64, so parity tests don't
   exercise the `True` branch. Single-precision users or anyone
   setting it `True` will diverge slightly. Low risk; flag for
   eventual port.

6. **2D `NoiseMatrix.make_sample`.** Implemented but only exercised
   through the GlobalLikelihood prior-draw path. Worth a targeted
   standalone test.

7. **`Pm` shape mismatch (cosmetic).** metamatrix's `matrix_inv`
   produces 3D batched-diagonal `(npsr, k, k)`; matrix.py's
   `VectorNoiseMatrix1D_var.make_inv` produces 2D `(npsr, k)`.
   Handled via `Pm.ndim` dispatch in `make_kernelproduct_gpcomponent`.
   Numerical results agree to rtol=1e-10 but the 3D path does more
   arithmetic than necessary. Efficiency item, not correctness.

#### Bridging code with a known sunset date

8. **The `getN` alias on `mh.NoiseMatrix`.** Exists so `signals.py:134`
   (`egp.Phi.getN`) keeps working under the patch. Dies naturally when
   `signals.py` switches to metamath construction directly (migration
   plan step 3).

9. **`tests/metamatrix/_patch.py`.** The key list IS the spec for what
   matrix.py symbols need metamath equivalents before deletion.
   Currently covers everything the parity suite exercises. Will be
   deleted alongside `matrix.py` once migration steps 3–4 land.

### Path to deletion

These are the remaining steps from `metamatrix_architecture.md`'s
"Migration plan: how matrix.py goes away" once the gaps above are
closed:

1. **Resolve `cglogL`** (gap #1). This is the only Tier-3 item that
   can't be mechanically rewritten — needs a design decision first.
2. **Rewrite `signals.py`** constructors to return metamath objects
   directly (drop `matrix.NoiseMatrix1D_var(...)` → `mh.NoiseMatrix1D(...)`).
   After this, the monkeypatch becomes a no-op.
3. **Rewrite `likelihood.py`** to import only metamath. The
   `isinstance(x, matrix.NoiseMatrix1D_var)` ndim dispatches become
   `isinstance(x, mh.NoiseMatrix1D)`.
4. **Delete `matrix.py`** and `tests/metamatrix/_patch.py`.
5. **Optionally rename** `metamath.py` → `matrix.py`. The "metamatrix"
   prefix in `tests/metamatrix/` becomes a historical artifact at that
   point.

Steps 2–4 are mostly mechanical once #1 is resolved.
