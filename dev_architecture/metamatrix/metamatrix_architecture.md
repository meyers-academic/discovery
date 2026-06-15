# Metamatrix Architecture

## End state

**`matrix.py` does not exist.** Every kernel operation, GP, and likelihood
component is built on the graph machinery in `metamatrix.py` (the DSL) and
`metamath.py` (the kernel/GP classes that use it). `signals.py` constructs
metamath objects directly. `likelihood.py` composes metamath graphs without
any matrix.py imports.

The current branch is a transitional state. `matrix.py` is still present,
still imported by `signals.py` and `likelihood.py`, and still functionally
correct — which is the only reason it is useful: **it is the oracle against
which the metamath replacements are tested**. The parity suite under
`tests/metamatrix/` exists to certify, row by row, that every method the
likelihood layer calls produces the same numerical result via the metamath
graph as via the matrix.py closure. When the parity suite covers every
path that any production user exercises, matrix.py gets deleted.

The motivation is not "make metamath also work" — it is "stop maintaining
matrix.py." matrix.py is costly to maintain (see the variant table below),
has known gaps (no `make_kernelproduct_gpcomponent` for non-VectorWoodbury,
no CG-MDL logdet outside one class, no `make_conditional` on
`VectorWoodburyKernel_varP`, etc.), and adding a feature means touching 4-6
sibling classes. The graph-based rewrite eliminates that combinatorial
maintenance surface entirely.

## Why this refactor exists

`matrix.py` evolved into a combinatorial explosion. For the Woodbury kernel
`Σ = N + F P F^T`, each of `N`, `F`, and `P` can be either fixed-at-trace-time
("constant") or parameter-dependent ("variable"). The current file enumerates
the variants by hand:

| Class | N | F | P |
|---|---|---|---|
| `WoodburyKernel_novar` | const | const | const |
| `WoodburyKernel_varP`  | const | const | var   |
| `WoodburyKernel_varN`  | var   | const | const |
| `WoodburyKernel_varNP` | var   | const | var   |
| `WoodburyKernel_varFP` | const | var   | var   |
| `VectorWoodburyKernel_varP` | const | const | var (per-pulsar) |
| ... | | | |

Plus per-class duplication of:
`make_kernelproduct`, `make_kernelproduct_vary`,
`make_kernelterms`, `make_kernelterms_vary`,
`make_kernelsolve`, `make_kernelsolve_vary`,
`make_kernelsolve_simple`, `make_solve_1d`, `make_solve_2d`,
`make_kernelproduct_gpcomponent`, `make_sample`...

Each of these does the same math, branching on whether things are callables vs
arrays, threading `params` only where needed, and pre-baking everything else.
The result is ~2000 lines of `matrix.py` where any nontrivial change has to be
mirrored across 4–6 sibling classes. New features (decentering, additives,
ExtSignals, CG-MDL logdet) get added to *one* class and the rest go stale.

**The metamatrix refactor exists to collapse this into a single generic path.**

## The core idea

Express every kernel operation as a **computation graph**, not a Python
closure. The graph is built once, declaratively, in terms of symbolic
operands. At runtime:

- **`fold_constants`** walks the graph and evaluates every node whose inputs
  are all constants. A "variable" thing becomes a constant simply by being a
  leaf with no free params — same code path.
- **`prune_graph`** removes nodes the requested output doesn't depend on.

This means the same source for `woodbury(y, Nsolve, F, Pinv)`:

- if `Nsolve`, `F`, `Pinv` are all constant → folds entirely into a single
  ConstLeaf at trace time. Equivalent to `WoodburyKernel_novar`.
- if `Pinv` is parameter-dependent → fold stops at the `cho_factor` node;
  everything upstream that doesn't depend on `Pinv` is still pre-baked.
  Equivalent to `WoodburyKernel_varP`.
- if `Nsolve` *and* `Pinv` are param-dependent → almost nothing folds;
  graph is evaluated end-to-end at runtime. Equivalent to `WoodburyKernel_varNP`.
- if `F` is callable → fold stops at the `F`-using nodes; same logic falls out.

**There is no "fixed N case" vs "variable N case" in the source.** There is
one expression. Folding decides what runs at trace time vs runtime.

This is why metamatrix's `mh.WoodburyKernel` doesn't have suffixes. There is
no `_var*` — there is just `WoodburyKernel`. The graph adapts.

## Graph primitives

`metamatrix.py` defines:

- **Leaves**
  - `ArgLeaf(name)` — runtime argument (e.g. residual vector passed each call).
  - `ConstLeaf(value)` — fixed array, baked into the graph.
  - `FuncLeaf(fn)` — callable that takes `params` and returns an array.
    Carries `fn.params` so the resulting graph callable knows what params to
    accept.
  - `GraphLeaf(graph)` — a nested graph, called via `Apply`.

- **Nodes**
  - `Node(op, inputs, description)` — a JAX-friendly op applied to upstream
    nodes/leaves by name.

- **The DSL** (`GraphBuilder`, `Sym`) — Python-level shorthand. `Sym`
  overloads `@`, `*`, `+`, `-`, `__call__`, `.T`, `.solve`, `.inv`, `.dot`,
  `.split`, `__iter__` so the math reads naturally:
  ```python
  @mm.graph
  def woodbury(g, y, Nsolve, F, Pinv):
      Nmy, lN = Nsolve(y)            # Apply on a GraphLeaf
      NmF, _  = Nsolve(F)
      FtNmy   = g.dot(NmF, y)        # F^T N^-1 y
      FtNmF   = g.dot(F, NmF)
      Pm, lP  = Pinv                 # destructure a 2-tuple result
      cf, lS  = g.cho_factor(Pm + FtNmF)
      ...
      logp = -0.5 * (g.dot(y, Nmy) - g.dot(FtNmy, mu)) - 0.5 * ld
  ```
  No `if callable(F): ...` branches. No `params = ... if var else None`.
  Just the math.

- **`mm.func(graph)`** → builds a JAX-jittable callable `f(*args, params={})`.
  Walks the (folded, pruned) graph and emits ops. Carries `f.params` (gathered
  from FuncLeaf / GraphLeaf params).

## Method contracts under metamatrix

Every kernel method that used to return a closure now returns a **graph**.
Where matrix.py exposed `make_kernelproduct(y) -> callable`, metamath exposes
`make_kernelproduct(y) -> Graph`. The caller composes graphs and converts to
a callable only at the outermost boundary (`ffunc` / `mm.func`) — i.e.,
inside `likelihood.py`'s `logL`, `conditional`, `clogL` cached_properties.

Concretely:

| Method | Returns | Notes |
|---|---|---|
| `NoiseMatrix.make_solve` | graph | input `y` → `(Nmy, lN)` |
| `NoiseMatrix.make_inv`   | graph | () → `(Nm, lN)` |
| `WoodburyKernel.make_solve` | graph (pruned to 'solve') | |
| `WoodburyKernel.make_kernelproduct(y)` | graph | scalar logp |
| `WoodburyKernel.make_kernelsolve(y, T)` | callable wrapping a graph | matrix.py contract for callsites that expect `ksolve(params)` |
| `WoodburyKernel.make_conditional(y)` | graph (pruned to 'cond') | |
| `VectorWoodburyKernel.make_kernelproduct(ys)` | graph | over a list of pulsars |
| `VectorWoodburyKernel.make_kernelproduct_gpcomponent(...)` | **graph** (target) | currently missing — see below |
| `make_sample` | plain callable | PRNG threading doesn't fit the graph DSL cleanly; exception |

Subgraphs compose naturally. When `WoodburyKernel.make_kernelproduct` needs
the inner noise solve, it doesn't *call* `self.N.make_solve()` to get an
array — it embeds `self.N.make_solve` as a `GraphLeaf`. The whole tree is
one composite graph that `fold_constants` simplifies in one pass.

## What this means for porting from matrix.py

The wrong way to port a matrix.py method to metamath:

```python
# DON'T — transliterates the variable/constant branching
def make_kernelproduct_gpcomponent(self, ys, ...):
    NmFs = [N.solve_2d(F) for N, F in zip(self.Ns, self.Fs)]   # const-path only
    FtNmFs = [F.T @ NmF for F, NmF in zip(self.Fs, NmFs)]
    FtNmF = jnparray(FtNmFs)
    ...
    def kernelproduct(params): ...
    return kernelproduct
```

This reproduces the matrix.py constraint ("N and F must be constant") and
adds nothing — it's matrix.py written in a different file. Every variant
class still implicitly exists; it's just been folded into runtime errors and
"don't pass variable N" assumptions.

The right way:

```python
# DO — express the math symbolically; let folding handle const-vs-var
@mm.graph
def vectorgpcomponent(g, ys, Nsolves, Fs, Pinv, reparams=(), additives=(), extsignals=()):
    # per-pulsar trace-time-or-runtime solves (graph picks)
    NmFs   = [Nsolve(F) for Nsolve, F in zip(Nsolves, Fs)]
    Nmys   = [Nsolve(y) for Nsolve, y in zip(Nsolves, ys)]
    FtNmFs = [g.dot(F, NmF[0]) for F, NmF in zip(Fs, NmFs)]
    NmFtys = [g.dot(NmF[0], y) for NmF, y in zip(NmFs, ys)]
    ytNmys = [g.dot(y, Nmy[0]) for y, Nmy in zip(ys, Nmys)]
    ldNs   = [Nmy[1] for Nmy in Nmys]
    ...
```

When `Nsolves[i]` is a constant (noisedict supplied), the `Nsolve(F)` subgraph
folds entirely. When it's parameter-dependent (free efacs), the same source
runs at runtime. **One code path, both cases.** No `solve_2d` vs
`make_solve_2d` split. No `_var` class. No `if callable(F)`.

## House rules for `metamath.py`

**We build graphs. We do not build closure factories.**

Every kernel-math method (`make_kernelproduct`, `make_solve`,
`make_conditional`, `make_kernelproduct_gpcomponent`, ...) returns a
metamatrix graph (a dict produced by an `@mm.graph` function), not a Python
callable. Composition with sub-objects (`N.make_solve`, `P.make_inv`, an
ExtSignal's coeff map, a reparam, `self.means`, ...) happens by passing them
into the graph as leaves — `GraphLeaf` for nested graphs, `FuncLeaf` for
param-dependent callables, `ConstLeaf` for arrays. `fold_constants` then
decides what runs at trace time vs runtime. That is the entire point.

Hard rules for new code in `metamath.py`:

1. **Do not call `mm.func(...)` inside a kernel method to evaluate a subgraph
   at construction time and capture the result in a closure.** That defeats
   folding/pruning across the composite graph and reinvents the matrix.py
   const-vs-var split this module exists to eliminate.
2. **Do not write `mm.func(subgraph)(args, params={})` to "materialize" a piece
   at trace time.** If a subexpression is constant, folding will bake it. If
   it isn't, it must stay in the graph.
3. **Do not branch on `callable(x)` / `isinstance(x, dict)` /
   `hasattr(self, 'prior')`** to pick between a constant path and a variable
   path. Express the math once; let folding pick.
4. **Do not introduce helpers like `_materialize` for kernel-math code.**
   `_materialize` exists only for `make_sample`, the documented exception
   (PRNG threading doesn't compose cleanly through the DSL). No other method
   gets that escape hatch.
5. **`mm.func` is called exactly once, at the outermost boundary** — in
   `likelihood.py`'s `logL` / `conditional` / `clogL` cached_properties. Not
   inside `metamath.py` kernel methods.

If you're translating something from `matrix.py` and find yourself reaching
for `mm.func`, stop. You are writing `matrix.py` in a different file. See
"What this means for porting from matrix.py" above for the right pattern.

## Practical guidance

1. **Stop thinking "constant" vs "variable" when writing metamath methods.**
   They are not two cases; they are two folding outcomes of one expression.
2. **Don't gate features on "N and F are constant."** If `make_kernelproduct_gpcomponent`
   needs to multiply `F^T N^-1 F`, it writes `g.dot(F, Nsolve(F))` and lets
   folding handle the rest. The runtime cost when N is variable is exactly the
   cost of actually solving it — there is no extra overhead because of the
   graph layer (folded paths bake out completely).
3. **Reparams / additives / extsignals are just leaves.** A reparam is a
   FuncLeaf returning `(c, ldL)`. An additive is a FuncLeaf returning a
   coefficient correction. An ExtSignal contributes a precomputable
   trace-time block (its `Fs` are constant) and a coefficient-map FuncLeaf.
   These compose into the same graph without bespoke handling.
4. **When porting matrix.py code, identify the math, then re-express it
   symbolically through `GraphBuilder`.** Discard the bookkeeping that exists
   only to manage the constant/variable split.
5. **Sampling (`make_sample`) is the documented exception** — `jax.random`'s
   key-threading model doesn't compose cleanly through the graph DSL, so
   `make_sample` returns a plain callable. This is the only kernel method
   that does.

## Composition example: decentering

To see how this scales, consider what `ArrayLikelihood.clogL` with
`decenter=True` should look like in metamath:

```python
@mm.graph
def gpcomponent_with_reparams(g, ys, Nsolves, Fs, Pinv, decenter, ...):
    # ... per-pulsar solves as above ...

    # decentering is a reparam on c. In the graph: it's a node returning (c, ldL).
    c0 = g.fold_params_to_array(...)   # xi -> c via index map
    c, ldL = decenter(c0, FtNmFs, NmFtys, Pinv)   # ldL is just another scalar node

    logpr = ...   # prior on c (post-reparam)
    c_total = c + sum(add(params) for add in additives)
    quad = -0.5 * jnp.einsum('ij,ijk,ik', c_total, FtNmF, c_total) + ...
    logp = quad + logpr + ldL + extcontrib(c_total, extsignals)
```

The decenter transform itself is a small graph that consumes precomputed
`FtNmFs`, `NmFtys`, and `Pinv` — and again, whether `Pinv` is "fixed"
(because all GP params are pinned) or variable doesn't change the source.
The graph adapts.

This is the eventual payoff: matrix.py's `WoodburyKernel_varP.make_kernelproduct_gpcomponent`
becomes one symbolic expression in metamath, valid for every combination of
fixed/variable N/F/P. **The 18 paths collapse to one.**

## Migration plan: how matrix.py goes away

The deletion is staged so the test suite stays green throughout. Roughly:

1. **Now (this branch's job).** Build metamath equivalents for every kernel
   class and method that any of `signals.py` / `likelihood.py` /
   `optimal.py` constructs or calls. The current monkeypatch in
   `tests/metamatrix/_patch.py` is the *spec*: every key it patches is a
   matrix.py symbol that must have a metamath replacement before deletion.
2. **Parity-test everything.** Every model topology that real users build
   (single-pulsar, GlobalLikelihood, ArrayLikelihood with/without
   commongp/globalgp, decentering, additives, ExtSignals, CG-MDL logdet)
   must have a row in `tests/metamatrix/` showing metamath ≈ matrix.py
   numerically.
3. **Rewrite `signals.py` constructors to return metamath objects directly.**
   Drop `matrix.NoiseMatrix1D_var(getnoise)` in favor of
   `mh.NoiseMatrix1D(getnoise)`, etc. After this step, the patch becomes
   a no-op.
4. **Rewrite `likelihood.py` to import only metamath.** Remove
   `from . import matrix`. The `isinstance(x, matrix.NoiseMatrix1D_var)`
   dimension dispatches become `isinstance(x, mh.NoiseMatrix1D)`. The
   1D/2D marker classes (introduced in metamath specifically to support
   this) carry the type discrimination matrix.py's hierarchy carried.
5. **Delete `matrix.py`.** Remove the file. Remove `_patch.py` (its purpose
   was to bridge the gap during transition).
6. **Rename `metamath.py` → `matrix.py` (optional).** At that point
   metamath IS the matrix subsystem; the name only mattered to distinguish
   the two during the migration.

`tests/metamatrix/` itself becomes test of the matrix subsystem proper
once matrix.py is gone — the "metamatrix" prefix is a historical artifact
of the transition.

## Current state and next steps

What's already in place (`tests/metamatrix/` validates):

- `mh.NoiseMatrix` (1D/2D markers), `mh.NoiseMatrixSM` (indexed
  Sherman-Morrison ecorr) — graph leaves and `make_solve` graphs.
- `mh.WoodburyKernel`, `mh.GlobalWoodburyKernel`, `mh.VectorWoodburyKernel`
  with `make_kernelproduct`, `make_conditional`, `make_kernelsolve` —
  all as graphs.
- `mh.CompoundGP` — Phi/F composition.
- `make_sample` — plain callable (the documented exception).

What is missing and must follow this architecture:

- `mh.VectorWoodburyKernel.make_kernelproduct_gpcomponent` — the
  `ArrayLikelihood.clogL` path, including `transform` (reparams), `additives`,
  and `extsignals`. Must be written as a graph, not a closure. This is the
  decentering branch's centerpiece and is the immediate next deliverable.
- A graph rewrite of `GlobalLikelihood.conditional` (currently uses
  `psl.N.make_kernelsolve` plus a closure over `Pinv + block_diag(FtNmF)`).
  Today it works via the `make_kernelsolve` wrapper, but it's a closure on
  top of a graph rather than one composite graph; eventually subsume it.
- A graph version of the CG-MDL / Lanczos logdet path (`cglogL`). This is
  the largest open design question — Lanczos is an iterative algorithm and
  doesn't fit a static graph; needs thought.

When in doubt while adding to metamath: **write the math symbolically, let
folding handle the rest.** If you find yourself writing `if callable(...)`
or `_var` / `_novar` branches, you're translating matrix.py instead of
rewriting it.

---

# Project status and handoff (as of this checkpoint)

## Goal recap

Delete `matrix.py`. Replace it with graph-based equivalents in `metamath.py`
(kernel/GP classes) and `metamatrix.py` (the graph DSL). Use `matrix.py` as
a numerical oracle during the transition: every method the production
likelihood layer (`PulsarLikelihood`, `GlobalLikelihood`, `ArrayLikelihood`)
calls must produce identical results via the metamath replacement. When
the parity suite under `tests/metamatrix/` covers every path real users
exercise, matrix.py gets deleted and the monkeypatch infrastructure goes
with it.

## What this checkpoint contains

### `tests/metamatrix/` — parity test scaffold

- `conftest.py` — `psr` (B1855+09) and `psrs` (3 pulsars) session fixtures.
- `_patch.py` — `metamatrix_patch()` context manager that swaps
  `matrix.NoiseMatrix*`, `matrix.WoodburyKernel`, `matrix.VectorWoodburyKernel_varP`,
  `matrix.CompoundGP`, etc. for their metamath equivalents. Models built
  inside the context capture mh objects; outside, they capture matrix.py
  objects. The key list IS the spec — every entry is a matrix.py symbol
  whose metamath replacement must reach parity before matrix.py deletion.
- `_comparison.py` — scale-aware `assert_close(kind=…)` helper (logL,
  residuals, coeffs, matrix; each picks tolerance from the scale of the
  reference value).
- `test_pulsar.py` — Tier 1: single-pulsar `PulsarLikelihood`.
  17 rows across `logL`, `conditional`, `clogL`, `sample`,
  `sample_conditional` × 9 model topologies (measurement only, ecorr-GP,
  ecorr-SM, timing, full RN, concat T/F, multi-vgp, variable timing).
- `test_global.py` — Tier 2: `GlobalLikelihood` with HD / monopole ORF.
  7 rows: `logL`, `conditional`, `sample`.
- `test_array.py` — Tier 3: `ArrayLikelihood` with commongp / globalgp.
  12 rows across `logL`, `conditional`, `clogL`, plus the new
  decenter/means/extsignal rows.

**Total: 35 passing, 1 xfailed.** The xfail is the only remaining gap
documented below.

### `metamath.py` — kernel/GP classes added during this work

- `NoiseMatrix` — single class replaces matrix.py's `NoiseMatrix1D_novar`,
  `NoiseMatrix1D_var`, `NoiseMatrix2D_var`, `VectorNoiseMatrix1D_var`
  (collapsed via the graph).
- `NoiseMatrix1D` / `NoiseMatrix2D` / `NoiseMatrix12D` — marker subclasses
  preserving the 1D-vs-2D type discrimination that `likelihood.py:468`
  uses for ndim dispatch. Same implementation as `NoiseMatrix`.
- `NoiseMatrixSM` — Sherman-Morrison indexed solve for ecorr-as-noise
  (`matrix.NoiseMatrixSM_var` replacement). Single graph node wrapping
  `matrix.SM_1d_indexed` / `SM_2d_indexed`, dispatching on `y.ndim`.
- `WoodburyKernel` — replaces all `WoodburyKernel_*` variants. Has
  `make_solve`, `make_kernelproduct`, `make_conditional`,
  `make_coefficientproduct`, `make_kernelsolve` (added during Tier 2),
  `make_sample`.
- `GlobalWoodburyKernel` — multi-pulsar global GP kernelproduct.
- `VectorWoodburyKernel` — vectorized per-pulsar kernel. Has
  `make_solve`, `make_kernelproduct`, `make_conditional`,
  `make_kernelproduct_gpcomponent` (added during this checkpoint:
  decenter/means/extsignals support; matches matrix.py b1bda23 signature).
- `CompoundGP` — Phi/F composition; `.index` produces list-of-dicts in
  vector mode and cumulative-offset flat dict in non-vector mode.

### `matrix.py` — decentering features ported in

- `matrix.ExtSignal` class (declarative; no math).
- `matrix.VectorWoodburyKernel_varP.make_kernelproduct_gpcomponent`
  rewritten to b1bda23 form: `transform` (list of reparams),
  `extsignals` (list of ExtSignal), and `self.means` (deterministic
  prior-center). The old single-`transform` signature is gone.
- This is the **oracle** for parity tests; the metamath equivalent on
  `mh.VectorWoodburyKernel` mirrors the same math.

### `signals.py` / `deterministic.py` additions

- `signals.make_extsignal_fourier(psrs, coefffunc, components, T, common, name)`
  — factory returning `matrix.ExtSignal`. Inspects `coefffunc`'s argspec
  to classify args as pulsar-attribute / common / per-pulsar; vmaps over
  pulsars. Independent of matrix-vs-mh path.
- `deterministic.makecw_extsignal` — CW factory wrapping
  `make_extsignal_fourier` with `makefourier_binary` as the coefficient
  map.

### `likelihood.py` changes

- `ArrayLikelihood.__init__` gains `decenter=False` and `extsignals=None`.
- `ArrayLikelihood.clogL` builds a `decenter_transform(params, c) -> (c, ldL)`
  reparam closure when `decenter=True`. Bridges matrix.py kernels (which
  expose `N.solve_2d(F)`) and metamath kernels (which expose
  `N.make_solve` as a graph) via `metamatrix.func(N.make_solve)(F, params={})`.
- Same `_eval_F` materialization for `self.vsm.Fs` items that may be
  metamath concat graphs (from `mh.CompoundGP.F`).
- `clogL` passes the combined `reparams + extsignals` to
  `make_kernelproduct_gpcomponent`.

## What's outstanding

### Immediate: the one xfail

`test_clogL_new_features[decenter+common_rn+global_hd]` — needs
`mh.CompoundGP` to handle the **mixed-Phi case** (commongp + globalgp).
matrix.VectorCompoundGP at this branch builds `multigp.prior = priorfunc`
(a callable) and sets `multigp.Phi = None`, side-stepping the
shape-incompatible Phi concat. mh.CompoundGP only does the concat path
and dies on shape mismatch ((3, 60) commongp vs (84, 84) globalgp Phi.N).

Concrete next step: in `mh.CompoundGP.__init__` or `.Phi`, detect when
`gp.Phi.N` shapes differ across gps and instead build a `prior` callable
following `matrix.VectorCompoundGP` lines 260-289. Then the existing
`hasattr(commongp, 'prior')` branch in `make_kernelproduct_gpcomponent`
takes care of the rest.

### Tier 3 still missing

- **`cglogL` (CG-MDL / Lanczos logdet)** in `ArrayLikelihood`. Entirely
  matrix.py-side, no metamath equivalent. This is the biggest remaining
  Tier-3 deliverable. Open design question: Lanczos is iterative and
  doesn't fit the static graph DSL cleanly. May need a callable carve-out
  similar to `make_sample`, or a different graph construct.

### Path-toward-deletion items

Per the staged migration plan above, after the test suite is exhaustive:

1. Rewrite `signals.py` constructors to return metamath objects directly
   (drop `matrix.NoiseMatrix1D_var(...)` in favor of
   `mh.NoiseMatrix1D(...)`). After this, the monkeypatch becomes a no-op.
2. Rewrite `likelihood.py` to import only metamath (`from . import matrix`
   removed). The `isinstance(x, matrix.NoiseMatrix1D_var)` ndim dispatch
   becomes `isinstance(x, mh.NoiseMatrix1D)`.
3. Delete `matrix.py` and `tests/metamatrix/_patch.py`.
4. Optionally rename `metamath.py` → `matrix.py` (it IS the matrix
   subsystem at that point).

These are mostly mechanical once the parity suite is complete.

## What still needs to be sanity-checked

Things I am reasonably confident about but haven't independently
verified beyond "the parity test passes":

- **`mh.NoiseMatrix.make_sample` for 2D Phi.** Tests cover 1D-diagonal
  noise sampling; the 2D-cholesky branch is implemented but only
  exercised through the GlobalLikelihood prior-draw path. Worth a
  targeted test for 2D `make_sample` standalone.
- **`mh.VectorWoodburyKernel.make_kernelproduct_gpcomponent`'s
  `self.prior` branch.** The `else` branch (`P_var_inv` path) is
  exercised by all current tests. The `if hasattr(self, 'prior')` branch
  is reached only when `mh.CompoundGP.prior` is set — which currently
  only happens via the (not yet implemented) mixed-Phi fallback. Once
  that lands, the `prior` branch should be parity-tested too.
- **Per-element vs full-matrix Pm shape under metamath.** The
  metamatrix `matrix_inv` produces 3D batched diagonals (npsr, k, k)
  where matrix.py's `VectorNoiseMatrix1D_var.make_inv` produces 2D
  (npsr, k). I handle both via `Pm.ndim` dispatch in
  `make_kernelproduct_gpcomponent`. The numerical results agree to
  rtol=1e-10 in tests, but the 3D path does more arithmetic than
  necessary — a small efficiency item, not a correctness one.
- **The `getN` alias on `mh.NoiseMatrix`.** Added so `signals.py:134`
  (`egp.Phi.getN`) keeps working under the patch. This is bridging code,
  expected to die when `signals.py` switches to metamath construction
  directly (migration step 1 above).
- **The `regularize_FtNmF` symmetrization** in the matrix.py
  `make_kernelproduct_gpcomponent`. I didn't carry this into the mh
  version. matrix.py defaults this to False for double precision, so
  the parity tests don't exercise the True branch. If anyone runs
  single-precision or sets `regularize_FtNmF=True`, the mh path may
  diverge slightly. Low risk for now but worth flagging.
- **The `staged` return shape (`(logp, c)` vs `logp`).** Both code paths
  match. Worth confirming downstream consumers handle the tuple form
  (the rest of the codebase plus any sampler integration). Search:
  `grep -rn 'clogL(' src/` after this branch settles.

## Where to start next session

1. Read this section and the parent architecture doc.
2. Run `python -m pytest tests/metamatrix/ --no-cov` to confirm the
   35-pass / 1-xfail baseline.
3. The natural next chunk of work is the `mh.CompoundGP` mixed-Phi
   prior fallback — it closes the last xfail and is the kind of
   targeted improvement (a single method, clear contract, existing
   parity oracle) that this branch is set up for.
4. After that, the `cglogL` / Lanczos question is the last Tier-3 item
   between here and "matrix.py can be deleted."

The architecture doc above explains the design discipline ("collapse
const/var into one graph, don't translate matrix.py"). The parity
suite under `tests/metamatrix/` is the safety net — every addition
should either flip an existing xfail to pass or add a new row covering
a topology not yet tested.

