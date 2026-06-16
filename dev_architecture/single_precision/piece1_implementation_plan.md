# Piece 1 — implementation & test plan (PSD/prior throttle)

Implementation plan for **Piece 1** of the single-precision work: the
scale-throttle decorator on PSD / prior functions. Companion to `README.md`
(design) and `repo_context_checkpoint.md` (codebase background) in this folder.
Read those first; this doc is the buildable/testable breakdown, plus the
corrections to the README prototype found while reading the code.

> **Route changed (session 2): log-space PSD bodies + build-time `scale`,
> not the decorator.** Once we decided `scale` is worth carrying (centering Φ to
> O(1) for f32 conditioning — README open-decision #4), we have to edit the PSD
> bodies anyway, so we fold the clip *into* the body in log-space (Patrick's
> prototype). This also buys **regime 1** (pure-f32 / x64-off, JAX's GPU default)
> correctness, which the decorator gave up. The `throttle(...)` decorator is
> demoted to an optional wrapper for *user-supplied* PSDs we don't own. §0/§1
> below are superseded by §0b and §8 (migration). Kept for the rationale trail.

## 0b. Locked decisions (session 2 — current)

1. **Log-space bodies.** Rewrite the power-law-family PSDs to compute
   `log10 Φ`, clip in log-space to `[low_clip, high_clip]`, then `10**`. No
   intermediate `f^-γ` overflow → correct in *both* precision regimes. The
   free-spectrum family (`10**(2ρ)`) is already overflow-safe; it gets the floor
   + scale only.

2. **Closure/factory architecture for config (`scale`, clips, fixed `gamma`).**
   Each PSD becomes a factory whose config lives as **closure constants** —
   they never appear in the returned function's signature, so they can't leak
   into sampled params and **no exclusion register changes** anywhere:

   ```python
   def make_powerlaw(*, gamma=None, scale=1.0, low_clip=-18.0, high_clip=-9.0):
       if gamma is None:
           def powerlaw(f, df, log10_A, gamma):   # gamma sampled (default)
               ...                                 # log-space body, clip, scale
       else:
           def powerlaw(f, df, log10_A):          # gamma FIXED (closure const)
               ...
       return powerlaw

   powerlaw = make_powerlaw()   # default ds.powerlaw — same sig & numerics as today
   ```

   This is the codebase's existing idiom (`makepowerlaw_crn`,
   `make_dmfourierbasis`). Three wins in one mechanism:
   - **No `partial`** to set scale/clips: `ds.make_powerlaw(scale=1e-6,
     high_clip=-6.0)`.
   - **Fixed-`gamma` power law** (long-requested; users dislike `partial`):
     `ds.make_powerlaw(gamma=4.33)` drops `gamma` from the signature → builders
     introspect only `log10_A` → no spurious sampled `gamma`.
   - **Old call sites unaffected** — `ds.powerlaw = make_powerlaw()` keeps the
     `(f, df, log10_A, gamma)` signature and (clip-inert) numerics. Existing
     models/recipes/tests pass `ds.powerlaw` unchanged.

   `scale` enters as `+2·log10(scale)` in log-space (multiplies Φ by `scale²`,
   s²); clips are per-factory-tunable (the `tndm` DM-unit case).

3. **Exclusion architecture: NOT touched** (user decision). The closure approach
   means scale/clip/fixed-gamma never enter signatures, so the existing
   `exclude=['f','df']` handling needs no change and old code keeps working
   quietly. The declarative-decorator cleanup of the `f`/`df` debt (former §9)
   is **dropped from this work**.

4. **Clips stay `[1e-18, 1e-9]`** (§0.2 below) — physical, `low` fixed.

## 0 (superseded). Session-1 decisions

1. **Route: decorator only.** Target the **mixed-precision** regime — x64 on
   globally, big linear algebra downcast to float32 (the intended GPU-speed
   configuration; README "regime 2"). The decorator evaluates Φ in float64,
   clamps, and casts to the working dtype. **Pure-float32 / x64-off** (JAX's
   default on GPU; README "regime 1") is *out of scope*: there an intermediate
   `f^-γ` overflow is clamped to a finite-but-wrong value, and correctness would
   need the log-space PSD body. We **assert x64-on** where the throttle's
   float64 guarantee is load-bearing, and document the limit rather than fix it.

2. **Clip bounds are physical, not just hazard bounds.** Detectable timing
   signals run ~O(10 ns) to O(10s of ms) RMS, so the *reachable* per-bin Φ sits
   within roughly `[1e-18, 1e-9]` s².
   - `low_clip = -18` — **fixed** (per Patrick). Doubles as the regime-2 floor:
     it caps `1/Φ ≤ 1e18` in the Woodbury core `C = diag(1/Φ) + FtNmF`, far under
     the float32 overflow at `3.4e38`. Any basis variance below `(1 ns)²` is
     below detectability, so flooring it is physical regularization, not
     corruption.
   - `high_clip = -9` — **default, configurable**. Can be bumped to `-8` or `-6`
     for long / steep red-noise processes whose lowest-frequency bin carries
     more power (see Phase C2 — measure headroom before shipping).

3. **API: throttle the shipped `powerlaw` / `freespectrum` in place**, with the
   inert clips above, *and* expose the parameterized `throttle(...)` factory for
   per-use custom bounds (the `tndm` DM-unit case, future signals). Default-on is
   ergonomic and, with inert clips, leaves the float64 parity suite unchanged.
   *(Confirm — the only API choice still open; the README's "decorate per use"
   idiom is preserved by keeping the factory public.)*

4. **Location: PSD boundary in the signals layer**, upstream of the metamath
   graph. Piece 1 is independent of the graph backward-pass precision work
   (Piece 2 / "Precision in the metamath graph"); it only needs to know the
   working dtype to cast to.

## 1. Corrections to the README prototype (found in code)

- **The README throttle has a real bug against the current PSDs.** It does
  `psd(f64, df64, *args, scale=scale, **kw)`, but `signals.powerlaw(f, df,
  log10_A, gamma)` and `freespectrum(f, df, log10_rho)` have **no `scale`
  parameter** → `TypeError: unexpected keyword argument 'scale'`. Verified.
  → **Drop `scale` forwarding.** Bounds are decoration-time constants; `scale`
  is not in the wrapped signature and nothing is forwarded. This *also* removes
  the `scale`/`exclude` "spurious sampled param" gotcha entirely — there is no
  `scale` to leak.

- **Introspection is preserved by `functools.wraps` + `__signature__`.**
  Verified that `inspect.getfullargspec(throttled)` returns the raw arg list and
  preserves the `typing.Sequence` annotation on `freespectrum.log10_rho` (so
  vector-param expansion in `makegp_fourier` still works), and that `.type` is
  carried for 2D-dispatch priors (`None` for `powerlaw`/`freespectrum`, so they
  stay `NoiseMatrix1D`). `getfullargspec` ignores `__wrapped__` but honors
  `__signature__`, which we set explicitly.

- **`f`/`df` reach the PSD as `np.float64`** (`signals.py:221,224`; the
  `jnparray` cast at `signals.py:278` is commented out). So with x64 *off* they
  truncate to float32 inside the first JAX op → `f^-γ` overflow. The throttle's
  `jnp.asarray(f, jnp.float64)` restores float64 evaluation — but only if x64 is
  on (hence the assertion).

- **README nominal bounds `[-18, -9]` stand** (physical, per §0.2). Earlier
  worry that they clip realistic Φ came from scanning unreachable parameter
  combinations; corrected.

## 2. The throttle (target implementation)

```python
# signals.py (near powerlaw/freespectrum)
def throttle(low_clip=-18.0, high_clip=-9.0, out_dtype=None):
    """Single-precision-safe PSD/prior wrapper (mixed-precision regime).

    Evaluates the PSD in float64 (no f^-gamma overflow), clamps the result to
    [10**low_clip, 10**high_clip] s^2, then casts to the working dtype. Requires
    x64 enabled. Bounds are decoration-time constants; no runtime `scale`.
    """
    def decorator(psd):
        @functools.wraps(psd)
        def wrapped(f, df, *args, **kw):
            f64  = jnp.asarray(f,  jnp.float64)
            df64 = jnp.asarray(df, jnp.float64)
            phi  = psd(f64, df64, *args, **kw)
            phi  = jnp.clip(phi, 10.0**low_clip, 10.0**high_clip)
            return phi.astype(out_dtype if out_dtype is not None
                              else utils.working_dtype())
        wrapped.__signature__ = inspect.signature(psd)
        if hasattr(psd, "type"):
            wrapped.type = psd.type
        return wrapped
    return decorator
```

Applied as `powerlaw = throttle()(_powerlaw)` etc. (raw bodies kept private as
`_powerlaw`/`_freespectrum` for the parity oracle and unit tests).

vmap-safe: `makecommongp_fourier` wraps the prior in `jax.vmap`; `asarray`,
`clip`, `astype` are all vmappable.

## 3. Shared infra — minimal mixed-precision config knob

The throttle needs to know what to cast to. Land the **Piece-1 slice** of the
README "Mixed-precision config" section (full `jnparray` downcast-helper polish
stays with Piece 2):

- In `utils.config()`: add a `working` dtype selection (default `float64`).
  - `working_dtype = jnp.float64` by default; `jnp.float32` in single mode.
  - Derive `single_precision = (working_dtype == jnp.float32)` **instead of**
    `not jax.config.x64_enabled`.
  - **Assert `jax.config.x64_enabled` whenever `working_dtype is float32`** —
    otherwise the float64 guarantee is vacuous.
  - `jnparray` default **unchanged** (still float64 — no behavior change).
- Add accessor `utils.working_dtype()` (and a `to_working(a)` helper for later
  use). `numpy` backend → `float64`.

Files touched: `utils.py` only.

## 4. Phases

Each phase is independently landable and testable. Test data = existing feather
fixtures in `tests/data` (B1855+09, J0023+0923, J0030+0451). Models = the
`recipes/` zoo (single source of truth for the parity suite). New tests live in
`tests/single_precision/` and reuse the `tests/metamatrix` `build_routes`
harness where a model is involved, so **both kernel backends** are exercised.

### Phase A — config knob + accessor
Implement §3.
**Tests** (`test_config.py`):
- A1 default `working_dtype()` is float64; `single_precision` False.
- A2 selecting float32 with x64 on → `working_dtype()` float32, `single_precision`
  True.
- A3 selecting float32 with x64 off → raises/asserts.
- A4 numpy backend → float64.

### Phase B — decorator, unit level (no model)
Implement §2. Pure-function tests against `_powerlaw`/`_freespectrum`.
**Tests** (`test_throttle_unit.py`):
- B1 **introspection**: `getfullargspec(powerlaw).args == ['f','df','log10_A',
  'gamma']`; `freespectrum` keeps `{'log10_rho': typing.Sequence}`; `.type`
  matches raw; no extra params.
- B2 **inert in float64** (working=float64): over a grid of *physically typical*
  params (e.g. `log10_A ∈ [-15,-13]`, `γ ∈ [0,7]`, a real 30-component basis),
  `throttled(f,df,θ) == raw(f,df,θ)` to ~1e-12 (clip never activates).
- B3 **clamp activates gracefully** at pathological θ (huge/tiny `log10_A`):
  output ∈ `[1e-18, 1e-9]`, finite, right shape.
- B4 **regime-2 microtest**: take a Φ that underflows toward 0; without the
  floor `1/Φ → inf`; with the floor `1/Φ ≤ 1e18`, finite. (Directly models the
  `diag(1/Φ)` term of the Woodbury core.)
- B5 **regime-1 limit (documentation test)**: with x64 *off*, raw `powerlaw`
  at low f / high γ → `inf`/`nan`; throttle returns finite-but-not-trustworthy.
  Asserts the documented behavior and justifies the x64-on assertion. Marked
  `xfail`/informational, not a correctness claim.

### Phase C — model parity, double precision, CPU
x64 on, working=float64 (default). Reuse `build_routes(factory)`.
Recipes: `full_rn`, `multi_vgp` (RN + DM GP), plus a freespectrum model
(`makegp_fourier(psr, ds.freespectrum, ...)`).
**Tests** (`test_throttle_parity.py`):
- C1 **logL parity**: throttled vs raw-PSD model `logL` agree to ~1e-8 over
  sampled params, for both `mh_native` and `mh_patched` routes. (Clip inert at
  realistic params ⇒ throttle is a no-op in float64.)
- C2 **headroom calibration**: across realistic priors — including a GWB-like
  point (`log10_A≈-14.5, γ≈3.2`) and the strongest reachable single-pulsar RN —
  record `max(Φ)` and its margin to the `1e-9` ceiling, and assert clip
  activation only touches sub-detectability bins. **Deliverable: a printed
  headroom report** that tells us whether to bump the ceiling to `-8`/`-6`
  before shipping.

### Phase D — single precision (mixed), CPU
x64 on, working=float32.
**Tests** (`test_throttle_single.py`):
- D1 **finiteness**: model `logL` finite (no nan) under float32 working dtype,
  including at prior edges where a raw float32 PSD would nan.
- D2 **accuracy**: `|logL_mixed − logL_f64|` within a float32-scale tolerance.
  *Caveat to document:* without Piece 2, the float32 Φ is re-promoted to float64
  the moment it meets the float64 `FtNmF` (JAX type promotion), so D2 chiefly
  validates the floor + cast, **not** an end-to-end float32 speedup. Piece 1 is
  NaN-safety, not yet speed.
- D3 **gradients**: `jax.grad(logL)` finite under float32 (sampler safety;
  `jnp.clip` subgradient is 0 at the bound, never nan).

### Phase E — GPU
Device fixture + `pytest.mark.gpu`, `skipif not jax.devices('gpu')`. Use
`jax.default_device(jax.devices('gpu')[0])` context.
**Tests** (`test_throttle_gpu.py`):
- E1 **mixed-mode on GPU** (x64 on, working float32) — the target deployment:
  `logL` finite and within tolerance of CPU float64. Run on `full_rn` and a
  multi-pulsar `ArrayLikelihood` recipe (where the float32 path actually pays).
- E2 **regime-1 demonstration on GPU default** (x64 off): raw model `logL` →
  nan; throttled → finite (documents the limit; not a correctness claim).
- E3 **perf smoke** (informational, not asserted): time a large multi-pulsar
  `logL` in float32 vs float64 on GPU; print the ratio.

### Phase F — wire-up & docs
- Apply the throttle to shipped `powerlaw`/`freespectrum` (§0.3).
- Update `README.md`: correct the `scale`-forwarding prototype, the bounds
  rationale, mark Piece 1 landed; cross-link this doc.
- Record follow-ups (not Piece 1): wrap `brokenpowerlaw`, `make_combined_crn`,
  `makepowerlaw_crn`; `tndm=True` DM-unit clip bounds via the factory; Piece 2
  (reference+delta) and the graph backward-pass that makes float32 actually
  fast end-to-end.

## 8. Migration (session-2 route: log-space bodies + scale)

Test-first, per function. Phases A–F above are adapted: the "decorator" is now
the log-space body; the config knob (Phase A) and the model-level parity /
single-precision / GPU tests (Phases C–E) carry over unchanged.

**Functions to migrate** (signals.py) — each becomes a `make_*` factory whose
default no-arg call reproduces today's public name (`powerlaw = make_powerlaw()`):
- power-law family (log-space rewrite, clip-in-log, `+2·log10(scale)`,
  optional fixed `gamma`): `make_powerlaw`, `make_brokenpowerlaw`,
  `make_powerlaw_brokencrn`, `make_brokenpowerlaw_brokencrn`, and the existing
  `makepowerlaw_crn` (×2 backend bodies; already a factory — extend it, keep the
  `crn_gamma='variable'` API working).
- free-spectrum family (floor + scale only; no overflow, no log rewrite):
  `make_freespectrum`, `makefreespectrum_crn` (×2).
- `make_combined_crn` — **no body change**: inherits safety from its argument
  PSDs. Verify its `exec`-built `combined` still introspects clean.

**Numerical care:**
- The broken fold `(1+(f/fb)^(1/κ))^(κγ)` with `κ=0.1` → `(f/fb)^10` overflows in
  f32. Log form: `κγ · logaddexp(0, (1/κ)(ln f − ln10·log10_fb))`. Use
  `jnp.logaddexp` / `jnp.log1p`, never the raw power.
- Log-space changes float64 rounding vs the current linear body → golden tests
  assert `allclose(rtol≈1e-10)`, **not** bit-exactness. (matrix-vs-metamath
  parity is unaffected: both backends share the same new body.)

**Step 1 — characterization tests (do first).** `tests/single_precision/
test_psd_characterization.py`. For each function above, an **independent
linear-space oracle** (a copy of the current formula) and assert the production
function matches it to `rtol≈1e-10` over a param grid chosen so every Φ stays
inside `(1e-18, 1e-9)` (clip inert). The test self-checks that grid bound. These
pass on current code now and must still pass after the rewrite.

**Step 2 — rewrite bodies as factories + add `scale`/clips/fixed-`gamma`.** Make
Step-1 tests pass (closure config defaults reproduce current numerics), then add
the *new-behavior* tests:
- `scale` semantics: `make_powerlaw(scale=s)(...) == s²·make_powerlaw()(...)`
  for the power-law family; the analogous log-shift for freespectrum.
- **fixed-`gamma`**: `make_powerlaw(gamma=g)` has signature `(f, df, log10_A)`,
  equals `make_powerlaw()(..., gamma=g)` numerically, and — built into a model —
  produces **no** `*_gamma` sampled param (`assert ...gamma not in
  model.logL.params`).
- clip activation at extremes (bounded, finite).
- no-leak guard: scale/clip never appear in `model.logL.params`.

**Exclusion registers: untouched** (per user). Closure config never enters a
signature, so the existing `exclude=['f','df']` handling and all old call sites
keep working unchanged. The no-leak / no-spurious-gamma guard tests prove it.

## 9. Parameter-exclusion architecture — DROPPED

The declarative-decorator cleanup of the positional `f`/`df` exclusion debt is
**not part of this work** (user decision): the closure/factory architecture
solves the scale/clip/fixed-gamma leak concern without touching any exclusion
register, and old code keeps working quietly. Recorded here only so a future
reader knows it was considered and deliberately deferred.

## 5. How to run

```bash
# CPU, double + single precision
pytest tests/single_precision -q
# just the parity (double precision) subset
pytest tests/single_precision/test_throttle_parity.py -q
# GPU (on a CUDA box)
pytest tests/single_precision/test_throttle_gpu.py -q -m gpu
```

x64 is set per-test-module (the suite toggles `working_dtype` via
`utils.config`); GPU tests select the device with a context manager so CPU
fixtures are unaffected.

## 6. Risks / open items

- **API default-on** (§0.3) — confirm before Phase F wires it in.
- **Ceiling headroom** — Phase C2 decides whether `-9` survives long/steep RN or
  needs `-8`/`-6`.
- **No float32 speedup yet** — type promotion re-floats Φ to f64 downstream
  until Piece 2; set expectations (Piece 1 = NaN-safety).
- **Pure-f32 unsupported** — asserted, documented (Phases B5/E2), not fixed.
```

## 7. Definition of done (Piece 1)

- Throttle + config knob landed; `powerlaw`/`freespectrum` throttled in place.
- Phases A–E green on CPU; E green on at least one GPU run (or skipped with a
  recorded reason).
- Headroom report (C2) reviewed; ceiling finalized.
- README updated; follow-ups filed.
