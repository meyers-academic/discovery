# Single-precision-safe calculations — design notes

Working notes for making Discovery's likelihood evaluation safe and efficient in
single precision. Target use cases: **large multi-pulsar analyses** and
**variable-white-noise single-pulsar runs with rich noise models** — i.e. the
regimes where the per-call linear algebra is large and float32 storage/throughput
actually pays. (For a single pulsar with *fixed* WN the per-call work is tiny and
this buys little — see "Where it pays".)

There are two distinct pieces, addressing two distinct failure modes.

## Piece 1 — scale throttle on PSD / prior functions

**Failure mode it addresses:** overflow/NaN *during* PSD evaluation. `powerlaw`
computes `f^(-γ)`; at `f ~ 1e-8 Hz`, `γ ~ 5` that's `~1e40`, which overflows
float32 (max ~3.4e38) → `inf` → `nan`. This happens *inside* the PSD, before any
GP object sees it, so a downstream clamp cannot recover it (`inf`→max is lossy,
`nan` stays `nan`).

**Design: a decorator**, so we don't rewrite every signal in `signals.py`. It
evaluates the PSD in float64 (which has the headroom — `1e40` is nothing for
float64), clamps the result, then casts to the working precision:

```python
def throttle(low_clip=-18.0, high_clip=-9.0, out_dtype=None):
    """Single-precision-safe PSD/prior wrapper.
    Evaluates in float64 (no f^-gamma overflow), clamps to
    [10**low_clip, 10**high_clip] * scale**2  (log10 of the variance in the
    function's native units, s^2 for powerlaw/freespectrum), casts to out_dtype.
    Requires x64 enabled."""
    def decorator(psd):
        @functools.wraps(psd)                 # __wrapped__ -> getfullargspec unwraps
        def wrapped(f, df, *args, scale=1.0, **kw):
            f64, df64 = jnp.asarray(f, jnp.float64), jnp.asarray(df, jnp.float64)
            phi = psd(f64, df64, *args, scale=scale, **kw)
            lo = 10.0**low_clip  * scale**2
            hi = 10.0**high_clip * scale**2
            return jnp.clip(phi, lo, hi).astype(out_dtype or _working_dtype())
        wrapped.__signature__ = inspect.signature(psd)
        if hasattr(psd, "type"): wrapped.type = psd.type     # 2D-dispatch marker
        return wrapped
    return decorator
```

Key points:

- **Requires x64 enabled.** The NaN-safety comes from the float64 evaluation; if
  x64 is off, `float64` silently degrades to float32 and overflow returns. So
  this decorator is valid only in the **mixed-precision** model (x64 on globally,
  downcast selectively). A *pure* all-float32 run cannot be fixed by any
  output-wrapping decorator — it would require each PSD body to compute in
  log-space and clip before `exp` (the form Patrick prototyped, below). The
  decorator is the chosen route precisely to avoid rewriting every PSD.
- **Introspection transparency.** `signals.py` uses `inspect.getfullargspec`,
  `typing.Sequence` annotations, `.type`, and positional calls. `functools.wraps`
  (via `__wrapped__`) preserves the argspec; `.type` is copied. **Gotcha:**
  `scale` is in `powerlaw`'s arg list, so `makegp_*` will mint a spurious
  `{psr}_{name}_scale` sampled parameter unless `scale` is in the builder's
  `exclude` set (or stripped from the advertised signature).
- **What the clamp protects.** We never form `F Φ Fᵀ` (ntoa×ntoa); the per-call
  Woodbury core is `C(θ) = diag(1/Φ) + FtNmF`. So the **floor** (`low_clip`) is
  load-bearing: it bounds `1/Φ` and stops `C` blowing up when `Φ` underflows. The
  basis scale lives in `FtNmF` (double), not in `Φ`.
- **Per-use configurable clips, for free.** Because `throttle(...)` is a
  parameterized factory, different bounds = decorate per use
  (`powerlaw_dm = throttle(low_clip=-20, high_clip=-9)(powerlaw)`), matching the
  `make_*` idiom. Bounds bound at decoration time stay out of the sampled
  signature, sidestepping the `scale`/`exclude` gotcha. Threading bounds through
  the `makegp_*` call site instead is possible but touches the builders.

### Units (resolved)

With the **standard** `dmfourierbasis`, the chromatic factor is `(f_ref/ν)²` — a
dimensionless frequency *ratio* — so `F_dm` is unitless and `Φ` from `powerlaw`
stays in **s²**, same as red noise. So **one s² clip covers both RN and DM**; no
per-signal units problem. `powerlaw` and `freespectrum` both output s².

The **only** exception is `tndm=True` (TempoNest convention), where the basis
folds in physical constants and the amplitude is reinterpreted in DM units
(pc cm⁻³) — there `Φ` is in DM-amplitude², so that GP wants different clip bounds.
This is the motivating case for per-use configurable clips.

### Reference: Patrick's prototype `powerlaw` (log-space clip, for the
all-float32 alternative)

```python
def powerlaw(f, df, log10_A, gamma, scale=1):
    log10_A = log10_A + jnp.log10(scale)
    log10_main = (2.0*log10_A - gamma*jnp.log10(f)
                  + (gamma - 3.0)*jnp.log10(ds.const.fyr)
                  - jnp.log10(12.0) - 2.0*jnp.log10(jnp.pi)) + jnp.log10(df)
    log10_main = jnp.clip(log10_main, low_clip + 2*jnp.log10(scale),
                                      high_clip + 2*jnp.log10(scale))
    return 10.0**log10_main
```

This is the in-body version (clip in log10 before `10**`); the decorator above
generalises it without per-signal rewrites, given x64-on.

## Piece 2 — reference + delta split of the likelihood

**Failure mode it addresses:** the absolute log-likelihood is dominated by large,
θ-independent numbers (`yᵀN⁻¹y`, `logdet N` — `logL ~ 1e6`), whose float32 ulp
(~0.06 at 1e6) swamps the `O(1)` variations we actually sample. The fix is to
compute relative to a user-supplied **reference point**: keep the big static
pieces in double, compute the small *changing* piece in single.

Woodbury decomposition of a single-pulsar GP likelihood:

```
C(θ)  = diag(1/Φ(θ)) + FtNmF                       # ngp×ngp  (ngp = 2*sum components)
logL  = -½[ ytNmy − FtNmyᵀ C⁻¹ FtNmy + logdetN + logdetΦ(θ) + logdetC(θ) + n·log2π ]
```

- **Static (fixed WN + basis):** `FtNmF`, `FtNmy`, `ytNmy`, `logdetN` — precompute
  once in double. **`F Φ Fᵀ` is never formed**; Woodbury keeps per-call work in
  the small ngp-space.
- **Per call:** `Φ(θ)`, `C(θ)`, one ngp×ngp solve + logdet.
- In `ΔlogL = logL(θ) − logL(θ_ref)`, the big static `ytNmy` and `logdetN`
  **cancel exactly** (θ-independent), so the varying computation never carries the
  huge baseline. For sampling only `ΔlogL` matters anyway.
- **Must compute the delta as a direct (low-rank) update, never as the difference
  of two big float32 numbers** — that subtraction is the catastrophic
  cancellation we're avoiding.

### Where it pays / where it's clean

- **WN fixed:** `FtNmF/FtNmy/ytNmy` static → double once; per-call ngp solve is
  trivial → single precision buys little *compute* for one pulsar. The
  reference→delta split is **clean** because the only variation is in `Φ`, which
  is low-rank (ngp).
- **WN sampled (the main case):** `N(θ)` changes → `FtNmF(θ)` etc. are per-call
  and **ntoa-sized** → single precision genuinely helps. **But** the variation is
  now in `N` (full ntoa diagonal), not a low-rank `Φ`, so `ΔΣ` is not rank-ngp and
  the reference→delta subtraction loses its clean low-rank form. **Open problem:**
  define the reference as a fixed reference WN point and analyse the (non-low-rank)
  delta's error — this is where the elegance does or doesn't survive.

## Connection to metamatrix

metamath's graph **folding already partitions** trace-time constants (the double
precompute set) from live nodes (the single delta). So the static/dynamic split
is largely the structure the DSL already produces; the precision typing (double
for folded constants, single for live nodes) + the clamp layer onto the
const→live seam rather than being bolted on. The throttle decorator sits at the
PSD boundary (signals layer), independent of kernel path.

## Open decisions to pin (next session)

1. **Confirm x64-on mixed precision is the target** (validates the f64-eval
   decorator; an all-float32 target would force log-space PSD bodies instead).
2. **`scale` / `exclude` handling** so config args never leak as sampled params.
3. **Reference-parameter scheme for the WN-sampled case** — the non-low-rank
   delta is the open analysis.
4. Whether to normalise all GP bases to O(1) and carry scales analytically
   (uniform clamp), or supply per-GP scale from the basis at build time. (For
   standard RN+DM this isn't needed; revisit if a signal's output units differ.)
