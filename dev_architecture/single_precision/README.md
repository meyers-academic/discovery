# Single-precision-safe calculations — design notes

Working notes for making Discovery's likelihood evaluation safe and efficient in
single precision. Target use cases: **large multi-pulsar analyses** and
**variable-white-noise single-pulsar runs with rich noise models** — i.e. the
regimes where the per-call linear algebra is large and float32 storage/throughput
actually pays. (For a single pulsar with *fixed* WN the per-call work is tiny and
this buys little — see "Where it pays".)

**Scope: metamath only.** These notes target the **metamath kernel path**
(`metamath.py` / `metamatrix.py`), not legacy `matrix.py`. This matters because
the two paths place precision differently: in `matrix.py` the const-vs-var split
is fixed by the chosen variant class, so a `jnparray()` cast at a known seam
works; in metamath, **graph folding decides const-vs-live at trace time**, so
there is no fixed seam and precision must be a rule applied at graph
materialization (see "Precision in the metamath graph"). The chosen precision
model is **mixed precision** (x64 on globally, downcast selectively) — see
"Mixed-precision config" — which also underpins Piece 2.

There are two distinct pieces, addressing two distinct failure modes.

## Piece 1 — scale throttle on PSD / prior functions

**Two distinct failure modes, two distinct regimes.** Both stem from `powerlaw`
computing `f^(-γ)` — at `f ~ 1e-8 Hz`, `γ ~ 5` that's `~1e40` — but *where* it
bites depends on the precision model, and the fix differs:

1. **Pure all-float32** (x64 off): `f^(-γ)` overflows *during* evaluation
   (`1e40 > 3.4e38` → `inf` → `nan`), inside the PSD before any GP sees it. A
   downstream clamp cannot recover it (`inf`→max is lossy, `nan` stays `nan`).
   The only fix is to compute the PSD body in **log-space and clip before
   `10**`** — Patrick's prototype, below. **No output-wrapping decorator can help
   here.**
2. **Mixed precision** (x64 on, downcast selectively): the PSD is evaluated in
   float64, where `1e40` is unremarkable, so **evaluation overflow cannot
   happen**. The hazard moves *downstream of the cast*: a tiny `Φ` underflowing in
   float32 makes `1/Φ` blow up to `inf` in the Woodbury core
   `C = diag(1/Φ) + FtNmF`. So here the load-bearing protection is the **floor**,
   not the ceiling — and a decorator suffices, because the danger is in the cast
   result, not the evaluation.

**These are not the same job, and the decorator only addresses regime 2.** The
overflow-during-evaluation framing belongs to regime 1; do not let it justify the
f64 decorator, which is a regime-2 tool. The decorator's real purpose is to put a
floor under `Φ` *before* it is cast to float32 and reciprocated.

**Design (regime 2): a decorator**, so we don't rewrite every signal in
`signals.py`. It evaluates the PSD in float64, clamps the result, then casts to
the working precision. Evaluating in f64 does **not** defeat the speedup: `Φ` is
an O(ngp) vector (~60–120 elements), while the cost this whole exercise targets is
the ntoa-sized / ngp×ngp linear algebra *downstream*. Computing a length-~100
vector in f64 and casting to f32 before it enters those matmuls is free even on a
GPU. The cast point is what matters, not the eval precision.

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

- **Requires x64 enabled, and that is itself a cost — not just a prerequisite.**
  The NaN-safety comes from the float64 evaluation; with x64 off, `float64`
  silently degrades to float32 and overflow returns. But "x64 on globally,
  downcast selectively" has a hazard the decorator does not remove: **any
  operation you forget to cast silently runs in float64**, losing the speedup with
  no error and no signal. The mixed model trades guaranteed NaN-safety for a
  silent-performance-leak risk that has to be policed by hand at every matmul. The
  pure-float32 + log-space-PSD route (regime 1) has no global flag to leak through
  and is regime-independent, at the cost of hand-writing each PSD body. This is the
  central route decision (see "Open decisions"), not a settled detail.
- **What the clamp actually protects (regime 2).** We never form `F Φ Fᵀ`
  (ntoa×ntoa); the per-call Woodbury core is `C(θ) = diag(1/Φ) + FtNmF`. So the
  **floor** (`low_clip`) is the load-bearing bound: it caps `1/Φ` and stops `C`
  blowing up when `Φ` underflows after the f32 cast. The ceiling (`high_clip`) is
  near-vestigial in this regime — f64 evaluation already precludes the overflow it
  guards against; it survives only as cheap insurance and to mirror the log-space
  body. The basis scale lives in `FtNmF` (double), not in `Φ`.
- **Introspection transparency.** `signals.py` uses `inspect.getfullargspec`,
  `typing.Sequence` annotations, `.type`, and positional calls. `functools.wraps`
  (via `__wrapped__`) preserves the argspec; `.type` is copied. **Gotcha:**
  `scale` is in `powerlaw`'s arg list, so `makegp_*` will mint a spurious
  `{psr}_{name}_scale` sampled parameter unless `scale` is in the builder's
  `exclude` set (or stripped from the advertised signature).
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

## Mixed-precision config (decided)

The current config equates single precision with x64-*off*
(`utils.py`: `single_precision = not jax.config.x64_enabled`; `jnparray` →
float32 only when x64 is off). That is the pure-float32 regime — incompatible with
the f64-eval decorator and with Piece 2's f64 baselines. We **decouple** them:

- **x64 stays on globally; default dtype stays float64.** Add a config knob
  `working_dtype` (the dtype we downcast *to*; float32 in single mode), and derive
  `single_precision = (working_dtype == float32)` instead of from `x64_enabled`.
  Assert x64 is on whenever `working_dtype` is float32 — otherwise the "keep some
  things f64" guarantee is vacuous.
- **`jnparray` default stays float64 (no behavior change); add an explicit
  downcast** — `jnparray(a, dtype='working')` or a `to_working(a)` helper — used
  deliberately where we want float32. Rationale: a precision *downcast* should
  always be a visible, greppable decision, never a default you can silently
  forget. The opposite default (jnparray→f32) flips the silent-failure direction
  from "accidentally slow" to "accidentally inaccurate," which is worse.
- The few **f64-must-stay** quantities are exactly Piece 2's static baselines
  (`ytNmy`, `logdetN`), so the override is load-bearing, not hypothetical.

This supersedes the `matrix.py`-flavored "cast at the seam" framing for the
metamath path — see next section: in metamath, dtype is assigned by a backward
pass from consumers, not at any fixed seam.

## Precision in the metamath graph

**Two independent axes — do not conflate them.** Folding answers *computed once
(constant) vs. recomputed every call (live)*. Precision answers *float64 vs.
working dtype (float32)*. A value can be any combination: a constant stored in f32
(`FtNmF` in a fixed-WN array), a live value in f64 (`ytNmy` under sampled WN),
etc. An earlier draft tied these together ("folded → f64, live → f32"); that was
wrong and is the source of the `FtNmF`-stored-f64 confusion. They are orthogonal.

**Precision is assigned by a backward pass from consumers, seeded by f64 pins —
not by the fold partition.** The rule, applied at materialization (`func()`):

1. **Working dtype (float32) is the default target** for the whole graph. We are
   in single precision to make things f32 *unless there is a reason not to*.
2. **f64 is injected by pins** — the quantities that need cancellation /
   conditioning safety: `ytNmy`, `logdetN`, the logdets. A pin marks an op (or one
   of its outputs) f64.
3. **f64 propagates backward from each pin** to the values that feed it. Anything
   on a path into a pin is materialized f64; everything else stays working dtype.
4. **Every value is born in the dtype the backward pass assigns it — constants
   included.** A folded constant whose consumers are all working-dtype is
   materialized **f32** (so it costs f32 memory and feeds the live calc directly,
   no per-call cast). A folded constant feeding an f64 pin is materialized **f64**.
   *Folding decides whether it's precomputed; the backward pass decides its dtype.*

This resolves the cases we worked through:

- **`FtNmF` list, fixed-WN array.** Each block folds to a *constant*, but its only
  consumer is the large live global Cholesky (working dtype). So each `FtNmF` is
  **born f32** — f32 memory, no per-call downcast — and the big factorization runs
  in f32, which is where it pays. (The f64 *storage* the earlier draft implied was
  pure waste here.)
- **`ytNmy`, same array, fixed WN.** Also a folded constant, but it's a pin (the
  catastrophic-cancellation term), so it's **born f64**. Same likelihood, two
  constants, two dtypes — each set by its consumer.
- **Mixed consumers** — a value feeding *both* a working-dtype live op and an f64
  pin — is the *only* case needing an f64 master + a downcast on the f32 edge.
  JAX type promotion forces this: without the explicit downcast, the f32 op
  re-promotes to f64 the moment it touches the f64 value.

**Separate lever — accumulation precision ≠ value dtype.** For a matmul that *is*
f32 (e.g. a live `FtNmF = FᵀN⁻¹F` under sampled WN), there's a further choice:
accumulate the ntoa-long sum in f32 (`precision=HIGHEST`, no TF32) or in f64
(`preferred_element_type=f64`, f32 operands / f64 running-sum). Set at
materialization via `dot_general`. Per Patrick: per-pulsar `FtNmF` has never shown
conditioning trouble (it's small, built per-pulsar, applied batched), and crucially
**`FtNmF` is never inverted on its own** — what gets factored is `FtNmF + Φ⁻¹`,
and the prior precision `Φ⁻¹` is what conditions that matrix. So the regularization
we rely on comes from `Φ`. (`utils.py` defines a `regularize_FtNmF` flag, but it is
used **only in `matrix.py`** — it does not exist in the metamath path — and isn't
generally relied on anyway.) Default is therefore plain **f32 accumulate**, with
f64-accumulate as an escalation only if a specific matmul ever needs it. The
genuinely load-bearing f64 is `ytNmy` (the pin), not `FtNmF`.

The throttle decorator (Piece 1) is unaffected by all this — it sits at the PSD
boundary in the signals layer, upstream of the kernel graph.

## Graph precision: implementation (stages 1 and 2 landed)

This section records how the precision-assignment above is actually built into
`metamatrix.py`, in two stages. **Both are landed.** Stage 1 was a probe (blanket
f32, no pins); stage 2 replaced it with op-level f64 pins via a dtype map +
cast-on-read. The stage-1 subsection below is kept for the rationale trail.

### Stage 1 — blanket working dtype at materialization (DONE)

A single gated cast at the materialization boundary
(`build_callable_from_graph`, reached via `func()` once at the likelihood
boundary). When `utils.single_precision` is true, every floating leaf — args,
constants, PSD/func-leaf outputs — is downcast to `utils.working_dtype()` as it
enters the runtime env; everything downstream inherits via JAX promotion. The
PSD factories (Piece 1) already cast their own output, so this extends the same
working dtype to the data-derived arrays (`y`, `F`, `N`).

- **Gated:** identity unless `working=float32`, so the float64 default is
  byte-identical (metamatrix parity suite green).
- **Runs at JIT runtime inside `f()`, not at fold time.** `fold_constants` is
  untouched: CPU-side constant folding and refcount eviction are unaffected, and
  nothing reaches the device earlier than before. Only post-prune/post-fold
  constants are seen, so pruned nodes are never cast.
- **Result:** the *entire* Woodbury (including the would-be f64 pins) runs in
  the working dtype — a **no-pins baseline**. Empirically (B1855 single pulsar,
  and a 3-pulsar array, fixed and variable WN): logL finite, accurate to
  ~1e-8 relative; sampling-relevant Δ-logL error ~1e-3 on Δ-logL of O(100). So
  blanket f32 is sampling-accurate at NANOGrav scale — no pin is *forced* here.
  The `ytNmy` cancellation is a known failure on **larger datasets** (seen
  elsewhere, not in our local data); the pin is required there regardless of
  whether it bites at NG scale.

Stage 1 was a **probe**, not the end state: it confirmed the mechanism works and
that f32 is viable, but it pins nothing. Stage 2 replaces its leaf-cast wholesale.

### Stage 2 — op-level pins via a dtype map + cast-on-read (DONE)

**Goal:** pin `ytNmy` and the logdets to f64 **uniformly — constant or live.**
We explicitly do *not* branch on fold status. Delineating "constant vs variable"
is exactly the combinatorial path `matrix.py` took and that metamatrix exists to
kill. Folding stays orthogonal: it decides *when* a value is computed (compile
vs per-call), never *whether* it is pinned. A pinned node is f64 whether it folds
to a constant or is recomputed every call. (That a fixed-WN `ytNmy` then happens
to be a folded f64 constant at no runtime cost is a *consequence*, not a case we
implement.)

**The mixed-consumer obstacle.** `ytNmy = yᵀN⁻¹y` (pin, f64) and
`FtNmF = FᵀN⁻¹F` (big op, want f32) share `N`. One dtype per value can't give
both: any f64 `N` promotes `FtNmF` to f64 and kills the f32 win. So precision
cannot be a per-*value* property — it must be per-*edge*.

**Mechanism — move the cast from leaves to edges:**

1. **Pin markers** on ops seed f64. `metamath.woodbury` tags `ytNmy` (i.e.
   `g.dot(y, Nmy)`) and the logdet `ld`. This is graph *intent*, set where the
   kernel math is written — not a `func()`/materialization call, so the house
   rule (methods return graphs) holds.
2. **Backward pass at materialization** → a per-node dtype map: a node is **f64
   if it is in any pin's ancestor cone, else working dtype**. (`N` lands f64
   because it feeds `ytNmy`.) Default target is the working dtype; f64 is
   injected only by pins and propagated backward. This map is computed once, in
   `func()`.
3. **Cast on read, to the *consumer's* dtype.** Each node stores its result at
   its own mapped dtype; when a node reads an input, the value is cast to *that
   consuming node's* dtype. So `N` is born f64; the `ytNmy` op reads it f64; the
   `FtNmF` op reads a **downcast f32 copy** of the same `N`. One producer, two
   edges, two dtypes — the big Cholesky stays f32, `ytNmy` stays f64, **no
   redundant solves**. Constants are simply born in their mapped dtype (a pinned
   folded constant → f64; a working-dtype-consumed folded constant → f32).

This **changes where casting happens**: stage 1 casts leaves once; stage 2 casts
on every input read, driven by the dtype map. Stage 1's leaf cast is removed (it
would otherwise clobber an f64-mapped folded constant).

**What stays f32 deliberately.** `lS` — the logdet of the small `Φ⁻¹ + FtNmF`
Cholesky — is left working dtype: its error is ~1e-5 (a sum of ~ngp logs), and
forcing it f64 would mean an f64 Cholesky, defeating the point.

**As built (the pin set).** Pinned f64: **`ytNmy`, `lN`, `lP`**. Working dtype:
**`lS` and `ld`**. Note `ld` (= `lN + lP + lS`) is *not* pinned, a deliberate
refinement of the earlier "`ld` = f64 sum of its parts" lean: pinning `ld` would
put `lS` in its ancestor cone and force the small Cholesky to f64. So `ld` reads
the f64 `lN`/`lP` *downcast* to f32 and sums in f32. This is consistent with the
limitation below — pins protect *building* each quantity, not the final sum.

**Code touchpoints (as built).** `Node` has a `pin: bool`; `GraphBuilder.pin_f64`
sets it; `_dtype_map` is the backward pass (pin → its ancestor cone f64, else
working); `build_callable_from_graph` casts each input on read to the consuming
node's dtype (`_cast_to`), replacing the stage-1 leaf cast; `woodbury` adds the
`g.pin_f64(...)` tags on `ytNmy`/`lN`/`lP`. Built in two steps, both green:
**(2a)** dtype-map + cast-on-read with an empty pin set (reproduces stage-1
numerics, parity suite green); **(2b)** the pins + tests. `visualize_graph` /
`print_graph` colour nodes by their resolved dtype and tag the pins (pass
`working=jnp.float32` to preview). Tests: `tests/single_precision/
test_graph_precision.py` (dtype-map unit tests, live-WN white-box, the
`test_pin_forces_f64_accumulation` cancellation proof, finiteness/accuracy).

**Limitation — pins protect *building* a quantity, not the final logL.** This is
the key thing to carry into Piece 2. A pin makes `ytNmy`/`lN`/`lP` *computed* in
f64 (their internal accumulation over ntoa is accurate). But the final
`logp = -0.5·(ytNmy − FtNmyᵀμ) − 0.5·ld` is assembled in **f32**, so:
- the accurate f64 `ytNmy` is **downcast to f32 before** the `ytNmy − FtNmyᵀμ`
  subtraction — the catastrophic cancellation in *that* subtraction is **not**
  protected by the pin;
- at logL ~ 1e6 one f32 step is ~0.01 (worse on bigger arrays), so the absolute
  logL simply cannot hold the 1e-2/1e-3 precision we sample to — regardless of
  cancellation.
- **You cannot up-convert your way out of it.** Casting `FtNmyᵀμ` back to f64
  doesn't recover bits it never had: `μ = chol_solve(cf, …)` is only f32-accurate
  because `cf` is the f32 Cholesky, and making `cf` f64 means an f64 factorization
  — the exact cost f32 was meant to avoid.

The fix is **Piece 2 (reference + delta)**: never form the absolute logL in f32;
compute `ΔlogL = logL(θ) − logL(θ_ref)` as a direct low-rank update so the huge
θ-independent baseline cancels *analytically* and only the O(1) change is carried
in f32. The pins are a **prerequisite** for that (the static baseline must be
accurate to subtract against), not a substitute. The hard open case is sampled
white noise, where the change is not low-rank — see "Piece 2" and
`future_planned_development.md`.

On our local data (B1855, 3-pulsar array) the pins are currently **inert**: f32
has no meaningful `ytNmy` cancellation at NANOGrav single-pulsar scale, so pinned
and blanket-f32 logL are bit-identical. The pins are insurance for larger arrays
where the cancellation is real. `test_pin_forces_f64_accumulation` exhibits the
behaviour on a constructed cancelling dot so it is actually exercised.

## Open decisions to pin (next session)

1. **`jnparray` downcast helper** — finalize the name/signature
   (`dtype='working'` kwarg vs a `to_working` helper) and land the `working_dtype`
   config knob + the x64-on assertion.
2. **`scale` / `exclude` handling** so config args never leak as sampled params.
3. **Reference-parameter scheme for the WN-sampled case** — the non-low-rank
   delta is the open analysis.
4. Whether to normalise all GP bases to O(1) and carry scales analytically
   (uniform clamp), or supply per-GP scale from the basis at build time. (For
   standard RN+DM this isn't needed; revisit if a signal's output units differ.)
