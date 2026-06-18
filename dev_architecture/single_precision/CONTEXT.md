# Single-precision likelihood (Piece 2)

Glossary for the reference+delta single-precision likelihood work in this folder.
Terms specific to making Discovery's metamath Woodbury likelihood safe and fast in
float32. Not a spec — just the canonical words.

After marginalizing over the GP coefficients, the likelihood is a Gaussian in the
residuals `y` with covariance `Σ = N + F Φ Fᵀ`, and we evaluate it via the Woodbury
lemma — `logL = −½[ yᵀΣ⁻¹y + logdet Σ + n·log2π ]` reduced to the small `n_gp` space
through `C = Φ⁻¹ + FᵀN⁻¹F`, so the `n_toa×n_toa` matrix `F Φ Fᵀ` is never formed. The
recursion is just this lemma applied to itself: when the noise `N` is *itself* of the
form `N₀ + F′Φ′F′ᵀ` (an inner GP already marginalized), the outer solve calls Woodbury
again with that `N` as its effective noise.

## Language

**Reference covariances (N_ref, Φ_ref)**:
The frozen covariance matrices the likelihood is expanded around — a reference white
noise `N_ref` and a reference GP prior `Φ_ref` for each *sampled* GP. These (not a
parameter point) are the primitive fed to the kernel as constant leaves; the f64
baselines are derived from them once. Fixed covariances need no reference — they fold
to constants. A *reference point* `θ_ref` is just one way to produce them (evaluate the
model at `θ_ref`); a per-pulsar Gibbs / single-pulsar estimate is another.
_Avoid_: reference *parameters* as the primitive; "reference point" as the thing
metamath consumes.

**Increment (ΔlogL)**:
The small O(1) change `logL(θ) − logL(θ_ref)` computed every call, formed as a direct
analytic object — never the float32 difference of two large totals.
_Avoid_: delta likelihood, residual logL.

**Pin**:
A graph node marked to be built in float64 while the rest of the graph runs in the
working dtype (float32). Protects the *building* (accumulation) of a quantity, not its
later combination into the final logL.
_Avoid_: f64 flag, double-tag.

**Leaf data term**:
The genuine white-noise quadratic `yᵀN₀⁻¹y` (N₀ = diagonal white noise), at the bottom
of the Woodbury recursion. Static under fixed white noise; the correct — and only —
data term to pin to float64.
_Avoid_: bare "ytNmy" except as the code symbol.

**Projected data term**:
A data term `yᵀÑ⁻¹y` seen at a higher recursion level, where Ñ is an effective noise
that already absorbs a marginalized GP (`Ñ = N₀ + F Φ Fᵀ`). It is a difference of large
numbers and depends on the marginalized GP's parameters — handled by reference+delta,
never by a pin.

**Recursive (fused) Woodbury**:
The nested Woodbury in which a per-pulsar (inner) GP is marginalized into an effective
per-pulsar noise that an outer (cross-pulsar) GP then sees. "Fused" = inner and outer
solves emitted as one batched graph (`vectorwoodburyjointsolve` + `globalwoodbury_fused`).

**Inner GP / outer GP**:
Inner = the per-pulsar GP marginalized first (e.g. intrinsic red noise, DM). Outer =
the GP marginalized on top of it (e.g. a common / Hellings–Downs process).

**Effective noise (N_eff)**:
The covariance an outer Woodbury level treats as its "noise": white noise plus every
inner GP already marginalized in (`N_eff = N₀ + Σ_inner F Φ Fᵀ`). When the inner GPs
are *fixed* (timing-model marginalization, ECORR) `N_eff` folds to an f64 constant;
when an inner GP is *sampled* (intrinsic red noise) `N_eff` is live and the outer level
is the nested-sampled-inner case.

**Flattening (rejected)**:
Collapsing the nested Woodbury into one big block-Φ solve over all sampled GPs. Makes
the math trivially the fixed-WN case but destroys the per-pulsar batching of the
intrinsic-RN marginalization — the core GPU win — so it is not used. See ADR 0002.
