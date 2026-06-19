# OS (`optimal.py`) memory / performance notes

Working notes on GPU-memory savings in `src/discovery/optimal.py`, plus the
`quadax` vs `scipy.integrate` swap for the Imhof CDF.

**Companion notes**

- [`schur_complement_S.md`](schur_complement_S.md) — full math for computing the
  per-pulsar GW-space matrix $S$ as a Cholesky–Schur complement (manifestly PSD,
  ridge-free), replacing the cancellation-prone difference form. This is the
  numerical-stability fix; §7 below summarizes it.

## 1. How `OS` is structured: precompute vs per-call

Every public entry point (`os`, `scramble`, `shift`, `Q`, `opQ`, `sample`,
`os_rhosigma`, ...) is a `@functools.cached_property` that runs a **setup block
once** and returns a **closure `f(params)`** that runs on every evaluation. The
memory/compute question is entirely about *which side of that line* each
quantity lands on.

### What is precomputed (runs once, on object construction / first access)

- `os_rhosigma`: `kernelsolves = [psl.N.make_kernelsolve(psl.y, gw.F)]`.
  Because the GW basis `gw.F` passed as `T` is a **constant array** (not a
  callable), `make_kernelsolve` takes the `else` branch in `matrix.py:996` and
  returns a closure with `kernelsolve.params == []`. It closes over the constant
  results `TtSy = Tᵀ Σ⁻¹ y` and `TtST = Tᵀ Σ⁻¹ T`. **These are fully
  precomputed** (and partly on the CPU/numpy via `sp.linalg.cho_solve`).
- `Q` / `opQ` / `sample`: the design-matrix products `FFt = Ftᵀ Ft`,
  `TTt = Ttᵀ Tt`, `FTt = Ftᵀ Tt`, and the slice bookkeeping (`inds`, `cnt`,
  `ngw`) are built once in the setup block.

### What is recomputed every `f(params)` call

- `N = getN(params)` — the **GW prior** `gw.Phi` (diagonal, length `ngw`).
  This is the *only* genuinely param-dependent input in `os_rhosigma`.
- `sN = sqrt(N)`, `ts`, `ds = sN ⊙ k[1] ⊙ sN`, and the normalization
  **`bs = [trace(ds[i] @ ds[j]) for pairs]`**.
- In `Q`/`opQ`/`sample`: the per-pulsar Cholesky `cs`, the Schur complements
  `Ss`, the regularized factors `As`, `Ds`, and again `bs`.

### Consequence for "fixed white noise, varying red noise"

Two distinct meanings — they behave very differently:

- **Varying the common GW process** (`gw_log10_A`, `gw_gamma`): flows through
  `getN` only. Everything per-pulsar (`kernelsolves`) stays constant; only the
  diagonal `Phi` and the quantities derived from it (`ds`, `bs`) recompute. This
  is the cheap, vmappable case and the one the trace fix below targets.
- **Varying per-pulsar intrinsic red noise**: that noise is *baked into*
  `psl.N` at fixed values when the `OS` object is built (`kernelsolve.params ==
  []`). Changing it requires **rebuilding the `OS` object**, not just calling
  `f(params)` with new params. So the precompute does not help you sweep
  intrinsic red noise; if that is the goal we need a different factoring.

> If the only thing varying is the GW prior shape, the dominant per-call work is
> `bs` (the pairwise traces). Since it's recomputed each call, optimizing it is
> worthwhile and it stays cheap enough to run on CPU.

## 2. Main memory win: `trace(A @ B)` → `sum(A * B)`

The pairwise normalization is computed everywhere as:

```python
bs = [matrix.jnp.trace(Ds[i] @ Ds[j]) for (i, j) in self.pairs]
```

`Ds[i] @ Ds[j]` **materializes a full `m × m` matmul** (m = number of GW basis
columns = `2 × n_freq`, e.g. 28 for `n_freq=14`) purely to take its trace —
O(m³) flops and an `m × m` temporary, repeated over `N_psr(N_psr-1)/2` pairs.

Both `Ds`/`ds` are symmetric (`D = diag(sPhi) · S · diag(sPhi)` with `S`
symmetric), so:

```
trace(A @ B) == sum(A * B)      # elementwise, for symmetric A, B
```

`sum(A * B)` is O(m²), allocates no `m × m` temporary, and launches one fused
elementwise+reduce instead of a matmul. Under `jax.vmap` over many param
samples / scrambles / shifts, the avoided intermediate is `(batch, m, m)` per
pair — that's where the GPU memory actually adds up.

### Occurrences to change

| location | line (approx) | context |
|---|---|---|
| `Q.get_Q` | 89 | `bs = [trace(Ds[i] @ Ds[j])]` |
| `opQ.get_opQ` | 140 | same |
| `sample.get_sample` | 195 | same |
| `sample_rhosigma_lowrank.xs2snrs` setup | 249 | same |
| `sample_rhosigma` setup | 293 | same |
| `os_rhosigma.get_rhosigma` (1D branch) | 356 | `bs = [trace(ds[i] @ ds[j])]` |
| `os_rhosigma.get_rhosigma` (2D branch) | 364 | same (here `ds = U k[1] Uᵀ`, also symmetric) |

All are the symmetric-trace pattern → drop-in `sum(A * B)`.

### Expected impact

- **Compute**: m³ → m² per pair. Modest in absolute flops at m≈28, but removes a
  matmul kernel launch per pair and the temporary.
- **Memory**: removes the `m × m` (or batched `(B, m, m)`) intermediate. The
  real benefit scales with `n_freq` and with `vmap` batch size.
- **Numerics**: equivalent up to FP reassociation (verified separately on SPD
  matrices; differences ~1e-12 relative).

### Measured impact (laptop, CPU, x64, Npsr=67, m=28, n_pairs=2211)

From `/tmp/bench_trace.py` (see §6 for the recipe):

```
single call    trace(A@B):   2.3 ms | temp= 13.87 MB
               sum(A*B)  :   1.0 ms | temp=  0.14 MB

vmap B=100     trace(A@B):  108 ms  | temp=  1.39 GB
               sum(A*B)  :    8 ms  | temp=  1.84 MB

vmap B=1000    trace(A@B): 1021 ms  | temp= 13.87 GB   <-- would OOM a GPU
               sum(A*B)  :   51 ms  | temp= 17.69 MB
```

The fix is a strict win on **both** time (~2x single, ~20x vmapped) and memory
(temp scratch ~750x smaller at B=1000). There is no recompute-vs-memory
tradeoff: `trace(A@B)` does *more* work (forms the full m x m product, then
discards everything but the diagonal), so computing it every call is simply
more expensive than `sum(A*B)`.

## 3. Prototyping GPU memory locally (no GPU / no MPS needed)

You do **not** need a GPU to see the memory blow-up. XLA computes the peak
device memory of the compiled executable at compile time, and that figure is
backend-independent. Get it on the CPU laptop with:

```python
f = jax.jit(your_fn)
mem = f.lower(args).compile().memory_analysis()
mem.temp_size_in_bytes      # peak scratch the executable needs (the OOM driver)
mem.output_size_in_bytes    # size of the returned arrays
```

The `temp_size_in_bytes` is what predicts a GPU OOM (the 13.87 GB above). This
is how the table in §2 was produced without a GPU present.

Two caveats on what this does and does not capture:

- It measures **device-resident peak**, not the host->device transfer of the
  precomputed constants (the `As`, design-matrix products, etc.). That transfer
  is a *one-time* cost paid when the constants are first moved to the GPU, not
  per call, and the constants here are small (~MB). It is a separate concern
  from the per-call scratch this fix targets.
- It reflects the compiled HLO for the *given input shapes*; re-check with the
  real `Npsr`, `m`, and `vmap` batch size you intend to run.

## 4. Secondary: dense `Q` assembly (lower priority)

`Q.get_Q` builds a dense `(N_psr·m)²` matrix via a Python loop of
`Q = Q.at[inds[i], inds[j]].add(Bij)` — once per pair. Under `jit` this unrolls
into `N_pairs` scatter-adds, each conceptually a fresh `(cnt, cnt)` array in the
trace graph (`cnt = N_psr·m`, e.g. 67·28 ≈ 1876 → ~28 MB per dense copy at
float64). Only `gx2cdf` consumes `Q` (for its eigenvalues), so this is **not** on
the hot vmapped path. Defer unless `gx2cdf` memory is a measured problem; if so,
build the block structure with a single `segment_sum`/scatter or operate on the
`opQ` matrix-free operator + Lanczos eigenvalues instead of forming `Q`.

## 5. `quadax` vs `scipy.integrate` for the Imhof CDF — VERIFIED

`eig2cdf` was changed from a per-point `scipy.integrate.quad` (with a host
round-trip via `float(imhof(...))`) to `quadax.quadgk` wrapped in `jax.vmap`
over `osxs`.

Comparison (`/tmp/test_quadax_imhof.py`, x64 enabled, three eigenvalue cases):

```
case 0: max|Δ|=6.39e-09   mean|Δ|=4.95e-10
case 1: max|Δ|=1.31e-12   mean|Δ|=3.77e-13
case 2: max|Δ|=1.57e-07   mean|Δ|=4.43e-08
```

All within `epsabs=1e-6`. quadax is fully in-JAX (vmappable, differentiable, no
host round-trip), so it should win on GPU even though CPU timing is a wash.

**Imhof integrand `u=0` fix (DONE).** The integrand
$\sin\theta(u)/(u\,\rho(u))$ has a removable $0/0$ singularity at $u=0$ — coded
naively it returns `nan`. The current pinned eigenvalues happen not to sample
the endpoint, but `quadax.quadgk` over $[0,\infty)$ *can*, which would poison
the whole integral. `imhof` now returns the finite limit

$$
\lim_{u\to0}\frac{\sin\theta(u)}{u\,\rho(u)} \;=\; \tfrac12\Big(\sum_\alpha\lambda_\alpha - x\Big)
$$

via `jnp.where(u == 0.0, limit, ...)` (also gradient-safe). Regression:
`test_imhof_u0_limit`.

**Follow-ups:** doc `docs/tutorials/optimal_statistic.rst:251` still says
"Passed to `scipy.integrate.quad`", and line 252 claims `gx2cdf` can't be
jitted/vmapped — the quadax rewrite likely lifts that restriction; re-check.

## 6. Proposed work order

1. ~~Apply `trace(A@B)` → `sum(A*B)` across the 7 sites in §2.~~ **DONE**
   (commit `46b680f`).
2. ~~Add a regression test pinning quadax `eig2cdf` to the scipy reference.~~
   **DONE** (`tests/test_optimal.py::test_gx2cdf_regression`).
3. ~~Stabilize $S$ via the Cholesky–Schur complement; drop the ridge.~~ **DONE**
   (§7, [`schur_complement_S.md`](schur_complement_S.md)).
4. ~~Fix the Imhof `u=0` nan and the `os_rhosigma_complex` `sN`-before-def bug.~~
   **DONE** (§5, §7).
5. Update `optimal_statistic.rst` (scipy→quadax; re-verify the `gx2cdf`
   jit/vmap claim). **TODO**
6. (Deferred) extend Cholesky–Schur into `matrix.make_kernelsolve` so the
   default `os`/`scramble` and `sample_rhosigma` paths get the same stability —
   a `matrix.py`/likelihood change, on purpose left for later.
7. (Optional) revisit dense `Q` assembly only if `gx2cdf` memory is measured to
   be a problem.

## 7. Numerical stability of $S$ + the two integrand/guard bugs (DONE)

Full math: [`schur_complement_S.md`](schur_complement_S.md). Short version:

The per-pulsar GW-space matrix

$$
S \;=\; \tilde G^{\mathsf T}\big(I + \tilde F B \tilde F^{\mathsf T}\big)^{-1}\tilde G
\;=\; \tilde G^{\mathsf T}\tilde G
- (\tilde F^{\mathsf T}\tilde G)^{\mathsf T}(B^{-1}+\tilde F^{\mathsf T}\tilde F)^{-1}(\tilde F^{\mathsf T}\tilde G)
$$

was formed as that **difference**. Because the GW Fourier basis $\tilde G$ lies
entirely inside the red-noise span $\tilde F$ (same frequencies, same baseline),
the two terms are large and nearly equal — catastrophic cancellation drops
$\sim15$ orders to $S$'s smallest eigenvalue, so the explicit $S$ stops being
numerically PSD. The `1e-10 tr(S)/n` (and adaptive-eigenvalue) **ridge** was a
band-aid for *self-inflicted* indefiniteness, and it swamped exactly the small
GW-mode directions the OS normalization needs.

Fix: never form the difference. Build the joint SPD matrix

$$
J=\begin{bmatrix}B^{-1}+\tilde F^{\mathsf T}\tilde F & \tilde F^{\mathsf T}\tilde G\\
\tilde G^{\mathsf T}\tilde F & \tilde G^{\mathsf T}\tilde G\end{bmatrix}
= L L^{\mathsf T},\qquad
L=\begin{bmatrix}L_{11}&0\\ L_{21}&L_{22}\end{bmatrix},
$$

whose Schur complement of the top-left block is $S$. The trailing Cholesky
block is then a Cholesky factor of $S$ directly:

$$
A \equiv L_{22},\qquad S = A A^{\mathsf T}\ \ (\text{PSD by construction, no ridge}).
$$

The subtraction now happens *inside* the factorization on well-scaled,
prior-regularized data (the matrix analogue of "subtract at the $O(1)$
whitened-vector level, then sum squares" from the projection note). Genuine
$\operatorname{cond}(S)\sim10^{15}$ is physical and untouched; we just stop
manufacturing indefiniteness.

Implemented as `optimal.schur_cholesky(Pinv, FFt, FTt, TTt)`, used in `Q`,
`opQ`, `sample`, `sample_rhosigma_lowrank`. Matches the old form to machine
precision in float64 (golden OS/$Q$/CDF unchanged); regression
`test_schur_cholesky_psd_no_ridge`.

Also fixed alongside:

- **Imhof `u=0` nan** — see §5.
- **`os_rhosigma_complex` guard** referenced `sN` before assignment
  (`NameError`); now checks `N.ndim`.
