# The OS per-pulsar matrix $S$: cancellation, a rejected Cholesky–Schur idea, and the ridge we actually need

Working note on the per-pulsar GW-space matrix $S_i$ in `optimal.py`: why it is
hard to factor, an elegant Cholesky–Schur idea that **looked** right on a small
test but **fails on real data**, and why an adaptive ridge is genuinely
required.

> **Status (corrected).** An earlier version of this note claimed the
> Cholesky–Schur construction made $S$ "manifestly PSD, no ridge needed." That
> is wrong on NANOGrav-15 data. $S$ is *genuinely* numerically indefinite for
> high-precision pulsars, `jnp.linalg.cholesky` returns `NaN` on it, and the
> ridge is mandatory. The code uses `ridge_cholesky`, not a Schur factorization.
> The cancellation analysis (§1–§2) still stands; the conclusion (§3–§4) is
> rewritten.

## 1. What $S$ is

Per pulsar, with white noise $K$, red-noise GP basis $F$ (prior covariance
$B=\operatorname{diag}(\mathbf P)$) and GW basis $G$, the OS needs

$$
S \;=\; G^{\mathsf T} C^{-1} G ,
\qquad
C \;=\; K + F B F^{\mathsf T}.
$$

In whitened coordinates $W=K^{-1/2}$, $\tilde F = WF$, $\tilde G = WG$, with
$\tilde C = I + \tilde F B \tilde F^{\mathsf T}$, the Woodbury identity reduces
the $n\times n$ inverse to the small GP space:

$$
S \;=\; \tilde G^{\mathsf T}\tilde G
\;-\;\big(\tilde F^{\mathsf T}\tilde G\big)^{\mathsf T}
\big(B^{-1} + \tilde F^{\mathsf T}\tilde F\big)^{-1}
\big(\tilde F^{\mathsf T}\tilde G\big).
\tag{$\star$}
$$

In code:

```python
FFt = Ft.T @ Ft     # F̃ᵀF̃
TTt = Tt.T @ Tt     # G̃ᵀG̃   (Tt is the *GW* whitened basis)
FTt = Ft.T @ Tt     # F̃ᵀG̃
c   = cho_factor(diag(1/Pvar) + FFt)        # (B⁻¹ + F̃ᵀF̃)
S   = TTt - FTt.T @ cho_solve(c, FTt)       # the difference (⋆)
```

## 2. Why ($\star$) is hard: catastrophic cancellation **and** real indefiniteness

$S$ is $\tilde G^{\mathsf T}\tilde G$ **minus** a correction. Both terms are PSD,
and their difference is small whenever $\tilde G$ is well explained by
$\tilde F$ — i.e. when $G$ lies in the span of $F$. In the standard PTA model
that overlap is *total*: the GW basis uses the first $n_{\rm gw}$ Fourier
frequencies $k/T$, the red-noise basis the first $n_{\rm rn}\ge n_{\rm gw}$ of
the *same* frequencies on the *same* baseline, so
$\operatorname{span}(G)\subseteq\operatorname{span}(F)$ exactly (every GW column
had overlap fraction $1.0$). The two terms then agree to many digits:

$$
\underbrace{\tilde G^{\mathsf T}\tilde G}_{\;\sim\,10^{16}}
\;-\;
\underbrace{(\cdots)}_{\;\sim\,10^{16}}
\;=\;
\underbrace{S}_{\text{smallest eig}\,\sim\,10^{1}\ \text{or}\ <0} .
$$

(The $10^{16}$ scale is just the whitening $1/\sqrt N\sim10^{6}$ squared times
$n$ — not the problem. The problem is the $\sim15$-order drop to the smallest
eigenvalue.) **Two distinct things are going on, and the distinction is the
whole point of this note:**

- **Cancellation** (a property of how we *compute* $S$): forming the explicit
  difference loses precision in the small directions.
- **Genuine ill-conditioning** (a property of $S$ *itself*): for high-precision
  pulsars the true $\operatorname{cond}(S)$ exceeds float64. Measured on NG15
  (`gw_log10_A=-14.5`, `gamma=3.2` per pulsar):

  | pulsar | $\min\mathrm{eig}(S)$ | $\max\mathrm{eig}(S)$ | $\operatorname{cond}$ | #neg eigs |
  |---|---|---|---|---|
  | J0437-4715 | $-1.93\times10^{1}$ | $9.24\times10^{15}$ | $4.8\times10^{14}$ | 5 |
  | J0406+3039 | $-3.48\times10^{-1}$ | $2.17\times10^{15}$ | $6.3\times10^{15}$ | 5 |
  | J0740+6620 | $-1.47\times10^{0}$ | $7.32\times10^{15}$ | $5.0\times10^{15}$ | 3 |

  The smallest eigenvalues come out **genuinely negative**. At
  $\operatorname{cond}\sim10^{15}$ no algorithm recovers a positive
  factorization in float64 — the small directions are below the noise floor of
  the representation.

## 3. The Cholesky–Schur idea — and why it fails on real data

The tempting "form the Gram, don't subtract" move (the projection-note
principle, see [`README.md`](README.md) §"Why whiten first"): assemble the joint
SPD matrix and read $S$'s Cholesky factor off its trailing block, never forming
the difference,

$$
J=\begin{bmatrix}B^{-1}+\tilde F^{\mathsf T}\tilde F & \tilde F^{\mathsf T}\tilde G\\
\tilde G^{\mathsf T}\tilde F & \tilde G^{\mathsf T}\tilde G\end{bmatrix}
= L L^{\mathsf T},\qquad
A \equiv L_{22},\quad S = A A^{\mathsf T}\ \ (J/J_{11}=L_{22}L_{22}^{\mathsf T}).
$$

This is exact in real arithmetic and elegant. **It does not work here**, for the
reason in §2: the trailing block factorization $L_{22}=\mathrm{chol}(J_{22}-L_{21}L_{21}^{\mathsf T})$
still has to Cholesky something equal to $S$, and when $S$ is numerically
indefinite that inner Cholesky hits a non-positive pivot. `jax.numpy.linalg.
cholesky` does **not raise** on failure — it returns a matrix full of `NaN`.

Measured consequence (NG15, 67 pulsars): `schur_cholesky` returned non-finite
$A$ for **35 of 67** pulsars. A single `NaN` block makes the assembled $Q$
all-`NaN`, so every eigenvalue is `NaN` and `gx2cdf` returns nonsense
(values outside $[0,1]$). The 3-pulsar regression fixture happened to contain
only well-conditioned pulsars ($S\succ0$), so it passed and hid the bug.

There is no free lunch: avoiding the *cancellation* does not avoid the *genuine
indefiniteness*. You must regularize.

## 4. What we use: `ridge_cholesky` (adaptive ridge)

Compute $S$ explicitly via ($\star$), symmetrize, and lift the smallest
eigenvalue just past zero before the Cholesky:

$$
\mathrm{ridge}=\max\!\big(0,\ -\lambda_{\min}(S)\big)+10^{-12}\max\!\big(\|\lambda(S)\|_\infty,\,1\big),
\qquad
A=\mathrm{chol}\!\big(S+\mathrm{ridge}\cdot I\big).
$$

```python
def ridge_cholesky(Pinv, FFt, FTt, TTt):
    c = cho_factor(diag(Pinv) + FFt)
    S = TTt - FTt.T @ cho_solve(c, FTt)
    S = 0.5 * (S + S.T)
    eigs  = eigvalsh(S)
    ridge = maximum(0.0, -eigs.min()) + 1e-12 * maximum(abs(eigs).max(), 1.0)
    A = cholesky(S + ridge * eye(S.shape[0]))
    return A, S          # bs uses the raw (unridged) S
```

The ridge adds only what is needed to make $S+\mathrm{ridge}\,I$ numerically
PD; it is $\mathcal O(\lambda_{\min}^-)$, vanishingly small next to the
eigenvalues that dominate the pairwise normalization $b_{ij}=\operatorname{sum}(D_i\!*\!D_j)$,
so the OS result is unaffected while the factorization is robust. Used in `Q`,
`opQ`, `sample`, `sample_rhosigma_lowrank`.

On the full NG15 array this gives finite $Q$ and a valid `gx2cdf` (monotone, in
$[0,1]$; SNR $\approx 5.3$ on the m3a chain sample). Regression guards:
`test_ridge_cholesky_indefinite` (factor stays finite on indefinite $S$) and
`test_Q_and_gx2cdf_finite_on_hard_pulsar` (J0437-4715 in the array).

## 5. Lessons

- `jax.numpy.linalg.cholesky` **returns `NaN` instead of raising** on
  non-PD input. Any code that Choleskys a possibly-indefinite matrix must either
  regularize first or check `isfinite`.
- Test fixtures must include a **high-precision pulsar** (J0437-4715 is the
  canonical one): well-conditioned-only fixtures hide exactly the failures that
  matter on real arrays.
- Distinguish "ill-conditioned because of how I computed it" from
  "ill-conditioned matrix." Only the first is fixable by reformulation; the
  second needs regularization or higher precision.

## 6. If we ever want to drop the ridge for real

The ridge is a pragmatic float64 fix. Genuinely removing it would mean not
forming $S$ in this basis at all — e.g. working in an orthonormalized GW basis
(QR of $\tilde G$ after projecting out the red-noise span) so the surviving
directions are $\mathcal O(1)$, or carrying the small directions in higher
precision. Out of scope for now; recorded here so the ridge is understood as a
deliberate choice, not an oversight.
