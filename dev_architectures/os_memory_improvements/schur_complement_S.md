# Stabilizing the OS per-pulsar matrix $S$ with a Cholesky–Schur complement

Working note on why the per-pulsar GW-space matrix $S_i$ in `optimal.py` lost
positive-definiteness (and needed a ridge), and how to compute it as a
manifest Gram instead. This is the projection note's "form the Gram, don't
subtract" principle (see [`README.md`](README.md) §"Why whiten first"),
generalized from the flat-prior timing case to a **finite** red-noise prior.

## 1. What $S$ is

Per pulsar, with white noise $K$, red-noise GP basis $F$ (prior covariance
$B=\operatorname{diag}(\mathbf P)$) and GW basis $G$, the OS needs

$$
S \;=\; G^{\mathsf T} C^{-1} G ,
\qquad
C \;=\; K + F B F^{\mathsf T},
$$

i.e. the GW basis projected through the *total* (white + red) noise covariance.
Working in whitened coordinates $W=K^{-1/2}$, $\tilde F = WF$, $\tilde G = WG$,
so $\tilde C = W C W^{\mathsf T} = I + \tilde F B \tilde F^{\mathsf T}$ and

$$
S \;=\; \tilde G^{\mathsf T}\big(I + \tilde F B \tilde F^{\mathsf T}\big)^{-1}\tilde G .
$$

The Woodbury identity reduces the $n\times n$ inverse to the small GP space,
$C = \Phi^{-1}+\tilde F^{\mathsf T}\tilde F$ space:

$$
S \;=\; \tilde G^{\mathsf T}\tilde G
\;-\;\big(\tilde F^{\mathsf T}\tilde G\big)^{\mathsf T}
\big(B^{-1} + \tilde F^{\mathsf T}\tilde F\big)^{-1}
\big(\tilde F^{\mathsf T}\tilde G\big).
\tag{$\star$}
$$

In code ($\star$) is, per pulsar,

```python
FFt = Ft.T @ Ft     # F̃ᵀF̃
TTt = Tt.T @ Tt     # G̃ᵀG̃   (Tt is the *GW* whitened basis)
FTt = Ft.T @ Tt     # F̃ᵀG̃
c   = cho_factor(diag(1/Pvar) + FFt)        # (B⁻¹ + F̃ᵀF̃)
S   = TTt - FTt.T @ cho_solve(c, FTt)       # the difference (⋆)
```

## 2. Why ($\star$) is numerically bad: catastrophic cancellation

$S$ is computed as $\tilde G^{\mathsf T}\tilde G$ **minus** a correction. Both
terms are positive semidefinite, and their difference is small whenever
$\tilde G$ is well explained by $\tilde F$ — i.e. when $G$ lies in the column
span of $F$.

In the standard PTA model that overlap is **total**: the GW basis uses the
first $n_{\rm gw}$ Fourier frequencies $k/T$, and the red-noise basis uses the
first $n_{\rm rn}\ge n_{\rm gw}$ of the *same* frequencies on the *same*
baseline $T$. So $\operatorname{span}(G)\subseteq\operatorname{span}(F)$
exactly. Empirically every GW column had overlap fraction $1.0$ with the
red-noise span. The two terms in ($\star$) then agree to many digits and their
difference is dominated by rounding:

$$
\underbrace{\tilde G^{\mathsf T}\tilde G}_{\;\sim\,10^{16}}
\;-\;
\underbrace{(\cdots)}_{\;\sim\,10^{16}}
\;=\;
\underbrace{S}_{\text{smallest eig}\,\sim\,10^{1}} .
$$

(The $10^{16}$ scale is just the whitening $1/\sqrt{N}\sim10^{6}$ squared,
times $n$; it is not the problem. The problem is the $\sim15$-order drop to the
smallest eigenvalue of $S$.) The result is an $S$ whose true condition number
$\sim10^{15}$ is at the edge of float64 and **past** float32, so the explicit
difference produces a matrix that is no longer numerically PSD — its Cholesky
fails. That is exactly what the
`S → ½(S+Sᵀ)` symmetrization and the `1e-10 tr(S)/n` (or adaptive
eigenvalue) **ridge** were patching.

The ridge makes the Cholesky *succeed*, but it does so by adding
$\mathcal O(10^{5})$ to a matrix whose smallest real eigenvalue is
$\mathcal O(10^{1})$ — i.e. it discards the small GW-mode directions, the very
ones the OS normalization $b_{ij}$ depends on.

## 3. The fix: read $S$'s Cholesky factor off a joint Cholesky

The clean, manifestly-PSD route never forms the difference. Assemble the joint
(small, $(m_F{+}m_G)\times(m_F{+}m_G)$) matrix

$$
J \;=\;
\begin{bmatrix}
B^{-1} + \tilde F^{\mathsf T}\tilde F & \tilde F^{\mathsf T}\tilde G\\[2pt]
\tilde G^{\mathsf T}\tilde F & \tilde G^{\mathsf T}\tilde G
\end{bmatrix}
\;=\;
\begin{bmatrix} J_{11} & J_{12}\\ J_{12}^{\mathsf T} & J_{22}\end{bmatrix},
$$

which is symmetric positive definite ($J_{11}\succ0$ because $B^{-1}\succ0$;
$J$ is the prior-scaled Gram of $[\tilde F B^{1/2},\,\tilde G]$ augmented by the
prior). Its **Schur complement of $J_{11}$ is exactly $S$**:

$$
J/J_{11}\;=\;J_{22}-J_{12}^{\mathsf T}J_{11}^{-1}J_{12}
\;=\;\tilde G^{\mathsf T}\tilde G-(\tilde F^{\mathsf T}\tilde G)^{\mathsf T}
(B^{-1}+\tilde F^{\mathsf T}\tilde F)^{-1}(\tilde F^{\mathsf T}\tilde G)
\;=\;S .
$$

Now factor $J$ by a block Cholesky $J = L L^{\mathsf T}$ with

$$
L=\begin{bmatrix}L_{11} & 0\\ L_{21} & L_{22}\end{bmatrix},
$$

The standard identity for the trailing block is

$$
L_{22}L_{22}^{\mathsf T}\;=\;J_{22}-L_{21}L_{21}^{\mathsf T}
\;=\;J_{22}-J_{12}^{\mathsf T}J_{11}^{-1}J_{12}\;=\;S .
$$

So the bottom-right block of the Cholesky factor **is** a Cholesky factor of
$S$:

$$
\boxed{\,A \equiv L_{22},\qquad S = A A^{\mathsf T}.\,}
$$

Two things matter here:

1. **No difference is ever formed.** The subtraction $J_{22}-L_{21}L_{21}^{\mathsf T}$
   happens *inside* the factorization, after $L_{21}=J_{12}^{\mathsf T}L_{11}^{-\mathsf T}$
   has been computed from the well-scaled, prior-regularized $L_{11}$. This is
   the matrix analogue of doing the cancellation "at the $O(1)$ whitened-vector
   level and then summing squares."
2. **$A$ is obtained directly**, so the separate `cholesky(S + ridge·I)` step
   disappears. $S=AA^{\mathsf T}$ is PSD by construction; there is nothing left
   to regularize.

The genuine conditioning of $S$ ($\sim10^{15}$, set by the power-law spectral
dynamic range over the band) is *physical* and unchanged — we are not pretending
$S$ is well conditioned. We are only refusing to *manufacture* indefiniteness
through cancellation.

### Code

```python
def schur_cholesky(Pinv, FFt, FTt, TTt):
    mF = FFt.shape[0]
    J = jnp.block([[jnp.diag(Pinv) + FFt, FTt],
                   [FTt.T,                TTt]])
    return jnp.linalg.cholesky(J)[mF:, mF:]      # = A,  S = A @ A.T
```

Downstream, the OS uses $A$ directly (for $Q$ blocks, samples) and reconstructs
$S = A A^{\mathsf T}$ where the pairwise normalization
$b_{ij}=\operatorname{sum}(D_i\!*\!D_j)$ needs it
($D=\operatorname{diag}(\sqrt\Phi)\,S\,\operatorname{diag}(\sqrt\Phi)$).

## 4. Verification

On the 3-pulsar regression fixture (`tests/test_optimal.py`), float64:

| pulsar | old $\min\mathrm{eig}(S)$ | new $\min\mathrm{eig}(S)$ | $\operatorname{cond}(S)$ | $\|S_{\rm new}-S_{\rm old}\|/\|S_{\rm old}\|$ |
|---|---|---|---|---|
| 0 | $1.359\times10^{13}$ | $1.359\times10^{13}$ | $9.1\times10^{2}$ | $7.2\times10^{-16}$ |
| 1 | $9.754$ (cancellation-corrupted) | $10.48$ | $8.0\times10^{14}$ | $5.7\times10^{-16}$ |
| 2 | $1.317\times10^{13}$ | $1.317\times10^{13}$ | $8.5\times10^{2}$ | $7.0\times10^{-16}$ |

- New $\min\mathrm{eig}(S)>0$ strictly for every pulsar — **no ridge**.
- On the ill-conditioned pulsar 1 the new value ($10.48$) is the *accurate*
  small eigenvalue; the old $9.754$ was already corrupted by the cancellation.
- Agreement to machine precision means the pinned golden OS / $Q$ / `gx2cdf`
  values are unchanged at `rtol=1e-8`.

Regression tests added: `test_schur_cholesky_psd_no_ridge` (PSD + equivalence
on a deliberately GW⊂red-noise near-singular case) and the existing OS/Q/CDF
goldens (unchanged).

## 5. Where it is applied / not applied

Applied (these formed $A$ via a ridged Cholesky):
`Q.get_Q`, `opQ.get_opQ`, `sample.get_sample`, `sample_rhosigma_lowrank`.

**Not** applied (no ridge there, so no crash today, but they carry the same
cancellation in their traces): the default `os`/`scramble` path via
`os_rhosigma` → `matrix.make_kernelsolve`, and `sample_rhosigma`. These build
$T^{\mathsf T}\Sigma^{-1}T$ with the same Woodbury difference inside
`matrix.py` and only use it for traces (never a Cholesky), so they degrade in
precision rather than failing. Extending the Cholesky–Schur treatment into
`make_kernelsolve` is a separate `matrix.py`/likelihood change, deferred.
