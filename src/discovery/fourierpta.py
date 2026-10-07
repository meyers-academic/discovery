"""Fourier-domain PTA likelihood (Valtolina & van Haasteren 2025, "VvH25"), built
on the standard discovery likelihoods.

Two steps:

1. Per pulsar, fix the red-noise spectrum at a reference eta0 and summarize what the
   data say about the red-noise Fourier coefficients a, with every other noise
   process (timing model, ECORR, white noise, ...) marginalized:

       p(a | dt, eta0) ~= N(a | ahat0, Sigma0)                       (VvH25 Eq. 12, 14)

   `summarize_pulsar` computes (ahat0, Sigma0) from a `PulsarLikelihood`, either at
   fixed noise parameters or averaged over samples of them (law of total expectation
   and covariance).

2. Over the array, swap the reference prior N(a | 0, phi0) for the model prior
   p(a | eta). Dividing out the reference prior leaves, as a function of a,

       log N(a | ahat0, Sigma0) - log N(a | 0, phi0)
           = -1/2 a^T (Sigma0^-1 - phi0^-1) a + a^T Sigma0^-1 ahat0 + const,

   which has exactly the form of an ordinary pulsar likelihood log p(dt | a),
   -1/2 a^T (F^T N^-1 F) a + a^T (F^T N^-1 dt) + const, with

       F^T N^-1 F  <->  TtNT = Sigma0^-1 - phi0^-1,
       F^T N^-1 dt <->  b0   = Sigma0^-1 ahat0.

   A `FourierSummary` is a stand-in pulsar carrying data (residuals, a basis, unit
   white noise) chosen to reproduce those two arrays, so step 2 is just a standard
   `ArrayLikelihood` over summaries: CURN/IRN through `commongp`, HD and
   anisotropic ORFs through `globalgp`, the marginalized `logL` (VvH25 Eq. 18) or the
   sampled-coefficient `clogL` with `decenter=True` (Eq. 16, CURN decentering).

The stand-in data are built from the eigendecomposition TtNT = U diag(lam) U^T,
keeping the r eigenvalues lam > 0:

    F_eff = diag(sqrt(lam)) U^T     (r x n),     N_eff = I_r,
    y_eff = diag(1/sqrt(lam)) U^T b0             (r,),

so that F_eff^T F_eff = TtNT and F_eff^T y_eff = b0 (b0 must lie in the range of
TtNT, which it does whenever TtNT is positive semi-definite and comes from data).
The likelihood differs from VvH25 Eq. 18 only by an eta-independent constant.
"""

import dataclasses
import warnings
from typing import Optional

import numpy as np

import jax
import jax.numpy as jnp

from . import _kernels as kernels


@dataclasses.dataclass
class FourierSummary:
    """Step-1 summary of one pulsar: the Gaussian N(a | ahat0, Sigma0) on its
    red-noise Fourier coefficients a at the reference spectrum phi0 = phi(eta0).

    Doubles as a stand-in pulsar for step 2 (`name`, `pos`, `residuals`, and the
    basis served by `summarybasis`), so it can be passed straight to
    `makecommongp_fourier(..., fourierbasis=summarybasis)` and
    `makeglobalgp_fourier(..., fourierbasis=summarybasis)`.
    """

    name: str
    pos: np.ndarray
    f: np.ndarray            # frequency of each coefficient (sin/cos repeated), length n
    df: np.ndarray           # frequency bin width of each coefficient, length n
    ahat0: np.ndarray        # mean of a, length n
    Sigma0_inv: np.ndarray   # precision of a, (n, n)
    phi0: np.ndarray         # reference prior variance of a, length n (diagonal)

    # per-sample conditionals (ahat(theta_k), Sigma(theta_k)) when step 1 averaged
    # over noise samples; these are the components of the GMM generalization
    ahat_samples: Optional[np.ndarray] = None    # (K, n)
    Sigma_samples: Optional[np.ndarray] = None   # (K, n, n)

    # eigenvalues of TtNT below rtol * max are roundoff, treated as zero (the
    # stand-in data then have fewer than n entries)
    rtol: float = 1e-14

    @property
    def ncoeff(self):
        return len(self.ahat0)

    @property
    def components(self):
        return self.ncoeff // 2

    @property
    def T(self):
        return 1.0 / self.f[0]

    @property
    def b0(self):
        """Sigma0^-1 ahat0, playing the role of F^T N^-1 dt."""
        return self.Sigma0_inv @ self.ahat0

    @property
    def TtNT(self):
        """Sigma0^-1 - phi0^-1, playing the role of F^T N^-1 F."""
        return self.Sigma0_inv - np.diag(1.0 / self.phi0)

    @property
    def Sigma0(self):
        return np.linalg.inv(self.Sigma0_inv)

    @property
    def _standin(self):
        if '_standin_cache' not in self.__dict__:
            lam, U = np.linalg.eigh(self.TtNT)
            if lam[0] < -self.rtol * lam[-1]:
                raise ValueError(
                    f"{self.name}: Sigma0^-1 - phi0^-1 has a negative eigenvalue "
                    f"({lam[0]:.3e}, largest {lam[-1]:.3e}). The data constrain some "
                    "combination of coefficients less than the reference prior phi0 does; "
                    "choose a reference spectrum eta0 with more power.")

            # Directions with lam ~ 0 are combinations of coefficients the data do not
            # constrain at all (e.g. low frequencies degenerate with the timing model in
            # a pulsar shorter than T). They are dropped; b0 should have no component
            # along them, and what component it has could shift the likelihood by up
            # to ~ (b0 . u)^2 / (2 lam_cut).
            cut = self.rtol * lam[-1]
            keep = lam > cut
            proj = U.T @ self.b0
            dropped = 0.5 * np.sum(proj[~keep]**2) / cut
            if dropped > 1e-3:
                warnings.warn(f"{self.name}: dropping {np.sum(~keep)} unconstrained directions of "
                              f"Sigma0^-1 - phi0^-1 along which b0 is not ~0 (log-likelihood "
                              f"effect up to {dropped:.2e}).")

            lam, U, proj = lam[keep], U[:, keep], proj[keep]
            self.__dict__['_standin_cache'] = (np.sqrt(lam)[:, None] * U.T,
                                               proj / np.sqrt(lam))
        return self.__dict__['_standin_cache']

    @property
    def Fmat(self):
        """Stand-in basis F_eff (r x n), with F_eff^T F_eff = TtNT."""
        return self._standin[0]

    @property
    def residuals(self):
        """Stand-in data y_eff (length r), with F_eff^T y_eff = b0."""
        return self._standin[1]


def summarybasis(psr, components, T=None):
    """`fourierbasis` replacement for `FourierSummary` stand-in pulsars.

    Returns the leading 2*components columns of the stand-in basis, so a GP with
    fewer components than the summary (e.g. a GWB on the lowest frequencies) picks
    out the matching coefficients.
    """
    n = 2 * components
    if n > psr.ncoeff:
        raise ValueError(f"{psr.name}: requested {components} components, summary has {psr.components}.")
    if T is not None and not np.isclose(T, psr.T, rtol=1e-10):
        raise ValueError(f"{psr.name}: T = {T} does not match the summary's T = {psr.T}.")

    return psr.f[:n], psr.df[:n], psr.Fmat[:, :n]


def makenoise_summary(psr):
    """Unit white noise for a `FourierSummary` stand-in pulsar."""
    return kernels.NoiseMatrix1D_novar(np.ones(len(psr.residuals)))


def _find_gp(psl, gp):
    """The variable GP in `psl` whose coefficient name contains `gp`, and its slice
    of the conditional mean."""
    found = [(sig, key) for sig in psl.signals
             for key in (getattr(sig, 'index', None) or {})
             if f'_{gp}_coefficients' in key]
    if len(found) != 1:
        raise ValueError(f"Expected exactly one GP named '{gp}' in the likelihood, found {len(found)}.")
    sig, key = found[0]

    return sig, psl.N.index[key]


def summarize_pulsar(psr, psl, params, gp='red_noise'):
    """Step 1 of VvH25: the Gaussian summary of `psr`'s `gp` Fourier coefficients.

    `psl` is a `PulsarLikelihood` that includes the `gp` (e.g. `makegp_fourier(...,
    name='red_noise')`) alongside whatever is to be marginalized (timing model,
    measurement noise, ECORR, DM GPs, ...). `params` sets every parameter of `psl`,
    with the `gp` spectrum at its reference value eta0:

      - all scalars: the summary is the conditional at those parameters
        (fixed white noise);
      - some arrays of length K (the rest scalars): samples theta_k of the noise
        parameters, combined by the law of total expectation and covariance
        (VvH25 Eq. 14):

            ahat0  = E_k[ahat(theta_k)],
            Sigma0 = E_k[Sigma(theta_k)] + Cov_k[ahat(theta_k)].

    The per-sample conditionals are kept on the summary for the GMM generalization.
    """
    sig, sl = _find_gp(psl, gp)

    cond = psl.conditional
    params = {par: jnp.asarray(params[par]) for par in cond.params}

    def mean_and_cov(p):
        mu, cf = cond(p)
        L = cf[0] if cf[1] else cf[0].T
        Sigma = jax.scipy.linalg.cho_solve((L, True), jnp.eye(L.shape[0]))
        return mu[sl], Sigma[sl, sl]

    sampled = [par for par, val in params.items() if val.ndim > 0]
    if sampled:
        nsamples = {params[par].shape[0] for par in sampled}
        if len(nsamples) != 1:
            raise ValueError(f"Sampled parameters have different lengths: {sorted(nsamples)}.")
        axes = {par: (0 if par in sampled else None) for par in params}
        mus, Sigmas = jax.jit(jax.vmap(mean_and_cov, in_axes=(axes,)))(params)
        mus, Sigmas = np.asarray(mus), np.asarray(Sigmas)

        ahat0 = mus.mean(axis=0)
        Sigma0 = Sigmas.mean(axis=0) + np.cov(mus.T, bias=False)
    else:
        mus = Sigmas = None
        ahat0, Sigma0 = map(np.asarray, jax.jit(mean_and_cov)(params))

    phi0 = np.asarray(sig.Phi.getN({par: params[par] for par in sig.Phi.getN.params}))
    if phi0.ndim != 1:
        raise NotImplementedError("Only diagonal reference priors phi0 are supported.")

    f, df = np.asarray(sig.f), np.asarray(sig.df)

    Sigma0_inv = np.linalg.inv(Sigma0)
    Sigma0_inv = 0.5 * (Sigma0_inv + Sigma0_inv.T)

    return FourierSummary(name=psr.name, pos=np.asarray(psr.pos), f=f, df=df,
                          ahat0=ahat0, Sigma0_inv=Sigma0_inv, phi0=phi0,
                          ahat_samples=mus, Sigma_samples=Sigmas)

