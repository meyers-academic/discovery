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
from . import signals


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

    # non-Gaussian replacement q(y) for N(y | 0, I) in the whitened coordinates
    # y = L0^-1 (a - ahat0): anything with `log_prob(y)`, e.g. `self.mixture(K)` or a
    # trained flowjax flow. None keeps the Gaussian summary (VvH25).
    density: Optional[object] = None

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
    def L0(self):
        """chol(Sigma0): the whitening y = L0^-1 (a - ahat0) under which the Gaussian
        summary is N(y | 0, I), and in which the corrections q(y) are defined."""
        return np.linalg.cholesky(self.Sigma0)

    def mixture(self, K=None):
        """The GMM generalization of the summary (paper Eq. 19, A8): the per-sample
        conditionals N(a | ahat(theta_k), Sigma(theta_k)), equally weighted, as a
        `GaussianMixture` in the whitened coordinates y. With K, use K samples evenly
        spaced through the step-1 samples."""
        if self.ahat_samples is None:
            raise ValueError(f"{self.name}: no per-sample conditionals; summarize over noise samples first.")

        idx = np.arange(len(self.ahat_samples)) if K is None else \
              np.linspace(0, len(self.ahat_samples) - 1, K).astype(int)
        L0 = self.L0
        Linv0 = np.linalg.inv(L0)

        mu = (self.ahat_samples[idx] - self.ahat0) @ Linv0.T
        C = Linv0 @ self.Sigma_samples[idx] @ Linv0.T
        return GaussianMixture(mu, np.linalg.cholesky(0.5 * (C + np.swapaxes(C, 1, 2))))

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
    out the matching coefficients. Any other pulsar gets the ordinary Fourier basis,
    so summaries and time-domain pulsars can share one array likelihood.
    """
    if not isinstance(psr, FourierSummary):
        return signals.fourierbasis(psr, components, T)

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


# ---------------------------------------------------------------------------
# Non-Gaussian corrections (paper Sec. 2.3-2.5)
#
# The Gaussian summary N(a | ahat0, Sigma0) is replaced by a better density q(a)
# per pulsar -- a Gaussian mixture or a normalizing flow -- defined in the whitened
# coordinates y = L0^-1 (a - ahat0). Step 2 then picks up, per pulsar,
#
#     log w(a) = log q(y) - log N(y | 0, I)                          (Eq. 28, 34)
#
# on top of the Gaussian-summary likelihood above (the Jacobians |L0| cancel). A
# correction is any object with a `log_prob(y)` method: a `GaussianMixture`, or a
# trained flowjax distribution.
# ---------------------------------------------------------------------------


class GaussianMixture:
    """Equally weighted Gaussian mixture (1/K) sum_k N(y | mu_k, L_k L_k^T)."""

    def __init__(self, mu, L):
        self.mu = jnp.asarray(mu)                                       # (K, d)
        self.L = jnp.asarray(L)                                         # (K, d, d) lower
        self.Linv = jnp.linalg.inv(self.L)
        self.logdet = jnp.sum(jnp.log(jnp.diagonal(self.L, axis1=1, axis2=2)), axis=1)   # log |L_k|

    @property
    def K(self):
        return self.mu.shape[0]

    def log_prob(self, y):
        z = jnp.einsum('kij,kj->ki', self.Linv, y - self.mu)
        return (jax.scipy.special.logsumexp(-0.5 * jnp.sum(z**2, axis=1) - self.logdet)
                - jnp.log(self.K) - 0.5 * y.shape[-1] * jnp.log(2 * jnp.pi))


def _gp_blocks(like):
    """Column counts of the GP blocks of each pulsar's coefficient vector, in the
    order `ArrayLikelihood.clogL` concatenates them (commongp(s), then globalgp)."""
    gps = like.commongp if isinstance(like.commongp, (list, tuple)) else [like.commongp]
    sizes = [np.shape(gp.F[0])[1] for gp in gps]
    if like.globalgp is not None:
        sizes.append(np.shape(like.globalgp.Fs[0])[1])
    return sizes


def make_correction(like, psrs):
    """The non-Gaussian correction sum_p log w_p(a_p) for `like.clogL`.

    `like` is the step-2 `ArrayLikelihood` over `psrs` (in the same order); the
    correction collects `density` from every `FourierSummary` among them that has one
    (time-domain pulsars and Gaussian summaries contribute nothing). Every GP on a
    `FourierSummary` covers the summary's lowest frequencies, so a pulsar's summary
    coefficients are the sum of its GP blocks, a_p = sum_g [c_g, 0...].

    Returns a function `(params, c) -> (c, sum_p log w_p)` in the form of a
    coefficient reparametrization, so it slots in after decentering:

        like = ds.ArrayLikelihood(pls, ..., decenter=True)
        like.transform = fpta.make_correction(like, psrs)

    (an identity map whose "log-Jacobian" is the correction).
    """
    sizes = _gp_blocks(like)
    rows = [i for i, s in enumerate(psrs) if getattr(s, 'density', None) is not None]
    terms = []
    for i in rows:
        s = psrs[i]
        if max(sizes) > s.ncoeff:
            raise ValueError(f"{s.name}: a GP has more coefficients than the summary.")
        embed = np.concatenate([np.eye(s.ncoeff)[:, :m] for m in sizes], axis=1)   # a_p = embed @ c_p
        terms.append((jnp.asarray(embed), jnp.asarray(s.ahat0), jnp.asarray(np.linalg.inv(s.L0)), s.density))

    def correction(params, c):
        logw = 0.0
        for i, (embed, ahat0, Linv0, q) in zip(rows, terms):
            y = Linv0 @ (embed @ c[i] - ahat0)
            logw = logw + q.log_prob(y) + 0.5 * y @ y + 0.5 * y.shape[0] * jnp.log(2 * jnp.pi)
        return c, logw
    correction.params = []

    return correction


def _diag_prior(commongp):
    """params -> (npsr, n) prior variances on each pulsar's summary coefficients,
    summing (lowest-frequency-aligned) commongp blocks, e.g. IRN + CRN."""
    gps = commongp if isinstance(commongp, (list, tuple)) else [commongp]

    def phi(params):
        blocks = [jnp.asarray(gp.Phi.getN(params)) for gp in gps]
        n = max(b.shape[1] for b in blocks)
        return sum(jnp.pad(b, ((0, 0), (0, n - b.shape[1]))) for b in blocks)
    phi.params = sorted(set(sum([list(gp.Phi.getN.params) for gp in gps], [])))

    return phi


def mixture_logL(summaries, commongp, K=None):
    """Marginalized step-2 likelihood with the GMM correction, for priors that do not
    correlate pulsars (SPNA, IRN, CURN): paper Eq. 21, which factorizes over pulsars.

    Per pulsar, each component k is a Gaussian summary N(a | ahat_k, Sigma_k), whose
    evidence against the prior swap phi0 -> phi(eta) is analytic:

        log Z_k = 1/2 b_k^T P_k^-1 b_k - 1/2 ahat_k^T b_k - 1/2 log|Sigma_k| - 1/2 log|P_k|
                  - 1/2 log|phi| + 1/2 log|phi0|,
        P_k = Sigma_k^-1 + phi^-1 - phi0^-1,    b_k = Sigma_k^-1 ahat_k,

    and the pulsar contributes logsumexp_k log Z_k - log K. With HD (which couples
    pulsars) use `clogL` with `make_correction` instead.
    """
    phi = _diag_prior(commongp)
    n = summaries[0].ncoeff

    comps = []
    for s in summaries:
        idx = np.arange(len(s.ahat_samples)) if K is None else \
              np.linspace(0, len(s.ahat_samples) - 1, K).astype(int)
        ahat, Sigma = s.ahat_samples[idx], s.Sigma_samples[idx]
        Sigma_inv = np.linalg.inv(Sigma)
        b = np.einsum('kij,kj->ki', Sigma_inv, ahat)
        const = (-0.5 * np.einsum('ki,ki->k', ahat, b) - 0.5 * np.linalg.slogdet(Sigma)[1]
                 + 0.5 * np.sum(np.log(s.phi0)) - np.log(len(idx)))
        comps.append((Sigma_inv - np.diag(1.0 / s.phi0), b, const))

    TtNT, b, const = (jnp.asarray(np.array(x)) for x in zip(*comps))    # (npsr, K, n, n), (npsr, K, n), (npsr, K)

    def loglike(params):
        phis = phi(params)                                                # (npsr, n)
        P = TtNT + jax.vmap(jnp.diag)(1.0 / phis)[:, None, :, :]
        cf = jnp.linalg.cholesky(P)
        x = jax.scipy.linalg.cho_solve((cf, True), b[..., None])[..., 0]
        logZ = (const + 0.5 * jnp.sum(b * x, axis=-1)
                - jnp.sum(jnp.log(jnp.diagonal(cf, axis1=-2, axis2=-1)), axis=-1)
                - 0.5 * jnp.sum(jnp.log(phis), axis=-1)[:, None])
        return jnp.sum(jax.scipy.special.logsumexp(logZ, axis=1))
    loglike.params = phi.params

    return loglike
