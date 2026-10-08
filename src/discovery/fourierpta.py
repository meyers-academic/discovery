r"""Fourier-domain PTA likelihood (Valtolina & van Haasteren 2025, "VvH25"), with the
Gaussian-mixture and normalizing-flow generalizations, built on the standard discovery
likelihoods.

The analysis has two steps.

**Step 1, per pulsar.** Fix the pulsar's red-noise spectrum at a reference
:math:`\boldsymbol\eta_0` and summarize what its data say about the red-noise Fourier
coefficients :math:`\mathbf a`, with every other noise process (timing model, white
noise, ECORR, DM and other chromatic GPs, ...) marginalized:

.. math::

    p(\mathbf a \mid \delta t, \boldsymbol\eta_0) \approx
    \mathcal N(\mathbf a \mid \hat{\mathbf a}_0, \boldsymbol\Sigma_0)
    \qquad \text{(draft Eq. 12, 14)}.

:func:`summarize_pulsar` computes this from a ``PulsarLikelihood`` and returns a
:class:`FourierSummary`.

**Step 2, over the array.** Replace the reference prior
:math:`\mathcal N(\mathbf a\mid 0,\boldsymbol\varphi_0)` by the model prior
:math:`p(\mathbf a\mid\boldsymbol\eta)` (intrinsic red noise, CURN, HD, ...) and infer
:math:`\boldsymbol\eta`. A :class:`FourierSummary` doubles as a *stand-in pulsar* whose
ordinary pulsar likelihood is exactly the step-2 likelihood, so step 2 uses the
standard ``makecommongp_fourier`` / ``makeglobalgp_fourier`` / ``ArrayLikelihood``
machinery unchanged. See :class:`FourierSummary` for how, and why it works.

Non-Gaussian summaries (draft Sec. 2.3-2.5) enter as a per-pulsar correction term,
:func:`makecorrection`, on the sampled coefficients; for priors that do not couple
pulsars, the Gaussian-mixture version can also be marginalized analytically,
:func:`mixture_logL`, and its coefficient conditional drawn, :func:`mixture_conditional`.

.. note::

    "VvH25" is Valtolina & van Haasteren, Phys. Rev. D 112, 043046 (2025). Equation,
    section and appendix numbers ("draft Eq. N") refer to the draft of Tresnjic &
    Meyers, *Capturing Non-Gaussianities in the Fourier-domain based Pulsar Timing
    Array Likelihood* (version of 7 October 2026), and will change with it.
"""

import dataclasses
import warnings
from typing import Optional

import numpy as np

import jax
import jax.numpy as jnp

from . import _kernels as kernels
from . import signals
from . import utils


@dataclasses.dataclass
class FourierSummary:
    r"""Step-1 summary of one pulsar, which doubles as a stand-in pulsar for step 2.

    **What it holds.** The Gaussian summary of the pulsar's red-noise Fourier
    coefficients :math:`\mathbf a` (length :math:`n = 2N_f`) at the reference spectrum
    :math:`\boldsymbol\varphi_0 = \boldsymbol\varphi(\boldsymbol\eta_0)`,

    .. math::

        p(\mathbf a \mid \delta t, \boldsymbol\eta_0) \approx
        \mathcal N(\mathbf a \mid \hat{\mathbf a}_0, \boldsymbol\Sigma_0),

    stored as ``ahat0`` and the precision ``Sigma0_inv``; the reference prior variances
    ``phi0``; optionally the per-sample conditionals that make up the Gaussian-mixture
    generalization (``ahat_samples``, ``Sigma_samples``) and a non-Gaussian ``density``;
    and ``logL0``, the step-1 marginalized likelihood at the reference spectrum. Build one with :func:`summarize_pulsar`.

    **The stand-in pulsar.** Step 2 swaps the reference prior for the model prior. As a
    function of :math:`\mathbf a`, the summary with the reference prior divided out is a
    quadratic,

    .. math::

        \log\mathcal N(\mathbf a\mid\hat{\mathbf a}_0,\boldsymbol\Sigma_0)
        - \log\mathcal N(\mathbf a\mid 0,\boldsymbol\varphi_0)
        = -\tfrac12\,\mathbf a^\top \mathbf M\,\mathbf a + \mathbf a^\top \mathbf b_0 + \text{const},
        \qquad
        \mathbf M = \boldsymbol\Sigma_0^{-1} - \boldsymbol\varphi_0^{-1},\quad
        \mathbf b_0 = \boldsymbol\Sigma_0^{-1}\hat{\mathbf a}_0

    (``TtNT`` and ``b0``). An ordinary pulsar with residuals :math:`\mathbf y`, design
    matrix :math:`\mathbf F` and white noise :math:`\mathbf N` has exactly the same form,

    .. math::

        \log p(\mathbf y\mid\mathbf a)
        = -\tfrac12(\mathbf y-\mathbf F\mathbf a)^\top\mathbf N^{-1}(\mathbf y-\mathbf F\mathbf a) + \text{const}
        = -\tfrac12\,\mathbf a^\top(\mathbf F^\top\mathbf N^{-1}\mathbf F)\,\mathbf a
          + \mathbf a^\top(\mathbf F^\top\mathbf N^{-1}\mathbf y) + \text{const},

    and the step-2 likelihood only ever sees the data through
    :math:`\mathbf F^\top\mathbf N^{-1}\mathbf F` and :math:`\mathbf F^\top\mathbf N^{-1}\mathbf y`
    (plus constants). So it is enough to find stand-in data with

    .. math::

        \mathbf F^\top\mathbf N^{-1}\mathbf F = \mathbf M, \qquad
        \mathbf F^\top\mathbf N^{-1}\mathbf y = \mathbf b_0 .

    With unit noise this asks for a "square root" of :math:`\mathbf M`. Diagonalize
    :math:`\mathbf M = \mathbf U\boldsymbol\Lambda\mathbf U^\top` and take

    .. math::

        \mathbf F = \boldsymbol\Lambda^{1/2}\mathbf U^\top
        \;\Rightarrow\; \mathbf F^\top\mathbf F = \mathbf U\boldsymbol\Lambda\mathbf U^\top = \mathbf M,
        \qquad
        \mathbf y = \boldsymbol\Lambda^{-1/2}\mathbf U^\top\mathbf b_0
        \;\Rightarrow\; \mathbf F^\top\mathbf y
        = \mathbf U\boldsymbol\Lambda^{1/2}\boldsymbol\Lambda^{-1/2}\mathbf U^\top\mathbf b_0 = \mathbf b_0 .

    In pulsar terms, each stand-in "TOA" :math:`y_i` is one eigen-combination
    :math:`\mathbf u_i^\top\mathbf a` of Fourier coefficients, measured with precision
    :math:`\lambda_i`. Everything after that is ordinary discovery: the stand-in gets the
    same red-noise / common / HD GPs as a real pulsar (their bases served by
    :func:`summarybasis`), and ``logL`` marginalizes the coefficients (draft Eq. 18) while
    ``clogL`` samples them (draft Eq. 16), with decentering, anisotropic ORFs, mixed arrays of
    summaries and time-domain pulsars, etc. all working as usual. A Cholesky factor of
    :math:`\mathbf M` would do as well; the eigendecomposition is used because it handles
    the two awkward cases cleanly:

    - :math:`\lambda_i \approx 0`: a combination of coefficients the data do not constrain
      at all -- for instance the lowest frequencies of a pulsar shorter than the array
      span, which the timing model absorbs. Its row of :math:`\mathbf F` would be zero
      and :math:`y_i = 0/0`, so it is dropped (eigenvalues below ``rtol`` times the
      largest are roundoff). :math:`\mathbf b_0` should have no component along it; a
      warning is raised if it does by enough to matter.
    - :math:`\lambda_i < 0`: :math:`\mathbf M` has no real square root. This happens when
      the reference prior is *narrower* than the summary in some direction, i.e.
      :math:`\boldsymbol\eta_0` is too quiet. The prior swap is importance sampling from
      :math:`\boldsymbol\varphi_0` to :math:`\boldsymbol\varphi(\boldsymbol\eta)`, which
      needs :math:`\boldsymbol\varphi_0` broader than anything step 2 explores; an error
      is raised. A louder :math:`\boldsymbol\eta_0` is always valid (just less efficient).

    **Normalization.** The stand-in's white noise is :math:`\mathbf N = s\,\mathbf I` (with
    :math:`\mathbf F` and :math:`\mathbf y` scaled by :math:`\sqrt s`), which leaves the two
    arrays above unchanged but adds :math:`-\tfrac r2\log s` to the log-likelihood
    (:math:`r` = number of kept eigenvalues). The scale :math:`s` (``noise``) is chosen so
    that the stand-in's likelihoods are *exactly*, not just up to a constant,

    .. math::

        \texttt{logL}(\boldsymbol\eta)
        = \log p(\delta t\mid\boldsymbol\eta) - \log p(\delta t\mid\boldsymbol\eta_0)
        = \log\!\int\! d\mathbf a\;
          \mathcal N(\mathbf a\mid\hat{\mathbf a}_0,\boldsymbol\Sigma_0)\,
          \frac{\mathcal N(\mathbf a\mid 0,\boldsymbol\varphi(\boldsymbol\eta))}
               {\mathcal N(\mathbf a\mid 0,\boldsymbol\varphi_0)},

    and correspondingly for ``clogL`` -- in discovery's convention, which omits factors
    of :math:`2\pi`, the same as for real pulsars. The step-1 marginalized likelihood at
    the reference spectrum, :math:`\log p(\delta t\mid\boldsymbol\eta_0)`, is ``logL0``: known exactly at fixed
    noise (then summaries plus :math:`\sum_p` ``logL0`` reproduce the time-domain array
    likelihood, constant included), and 0 when step 1 marginalized noise parameters by
    sampling (step-2 numbers are then relative to :math:`\boldsymbol\eta_0`, which is
    exact for Bayes factors between step-2 models built on the same summaries). It is
    kept separate because it is far too large (:math:`\sim 10^5` for a real pulsar) to
    fold into :math:`s`.

    Examples
    --------
    Step 1 at fixed white noise, with DM marginalized as a GP with fixed
    hyperparameters (pass arrays of samples for any parameter to marginalize over it
    by sampling instead):

    >>> import discovery as ds
    >>> from discovery import fourierpta as fpta
    >>> ds.config(kernels='metamath')
    >>> T = ds.getspan(psrs)
    >>> psl = ds.PulsarLikelihood([psr.residuals,
    ...                            ds.makenoise_measurement(psr, psr.noisedict),
    ...                            ds.makegp_timing(psr, svd=True),
    ...                            ds.makegp_fourier(psr, ds.powerlaw, 30, T=T,
    ...                                              fourierbasis=ds.dmfourierbasis, name='dm_gp'),
    ...                            ds.makegp_fourier(psr, ds.powerlaw, 30, T=T, name='red_noise')])
    >>> eta0 = {f'{psr.name}_red_noise_log10_A': -11.1, f'{psr.name}_red_noise_gamma': 3.0}
    >>> summary = fpta.summarize_pulsar(psr, psl, {**psr.noisedict, **dm_params, **eta0})

    The stand-in pulsar is just data:

    >>> summary.residuals.shape, summary.Fmat.shape     # (r,), (r, n); r = n unless directions dropped
    >>> np.allclose(summary.Fmat.T @ summary.Fmat / summary.noise, summary.TtNT)
    True

    Step 2, single pulsar (Gaussian summary, coefficients marginalized):

    >>> rn = ds.makegp_fourier(summary, ds.powerlaw, 30, T=T, fourierbasis=fpta.summarybasis,
    ...                        name='red_noise')
    >>> psl2 = ds.PulsarLikelihood([summary.residuals, fpta.makenoise_summary(summary), rn])
    >>> psl2.logL(params)          # log p(dt | eta) - log p(dt | eta0)

    Step 2, array with Hellings-Downs (any mix of summaries and real pulsars):

    >>> pls = [ds.PulsarLikelihood([s.residuals, fpta.makenoise_summary(s)]) for s in summaries]
    >>> irn = ds.makecommongp_fourier(summaries, ds.powerlaw, 30, T,
    ...                               fourierbasis=fpta.summarybasis, name='red_noise')
    >>> gw = ds.makeglobalgp_fourier(summaries, ds.powerlaw, ds.hd_orf, 14, T,
    ...                              fourierbasis=fpta.summarybasis, name='gw')
    >>> like = ds.ArrayLikelihood(pls, commongp=irn, globalgp=gw, decenter=True)
    >>> like.clogL(params)         # (log-density, physical coefficients)

    With a non-Gaussian summary density (here the Gaussian mixture of the step-1
    conditionals), add the correction term to each stand-in pulsar:

    >>> summary.density = summary.mixture(512)
    >>> psl2 = ds.PulsarLikelihood([summary.residuals, fpta.makenoise_summary(summary),
    ...                             fpta.makecorrection(summary)])
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

    # log p(dt | eta0), the step-1 marginalized likelihood at the reference spectrum,
    # in discovery's convention (PulsarLikelihood.logL); 0 if unknown. Step-2
    # likelihoods are relative to it -- add it for absolute likelihoods.
    logL0: float = 0.0

    # eigenvalues of TtNT below rtol * max are roundoff, treated as zero (the
    # stand-in data then have fewer than n entries)
    rtol: float = 1e-14

    @property
    def ncoeff(self):
        r"""Number of Fourier coefficients :math:`n = 2N_f` (a sine and a cosine per frequency)."""
        return len(self.ahat0)

    @property
    def components(self):
        r"""Number of frequencies :math:`N_f`."""
        return self.ncoeff // 2

    @property
    def T(self):
        r"""Span :math:`T` of the Fourier basis (the frequencies are :math:`f_j = j/T`)."""
        return 1.0 / self.f[0]

    @property
    def b0(self):
        r""":math:`\mathbf b_0 = \boldsymbol\Sigma_0^{-1}\hat{\mathbf a}_0`, which plays the role of
        :math:`\mathbf F^\top\mathbf N^{-1}\delta t` for the stand-in pulsar."""
        return self.Sigma0_inv @ self.ahat0

    @property
    def TtNT(self):
        r""":math:`\mathbf M = \boldsymbol\Sigma_0^{-1} - \boldsymbol\varphi_0^{-1}`, which plays the
        role of :math:`\mathbf F^\top\mathbf N^{-1}\mathbf F` for the stand-in pulsar: the
        information the data add to the reference prior."""
        return self.Sigma0_inv - np.diag(1.0 / self.phi0)

    @property
    def Sigma0(self):
        r"""Summary covariance :math:`\boldsymbol\Sigma_0` (the inverse of ``Sigma0_inv``)."""
        return np.linalg.inv(self.Sigma0_inv)

    @property
    def L0(self):
        r""":math:`\mathbf L_0 = \operatorname{chol}(\boldsymbol\Sigma_0)`, the whitening
        :math:`\mathbf y = \mathbf L_0^{-1}(\mathbf a - \hat{\mathbf a}_0)` under which the Gaussian
        summary is :math:`\mathcal N(\mathbf y\mid 0,\mathbf I)`, and in which non-Gaussian
        densities ``density`` are defined."""
        return np.linalg.cholesky(self.Sigma0)

    def mixture(self, K=None):
        r"""The Gaussian-mixture generalization of the summary (draft Eq. 19, A8),

        .. math::

            q(\mathbf a) = \frac1K\sum_{k=1}^K
            \mathcal N\big(\mathbf a\mid\hat{\mathbf a}(\boldsymbol\theta_k),\boldsymbol\Sigma(\boldsymbol\theta_k)\big),

        the equally weighted per-sample conditionals from step 1, returned as a
        :class:`GaussianMixture` in the whitened coordinates :math:`\mathbf y` (see ``L0``).
        With ``K``, use ``K`` samples spaced evenly through the step-1 samples; the
        non-Gaussian tails are carried by rare components, so ``K`` needs checking."""
        ahat, Sigma = self._components(K)
        Linv0 = np.linalg.inv(self.L0)

        mu = (ahat - self.ahat0) @ Linv0.T
        C = Linv0 @ Sigma @ Linv0.T
        return GaussianMixture(mu, np.linalg.cholesky(0.5 * (C + np.swapaxes(C, 1, 2))))

    def _components(self, K=None):
        """The per-sample conditionals (ahat_k, Sigma_k): all of them, or K spaced evenly
        through the step-1 samples."""
        if self.ahat_samples is None:
            raise ValueError(f"{self.name}: no per-sample conditionals; summarize over noise samples first.")

        idx = np.arange(len(self.ahat_samples)) if K is None else \
              np.linspace(0, len(self.ahat_samples) - 1, K).astype(int)
        return self.ahat_samples[idx], self.Sigma_samples[idx]

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
            F, y = np.sqrt(lam)[:, None] * U.T, proj / np.sqrt(lam)

            # Normalization. With N = s I, the stand-in's clogL is
            #   -1/2 |y - F a|^2 - r/2 log s + log N(a | 0, phi),
            # and we want -1/2 |y - F a|^2 - r/2 log s = log N(a | ahat0, Sigma0) - log N(a | 0, phi0)
            #   = -1/2 a^T TtNT a + a^T b0 - 1/2 ahat0^T b0 - 1/2 log|Sigma0| + 1/2 log|phi0|
            # (the 2 pi's cancel in the ratio). Matching the a-independent parts:
            logdet_Sigma0 = -2.0 * np.sum(np.log(np.diag(np.linalg.cholesky(self.Sigma0_inv))))
            const = (-0.5 * self.ahat0 @ self.b0 - 0.5 * logdet_Sigma0
                     + 0.5 * np.sum(np.log(self.phi0)) + 0.5 * y @ y)
            # NB: discovery's likelihoods omit factors of 2 pi; if they gain them, the
            # stand-in's -r/2 log s becomes -r/2 log(2 pi s), so subtract log(2 pi) here
            # (test_standin_normalization and test_summaries_reproduce_time_domain_likelihood
            # will flag it).
            log_s = -2.0 * const / len(y)

            sqrt_s = np.exp(0.5 * log_s)
            self.__dict__['_standin_cache'] = (sqrt_s * F, sqrt_s * y, np.exp(log_s))
        return self.__dict__['_standin_cache']

    @property
    def Fmat(self):
        r"""Stand-in design matrix :math:`\mathbf F` (:math:`r\times n`), with
        :math:`\mathbf F^\top\mathbf N^{-1}\mathbf F = \mathbf M` (``TtNT``)."""
        return self._standin[0]

    @property
    def residuals(self):
        r"""Stand-in data :math:`\mathbf y` (length :math:`r`), with
        :math:`\mathbf F^\top\mathbf N^{-1}\mathbf y = \mathbf b_0` (``b0``)."""
        return self._standin[1]

    @property
    def noise(self):
        r"""Stand-in white-noise variance :math:`s` (:math:`\mathbf N = s\mathbf I`); it sets the
        likelihood normalization without changing :math:`\mathbf M` or :math:`\mathbf b_0`."""
        return self._standin[2]


def summarybasis(psr, components, T=None):
    r"""Fourier basis for :class:`FourierSummary` stand-in pulsars, for the
    ``fourierbasis=`` argument of ``makegp_fourier``, ``makecommongp_fourier`` and
    ``makeglobalgp_fourier``.

    For a summary, returns the frequencies and the leading :math:`2\times`
    ``components`` columns of the stand-in design matrix :math:`\mathbf F`, so that a GP
    with fewer components than the summary (e.g. a GWB on the lowest frequencies) acts
    on the matching lowest-frequency coefficients. For any other pulsar, returns the
    ordinary ``signals.fourierbasis``, so summaries and time-domain pulsars can share
    one array likelihood.

    Parameters
    ----------
    psr : FourierSummary or Pulsar
    components : int
        Number of frequencies :math:`N_f`; at most the summary's.
    T : float, optional
        Basis span; must match the summary's (checked).

    Returns
    -------
    f, df, F
        Frequencies and bin widths (one per coefficient, sine and cosine repeated) and
        the basis, as for ``signals.fourierbasis``.
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
    r"""White-noise kernel :math:`\mathbf N = s\,\mathbf I` for a :class:`FourierSummary`
    stand-in pulsar, with :math:`s` = ``psr.noise`` (which sets the likelihood
    normalization; see :class:`FourierSummary`)."""
    return kernels.NoiseMatrix1D_novar(np.full(len(psr.residuals), psr.noise))


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
    r"""Step 1: the Gaussian summary of one pulsar's Fourier coefficients for the GP
    ``gp``, with everything else in ``psl`` marginalized.

    For fixed parameters :math:`\boldsymbol\theta` (and the GP's spectrum at its
    reference value :math:`\boldsymbol\eta_0`) the coefficients are exactly Gaussian,

    .. math::

        p(\mathbf a\mid\delta t,\boldsymbol\theta,\boldsymbol\eta_0)
        = \mathcal N\big(\mathbf a\mid\hat{\mathbf a}(\boldsymbol\theta),\boldsymbol\Sigma(\boldsymbol\theta)\big),

    the pulsar likelihood's ``conditional``, restricted to the ``gp`` block (all other
    GP coefficients -- timing model, ECORR, DM, ... -- integrated out).

    - If every value in ``params`` is a scalar, the summary is that conditional
      (fixed noise).
    - If some values are arrays of length :math:`K` (samples
      :math:`\boldsymbol\theta_k`, typically from a step-1 sampler with the ``gp``
      spectrum fixed at :math:`\boldsymbol\eta_0`), the conditionals are combined by
      the law of total expectation and covariance (draft Eq. 14),

      .. math::

          \hat{\mathbf a}_0 = \mathbb E_k\big[\hat{\mathbf a}(\boldsymbol\theta_k)\big],
          \qquad
          \boldsymbol\Sigma_0 = \mathbb E_k\big[\boldsymbol\Sigma(\boldsymbol\theta_k)\big]
          + \operatorname{Cov}_k\big[\hat{\mathbf a}(\boldsymbol\theta_k)\big],

      and the per-sample pairs are kept (``ahat_samples``, ``Sigma_samples``): they are
      the components of the Gaussian-mixture summary, :meth:`FourierSummary.mixture`.

    The step-1 marginalized likelihood at the reference spectrum,
    :math:`\log p(\delta t\mid\boldsymbol\eta_0)`, is stored as ``logL0``: at fixed noise
    it is ``psl.logL(params)``; with sampled noise it is also integrated over
    :math:`\boldsymbol\theta`, which is not computed here, so
    ``logL0`` is 0 and step-2 likelihoods are relative to it (exact for comparing
    step-2 models built on the same summaries). Set ``summary.logL0`` if you have it.

    Parameters
    ----------
    psr : Pulsar
        The pulsar (for its name and sky position).
    psl : PulsarLikelihood
        Includes the GP ``gp`` (e.g. ``makegp_fourier(..., name='red_noise')``) and
        whatever is to be marginalized.
    params : dict
        Every parameter of ``psl``, with the ``gp`` spectrum at :math:`\boldsymbol\eta_0`;
        arrays of samples for parameters to average over.
    gp : str
        Name of the GP whose coefficients are summarized.

    Returns
    -------
    FourierSummary
    """
    sig, sl = _find_gp(psl, gp)

    cond = psl.conditional
    allparams, params = params, {par: jnp.asarray(params[par]) for par in cond.params}

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
        logL0 = 0.0
    else:
        mus = Sigmas = None
        ahat0, Sigma0 = map(np.asarray, jax.jit(mean_and_cov)(params))
        logL0 = float(psl.logL({par: allparams[par] for par in psl.logL.params}))

    phi0 = np.asarray(sig.Phi.getN({par: params[par] for par in sig.Phi.getN.params}))
    if phi0.ndim != 1:
        raise NotImplementedError("Only diagonal reference priors phi0 are supported.")

    f, df = np.asarray(sig.f), np.asarray(sig.df)

    Sigma0_inv = np.linalg.inv(Sigma0)
    Sigma0_inv = 0.5 * (Sigma0_inv + Sigma0_inv.T)

    return FourierSummary(name=psr.name, pos=np.asarray(psr.pos), f=f, df=df,
                          ahat0=ahat0, Sigma0_inv=Sigma0_inv, phi0=phi0,
                          ahat_samples=mus, Sigma_samples=Sigmas, logL0=logL0)


# ---------------------------------------------------------------------------
# Non-Gaussian corrections (draft Sec. 2.3-2.5)
#
# The Gaussian summary N(a | ahat0, Sigma0) is replaced by a better density q(a)
# per pulsar -- a Gaussian mixture or a normalizing flow -- defined in the whitened
# coordinates y = L0^-1 (a - ahat0). Step 2 then picks up, per pulsar,
#
#     log w(a) = log q(y) - log N(y | 0, I)                    (draft Eq. 28, 34)
#
# on top of the Gaussian-summary likelihood above (the Jacobians |L0| cancel). A
# density is any object with a `log_prob(y)` method: a `GaussianMixture`, or a
# trained flowjax distribution; `makecorrection` turns it into a term of the
# summary's PulsarLikelihood.
# ---------------------------------------------------------------------------


class GaussianMixture:
    r"""Gaussian mixture

    .. math::

        q(\mathbf y) = \sum_{k=1}^K w_k\,
        \mathcal N\big(\mathbf y\mid\boldsymbol\mu_k,\mathbf L_k\mathbf L_k^\top\big),
        \qquad \sum_k w_k = 1,

    usable as a :class:`FourierSummary` ``density``. :meth:`FourierSummary.mixture` builds
    the equally weighted mixture of the step-1 conditionals in the whitened coordinates
    :math:`\mathbf y`; :func:`reduce_mixture` merges it into fewer, unequally weighted
    components.

    Parameters
    ----------
    mu : array, shape (K, d)
        Component means :math:`\boldsymbol\mu_k`.
    L : array, shape (K, d, d)
        Lower Cholesky factors :math:`\mathbf L_k` of the component covariances.
    weights : array, shape (K,), optional
        Component weights :math:`w_k` (normalized here); equal if omitted.
    """

    def __init__(self, mu, L, weights=None):
        self.mu = jnp.asarray(mu)                                       # (K, d)
        self.L = jnp.asarray(L)                                         # (K, d, d) lower
        self.Linv = jnp.linalg.inv(self.L)
        self.logdet = jnp.sum(jnp.log(jnp.diagonal(self.L, axis1=1, axis2=2)), axis=1)   # log |L_k|
        w = np.full(self.mu.shape[0], 1.0) if weights is None else np.asarray(weights, dtype=float)
        self.logw = jnp.asarray(np.log(w / w.sum()))                   # log w_k

    @property
    def K(self):
        """Number of components."""
        return self.mu.shape[0]

    @property
    def weights(self):
        r"""Component weights :math:`w_k`."""
        return jnp.exp(self.logw)

    def log_prob(self, y):
        r""":math:`\log q(\mathbf y)`. The mixture is a sum of probabilities, so the log is a
        log-sum-exp over the components' weighted log-densities
        :math:`\log w_k - \tfrac12|\mathbf L_k^{-1}(\mathbf y-\boldsymbol\mu_k)|^2 - \log|\mathbf L_k|`
        (computed stably), minus the shared :math:`\tfrac d2\log 2\pi`."""
        z = jnp.einsum('kij,kj->ki', self.Linv, y - self.mu)
        return (jax.scipy.special.logsumexp(self.logw - 0.5 * jnp.sum(z**2, axis=1) - self.logdet)
                - 0.5 * y.shape[-1] * jnp.log(2 * jnp.pi))

    def sample(self, key, n):
        """``n`` draws from the mixture: a component by weight, then a Gaussian draw from it."""
        kc, kz = jax.random.split(key)
        k = jax.random.categorical(kc, self.logw, shape=(n,))
        z = jax.random.normal(kz, (n, self.mu.shape[1]))
        return self.mu[k] + jnp.einsum('nij,nj->ni', self.L[k], z)


def _merge_cost(w, mu, C, logdetC, i, idx):
    """Runnalls' upper bound on the KL cost of merging component i with each of idx:
    B = 1/2 [(w_i + w_j) log|C_ij| - w_i log|C_i| - w_j log|C_j|], with C_ij the
    moment-preserving merge."""
    wi, wj = w[i], w[idx]
    wij = wi + wj
    dmu = mu[idx] - mu[i]
    Cij = ((wi * C[i])[None] + wj[:, None, None] * C[idx]) / wij[:, None, None] \
          + (wi * wj / wij**2)[:, None, None] * dmu[:, :, None] * dmu[:, None, :]
    logdetCij = 2 * np.sum(np.log(np.diagonal(np.linalg.cholesky(Cij), axis1=1, axis2=2)), axis=1)
    return 0.5 * (wij * logdetCij - wi * logdetC[i] - wj * logdetC[idx])


def reduce_mixture(q, K, method='runnalls'):
    r"""Reduce a Gaussian mixture to ``K`` components (the reduction announced in
    draft Sec. 2.3).

    ``method='runnalls'`` (Runnalls 2007, IEEE Trans. Aerosp. Electron. Syst. 43, 989;
    the method used by A. Tresnjic) greedily merges the pair of components with the smallest upper bound on the
    Kullback-Leibler cost of merging,

    .. math::

        B_{ij} = \tfrac12\big[(w_i + w_j)\log|\mathbf C_{ij}| - w_i\log|\mathbf C_i| - w_j\log|\mathbf C_j|\big],

    replacing them by the single Gaussian with the same total weight, mean and
    covariance,

    .. math::

        w_{ij} = w_i + w_j,\quad
        \boldsymbol\mu_{ij} = \frac{w_i\boldsymbol\mu_i + w_j\boldsymbol\mu_j}{w_{ij}},\quad
        \mathbf C_{ij} = \frac{w_i\mathbf C_i + w_j\mathbf C_j}{w_{ij}}
          + \frac{w_i w_j}{w_{ij}^2}(\boldsymbol\mu_i-\boldsymbol\mu_j)(\boldsymbol\mu_i-\boldsymbol\mu_j)^\top .

    Each merge preserves the mixture's overall mean and covariance, so the reduced
    mixture always has the same first two moments; what is lost is shape (tails,
    skewness). The Kullback-Leibler divergence is unchanged by affine changes of
    coordinates, so the reduction can be done directly in the whitened coordinates.
    Check the loss with :func:`mixture_kl`.

    Parameters
    ----------
    q : GaussianMixture
    K : int
        Number of components to keep.
    method : {'runnalls'}
        Reduction method.

    Returns
    -------
    GaussianMixture
        With unequal weights.
    """
    if method == 'runnalls':
        return _reduce_runnalls(q, K)
    raise ValueError(f"Unknown mixture reduction method '{method}'.")


def _reduce_runnalls(q, K):
    w = np.asarray(q.weights, dtype=float).copy()
    mu = np.asarray(q.mu, dtype=float).copy()
    L = np.asarray(q.L, dtype=float)
    C = L @ np.swapaxes(L, 1, 2)
    logdetC = 2 * np.asarray(q.logdet, dtype=float).copy()
    alive = np.ones(len(w), dtype=bool)

    # cost[i, j] for j > i, i.e. each pair once; recomputed for a merged component only
    n = len(w)
    cost = np.full((n, n), np.inf)
    for i in range(n - 1):
        cost[i, i + 1:] = _merge_cost(w, mu, C, logdetC, i, np.arange(i + 1, n))

    for _ in range(n - K):
        i, j = np.unravel_index(np.argmin(cost), cost.shape)
        wij = w[i] + w[j]
        dmu = mu[i] - mu[j]
        C[i] = (w[i] * C[i] + w[j] * C[j]) / wij + (w[i] * w[j] / wij**2) * np.outer(dmu, dmu)
        mu[i] = (w[i] * mu[i] + w[j] * mu[j]) / wij
        w[i] = wij
        logdetC[i] = 2 * np.sum(np.log(np.diag(np.linalg.cholesky(C[i]))))

        alive[j] = False
        cost[j, :] = cost[:, j] = np.inf
        others = np.flatnonzero(alive)
        others = others[others != i]
        if len(others):
            c = _merge_cost(w, mu, C, logdetC, i, others)
            lo, hi = others < i, others > i
            cost[others[lo], i] = c[lo]
            cost[i, others[hi]] = c[hi]

    keep = np.flatnonzero(alive)
    return GaussianMixture(mu[keep], np.linalg.cholesky(C[keep]), weights=w[keep])


def mixture_kl(p, q, key, n=10000):
    r"""Monte Carlo estimate of the Kullback-Leibler divergence

    .. math::

        D_\mathrm{KL}(p\,\|\,q) = \mathbb E_{\mathbf y\sim p}\big[\log p(\mathbf y) - \log q(\mathbf y)\big]

    from ``n`` draws of ``p``, e.g. between a full mixture and a reduced one
    (:func:`reduce_mixture`).

    It is the average, over where ``p`` puts its probability, of how much lower the log
    density is under ``q`` than under ``p``. With ``p`` the full summary density and ``q``
    its replacement, the correction term (:func:`makecorrection`) changes this pulsar's
    log-likelihood by exactly :math:`\log q - \log p` at each :math:`\mathbf a`, so this is
    the log-likelihood lost on average at the coefficients the data favor, in nats. That is
    why ``p`` comes first: it penalizes ``q`` most for missing tails that ``p`` has.

    What to aim for:

    - :math:`\lesssim 0.01`: negligible; the reduced mixture is as good as the full one.
    - :math:`\sim 0.1`: borderline; compare posteriors, or :func:`mixture_logL` before and
      after the reduction over the prior range.
    - :math:`\gtrsim 1`: the reduction has lost real structure (often the tails); keep more
      components.

    It is per pulsar, and the errors add over the pulsars in an array, so the sum over
    pulsars should stay small too. An estimate within a few standard errors of zero is
    consistent with no loss at all. For J1738+0333 (tutorial), reducing 512 components to
    16 gives a value consistent with 0.

    Parameters
    ----------
    p : GaussianMixture
        The reference density, sampled from (e.g. the full ``summary.mixture(K)``).
    q : object with ``log_prob``
        The approximation to it (e.g. ``reduce_mixture(p, K_reduced)``), in the same
        whitened coordinates.
    key : jax.random.PRNGKey
        Random key for the draws.
    n : int, optional
        Number of draws (default 10000); the standard error falls as :math:`1/\sqrt n`.

    Returns
    -------
    estimate : float
        The Monte Carlo estimate of :math:`D_\mathrm{KL}(p\,\|\,q)`, in nats (always
        :math:`\ge 0` in expectation).
    error : float
        Its standard error.
    """
    y = p.sample(key, n)
    d = np.asarray(jax.vmap(p.log_prob)(y) - jax.vmap(q.log_prob)(y))
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(n))


class SummaryCorrection(utils.CoefficientTerm):
    r"""The non-Gaussian correction :math:`\log w(\mathbf a)` of one :class:`FourierSummary`,
    as a coefficient term of its stand-in ``PulsarLikelihood``. Build it with
    :func:`makecorrection`."""

    def __init__(self, summary, density):
        self.n = summary.ncoeff
        self.ahat0 = jnp.asarray(summary.ahat0)
        self.Linv0 = jnp.asarray(np.linalg.inv(summary.L0))
        self.density = density

    def logL(self, cs):
        r""":math:`\log w(\mathbf a)` at the pulsar's physical GP coefficients ``cs``
        ({name: vector}), with :math:`\mathbf a = \sum_g [\mathbf c_g, 0, \ldots]`."""
        # every GP on a stand-in pulsar covers the summary's lowest frequencies, so
        # the summary coefficients are the sum of the GP blocks, a = sum_g [c_g, 0...]
        a = sum(jnp.pad(c, (0, self.n - c.shape[0])) for c in cs.values())
        y = self.Linv0 @ (a - self.ahat0)
        return self.density.log_prob(y) + 0.5 * y @ y + 0.5 * self.n * jnp.log(2 * jnp.pi)


def makecorrection(summary, density=None):
    r"""The non-Gaussian correction term for a :class:`FourierSummary` stand-in pulsar,

    .. math::

        \log w(\mathbf a) = \log q(\mathbf y) - \log\mathcal N(\mathbf y\mid 0,\mathbf I),
        \qquad \mathbf y = \mathbf L_0^{-1}(\mathbf a - \hat{\mathbf a}_0)
        \qquad \text{(draft Eq. 28, 34)},

    which turns the Gaussian-summary target into the one with summary density
    :math:`q` = ``density`` (default ``summary.density``): the Gaussian
    :math:`\mathcal N(\mathbf a\mid\hat{\mathbf a}_0,\boldsymbol\Sigma_0)` already in the
    stand-in likelihood is divided out and :math:`q` multiplied in. The whitening
    Jacobian :math:`|\mathbf L_0|` cancels in the ratio.

    Include it among the stand-in pulsar's components,

    >>> ds.PulsarLikelihood([s.residuals, fpta.makenoise_summary(s), ..., fpta.makecorrection(s)])

    and ``clogL`` -- of the pulsar, or of any ``ArrayLikelihood`` containing it -- adds
    it at the sampled *physical* coefficients (after any decentering, whose Jacobian
    is accounted for separately). The marginalized ``logL`` (and ``conditional``) of a
    likelihood containing it raise: a non-Gaussian correction cannot be integrated out
    analytically, except for the Gaussian mixture without inter-pulsar correlations
    (:func:`mixture_logL`, :func:`mixture_conditional`).
    """
    density = summary.density if density is None else density
    if density is None:
        raise ValueError(f"{summary.name}: no density to correct with; set summary.density or pass one.")

    return SummaryCorrection(summary, density)


def mixture_logL(summaries, commongp, K=None):
    r"""Marginalized step-2 likelihood with the Gaussian-mixture summary (draft Eq. 21),
    for priors that do not correlate pulsars (SPNA, IRN, CURN), where it factorizes
    over pulsars.

    Per pulsar, each component :math:`k` is a Gaussian summary
    :math:`\mathcal N(\mathbf a\mid\hat{\mathbf a}_k,\boldsymbol\Sigma_k)`. Read as a
    posterior under the reference prior, dividing out that prior leaves the likelihood,
    :math:`p_k(\delta t\mid\mathbf a)\propto\mathcal N(\mathbf a\mid\hat{\mathbf a}_k,\boldsymbol\Sigma_k)
    /\mathcal N(\mathbf a\mid 0,\boldsymbol\varphi_0)`. Integrating it against the step-2
    prior (prior swap :math:`\boldsymbol\varphi_0\to\boldsymbol\varphi(\boldsymbol\eta)`)
    gives the component's marginalized likelihood at :math:`\boldsymbol\eta`, relative
    to the reference spectrum,

    .. math::

        \mathcal L_k(\boldsymbol\eta) = \frac{p_k(\delta t\mid\boldsymbol\eta)}{p_k(\delta t\mid\boldsymbol\eta_0)}
        = \int \mathcal N(\mathbf a\mid\hat{\mathbf a}_k,\boldsymbol\Sigma_k)\,
          \frac{\mathcal N(\mathbf a\mid 0,\boldsymbol\varphi)}{\mathcal N(\mathbf a\mid 0,\boldsymbol\varphi_0)}\,
          d\mathbf a ,

    which is analytic:

    .. math::

        \log\mathcal L_k = \tfrac12\mathbf b_k^\top\mathbf P_k^{-1}\mathbf b_k
                  - \tfrac12\hat{\mathbf a}_k^\top\mathbf b_k
                  - \tfrac12\log|\boldsymbol\Sigma_k| - \tfrac12\log|\mathbf P_k|
                  - \tfrac12\log|\boldsymbol\varphi| + \tfrac12\log|\boldsymbol\varphi_0|,
        \qquad
        \mathbf P_k = \boldsymbol\Sigma_k^{-1} + \boldsymbol\varphi^{-1} - \boldsymbol\varphi_0^{-1},\quad
        \mathbf b_k = \boldsymbol\Sigma_k^{-1}\hat{\mathbf a}_k .

    The pulsar contributes :math:`\operatorname{logsumexp}_k (\log w_k + \log\mathcal L_k)` (the
    mixture's marginalized likelihood relative to the reference spectrum) plus ``logL0`` (the step-1
    marginalized likelihood at the reference spectrum), normalized like the stand-in likelihoods of
    :class:`FourierSummary`. Each call costs one Cholesky factorization per component.
    With HD (which couples pulsars) use ``clogL`` with :func:`makecorrection` terms
    instead.

    Parameters
    ----------
    summaries : list of FourierSummary
        Summaries with per-sample conditionals (``ahat_samples``, ``Sigma_samples``).
    commongp : VariableGP
        The step-2 prior as a single common GP over the summaries (``makecommongp_fourier``);
        only its prior spectrum is used. For intrinsic red noise plus a common process
        (CURN), build the spectrum with ``make_combined_crn``, as for time-domain arrays.
    K : int, optional
        Use ``K`` step-1 conditionals spaced evenly through the samples
        (:meth:`FourierSummary.mixture`). If omitted, a summary whose ``density`` is a
        :class:`GaussianMixture` (e.g. reduced with :func:`reduce_mixture`) uses that, and
        otherwise all its step-1 conditionals.
    """
    terms = _mixture_terms(summaries, commongp, K)
    logL0 = sum(s.logL0 for s in summaries)

    def loglike(params):
        logL_comp, _, _ = terms(params)
        return jnp.sum(jax.scipy.special.logsumexp(logL_comp, axis=1)) + logL0
    loglike.params = terms.params

    return loglike


def _mixture_terms(summaries, commongp, K=None):
    """Per pulsar and component, at the step-2 spectrum: log w_k + log L_k, with L_k component k's
    marginalized likelihood relative to the reference spectrum; the Cholesky factor of P_k; and P_k^-1 b_k
    (see mixture_logL and mixture_conditional)."""
    phi = commongp.Phi.getN

    comps = []
    for s in summaries:
        q = s.density if (K is None and isinstance(s.density, GaussianMixture)) else s.mixture(K)

        # components back in the coefficients a = ahat0 + L0 y: mean ahat0 + L0 mu_k,
        # covariance with Cholesky factor L0 L_k
        L0 = s.L0
        ahat = s.ahat0 + np.asarray(q.mu) @ L0.T
        L = L0 @ np.asarray(q.L)
        Linv = np.linalg.inv(L)
        Sigma_inv = np.swapaxes(Linv, 1, 2) @ Linv
        b = np.einsum('kij,kj->ki', Sigma_inv, ahat)
        logdet = 2 * np.sum(np.log(np.diagonal(L, axis1=1, axis2=2)), axis=1)
        const = (-0.5 * np.einsum('ki,ki->k', ahat, b) - 0.5 * logdet
                 + 0.5 * np.sum(np.log(s.phi0)) + np.asarray(q.logw))
        comps.append((Sigma_inv - np.diag(1.0 / s.phi0), b, const))

    TtNT, b, const = (jnp.asarray(np.array(x)) for x in zip(*comps))    # (npsr, K, n, n), (npsr, K, n), (npsr, K)

    def terms(params):
        phis = jnp.asarray(phi(params))                                   # (npsr, n)
        P = TtNT + jax.vmap(jnp.diag)(1.0 / phis)[:, None, :, :]
        cf = jnp.linalg.cholesky(P)
        x = jax.scipy.linalg.cho_solve((cf, True), b[..., None])[..., 0]
        # log w_k + log L_k: weighted per-component log-likelihoods, relative to eta0
        logL_comp = (const + 0.5 * jnp.sum(b * x, axis=-1)
                - jnp.sum(jnp.log(jnp.diagonal(cf, axis1=-2, axis2=-1)), axis=-1)
                - 0.5 * jnp.sum(jnp.log(phis), axis=-1)[:, None])
        return logL_comp, cf, x
    terms.params = phi.params

    return terms


def mixture_conditional(summaries, commongp, K=None):
    r"""Conditional distribution of the Fourier coefficients under the Gaussian-mixture
    summary, :math:`p(\mathbf a\mid\delta t,\boldsymbol\eta)`, for the same priors as
    :func:`mixture_logL` (no correlations between pulsars), where it factorizes over pulsars.

    Swapping the reference prior for the step-2 prior turns each summary component into
    a Gaussian times :math:`\mathcal L_k(\boldsymbol\eta)`, the component's marginalized
    likelihood relative to the reference spectrum (see :func:`mixture_logL`),

    .. math::

        \mathcal N(\mathbf a\mid\hat{\mathbf a}_k,\boldsymbol\Sigma_k)\,
        \frac{\mathcal N(\mathbf a\mid 0,\boldsymbol\varphi)}{\mathcal N(\mathbf a\mid 0,\boldsymbol\varphi_0)}
        = \mathcal L_k\,\mathcal N(\mathbf a\mid\mathbf m_k,\mathbf P_k^{-1}),
        \qquad \mathbf m_k = \mathbf P_k^{-1}\mathbf b_k ,

    so the conditional is again a mixture, with weights updated by how strongly each
    component's marginalized likelihood favors the spectrum :math:`\boldsymbol\eta`:

    .. math::

        p(\mathbf a\mid\delta t,\boldsymbol\eta)
        = \sum_k \tilde w_k\,\mathcal N(\mathbf a\mid\mathbf m_k,\mathbf P_k^{-1}),
        \qquad \tilde w_k = \frac{w_k\mathcal L_k}{\sum_j w_j\mathcal L_j} .

    With one component this is the stand-in pulsar's ``PulsarLikelihood.conditional``.

    It is conditional on the step-2 hyperparameters :math:`\boldsymbol\eta`; evaluate it at
    draws of :math:`\boldsymbol\eta` from the step-2 posterior to marginalize over them.
    What step 1 integrated out (timing model, chromatic noise) or sampled (white noise
    over the per-sample conditionals) is already marginalized.

    Returns ``cond(params) -> (logw, m, cf)``, per pulsar and component: the log weights
    :math:`\log\tilde w_k` (npsr, K), the means :math:`\mathbf m_k` (npsr, K, n), and the
    lower Cholesky factors of the precisions :math:`\mathbf P_k` (npsr, K, n, n).
    Arguments as in :func:`mixture_logL`.
    """
    terms = _mixture_terms(summaries, commongp, K)

    def cond(params):
        logL_comp, cf, m = terms(params)
        return logL_comp - jax.scipy.special.logsumexp(logL_comp, axis=1, keepdims=True), m, cf
    cond.params = terms.params

    return cond


def sample_mixture_conditional(summaries, commongp, K=None):
    r"""Draws from :func:`mixture_conditional`: per pulsar, a component :math:`k` with
    probability :math:`\tilde w_k`, then :math:`\mathbf a = \mathbf m_k + \mathbf L_k^{-\top}\mathbf z`
    with :math:`\mathbf P_k = \mathbf L_k\mathbf L_k^\top` and :math:`\mathbf z` standard normal.

    Returns ``sample(key, params) -> (key, {coefficient name: a_p})``, like
    ``PulsarLikelihood.sample_conditional``; the names are the ``commongp`` coefficient
    names. To see a draw as a time series, multiply by the real pulsar's Fourier basis
    on the same frequencies (the stand-in's own ``Fmat`` is not a time-domain basis).
    """
    cond = mixture_conditional(summaries, commongp, K)
    names = list(commongp.index)

    def sample(key, params):
        logw, m, cf = cond(params)
        key, kc, kz = jax.random.split(key, 3)
        k = jax.random.categorical(kc, logw, axis=1)                      # (npsr,)
        rows = jnp.arange(m.shape[0])
        z = jax.random.normal(kz, (m.shape[0], m.shape[2]))               # (npsr, n)
        a = m[rows, k] + jax.vmap(
            lambda L, zi: jax.scipy.linalg.solve_triangular(L.T, zi, lower=False))(cf[rows, k], z)
        return key, dict(zip(names, a))
    sample.params = cond.params

    return sample
