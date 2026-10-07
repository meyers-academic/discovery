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
:func:`mixture_logL`.

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
    and the step-1 log-evidence ``logL0``. Build one with :func:`summarize_pulsar`.

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
    of :math:`2\pi`, the same as for real pulsars. The step-1 log-evidence
    :math:`\log p(\delta t\mid\boldsymbol\eta_0)` is ``logL0``: known exactly at fixed
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

    # log p(dt | eta0), the step-1 log-evidence at the reference spectrum, in
    # discovery's convention (PulsarLikelihood.logL); 0 if unknown. Step-2
    # likelihoods are relative to it -- add it for absolute evidences.
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
            # (test_standin_normalization and test_summaries_reproduce_time_domain_evidence
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

    The step-1 log-evidence :math:`\log p(\delta t\mid\boldsymbol\eta_0)` is stored as
    ``logL0``: at fixed noise it is ``psl.logL(params)``; with sampled noise it is an
    evidence integral over :math:`\boldsymbol\theta` that is not computed here, so
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
    r"""Equally weighted Gaussian mixture

    .. math::

        q(\mathbf y) = \frac1K\sum_{k=1}^K
        \mathcal N\big(\mathbf y\mid\boldsymbol\mu_k,\mathbf L_k\mathbf L_k^\top\big),

    usable as a :class:`FourierSummary` ``density`` (built by
    :meth:`FourierSummary.mixture` in the whitened coordinates :math:`\mathbf y`).

    Parameters
    ----------
    mu : array, shape (K, d)
        Component means :math:`\boldsymbol\mu_k`.
    L : array, shape (K, d, d)
        Lower Cholesky factors :math:`\mathbf L_k` of the component covariances.
    """

    def __init__(self, mu, L):
        self.mu = jnp.asarray(mu)                                       # (K, d)
        self.L = jnp.asarray(L)                                         # (K, d, d) lower
        self.Linv = jnp.linalg.inv(self.L)
        self.logdet = jnp.sum(jnp.log(jnp.diagonal(self.L, axis1=1, axis2=2)), axis=1)   # log |L_k|

    @property
    def K(self):
        return self.mu.shape[0]

    def log_prob(self, y):
        r""":math:`\log q(\mathbf y)`. The mixture is a sum of probabilities, so the log is a
        log-sum-exp over the components' log-densities
        :math:`-\tfrac12|\mathbf L_k^{-1}(\mathbf y-\boldsymbol\mu_k)|^2 - \log|\mathbf L_k|`
        (computed stably), minus :math:`\log K` and the shared
        :math:`\tfrac d2\log 2\pi`."""
        z = jnp.einsum('kij,kj->ki', self.Linv, y - self.mu)
        return (jax.scipy.special.logsumexp(-0.5 * jnp.sum(z**2, axis=1) - self.logdet)
                - jnp.log(self.K) - 0.5 * y.shape[-1] * jnp.log(2 * jnp.pi))


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
    is accounted for separately). The marginalized ``logL`` does not see it: a
    non-Gaussian correction cannot be integrated out analytically, except for the
    Gaussian mixture without inter-pulsar correlations (:func:`mixture_logL`).
    """
    density = summary.density if density is None else density
    if density is None:
        raise ValueError(f"{summary.name}: no density to correct with; set summary.density or pass one.")

    return SummaryCorrection(summary, density)


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
    r"""Marginalized step-2 likelihood with the Gaussian-mixture summary (draft Eq. 21),
    for priors that do not correlate pulsars (SPNA, IRN, CURN), where it factorizes
    over pulsars.

    Per pulsar, each component :math:`k` is a Gaussian summary
    :math:`\mathcal N(\mathbf a\mid\hat{\mathbf a}_k,\boldsymbol\Sigma_k)`, whose evidence
    against the prior swap :math:`\boldsymbol\varphi_0\to\boldsymbol\varphi(\boldsymbol\eta)`
    is analytic:

    .. math::

        \log Z_k = \tfrac12\mathbf b_k^\top\mathbf P_k^{-1}\mathbf b_k
                  - \tfrac12\hat{\mathbf a}_k^\top\mathbf b_k
                  - \tfrac12\log|\boldsymbol\Sigma_k| - \tfrac12\log|\mathbf P_k|
                  - \tfrac12\log|\boldsymbol\varphi| + \tfrac12\log|\boldsymbol\varphi_0|,
        \qquad
        \mathbf P_k = \boldsymbol\Sigma_k^{-1} + \boldsymbol\varphi^{-1} - \boldsymbol\varphi_0^{-1},\quad
        \mathbf b_k = \boldsymbol\Sigma_k^{-1}\hat{\mathbf a}_k .

    The pulsar contributes :math:`\operatorname{logsumexp}_k \log Z_k - \log K` plus its
    step-1 log-evidence ``logL0``, normalized like the stand-in likelihoods of
    :class:`FourierSummary`. Each call costs one Cholesky factorization per component.
    With HD (which couples pulsars) use ``clogL`` with :func:`makecorrection` terms
    instead.

    Parameters
    ----------
    summaries : list of FourierSummary
        Summaries with per-sample conditionals (``ahat_samples``, ``Sigma_samples``).
    commongp : VariableGP or list of VariableGP
        The step-2 prior, as common GPs (e.g. from ``makecommongp_fourier``); only their
        prior spectra are used.
    K : int, optional
        Use ``K`` components spaced evenly through the samples (default: all).
    """
    phi = _diag_prior(commongp)
    n = summaries[0].ncoeff

    comps = []
    for s in summaries:
        idx = np.arange(len(s.ahat_samples)) if K is None else \
              np.linspace(0, len(s.ahat_samples) - 1, K).astype(int)
        ahat, Sigma = s.ahat_samples[idx], s.Sigma_samples[idx]
        L = np.linalg.cholesky(Sigma)
        Linv = np.linalg.inv(L)
        Sigma_inv = np.swapaxes(Linv, 1, 2) @ Linv
        b = np.einsum('kij,kj->ki', Sigma_inv, ahat)
        logdet = 2 * np.sum(np.log(np.diagonal(L, axis1=1, axis2=2)), axis=1)
        const = (-0.5 * np.einsum('ki,ki->k', ahat, b) - 0.5 * logdet
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
        return jnp.sum(jax.scipy.special.logsumexp(logZ, axis=1)) + logL0
    loglike.params = phi.params
    logL0 = sum(s.logL0 for s in summaries)

    return loglike
