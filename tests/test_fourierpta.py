#!/usr/bin/env python3
"""Tests for discovery.fourierpta: the Fourier-domain PTA likelihood of
Valtolina & van Haasteren (2025) built on the standard likelihoods.

Step 1 (`summarize_pulsar`) is checked against the pulsar likelihood's own
conditional; step 2 (`ArrayLikelihood` over `FourierSummary` stand-in pulsars) is
checked against a direct implementation of draft Eq. 18 (marginalized) and Eq. 16
(sampled coefficients). Step-2 likelihoods agree up to an eta-independent constant,
so differences between parameter points are compared.

The direct implementations follow S. Valtolina / A. Tresnjic's reference code
(`log_fourier_likelihood`), which these were also checked against to ~1e-9.
"""

from pathlib import Path

import numpy as np
import pytest

import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

import discovery as ds
from discovery import fourierpta as fpta


NC, NGW = 5, 3
N = 2 * NC
ETA0 = {'log10_A': -13.5, 'gamma': 3.0}


@pytest.fixture(scope='module', autouse=True)
def metamath():
    ds.config(kernels='metamath')
    yield
    ds.config(kernels='matrix')


@pytest.fixture(scope='module')
def psrs():
    data_dir = Path(__file__).resolve().parent.parent / 'data'
    # J0023+0923 is shorter than the full span, so its lowest modes are degenerate
    # with the timing model: exercises the rank-deficient stand-in
    names = ['B1855+09', 'J0023+0923', 'J0030+0451']
    return [ds.Pulsar.read_feather(data_dir / f'v1p1_de440_pint_bipm2019-{nm}.feather') for nm in names]


@pytest.fixture(scope='module')
def T(psrs):
    return ds.getspan(psrs)


def _psl(psr, T, noisedict={}):
    return ds.PulsarLikelihood([psr.residuals, ds.makegp_timing(psr, svd=True),
                                ds.makenoise_measurement(psr, noisedict=noisedict, ecorr=True),
                                ds.makegp_fourier(psr, ds.powerlaw, NC, T=T, name='red_noise')])


def _eta0(psr):
    return {f'{psr.name}_red_noise_{k}': v for k, v in ETA0.items()}


@pytest.fixture(scope='module')
def summaries(psrs, T):
    return [fpta.summarize_pulsar(psr, _psl(psr, T, psr.noisedict), {**psr.noisedict, **_eta0(psr)})
            for psr in psrs]


def _rand_params(psrs, rng):
    p = {'gw_log10_A': rng.uniform(-15, -13.5), 'gw_gamma': rng.uniform(2, 5)}
    for psr in psrs:
        p[f'{psr.name}_red_noise_log10_A'] = rng.uniform(-15, -13)
        p[f'{psr.name}_red_noise_gamma'] = rng.uniform(1, 5)
    return p


def _phi_dense(summaries, orf, p):
    """Full prior covariance over all pulsars' coefficients: IRN on every mode,
    plus orf (x) S on the lowest NGW frequencies."""
    npsr, f, df = len(summaries), summaries[0].f, summaries[0].df
    S = np.asarray(ds.powerlaw(f[:2*NGW], df[:2*NGW], p['gw_log10_A'], p['gw_gamma']))
    Phi = np.zeros((npsr * N, npsr * N))
    for a, s in enumerate(summaries):
        irn = np.asarray(ds.powerlaw(f, df, p[f'{s.name}_red_noise_log10_A'], p[f'{s.name}_red_noise_gamma']))
        Phi[a*N:(a+1)*N, a*N:(a+1)*N] += np.diag(irn)
        for c in range(npsr):
            Phi[a*N:a*N+2*NGW, c*N:c*N+2*NGW] += orf[a, c] * np.diag(S)
    return Phi


def _eq18(summaries, orf, p):
    """Draft Eq. 18 up to a constant: 1/2 b^T Sigma b - 1/2 log|Sigma^-1| - 1/2 log|Phi|."""
    b = np.concatenate([s.b0 for s in summaries])
    TtNT = jax.scipy.linalg.block_diag(*[s.TtNT for s in summaries])
    Phi = _phi_dense(summaries, orf, p)
    Sigma_inv = np.asarray(TtNT) + np.linalg.inv(Phi)
    return (0.5 * b @ np.linalg.solve(Sigma_inv, b) - 0.5 * np.linalg.slogdet(Sigma_inv)[1]
            - 0.5 * np.linalg.slogdet(Phi)[1])


def _stand_ins(summaries, T):
    pls = [ds.PulsarLikelihood([s.residuals, fpta.makenoise_summary(s)]) for s in summaries]
    irn = ds.makecommongp_fourier(summaries, ds.powerlaw, NC, T, fourierbasis=fpta.summarybasis, name='red_noise')
    return pls, irn


def test_summary_matches_conditional(psrs, T, summaries):
    for psr, s in zip(psrs, summaries):
        mu, cf = _psl(psr, T, psr.noisedict).conditional({**psr.noisedict, **_eta0(psr)})
        Sigma_inv = cf[0] @ cf[0].T
        assert np.allclose(s.ahat0, mu, rtol=1e-9, atol=0)
        assert np.allclose(s.Sigma0_inv, Sigma_inv, rtol=1e-9, atol=1e-9 * np.abs(Sigma_inv).max())
        assert np.allclose(s.phi0, ds.powerlaw(s.f, s.df, **ETA0))


def test_standin_reproduces_TtNT_and_b0(summaries):
    for s in summaries:
        F, y = s.Fmat, s.residuals
        assert np.allclose(F.T @ F / s.noise, s.TtNT, atol=1e-12 * np.abs(s.TtNT).max())
        assert np.allclose(F.T @ y / s.noise, s.b0, atol=1e-8 * np.abs(s.b0).max())


def test_standin_rank_deficient():
    # data that constrain only 3 of 4 coefficient combinations (as when low
    # frequencies are degenerate with the timing model): TtNT is singular
    rng = np.random.default_rng(4)
    A = rng.normal(size=(3, 4))
    TtNT, phi0 = A.T @ A, np.full(4, 0.5)
    ahat0 = np.linalg.solve(TtNT + np.diag(1 / phi0), A.T @ rng.normal(size=3))
    s = fpta.FourierSummary(name='fake', pos=np.ones(3) / np.sqrt(3), f=np.repeat([1.0, 2.0], 2), df=np.ones(4),
                            ahat0=ahat0, Sigma0_inv=TtNT + np.diag(1 / phi0), phi0=phi0)

    assert len(s.residuals) == 3
    assert np.allclose(s.Fmat.T @ s.Fmat / s.noise, TtNT)
    assert np.allclose(s.Fmat.T @ s.residuals / s.noise, s.b0)


def test_sampled_noise_law_of_total_covariance(psrs, T):
    psr = psrs[0]
    psl = _psl(psr, T)                                 # free white noise
    rng = np.random.default_rng(0)
    K = 3
    wn = [par for par in psl.conditional.params if 'red_noise' not in par]
    samples = {par: psr.noisedict[par] + 0.05 * rng.normal(size=K) for par in wn}

    s = fpta.summarize_pulsar(psr, psl, {**samples, **_eta0(psr)})

    mus, Sigmas = [], []
    for k in range(K):
        sk = fpta.summarize_pulsar(psr, psl, {**{par: v[k] for par, v in samples.items()}, **_eta0(psr)})
        mus.append(sk.ahat0)
        Sigmas.append(sk.Sigma0)
    mus, Sigmas = np.array(mus), np.array(Sigmas)

    assert np.allclose(s.ahat0, mus.mean(axis=0), rtol=1e-8)
    assert np.allclose(s.Sigma0, Sigmas.mean(axis=0) + np.cov(mus.T), rtol=1e-6)
    assert s.ahat_samples.shape == (K, N) and s.Sigma_samples.shape == (K, N, N)


@pytest.mark.parametrize('model', ['hd', 'curn'])
def test_marginalized_matches_eq18(psrs, T, summaries, model):
    pls, irn = _stand_ins(summaries, T)
    npsr = len(summaries)

    if model == 'hd':
        gw = ds.makeglobalgp_fourier(summaries, ds.powerlaw, ds.hd_orf, NGW, T, fourierbasis=fpta.summarybasis, name='gw')
        like = ds.ArrayLikelihood(pls, commongp=irn, globalgp=gw)
        orf = np.array([[ds.hd_orf(a.pos, b.pos) for a in summaries] for b in summaries])
        topar = lambda p: p
    else:
        crn = ds.makecommongp_fourier(summaries, ds.powerlaw, NGW, T, fourierbasis=fpta.summarybasis,
                                      common=['crn_log10_A', 'crn_gamma'], name='crn')
        like = ds.ArrayLikelihood(pls, commongp=[irn, crn])
        orf = np.eye(npsr)
        topar = lambda p: {**{k: v for k, v in p.items() if not k.startswith('gw')},
                           'crn_log10_A': p['gw_log10_A'], 'crn_gamma': p['gw_gamma']}

    logL = jax.jit(like.logL)
    rng = np.random.default_rng(1)
    ps = [_rand_params(psrs, rng) for _ in range(4)]
    new = np.array([float(logL(topar(p))) for p in ps])
    ref = np.array([_eq18(summaries, orf, p) for p in ps])

    assert np.allclose(new[1:] - new[0], ref[1:] - ref[0], rtol=1e-7, atol=1e-6)


def test_sampled_coefficients_match_eq16(psrs, T, summaries):
    pls, irn = _stand_ins(summaries, T)
    gw = ds.makeglobalgp_fourier(summaries, ds.powerlaw, ds.hd_orf, NGW, T, fourierbasis=fpta.summarybasis, name='gw')
    clogL = jax.jit(ds.ArrayLikelihood(pls, commongp=irn, globalgp=gw).clogL)
    orf = np.array([[ds.hd_orf(a.pos, b.pos) for a in summaries] for b in summaries])

    def ref(p, c_irn, c_gw):
        # a_p = c_irn,p + c_gw,p (on the lowest modes); summary data term + both priors
        out = 0.0
        for s, ci, cg in zip(summaries, c_irn, c_gw):
            a = ci + np.concatenate([cg, np.zeros(N - 2*NGW)])
            out += -0.5 * a @ s.TtNT @ a + a @ s.b0
            phi = np.asarray(ds.powerlaw(s.f, s.df, p[f'{s.name}_red_noise_log10_A'], p[f'{s.name}_red_noise_gamma']))
            out += -0.5 * np.sum(ci**2 / phi) - 0.5 * np.sum(np.log(2 * np.pi * phi))
        f, df = summaries[0].f, summaries[0].df
        Phi = np.kron(orf, np.diag(ds.powerlaw(f[:2*NGW], df[:2*NGW], p['gw_log10_A'], p['gw_gamma'])))
        cg = np.concatenate(c_gw)
        return out - 0.5 * cg @ np.linalg.solve(Phi, cg) - 0.5 * np.linalg.slogdet(2 * np.pi * Phi)[1]

    rng = np.random.default_rng(2)
    new, old = [], []
    for _ in range(4):
        p = _rand_params(psrs, rng)
        c_irn = [s.ahat0 + 1e-7 * rng.normal(size=N) for s in summaries]
        c_gw = [1e-7 * rng.normal(size=2*NGW) for _ in summaries]
        q = dict(p)
        for s, ci, cg in zip(summaries, c_irn, c_gw):
            q[f'{s.name}_red_noise_coefficients({N})'] = jnp.asarray(ci)
            q[f'{s.name}_gw_coefficients({2*NGW})'] = jnp.asarray(cg)
        new.append(float(clogL(q)))
        old.append(ref(p, c_irn, c_gw))
    new, old = np.array(new), np.array(old)

    assert np.allclose(new[1:] - new[0], old[1:] - old[0], rtol=1e-8, atol=1e-6)


def test_decentered_hd_runs(psrs, T, summaries):
    pls, irn = _stand_ins(summaries, T)
    gw = ds.makeglobalgp_fourier(summaries, ds.powerlaw, ds.hd_orf, NGW, T, fourierbasis=fpta.summarybasis, name='gw')
    like = ds.ArrayLikelihood(pls, commongp=irn, globalgp=gw, decenter=True)

    q = _rand_params(psrs, np.random.default_rng(3))
    for s in summaries:
        q[f'{s.name}_red_noise_coefficients({N})'] = jnp.zeros(N)
        q[f'{s.name}_gw_coefficients({2*NGW})'] = jnp.zeros(2*NGW)

    logp, c = jax.jit(like.clogL)(q)
    grad = jax.jit(jax.grad(lambda q: like.clogL(q)[0]))(q)
    assert np.isfinite(logp)
    assert all(np.all(np.isfinite(g)) for g in jax.tree_util.tree_leaves(grad))


def test_summarybasis_checks(summaries, T):
    s = summaries[0]
    f, df, F = fpta.summarybasis(s, NGW, T)
    assert F.shape == (len(s.residuals), 2*NGW) and np.allclose(f, s.f[:2*NGW])
    with pytest.raises(ValueError):
        fpta.summarybasis(s, NC + 1, T)
    with pytest.raises(ValueError):
        fpta.summarybasis(s, NC, 2 * T)


# ---------------------------------------------------------------------------
# non-Gaussian corrections, mixing with time-domain pulsars


def _fake_summary(rng, d=4, K=3, name='fake', phi0=None):
    """A small summary with K per-sample conditionals, all well inside the prior phi0."""
    phi0 = np.full(d, 1.0) if phi0 is None else phi0
    sc = np.sqrt(phi0)
    ahat = 0.3 * rng.normal(size=(K, d)) * sc
    A = 0.3 * rng.normal(size=(K, d, d))
    Sigma = sc[:, None] * (0.1 * np.eye(d) + A @ np.swapaxes(A, 1, 2) * 0.1) * sc[None, :]
    ahat0 = ahat.mean(0)
    Sigma0 = Sigma.mean(0) + np.cov(ahat.T)
    return fpta.FourierSummary(name=name, pos=np.array([0.0, 0.0, 1.0]), f=np.repeat(np.arange(1, d//2 + 1) / 10.0, 2),
                               df=np.full(d, 0.1), ahat0=ahat0, Sigma0_inv=np.linalg.inv(Sigma0), phi0=phi0,
                               ahat_samples=ahat, Sigma_samples=Sigma)


def test_mixture_density_normalized():
    s = _fake_summary(np.random.default_rng(5), d=2, K=4)
    q = s.mixture()
    x = np.linspace(-12, 12, 801)
    X, Y = np.meshgrid(x, x)
    pts = jnp.asarray(np.stack([X.ravel(), Y.ravel()], axis=1))
    integral = np.sum(np.exp(jax.vmap(q.log_prob)(pts))) * (x[1] - x[0])**2
    assert np.isclose(integral, 1.0, rtol=1e-6)


def test_mixture_logL_monte_carlo():
    # L_k = E_{a ~ N(ahat_k, Sigma_k)} [N(a | 0, phi) / N(a | 0, phi0)], checked by sampling
    rng = np.random.default_rng(6)
    d = 4
    f, df = np.repeat(np.arange(1, d//2 + 1) / 10.0, 2), np.full(d, 0.1)
    p = {'fake_red_noise_log10_A': -14.0, 'fake_red_noise_gamma': 3.0}
    phi = np.asarray(ds.powerlaw(f, df, -14.0, 3.0))

    # reference prior a bit broader than the model prior at p
    s = _fake_summary(rng, d=d, phi0=1.5 * phi)
    irn = ds.makecommongp_fourier([s], ds.powerlaw, d // 2, s.T, fourierbasis=fpta.summarybasis, name='red_noise')
    logL = fpta.mixture_logL([s], irn)
    assert np.allclose(np.asarray(irn.Phi.getN(p))[0], phi)

    def lognorm(a, var):
        return -0.5 * np.sum(a**2 / var, axis=-1) - 0.5 * np.sum(np.log(2 * np.pi * var))

    L = []
    for ahat, Sigma in zip(s.ahat_samples, s.Sigma_samples):
        a = rng.multivariate_normal(ahat, Sigma, size=400000)
        L.append(np.mean(np.exp(lognorm(a, phi) - lognorm(a, s.phi0))))
    assert np.isclose(float(logL(p)), np.log(np.mean(L)), atol=5e-3)


def _coefficients(summaries, rng, gw=True):
    q = {}
    for s in summaries:
        q[f'{s.name}_red_noise_coefficients({N})'] = jnp.asarray(s.ahat0 + s.L0 @ rng.normal(size=N))
        if gw:
            q[f'{s.name}_gw_coefficients({2*NGW})'] = jnp.asarray(1e-7 * rng.normal(size=2*NGW))
    return q


def _logw(s, density, a):
    y = np.linalg.solve(s.L0, np.asarray(a) - s.ahat0)
    return float(density.log_prob(jnp.asarray(y))) + 0.5 * y @ y + 0.5 * len(y) * np.log(2 * np.pi)


def _density(rng):
    return fpta.GaussianMixture(0.3 * rng.normal(size=(2, N)), np.stack([1.2 * np.eye(N), 0.8 * np.eye(N)]))


def test_correction_single_pulsar(psrs, T, summaries):
    # the term rides on the summary's own PulsarLikelihood: clogL adds log w(a)
    s, rng = summaries[0], np.random.default_rng(9)
    density = _density(rng)
    rn = lambda: ds.makegp_fourier(s, ds.powerlaw, NC, T=T, fourierbasis=fpta.summarybasis, name='red_noise')
    plain = ds.PulsarLikelihood([s.residuals, fpta.makenoise_summary(s), rn()])
    corrected = ds.PulsarLikelihood([s.residuals, fpta.makenoise_summary(s), rn(), fpta.makecorrection(s, density)])

    q = {**_rand_params(psrs, rng), **_coefficients([s], rng, gw=False)}
    a = q[f'{s.name}_red_noise_coefficients({N})']
    assert np.isclose(float(corrected.clogL(q)) - float(plain.clogL(q)), _logw(s, density, a), rtol=1e-10, atol=1e-8)

    # a is not marginalizable analytically under the term: logL and conditional refuse
    for method in ('logL', 'conditional', 'sample_conditional', 'sample'):
        with pytest.raises(NotImplementedError, match='coefficient terms'):
            getattr(corrected, method)


def test_gaussian_density_is_no_correction(psrs, T, summaries):
    # a "correction" equal to N(y | 0, I) leaves the decentered HD clogL unchanged
    pls, irn = _stand_ins(summaries, T)
    gw = ds.makeglobalgp_fourier(summaries, ds.powerlaw, ds.hd_orf, NGW, T, fourierbasis=fpta.summarybasis, name='gw')
    plain = ds.ArrayLikelihood(pls, commongp=irn, globalgp=gw, decenter=True)

    unit = fpta.GaussianMixture(np.zeros((1, N)), np.eye(N)[None])
    pls_c = [ds.PulsarLikelihood([s.residuals, fpta.makenoise_summary(s), fpta.makecorrection(s, unit)])
             for s in summaries]
    corrected = ds.ArrayLikelihood(pls_c, commongp=irn, globalgp=gw, decenter=True)

    rng = np.random.default_rng(8)
    q = {**_rand_params(psrs, rng), **_coefficients(summaries, rng)}
    lp, c = plain.clogL(q)
    lc, cc = corrected.clogL(q)
    assert np.isclose(float(lp), float(lc), rtol=1e-12) and np.allclose(c, cc)


def test_correction_swept_up_by_array(psrs, T, summaries):
    # ArrayLikelihood.clogL adds each pulsar's term at that pulsar's coefficients a_p
    # (IRN + GW block); with decentering, at the physical coefficients it returns
    pls, irn = _stand_ins(summaries, T)
    gw = ds.makeglobalgp_fourier(summaries, ds.powerlaw, ds.hd_orf, NGW, T, fourierbasis=fpta.summarybasis, name='gw')
    rng = np.random.default_rng(11)
    density = _density(rng)
    pls_c = [ds.PulsarLikelihood([summaries[0].residuals, fpta.makenoise_summary(summaries[0]),
                                  fpta.makecorrection(summaries[0], density)])] + pls[1:]

    q = {**_rand_params(psrs, rng), **_coefficients(summaries, rng)}
    for decenter in (False, True):
        plain = ds.ArrayLikelihood(pls, commongp=irn, globalgp=gw, decenter=decenter)
        corrected = ds.ArrayLikelihood(pls_c, commongp=irn, globalgp=gw, decenter=decenter)
        if decenter:
            (lp, c), (lc, _) = plain.clogL(q), corrected.clogL(q)
            c0 = np.asarray(c[0])                        # [irn (N), gw (2 NGW)]
        else:
            lp, lc = plain.clogL(q), corrected.clogL(q)
            s0 = summaries[0].name
            c0 = np.concatenate([q[f'{s0}_red_noise_coefficients({N})'], q[f'{s0}_gw_coefficients({2*NGW})']])
        a0 = c0[:N] + np.concatenate([c0[N:], np.zeros(N - 2*NGW)])
        assert np.isclose(float(lc) - float(lp), _logw(summaries[0], density, a0), rtol=1e-9, atol=1e-8)


def test_correction_guards_array(psrs, T, summaries):
    # every likelihood method that marginalizes or conditions on a refuses coefficient
    # terms; clogL (which samples a) still works on the same model
    pls, irn = _stand_ins(summaries, T)
    gw = ds.makeglobalgp_fourier(summaries, ds.powerlaw, ds.hd_orf, NGW, T, fourierbasis=fpta.summarybasis, name='gw')
    density = _density(np.random.default_rng(12))
    pls_c = [ds.PulsarLikelihood([summaries[0].residuals, fpta.makenoise_summary(summaries[0]),
                                  fpta.makecorrection(summaries[0], density)])] + pls[1:]

    for kwargs in ({}, dict(commongp=irn), dict(commongp=irn, globalgp=gw)):
        like = ds.ArrayLikelihood(pls_c, **kwargs)
        for method in ('logL', 'conditional', 'sample_conditional'):
            with pytest.raises(NotImplementedError, match='coefficient terms'):
                getattr(like, method)
        with pytest.raises(NotImplementedError, match='coefficient terms'):
            like.cglogL()
        if kwargs:                                       # (without a GP, nothing to sample)
            assert callable(like.clogL)

    glike = ds.GlobalLikelihood(pls_c, globalgp=gw)
    for method in ('logL', 'plogL', 'conditional', 'sample_conditional', 'sample'):
        with pytest.raises(NotImplementedError, match='coefficient terms'):
            getattr(glike, method)


def _log_likelihood_ratio(s, phi):
    """log int da N(a | ahat0, Sigma0) N(a | 0, phi) / N(a | 0, phi0), in closed form."""
    P = s.Sigma0_inv + np.diag(1 / phi - 1 / s.phi0)
    b = s.b0
    return (0.5 * b @ np.linalg.solve(P, b) - 0.5 * s.ahat0 @ b + 0.5 * np.linalg.slogdet(s.Sigma0_inv)[1]
            - 0.5 * np.linalg.slogdet(P)[1] - 0.5 * np.sum(np.log(phi)) + 0.5 * np.sum(np.log(s.phi0)))


def test_standin_normalization(psrs, T, summaries):
    # logL = log p(dt | eta) - log p(dt | eta0) + logL0 exactly, not just up to a constant;
    # clogL = log N(a | ahat0, Sigma0) - log N(a | 0, phi0) + log N(a | 0, phi) (+ discovery's
    # missing 2 pi of the coefficient prior) + logL0
    rng = np.random.default_rng(12)
    for s in summaries:
        rn = ds.makegp_fourier(s, ds.powerlaw, NC, T=T, fourierbasis=fpta.summarybasis, name='red_noise')
        psl = ds.PulsarLikelihood([s.residuals, fpta.makenoise_summary(s), rn])
        p = {f'{s.name}_red_noise_log10_A': -14.2, f'{s.name}_red_noise_gamma': 4.0}
        phi = np.asarray(rn.Phi.getN(p))
        assert np.isclose(float(psl.logL(p)), _log_likelihood_ratio(s, phi), rtol=1e-10, atol=1e-6)

        a = s.ahat0 + s.L0 @ rng.normal(size=N)
        lognorm = lambda x, m, C: -0.5 * (x - m) @ np.linalg.solve(C, x - m) - 0.5 * np.linalg.slogdet(2 * np.pi * C)[1]
        ref = (lognorm(a, s.ahat0, s.Sigma0) - lognorm(a, 0, np.diag(s.phi0)) + lognorm(a, 0, np.diag(phi))
               + 0.5 * N * np.log(2 * np.pi))
        q = {**p, f'{s.name}_red_noise_coefficients({N})': jnp.asarray(a)}
        assert np.isclose(float(psl.clogL(q)), ref, rtol=1e-10, atol=1e-6)


def test_mixture_logL_one_component_is_standin(psrs, T, summaries):
    # a one-component "mixture" at (ahat0, Sigma0) is the Gaussian summary, normalization included
    import copy
    s = copy.copy(summaries[0])
    s.ahat_samples, s.Sigma_samples = s.ahat0[None], s.Sigma0[None]
    s.logL0 = 123.4
    rn = ds.makegp_fourier(s, ds.powerlaw, NC, T=T, fourierbasis=fpta.summarybasis, name='red_noise')
    psl = ds.PulsarLikelihood([s.residuals, fpta.makenoise_summary(s), rn])
    irn = ds.makecommongp_fourier([s], ds.powerlaw, NC, T, fourierbasis=fpta.summarybasis, name='red_noise')
    p = {f'{s.name}_red_noise_log10_A': -14.2, f'{s.name}_red_noise_gamma': 4.0}
    assert np.isclose(float(fpta.mixture_logL([s], irn)(p)), float(psl.logL(p)) + s.logL0, rtol=1e-10, atol=1e-6)


def test_summaries_reproduce_time_domain_likelihood(psrs, T, summaries):
    # at fixed white noise the Gaussian summary is exact: with each pulsar's step-1
    # marginalized likelihood at the reference spectrum, logL0, added, the step-2 HD logL over summaries equals the
    # time-domain HD logL, constant included
    pls_real = [ds.PulsarLikelihood([psr.residuals, ds.makegp_timing(psr, svd=True),
                                     ds.makenoise_measurement(psr, noisedict=psr.noisedict, ecorr=True)])
                for psr in psrs]

    def hd_logL(members, pls):
        irn = ds.makecommongp_fourier(members, ds.powerlaw, NC, T, fourierbasis=fpta.summarybasis, name='red_noise')
        gw = ds.makeglobalgp_fourier(members, ds.powerlaw, ds.hd_orf, NGW, T, fourierbasis=fpta.summarybasis, name='gw')
        return jax.jit(ds.ArrayLikelihood(pls, commongp=irn, globalgp=gw).logL)

    pls_sum, _ = _stand_ins(summaries, T)
    time_domain = hd_logL(psrs, pls_real)
    fourier = hd_logL(summaries, pls_sum)
    logL0 = sum(s.logL0 for s in summaries)

    rng = np.random.default_rng(13)
    for p in [_rand_params(psrs, rng) for _ in range(3)]:
        assert np.isclose(float(fourier(p)) + logL0, float(time_domain(p)), rtol=1e-10, atol=1e-5)


def test_mixed_time_domain_and_summaries(psrs, T, summaries):
    # at fixed white noise the Gaussian summary is exact, so replacing one summary by
    # the pulsar's full time-domain likelihood changes the step-2 logL only by that
    # summary's step-1 marginalized likelihood at the reference spectrum, logL0
    psr = psrs[0]
    real = ds.PulsarLikelihood([psr.residuals, ds.makegp_timing(psr, svd=True),
                                ds.makenoise_measurement(psr, noisedict=psr.noisedict, ecorr=True)])
    mixed = [psr] + summaries[1:]

    def hd_logL(members, pls):
        irn = ds.makecommongp_fourier(members, ds.powerlaw, NC, T, fourierbasis=fpta.summarybasis, name='red_noise')
        gw = ds.makeglobalgp_fourier(members, ds.powerlaw, ds.hd_orf, NGW, T, fourierbasis=fpta.summarybasis, name='gw')
        return jax.jit(ds.ArrayLikelihood(pls, commongp=irn, globalgp=gw).logL)

    pls_sum, _ = _stand_ins(summaries, T)
    all_summ = hd_logL(summaries, pls_sum)
    with_real = hd_logL(mixed, [real] + pls_sum[1:])

    rng = np.random.default_rng(10)
    ps = [_rand_params(psrs, rng) for _ in range(4)]
    a = np.array([float(all_summ(p)) for p in ps])
    b = np.array([float(with_real(p)) for p in ps])
    assert np.allclose(a + summaries[0].logL0, b, rtol=1e-10, atol=1e-5)


def test_mixture_logL_curn_combined(psrs, T, summaries):
    # CURN as one common GP (make_combined_crn, as for time-domain arrays): a
    # one-component mixture equals the Gaussian stand-in array likelihood
    import copy
    sums = [copy.copy(s) for s in summaries]
    for s in sums:
        s.ahat_samples, s.Sigma_samples = s.ahat0[None], s.Sigma0[None]

    psd, common = ds.make_combined_crn(NGW, ds.powerlaw, ds.powerlaw)
    curn = ds.makecommongp_fourier(sums, psd, NC, T, fourierbasis=fpta.summarybasis, name='red_noise', common=common)
    pls = [ds.PulsarLikelihood([s.residuals, fpta.makenoise_summary(s)]) for s in sums]
    gaussian = jax.jit(ds.ArrayLikelihood(pls, commongp=curn).logL)
    mixture = jax.jit(fpta.mixture_logL(sums, curn))

    rng = np.random.default_rng(14)
    for _ in range(3):
        p = {par: rng.uniform(*((-15, -13) if 'log10_A' in par else (1, 5))) for par in gaussian.params}
        assert np.isclose(float(mixture(p)), float(gaussian(p)) + sum(s.logL0 for s in sums), rtol=1e-10, atol=1e-6)


# ---------------------------------------------------------------------------
# mixture reduction


def _moments(q):
    w, mu = np.asarray(q.weights), np.asarray(q.mu)
    C = np.asarray(q.L) @ np.swapaxes(np.asarray(q.L), 1, 2)
    mean = w @ mu
    d = mu - mean
    return mean, np.einsum('k,kij->ij', w, C) + np.einsum('k,ki,kj->ij', w, d, d)


def test_weighted_mixture_normalized():
    rng = np.random.default_rng(15)
    q = fpta.GaussianMixture(rng.normal(size=(3, 2)), np.stack([np.eye(2) * s for s in (0.5, 1.0, 1.5)]),
                             weights=[0.2, 0.5, 0.3])
    x = np.linspace(-12, 12, 801)
    X, Y = np.meshgrid(x, x)
    pts = jnp.asarray(np.stack([X.ravel(), Y.ravel()], axis=1))
    assert np.isclose(np.sum(np.exp(jax.vmap(q.log_prob)(pts))) * (x[1] - x[0])**2, 1.0, rtol=1e-6)


def test_reduce_mixture_preserves_moments():
    s = _fake_summary(np.random.default_rng(16), d=4, K=40)
    q = s.mixture()
    mean, cov = _moments(q)
    for K in (40, 10, 3, 1):
        r = fpta.reduce_mixture(q, K, method='runnalls')
        assert r.K == K and np.isclose(float(np.sum(r.weights)), 1.0)
        m, c = _moments(r)
        assert np.allclose(m, mean, atol=1e-10) and np.allclose(c, cov, atol=1e-10)

    # one component: the moment-matched Gaussian, centred on the summary mean (y = 0).
    # (Its covariance is the mixture's, which differs from Sigma0 = I only by Sigma0's
    # 1/(K-1) sample-covariance normalization of the between-sample term.)
    one = fpta.reduce_mixture(q, 1)
    assert np.allclose(one.mu[0], 0, atol=1e-10)

    with pytest.raises(ValueError):
        fpta.reduce_mixture(q, 3, method='unknown')


def test_mixture_kl():
    s = _fake_summary(np.random.default_rng(17), d=4, K=40)
    q = s.mixture()
    key = jax.random.key(0)
    kl0, err0 = fpta.mixture_kl(q, q, key, n=2000)
    assert kl0 == 0.0 and err0 == 0.0
    kl, err = fpta.mixture_kl(q, fpta.reduce_mixture(q, 1), key, n=20000)
    assert kl > 3 * err                                # merging everything loses information


def test_mixture_logL_uses_reduced_density():
    # a reduced, weighted mixture attached as the density is what mixture_logL uses;
    # with K = all components it reproduces the full mixture
    rng = np.random.default_rng(18)
    d = 4
    f, df = np.repeat(np.arange(1, d//2 + 1) / 10.0, 2), np.full(d, 0.1)
    phi = np.asarray(ds.powerlaw(f, df, -14.0, 3.0))
    s = _fake_summary(rng, d=d, K=12, phi0=1.5 * phi)
    irn = ds.makecommongp_fourier([s], ds.powerlaw, d // 2, s.T, fourierbasis=fpta.summarybasis, name='red_noise')
    p = {'fake_red_noise_log10_A': -14.0, 'fake_red_noise_gamma': 3.0}

    full = float(fpta.mixture_logL([s], irn)(p))
    s.density = fpta.reduce_mixture(s.mixture(), 12)              # no merges
    assert np.isclose(float(fpta.mixture_logL([s], irn)(p)), full, rtol=1e-10)

    s.density = fpta.reduce_mixture(s.mixture(), 3)
    reduced = float(fpta.mixture_logL([s], irn)(p))
    assert np.isfinite(reduced) and reduced != full


def test_mixture_conditional_one_component_is_standin(psrs, T, summaries):
    # one component at (ahat0, Sigma0): the stand-in pulsar's own conditional
    import copy
    s = copy.copy(summaries[0])
    s.ahat_samples, s.Sigma_samples = s.ahat0[None], s.Sigma0[None]
    rn = ds.makegp_fourier(s, ds.powerlaw, NC, T=T, fourierbasis=fpta.summarybasis, name='red_noise')
    psl = ds.PulsarLikelihood([s.residuals, fpta.makenoise_summary(s), rn])
    irn = ds.makecommongp_fourier([s], ds.powerlaw, NC, T, fourierbasis=fpta.summarybasis, name='red_noise')
    p = {f'{s.name}_red_noise_log10_A': -14.2, f'{s.name}_red_noise_gamma': 4.0}

    logw, m, cf = fpta.mixture_conditional([s], irn)(p)
    mu, cf0 = psl.conditional(p)
    P, P0 = (np.asarray(L @ L.T) for L in (cf[0, 0], cf0[0]))   # both lower Cholesky factors of the precision
    assert np.isclose(float(logw[0, 0]), 0.0, atol=1e-12)
    assert np.allclose(m[0, 0], mu, rtol=1e-8, atol=1e-12 * np.max(np.abs(mu)))
    assert np.allclose(P, P0, rtol=1e-8)


def _conditional_logpdf(logw, m, cf, a):
    # log of sum_k w_k N(a | m_k, P_k^-1), with P_k = L_k L_k^T
    z = np.einsum('kji,kj->ki', np.asarray(cf), np.asarray(a - m))    # L_k^T (a - m_k)
    logdet = np.sum(np.log(np.diagonal(np.asarray(cf), axis1=1, axis2=2)), axis=1)
    return float(jax.scipy.special.logsumexp(np.asarray(logw) - 0.5 * np.sum(z**2, axis=1) + logdet))


def test_mixture_conditional_is_posterior():
    # the conditional is q(a) N(a | 0, phi) / N(a | 0, phi0), normalized: their log ratio
    # is the same at every a
    rng = np.random.default_rng(21)
    d = 4
    f, df = np.repeat(np.arange(1, d//2 + 1) / 10.0, 2), np.full(d, 0.1)
    phi = np.asarray(ds.powerlaw(f, df, -14.0, 3.0))
    s = _fake_summary(rng, d=d, K=5, phi0=1.5 * phi)
    irn = ds.makecommongp_fourier([s], ds.powerlaw, d // 2, s.T, fourierbasis=fpta.summarybasis, name='red_noise')
    p = {'fake_red_noise_log10_A': -14.0, 'fake_red_noise_gamma': 3.0}
    logw, m, cf = (x[0] for x in fpta.mixture_conditional([s], irn)(p))
    assert np.isclose(float(jax.scipy.special.logsumexp(logw)), 0.0, atol=1e-12)

    def lognorm(a, mean, cov):
        r = a - mean
        return -0.5 * r @ np.linalg.solve(cov, r) - 0.5 * np.linalg.slogdet(2 * np.pi * cov)[1]

    def logtarget(a):
        q = np.log(np.mean([np.exp(lognorm(a, ah, S)) for ah, S in zip(s.ahat_samples, s.Sigma_samples)]))
        return q + lognorm(a, 0, np.diag(phi)) - lognorm(a, 0, np.diag(s.phi0))

    ratios = [_conditional_logpdf(logw, m, cf, a) - logtarget(a)
              for a in s.ahat_samples + 0.5 * rng.normal(size=(5, d)) * np.sqrt(phi)]
    assert np.allclose(ratios, ratios[0], atol=1e-8)


def test_sample_mixture_conditional_moments():
    # draws reproduce the mixture's mean and covariance; names are the commongp coefficients
    rng = np.random.default_rng(22)
    d = 4
    f, df = np.repeat(np.arange(1, d//2 + 1) / 10.0, 2), np.full(d, 0.1)
    phi = np.asarray(ds.powerlaw(f, df, -14.0, 3.0))
    sums = [_fake_summary(rng, d=d, K=4, phi0=1.5 * phi, name=nm) for nm in ('fake0', 'fake1')]
    irn = ds.makecommongp_fourier(sums, ds.powerlaw, d // 2, sums[0].T, fourierbasis=fpta.summarybasis,
                                  name='red_noise')
    p = {f'{nm}_red_noise_{x}': v for nm in ('fake0', 'fake1') for x, v in (('log10_A', -14.0), ('gamma', 3.0))}

    logw, m, cf = fpta.mixture_conditional(sums, irn)(p)
    sample = jax.jit(jax.vmap(fpta.sample_mixture_conditional(sums, irn), in_axes=(0, None)))
    _, draws = sample(jax.random.split(jax.random.PRNGKey(0), 200000), p)
    assert list(draws) == list(irn.index)

    for i, name in enumerate(irn.index):
        w = np.exp(np.asarray(logw[i]))
        C = np.linalg.inv(np.asarray(cf[i] @ np.swapaxes(cf[i], 1, 2)))
        mean = w @ np.asarray(m[i])
        cov = np.einsum('k,kij->ij', w, C + np.einsum('ki,kj->kij', m[i] - mean, m[i] - mean))
        a = np.asarray(draws[name])
        se = np.sqrt(np.diag(cov) / len(a))
        assert np.all(np.abs(a.mean(0) - mean) < 5 * se)
        assert np.allclose(np.cov(a.T), cov, rtol=0.03, atol=0.03 * np.max(np.diag(cov)))
