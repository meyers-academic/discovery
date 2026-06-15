"""Tier-1 parity table: single-pulsar PulsarLikelihood.

Each row builds the same model under three routes (matrix.py reference,
metamath-via-monkeypatch, metamath-native via likelihood_metamath.py) and
asserts the output method (logL / conditional / clogL / ...) and param set
agree across all three. See ``_routes.build_routes`` for the route
definitions.

xfail rows pin known gaps in the metamath coverage; they flip to failure
when the underlying gap closes (strict=True).
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import discovery as ds


def _toy_delay(toas):
    # deterministic, parameter-free delay (args come only from psr attributes);
    # exercises the CompoundDelay path without introducing unsampleable params.
    return 1e-9 * jnp.sin(2.0 * jnp.pi * (toas - toas.min()) / 3.16e8)

from ._comparison import assert_close, assert_params_equal
from ._routes import build_routes


# ---------- model builders ----------
# each takes a psr, returns a PulsarLikelihood. invoked twice — once inside
# the patch context, once outside — so the body is just a recipe.

def _measurement_simple(psr):
    return ds.PulsarLikelihood([
        psr.residuals,
        ds.makenoise_measurement_simple(psr),
    ])

def _measurement_white(psr):
    return ds.PulsarLikelihood([
        psr.residuals,
        ds.makenoise_measurement(psr),
    ])

def _ecorr_gp(psr):
    # ecorr-as-GP: VariableGP with NoiseMatrix1D_var Phi
    return ds.PulsarLikelihood([
        psr.residuals,
        ds.makenoise_measurement(psr),
        ds.makegp_ecorr(psr),
    ])

def _ecorr_sm(psr):
    # ecorr folded into measurement noise via Sherman-Morrison.
    # SM noise classes don't define make_kernelproduct directly — they must be
    # wrapped in a Woodbury, so we add a constant timing GP for a minimal combo.
    return ds.PulsarLikelihood([
        psr.residuals,
        ds.makenoise_measurement(psr, psr.noisedict, ecorr=True),
        ds.makegp_timing(psr, svd=True),
    ])

def _meas_timing(psr):
    return ds.PulsarLikelihood([
        psr.residuals,
        ds.makenoise_measurement(psr, psr.noisedict),
        ds.makegp_timing(psr, svd=True),
    ])

def _full_rn(psr):
    # the realistic single-pulsar model: WN + ecorr-GP + timing + power-law RN
    return ds.PulsarLikelihood([
        psr.residuals,
        ds.makenoise_measurement(psr, psr.noisedict),
        ds.makegp_ecorr(psr, psr.noisedict),
        ds.makegp_timing(psr, svd=True),
        ds.makegp_fourier(psr, ds.powerlaw, components=30, name="rednoise"),
    ])

def _full_rn_concat_false(psr):
    # same as _full_rn but PulsarLikelihood(... concat=False) → chained Woodburys
    return ds.PulsarLikelihood([
        psr.residuals,
        ds.makenoise_measurement(psr, psr.noisedict),
        ds.makegp_ecorr(psr, psr.noisedict),
        ds.makegp_timing(psr, svd=True),
        ds.makegp_fourier(psr, ds.powerlaw, components=30, name="rednoise"),
    ], concat=False)

def _multi_vgp(psr):
    # two variable GPs (RN + freespectrum-like), exercises CompoundGP variable branch
    return ds.PulsarLikelihood([
        psr.residuals,
        ds.makenoise_measurement(psr, psr.noisedict),
        ds.makegp_ecorr(psr, psr.noisedict),
        ds.makegp_timing(psr, svd=True),
        ds.makegp_fourier(psr, ds.powerlaw, components=30, name="rednoise"),
        # second variable GP under the dmgp prefix (has its own priors)
        ds.makegp_fourier(psr, ds.powerlaw, components=14, name="dmgp"),
    ])

def _variable_timing(psr):
    # makegp_timing(variable=True) — turns a constant timing GP into a variable one
    return ds.PulsarLikelihood([
        psr.residuals,
        ds.makenoise_measurement(psr, ecorr=True),
        ds.makegp_timing(psr, svd=True, variable=True),
        ds.makegp_fourier(psr, ds.powerlaw, components=30, name="rednoise"),
    ])


def _fftcov_2d(psr):
    # fftcov GP emits a *2D* covariance noise matrix (NoiseMatrix2D_var via the
    # NoiseMatrix12D_var dispatcher). Exercises the 2D noise path that the fftcov
    # family (os_example / numpyro_example) uses but no 1D-PSD row reaches.
    return ds.PulsarLikelihood([
        psr.residuals,
        ds.makenoise_measurement(psr, psr.noisedict),
        ds.makegp_timing(psr, svd=True),
        # name 'rednoise' so the powerlaw params get standard sampling priors
        ds.makegp_fftcov(psr, ds.powerlaw, components=31, name="rednoise"),
    ])

def _delay(psr):
    # makedelay -> CompoundDelay path (used by the CW examples).
    return ds.PulsarLikelihood([
        psr.residuals,
        ds.makenoise_measurement(psr, psr.noisedict),
        ds.makegp_timing(psr, svd=True),
        ds.makedelay(psr, _toy_delay, name="toydelay"),
    ])

def _fourier_variance_fixed(psr):
    # makegp_fourier_variance with the variance matrix supplied -> ConstantGP
    # wrapping NoiseMatrix2D_novar (the fixed 2D noise constructor).
    comps = 10
    argname = f"{psr.name}_fourierGP_variance({comps*2},{comps*2})"
    cov = np.diag(np.full(comps * 2, 1e-16))
    return ds.PulsarLikelihood([
        psr.residuals,
        ds.makenoise_measurement(psr, psr.noisedict),
        ds.makegp_timing(psr, svd=True),
        ds.makegp_fourier_variance(psr, components=comps, noisedict={argname: cov}),
    ])


# ---------- logL rows ----------

LOGL_ROWS = [
    pytest.param(_measurement_simple, id="measurement_simple"),
    pytest.param(_measurement_white,  id="measurement_white"),
    pytest.param(_ecorr_gp,             id="measurement+ecorr_gp"),
    pytest.param(_ecorr_sm,             id="ecorr_sm+timing"),
    pytest.param(_meas_timing,          id="measurement+timing_svd"),
    pytest.param(_full_rn,              id="full_rn"),
    pytest.param(_full_rn_concat_false, id="full_rn_concat_false"),
    pytest.param(_multi_vgp,            id="multi_vgp"),
    pytest.param(_variable_timing,      id="variable_timing"),
    pytest.param(_fftcov_2d,            id="fftcov_2d"),
    pytest.param(_delay,                id="delay"),
    pytest.param(
        _fourier_variance_fixed, id="fourier_variance_fixed",
        marks=pytest.mark.xfail(strict=True, reason=(
            "Phase 3 known gap: an all-constant 2D GP prior (ConstantGP from "
            "makegp_fourier_variance, NoiseMatrix2D_novar) is unsupported in the "
            "metamath path -- metamath CompoundGP._build_mixed_logprior requires "
            "gp.index, which a marginalized constant GP lacks. The kernel maps "
            "(NoiseMatrix2D_novar -> mh.NoiseMatrix2D); the likelihood path does "
            "not. Close in Phase 4; see docs/components/phase3_coverage.md.")),
    ),
]


# ---------- helpers ----------

ALT_ROUTES = ("mh_patched", "mh_native")


def _routes(build, psr):
    """Build the model under each route. Returns the dict from
    ``build_routes``."""
    return build_routes(lambda: build(psr))


def _draw_params(model, seed=0):
    np.random.seed(seed)
    return ds.sample_uniform(model.logL.params)


# ---------- tests ----------

@pytest.mark.parametrize("build", LOGL_ROWS)
def test_logL(psr, build):
    r = _routes(build, psr)
    ref = r["matrix"]
    p0 = _draw_params(ref)
    ref_val = float(ref.logL(p0))

    for route in ALT_ROUTES:
        assert_params_equal(r[route].logL, ref.logL,
                            name=f"{build.__name__}[{route}]")
        val = float(r[route].logL(p0))
        assert_close(val, ref_val, kind="logL",
                     name=f"{build.__name__}[{route}]")


# conditional / clogL only make sense with at least one variable GP

CONDITIONAL_ROWS = [
    pytest.param(_full_rn, id="full_rn"),
    pytest.param(_multi_vgp, id="multi_vgp"),
]


@pytest.mark.parametrize("build", CONDITIONAL_ROWS)
def test_conditional(psr, build):
    r = _routes(build, psr)
    ref = r["matrix"]
    p0 = _draw_params(ref)
    mu_ref, cf_ref = ref.conditional(p0)

    for route in ALT_ROUTES:
        assert_params_equal(r[route].conditional, ref.conditional,
                            name=f"{build.__name__}[{route}]")
        mu, cf = r[route].conditional(p0)
        assert_close(np.asarray(mu), np.asarray(mu_ref), kind="coeffs",
                     name=f"{build.__name__}[{route}].mu")
        assert_close(np.asarray(cf[0]), np.asarray(cf_ref[0]), kind="matrix",
                     name=f"{build.__name__}[{route}].cf")


@pytest.mark.parametrize("build", CONDITIONAL_ROWS)
def test_clogL(psr, build):
    r = _routes(build, psr)
    ref = r["matrix"]

    # clogL params include latent coefficient vectors like "..._coefficients(60)"
    # for which there is no standard prior; draw those manually.
    np.random.seed(0)
    scalar_params = [p for p in ref.clogL.params if not p.endswith(")")]
    p0 = ds.sample_uniform(scalar_params)
    for p in ref.clogL.params:
        if p.endswith(")"):
            n = int(p[p.index("(") + 1: -1])
            p0[p] = 1e-6 * np.random.randn(n)
    ref_val = float(ref.clogL(p0))

    for route in ALT_ROUTES:
        assert_params_equal(r[route].clogL, ref.clogL,
                            name=f"{build.__name__}[{route}]")
        val = float(r[route].clogL(p0))
        assert_close(val, ref_val, kind="logL",
                     name=f"{build.__name__}[{route}]")


SAMPLE_ROWS = [
    pytest.param(_full_rn, id="full_rn"),
    pytest.param(_multi_vgp, id="multi_vgp"),
]


@pytest.mark.parametrize("build", SAMPLE_ROWS)
def test_sample(psr, build):
    r = _routes(build, psr)
    ref = r["matrix"]

    p0 = _draw_params(ref)
    key = jax.random.PRNGKey(42)
    _, y_ref = ref.sample(key, p0)

    for route in ALT_ROUTES:
        _, y = r[route].sample(key, p0)
        assert_close(np.asarray(y), np.asarray(y_ref), kind="residuals",
                     name=f"{build.__name__}[{route}]")


@pytest.mark.parametrize("build", CONDITIONAL_ROWS)
def test_sample_conditional(psr, build):
    r = _routes(build, psr)
    ref = r["matrix"]

    p0 = _draw_params(ref)
    key = jax.random.PRNGKey(42)
    _, c_ref = ref.sample_conditional(key, p0)

    for route in ALT_ROUTES:
        _, c = r[route].sample_conditional(key, p0)
        assert set(c.keys()) == set(c_ref.keys())
        for k in c_ref:
            assert_close(np.asarray(c[k]), np.asarray(c_ref[k]),
                         kind="coeffs",
                         name=f"{build.__name__}[{route}].{k}")
