"""Tier-1 parity table: single-pulsar PulsarLikelihood.

Each row builds the same model two ways:
  - 'old' : stock matrix.py
  - 'new' : matrix.py monkeypatched to metamath classes

and asserts the output method (logL/conditional/clogL/...) and param set agree.

xfail rows pin known gaps in the monkeypatch coverage; they flip to a failure
when the underlying gap closes (strict=True).
"""

import numpy as np
import pytest

import jax

import discovery as ds

from ._comparison import assert_close, assert_params_equal
from ._patch import metamatrix_patch


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
]


# ---------- helpers ----------

def _both(build, psr):
    """Build the model twice (old + new). Returns (old, new)."""
    old = build(psr)
    _ = old.logL  # force closure capture before patch
    with metamatrix_patch():
        new = build(psr)
        _ = new.logL
    return old, new


def _draw_params(model, seed=0):
    np.random.seed(seed)
    return ds.sample_uniform(model.logL.params)


# ---------- tests ----------

@pytest.mark.parametrize("build", LOGL_ROWS)
def test_logL(psr, build):
    old, new = _both(build, psr)
    assert_params_equal(new.logL, old.logL, name=build.__name__)

    p0 = _draw_params(old)
    lo = float(old.logL(p0))
    ln = float(new.logL(p0))
    assert_close(ln, lo, kind="logL", name=build.__name__)


# conditional / clogL only make sense with at least one variable GP

CONDITIONAL_ROWS = [
    pytest.param(_full_rn, id="full_rn"),
    pytest.param(_multi_vgp, id="multi_vgp"),
]


@pytest.mark.parametrize("build", CONDITIONAL_ROWS)
def test_conditional(psr, build):
    old, new = _both(build, psr)
    # force conditional closures too
    _ = old.conditional
    with metamatrix_patch():
        _ = new.conditional

    assert_params_equal(new.conditional, old.conditional, name=build.__name__)

    p0 = _draw_params(old)

    mu_o, cf_o = old.conditional(p0)
    mu_n, cf_n = new.conditional(p0)

    assert_close(np.asarray(mu_n), np.asarray(mu_o), kind="coeffs",
                 name=f"{build.__name__}.mu")
    # cf is (matrix, lower_flag); only the matrix is numeric
    assert_close(np.asarray(cf_n[0]), np.asarray(cf_o[0]), kind="matrix",
                 name=f"{build.__name__}.cf")


@pytest.mark.parametrize("build", CONDITIONAL_ROWS)
def test_clogL(psr, build):
    old, new = _both(build, psr)
    _ = old.clogL
    with metamatrix_patch():
        _ = new.clogL

    assert_params_equal(new.clogL, old.clogL, name=build.__name__)

    # clogL params include latent coefficient vectors like "..._coefficients(60)"
    # for which there is no standard prior; draw those manually.
    np.random.seed(0)
    scalar_params = [p for p in old.clogL.params if not p.endswith(")")]
    p0 = ds.sample_uniform(scalar_params)
    for p in old.clogL.params:
        if p.endswith(")"):
            n = int(p[p.index("(") + 1 : -1])
            p0[p] = 1e-6 * np.random.randn(n)

    lo = float(old.clogL(p0))
    ln = float(new.clogL(p0))
    assert_close(ln, lo, kind="logL", name=build.__name__)


SAMPLE_ROWS = [
    pytest.param(_full_rn, id="full_rn"),
    pytest.param(_multi_vgp, id="multi_vgp"),
]


@pytest.mark.parametrize("build", SAMPLE_ROWS)
def test_sample(psr, build):
    old, new = _both(build, psr)
    _ = old.sample
    with metamatrix_patch():
        _ = new.sample

    p0 = _draw_params(old)
    key = jax.random.PRNGKey(42)
    _, yo = old.sample(key, p0)
    _, yn = new.sample(key, p0)

    assert_close(np.asarray(yn), np.asarray(yo), kind="residuals",
                 name=build.__name__)


@pytest.mark.parametrize("build", CONDITIONAL_ROWS)
def test_sample_conditional(psr, build):
    old, new = _both(build, psr)
    _ = old.sample_conditional
    with metamatrix_patch():
        _ = new.sample_conditional

    p0 = _draw_params(old)
    key = jax.random.PRNGKey(42)
    _, co = old.sample_conditional(key, p0)
    _, cn = new.sample_conditional(key, p0)

    assert set(co.keys()) == set(cn.keys())
    for k in co:
        assert_close(np.asarray(cn[k]), np.asarray(co[k]), kind="coeffs",
                     name=f"{build.__name__}.{k}")
