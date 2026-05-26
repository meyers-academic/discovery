"""Tier-2 parity table: GlobalLikelihood.

GlobalLikelihood wraps a list of PulsarLikelihoods plus optional globalgp
(correlated GP across pulsars). With the monkeypatch, globalgp.Phi becomes
mh.NoiseMatrix, which triggers the metamath GlobalWoodburyKernel path in
likelihood.py:307.
"""

import numpy as np
import pytest

import jax

import discovery as ds

from ._comparison import assert_close, assert_params_equal
from ._patch import metamatrix_patch


# ---------- per-psr builder ----------

def _psl(psr, T):
    return ds.PulsarLikelihood([
        psr.residuals,
        ds.makenoise_measurement(psr, psr.noisedict),
        ds.makegp_ecorr(psr, psr.noisedict),
        ds.makegp_timing(psr, svd=True),
        ds.makegp_fourier(psr, ds.powerlaw, components=30, T=T, name="rednoise"),
    ])


# ---------- GlobalLikelihood builders ----------

def _no_global(psrs):
    """No globalgp → sum of psl.logL."""
    T = ds.getspan(psrs)
    return ds.GlobalLikelihood([_psl(p, T) for p in psrs])


def _global_hd(psrs):
    """HD-correlated GW global GP (2D Phi)."""
    T = ds.getspan(psrs)
    return ds.GlobalLikelihood(
        [_psl(p, T) for p in psrs],
        globalgp=ds.makeglobalgp_fourier(psrs, ds.powerlaw, ds.hd_orf,
                                         components=14, T=T, name="gw"),
    )


def _global_monopole(psrs):
    """Monopole-correlated global GP — simpler 2D Phi structure."""
    T = ds.getspan(psrs)
    return ds.GlobalLikelihood(
        [_psl(p, T) for p in psrs],
        globalgp=ds.makeglobalgp_fourier(psrs, ds.powerlaw, ds.monopole_orf,
                                         components=14, T=T, name="gw"),
    )


# ---------- tables ----------

LOGL_ROWS = [
    pytest.param(_no_global,       id="no_global"),
    pytest.param(_global_hd,       id="global_hd"),
    pytest.param(_global_monopole, id="global_monopole"),
]


# ---------- helpers ----------

def _both(build, psrs):
    old = build(psrs)
    _ = old.logL
    with metamatrix_patch():
        new = build(psrs)
        _ = new.logL
    return old, new


# ---------- tests ----------

@pytest.mark.parametrize("build", LOGL_ROWS)
def test_logL(psrs, build):
    old, new = _both(build, psrs)
    assert_params_equal(new.logL, old.logL, name=build.__name__)

    np.random.seed(0)
    p0 = ds.sample_uniform(old.logL.params)

    lo = float(old.logL(p0))
    ln = float(new.logL(p0))
    assert_close(ln, lo, kind="logL", name=build.__name__)


# conditional only meaningful with globalgp
CONDITIONAL_ROWS = [
    pytest.param(_global_hd,       id="global_hd"),
    pytest.param(_global_monopole, id="global_monopole"),
]


@pytest.mark.parametrize("build", CONDITIONAL_ROWS)
def test_conditional(psrs, build):
    """GlobalLikelihood.conditional has its own bespoke matrix.py path; with
    monkeypatch, globalgp.Phi.make_inv etc still resolve through metamath.
    """
    old, new = _both(build, psrs)
    _ = old.conditional
    with metamatrix_patch():
        _ = new.conditional

    assert_params_equal(new.conditional, old.conditional, name=build.__name__)

    np.random.seed(0)
    p0 = ds.sample_uniform(old.conditional.params)

    mu_o, cf_o = old.conditional(p0)
    mu_n, cf_n = new.conditional(p0)

    assert_close(np.asarray(mu_n), np.asarray(mu_o), kind="coeffs",
                 name=f"{build.__name__}.mu")
    assert_close(np.asarray(cf_n[0]), np.asarray(cf_o[0]), kind="matrix",
                 name=f"{build.__name__}.cf")


SAMPLE_ROWS = [
    pytest.param(_no_global, id="no_global"),
    pytest.param(_global_hd, id="global_hd"),
]


@pytest.mark.parametrize("build", SAMPLE_ROWS)
def test_sample(psrs, build):
    """Per-pulsar prior draws plus (when globalgp set) a correlated Phi draw."""
    old, new = _both(build, psrs)
    _ = old.sample
    with metamatrix_patch():
        _ = new.sample

    np.random.seed(0)
    p0 = ds.sample_uniform(old.logL.params)
    key = jax.random.PRNGKey(42)

    _, ys_o = old.sample(key, p0)
    _, ys_n = new.sample(key, p0)

    assert len(ys_o) == len(ys_n) == len(psrs)
    for i, (yo, yn) in enumerate(zip(ys_o, ys_n)):
        assert_close(np.asarray(yn), np.asarray(yo), kind="residuals",
                     name=f"{build.__name__}.psr{i}")
