"""Tier-3 parity table: ArrayLikelihood.

ArrayLikelihood wraps a vector of PulsarLikelihoods plus optional commongp
(shared GP basis across pulsars) and optional globalgp (correlated GP across
pulsars, e.g. HD). The monkeypatch routes:
    matrix.VectorWoodburyKernel_varP → mh.VectorWoodburyKernel
    matrix.VectorCompoundGP          → mh.CompoundGP
"""

import numpy as np
import pytest

import jax

import discovery as ds

from ._comparison import assert_close, assert_params_equal
from ._patch import metamatrix_patch
from ._routes import build_routes


# ---------- per-pulsar PSL skeleton (no per-psr RN — RN goes in commongp) ----------

def _psl_skeleton(psr):
    return ds.PulsarLikelihood([
        psr.residuals,
        ds.makenoise_measurement(psr, psr.noisedict),
        ds.makegp_ecorr(psr, psr.noisedict),
        ds.makegp_timing(psr, svd=True),
    ])


# ---------- ArrayLikelihood builders ----------

def _no_common(psrs):
    """No commongp, no globalgp → sum of psl.logL (per-psr RN inline)."""
    T = ds.getspan(psrs)
    return ds.ArrayLikelihood([
        ds.PulsarLikelihood([
            psr.residuals,
            ds.makenoise_measurement(psr, psr.noisedict),
            ds.makegp_ecorr(psr, psr.noisedict),
            ds.makegp_timing(psr, svd=True),
            ds.makegp_fourier(psr, ds.powerlaw, components=30, T=T, name="rednoise"),
        ]) for psr in psrs
    ])


def _common_rn(psrs):
    """Single commongp: shared-basis RN. Triggers VectorWoodburyKernel path."""
    T = ds.getspan(psrs)
    return ds.ArrayLikelihood(
        [_psl_skeleton(p) for p in psrs],
        commongp=ds.makecommongp_fourier(psrs, ds.powerlaw, components=30,
                                         T=T, name="rednoise"),
    )


def _common_rn_and_crn(psrs):
    """Two commongps: per-psr RN + CRN with shared params."""
    T = ds.getspan(psrs)
    return ds.ArrayLikelihood(
        [_psl_skeleton(p) for p in psrs],
        commongp=[
            ds.makecommongp_fourier(psrs, ds.powerlaw, components=30,
                                    T=T, name="rednoise"),
            ds.makecommongp_fourier(psrs, ds.powerlaw, components=14,
                                    T=T, name="crn",
                                    common=["crn_log10_A", "crn_gamma"]),
        ],
    )


def _common_rn_plus_global_hd(psrs):
    """commongp=RN + globalgp=HD-correlated GW. Exercises the fused metamath path."""
    T = ds.getspan(psrs)
    return ds.ArrayLikelihood(
        [_psl_skeleton(p) for p in psrs],
        commongp=ds.makecommongp_fourier(psrs, ds.powerlaw, components=30,
                                         T=T, name="rednoise"),
        globalgp=ds.makeglobalgp_fourier(psrs, ds.powerlaw, ds.hd_orf,
                                         components=14, T=T, name="gw"),
    )


# ---------- tables ----------

LOGL_ROWS = [
    pytest.param(_no_common,                  id="no_common"),
    pytest.param(_common_rn,                  id="common_rn"),
    pytest.param(_common_rn_and_crn,          id="common_rn+crn"),
    pytest.param(_common_rn_plus_global_hd,   id="common_rn+global_hd"),
]


# ---------- helpers ----------

ALT_ROUTES = ("mh_patched", "mh_native")


def _routes(build, psrs):
    return build_routes(lambda: build(psrs))


# ---------- tests ----------

@pytest.mark.parametrize("build", LOGL_ROWS)
def test_logL(psrs, build):
    r = _routes(build, psrs)
    ref = r["matrix"]
    np.random.seed(0)
    p0 = ds.sample_uniform(ref.logL.params)
    ref_val = float(ref.logL(p0))

    for route in ALT_ROUTES:
        assert_params_equal(r[route].logL, ref.logL,
                            name=f"{build.__name__}[{route}]")
        val = float(r[route].logL(p0))
        assert_close(val, ref_val, kind="logL",
                     name=f"{build.__name__}[{route}]")


# conditional — only meaningful with commongp.
# matrix.VectorWoodburyKernel_varP has no make_conditional, so this is a
# *metamath-only* capability; we run it standalone (no parity check).
CONDITIONAL_ROWS = [
    pytest.param(_common_rn,         id="common_rn"),
    pytest.param(_common_rn_and_crn, id="common_rn+crn"),
]


@pytest.mark.parametrize("build", CONDITIONAL_ROWS)
def test_conditional_metamath_only(psrs, build):
    """Smoke test: ArrayLikelihood.conditional only exists on the metamath path.

    matrix.VectorWoodburyKernel_varP doesn't define make_conditional, so the
    stock matrix.py path raises NotImplementedError. We exercise the new path
    and check shape/finiteness, since there's no oracle to compare against.
    """
    with metamatrix_patch():
        mdl = build(psrs)
        cond = mdl.conditional

        np.random.seed(0)
        p0 = ds.sample_uniform(cond.params)
        mu, cf = cond(p0)

    mu, cf0 = np.asarray(mu), np.asarray(cf[0])
    assert mu.ndim >= 1 and np.all(np.isfinite(mu)), f"{build.__name__}.mu bad"
    assert cf0.ndim >= 2 and np.all(np.isfinite(cf0)), f"{build.__name__}.cf bad"


@pytest.mark.parametrize("build", CONDITIONAL_ROWS)
def test_clogL(psrs, build):
    r = _routes(build, psrs)
    ref = r["matrix"]

    np.random.seed(0)
    scalar = [p for p in ref.clogL.params if not p.endswith(")")]
    p0 = ds.sample_uniform(scalar)
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


# ============================================================================
# Decentering / means / extsignals — new capabilities ported from b1bda23.
# Each row's builder configures one of the three new clogL features.
# ============================================================================

def _decenter_common_rn(psrs):
    T = ds.getspan(psrs)
    return ds.ArrayLikelihood(
        [_psl_skeleton(p) for p in psrs],
        commongp=ds.makecommongp_fourier(psrs, ds.powerlaw, components=30,
                                         T=T, name="rednoise"),
        decenter=True,
    )


def _decenter_common_rn_global_hd(psrs):
    T = ds.getspan(psrs)
    return ds.ArrayLikelihood(
        [_psl_skeleton(p) for p in psrs],
        commongp=ds.makecommongp_fourier(psrs, ds.powerlaw, components=30,
                                         T=T, name="rednoise"),
        globalgp=ds.makeglobalgp_fourier(psrs, ds.powerlaw, ds.hd_orf,
                                         components=14, T=T, name="gw"),
        decenter=True,
    )


def _means_on_commongp(psrs):
    """commongp with a `means` callable that shifts the prior center.

    The mean uses a per-pulsar scalar param ``{psr.name}_mean_amp`` that maps
    to a flat coefficient shift across the Fourier basis.
    """
    import jax.numpy as jnp

    def my_means(f, df, mean_amp):
        # length-(2*components,) shift; sized by f
        return mean_amp * jnp.ones_like(f)

    T = ds.getspan(psrs)
    return ds.ArrayLikelihood(
        [_psl_skeleton(p) for p in psrs],
        commongp=ds.makecommongp_fourier(psrs, ds.powerlaw, components=30,
                                         T=T, name="rednoise",
                                         means=my_means),
    )


def _extsignal_cw(psrs):
    """commongp + a CW external signal on its own (higher-freq) basis."""
    T = ds.getspan(psrs)
    return ds.ArrayLikelihood(
        [_psl_skeleton(p) for p in psrs],
        commongp=ds.makecommongp_fourier(psrs, ds.powerlaw, components=30,
                                         T=T, name="rednoise"),
        extsignals=[
            ds.makecw_extsignal(psrs, components=50, T=T,
                                pulsarterm=True, name="cw"),
        ],
    )


NEW_CLOGL_ROWS = [
    pytest.param(_decenter_common_rn,             id="decenter+common_rn"),
    pytest.param(_decenter_common_rn_global_hd, id="decenter+common_rn+global_hd"),
    pytest.param(_means_on_commongp,              id="means_on_commongp"),
    pytest.param(_extsignal_cw,                   id="extsignal_cw"),
]


def _fill_clogL_p0(build, psrs):
    """Build all three routes, sample p0 with array coeffs + non-standard scalars."""
    r = _routes(build, psrs)
    ref = r["matrix"]

    np.random.seed(0)
    params = ref.clogL.params

    # scalars with known priors via sample_uniform; array-coeff and any
    # unrecognized scalars get manual fills.
    scalar_known, scalar_unknown, array_p = [], [], []
    for p in params:
        if p.endswith(")"):
            array_p.append(p)
        else:
            try:
                ds.sample_uniform([p])
                scalar_known.append(p)
            except KeyError:
                scalar_unknown.append(p)

    p0 = ds.sample_uniform(scalar_known)
    for p in scalar_unknown:
        p0[p] = float(np.random.randn())
    for p in array_p:
        n = int(p[p.index("(") + 1: -1])
        p0[p] = 1e-6 * np.random.randn(n)
    return r, p0


def _compare_clogL(lo, ln, *, name):
    """clogL may return (logp, c) when staged (reparams applied) or scalar."""
    if isinstance(lo, tuple):
        lo_logp, lo_c = float(lo[0]), np.asarray(lo[1])
        ln_logp, ln_c = float(ln[0]), np.asarray(ln[1])
        assert_close(ln_logp, lo_logp, kind="logL", name=f"{name}.logp")
        assert_close(ln_c, lo_c, kind="coeffs", name=f"{name}.c")
    else:
        assert_close(float(ln), float(lo), kind="logL", name=name)


@pytest.mark.parametrize("build", NEW_CLOGL_ROWS)
def test_clogL_new_features(psrs, build):
    r, p0 = _fill_clogL_p0(build, psrs)
    ref = r["matrix"]
    lo = ref.clogL(p0)

    for route in ALT_ROUTES:
        assert_params_equal(r[route].clogL, ref.clogL,
                            name=f"{build.__name__}[{route}]")
        ln = r[route].clogL(p0)
        _compare_clogL(lo, ln, name=f"{build.__name__}[{route}]")
