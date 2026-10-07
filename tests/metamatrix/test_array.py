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
import jax.numpy as jnp

import discovery as ds

from ._comparison import assert_close, assert_params_equal
from ._patch import metamatrix_patch
from ._routes import build_routes
import discovery.recipes as R


# Model builders live in `discovery.recipes` (in discovery.recipes, shared with the docs cookbook).

# ---------- tables ----------

LOGL_ROWS = [
    pytest.param(R.no_common,                id="no_common"),
    pytest.param(R.intrinsic_rn,                id="intrinsic_rn"),
    pytest.param(R.intrinsic_plus_crn,        id="intrinsic_rn+crn"),
    pytest.param(R.intrinsic_rn_plus_global_hd, id="intrinsic_rn+global_hd"),
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
    pytest.param(R.intrinsic_rn,         id="intrinsic_rn"),
    pytest.param(R.intrinsic_plus_crn, id="intrinsic_rn+crn"),
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
# Decentering / means / extsignals — clogL features (recipes in discovery.recipes).
# ============================================================================

NEW_CLOGL_ROWS = [
    pytest.param(R.decenter_intrinsic_rn,            id="decenter+intrinsic_rn"),
    pytest.param(R.decenter_intrinsic_rn_global_hd,  id="decenter+intrinsic_rn+global_hd"),
    pytest.param(R.means_on_commongp,             id="means_on_commongp"),
    pytest.param(R.extsignal_cw,                  id="extsignal_cw"),
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


# ============================================================================
# HD global GP: Kronecker Phi (signals.makeglobalgp_fourier) and the
# sampled-coefficient prior (metamath.CompoundGP._build_mixed_logprior).
#
# For a constant scalar ORF Gamma and spectrum S, Phi = Gamma (x) S (pulsar-major,
# mode-inner) and Phi^-1 = Gamma^-1 (x) S^-1. makeglobalgp_fourier builds both as
# Kronecker products and exposes the factors as gp.Phi_inv_kron; _build_mixed_logprior
# uses them (else gp.Phi_inv, else a dense solve of Phi), so the decentered clogL never
# forms or factorizes a (npsr m) x (npsr m) matrix. Gradients through JAX's CPU Cholesky
# are only reproducible with OMP_NUM_THREADS=1 when two OpenMP runtimes are loaded
# (e.g. torch + conda).
# ============================================================================

# small bases keep these fast; N_GW is chosen so the global size (npsr * 2 N_GW) differs from the
# per-pulsar size 2 (N_RED + N_GW), which legitimately has a Cholesky in the decentering
N_RED, N_GW = 10, 4


def _phi_matrix(gp, params):
    """The global prior covariance as an array, whichever kernel class holds it."""
    getN = getattr(gp.Phi, "getN", None)
    return np.asarray(getN(params) if callable(getN) else gp.Phi.N(params))


@pytest.fixture(scope="module")
def hd_span(psrs):
    return ds.getspan(psrs)


@pytest.fixture(scope="module")
def hd_params():
    return {"gw_log10_A": -14.3, "gw_gamma": 3.7}


def test_hd_phi_is_kronecker(psrs, hd_span, hd_params):
    gp = ds.makeglobalgp_fourier(psrs, ds.powerlaw, ds.hd_orf, components=N_GW, T=hd_span, name="gw")
    f, df, _ = ds.fourierbasis(psrs[0], N_GW, hd_span)
    phi = np.asarray(ds.powerlaw(f, df, hd_params["gw_log10_A"], hd_params["gw_gamma"]))
    orf = np.array([[ds.hd_orf(a.pos, b.pos) for a in psrs] for b in psrs])

    expected = np.block([[orf[a, b] * np.diag(phi) for b in range(len(psrs))] for a in range(len(psrs))])
    Phi = _phi_matrix(gp, hd_params)
    np.testing.assert_allclose(Phi, expected, rtol=1e-12, atol=0)

    Phi_inv, logdet = gp.Phi_inv(hd_params)
    np.testing.assert_allclose(np.asarray(Phi_inv) @ expected, np.eye(len(expected)), atol=1e-8)
    np.testing.assert_allclose(float(logdet), np.linalg.slogdet(expected)[1], rtol=1e-12)


def test_hd_kron_factors_match_phi_inv(psrs, hd_span, hd_params):
    gp = ds.makeglobalgp_fourier(psrs, ds.powerlaw, ds.hd_orf, components=N_GW, T=hd_span, name="gw")
    assert gp.Phi_inv_kron is not None
    orf_inv, spectrum_inv = gp.Phi_inv_kron
    S_inv, logdet = spectrum_inv(hd_params)
    Phi_inv, logdet_ref = gp.Phi_inv(hd_params)
    np.testing.assert_allclose(np.kron(np.asarray(orf_inv), np.diag(np.asarray(S_inv))), np.asarray(Phi_inv),
                               rtol=1e-12, atol=0)
    np.testing.assert_allclose(float(logdet), float(logdet_ref), rtol=1e-12)


def _decentered_clogL(psrs, hd_span, prior_path):
    """Decentered commongp (intrinsic RN) + HD globalgp clogL with a chosen HD-prior path."""
    with metamatrix_patch():
        gw = ds.makeglobalgp_fourier(psrs, ds.powerlaw, ds.hd_orf, components=N_GW, T=hd_span, name="gw")
        if prior_path in ("phi_inv", "dense"):
            gw.Phi_inv_kron = None
        if prior_path == "dense":
            gw.Phi_inv = None
        model = ds.ArrayLikelihood(
            [ds.PulsarLikelihood([psr.residuals,
                                  ds.makenoise_measurement(psr, noisedict=psr.noisedict, ecorr=True),
                                  ds.makegp_timing(psr, svd=True)]) for psr in psrs],
            commongp=ds.makecommongp_fourier(psrs, ds.powerlaw, components=N_RED, T=hd_span, name="red_noise"),
            globalgp=gw, decenter=True)
        return model.clogL


@pytest.fixture(scope="module")
def clogL_params(psrs, hd_params):
    rng = np.random.default_rng(0)
    params = dict(hd_params)
    for psr in psrs:
        params[f"{psr.name}_red_noise_log10_A"] = -14.5
        params[f"{psr.name}_red_noise_gamma"] = 3.0
        params[f"{psr.name}_red_noise_coefficients({2 * N_RED})"] = jnp.asarray(rng.standard_normal(2 * N_RED))
        params[f"{psr.name}_gw_coefficients({2 * N_GW})"] = jnp.asarray(rng.standard_normal(2 * N_GW))
    return params


def test_mixed_logprior_paths_agree(psrs, hd_span, clogL_params):
    results = {}
    for path in ("kron", "phi_inv", "dense"):
        clogL = _decentered_clogL(psrs, hd_span, path)
        value, grad = jax.jit(jax.value_and_grad(lambda q: clogL(q)[0]))(clogL_params)
        results[path] = (float(value), grad)

    ref_value, ref_grad = results["dense"]
    for path in ("kron", "phi_inv"):
        value, grad = results[path]
        np.testing.assert_allclose(value, ref_value, rtol=1e-12, err_msg=f"{path} value")
        for key in ref_grad:
            np.testing.assert_allclose(np.asarray(grad[key]), np.asarray(ref_grad[key]), rtol=1e-8,
                                       atol=1e-10 * float(np.max(np.abs(ref_grad[key])) + 1e-30),
                                       err_msg=f"{path} gradient {key}")


def test_kron_path_has_no_dense_global_solve(psrs, hd_span, clogL_params):
    size = len(psrs) * 2 * N_GW
    clogL = _decentered_clogL(psrs, hd_span, "kron")          # build outside the trace
    jaxpr = jax.make_jaxpr(lambda q: clogL(q)[0])(clogL_params).jaxpr

    def dense_solves(jaxpr):
        found = []
        for eqn in jaxpr.eqns:
            inner = eqn.params.get("jaxpr") or eqn.params.get("call_jaxpr")
            inner = getattr(inner, "jaxpr", inner)
            if inner is not None and hasattr(inner, "eqns"):
                found += dense_solves(inner)
            elif eqn.primitive.name in ("lu", "custom_linear_solve", "cholesky", "triangular_solve"):
                if any(getattr(v.aval, "shape", ())[-2:] == (size, size) for v in eqn.invars):
                    found.append(eqn.primitive.name)
        return found

    assert dense_solves(jaxpr) == [], f"dense ({size} x {size}) factorization left in clogL"
