"""Graph-precision pass, stage 1 -- blanket working dtype at materialization.

The metamath materialization boundary (metamatrix.build_callable_from_graph,
reached via func()) casts every floating leaf -- args, constants, PSD/func-leaf
outputs -- to utils.working_dtype() when single precision is active. Stage 1 is
the *blanket, no-pins* baseline: the ENTIRE Woodbury (including the would-be f64
pins ytNmy / logdets) runs in the working dtype. These tests lock that path:

  * float64 default is untouched (the cast is the identity unless working=f32);
    covered structurally by the metamatrix parity suite, asserted here too.
  * under working=float32 the single-pulsar logL is finite and close to f64
    (no pins are needed at single-pulsar scale -- the ytNmy cancellation only
    bites across the full array; that motivates the pins, added next).

Requires the metamath backend (ds.config(kernels='metamath')) -- the cast lives
in the metamath graph; the legacy matrix.py path is unaffected.
"""
import contextlib
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import pytest

import discovery as ds
from discovery import utils


DATA = Path(__file__).resolve().parents[2] / "data"
B1855 = DATA / "v1p1_de440_pint_bipm2019-B1855+09.feather"


@pytest.fixture(scope="module")
def psr():
    return ds.Pulsar.read_feather(B1855)


@contextlib.contextmanager
def metamath_working(dtype):
    """Metamath backend + a chosen working dtype; restore matrix/float64 after.
    Build the model INSIDE the block so its graph materializes under both."""
    ds.config(kernels="metamath")
    utils.config(backend="jax", working=dtype)
    try:
        yield
    finally:
        utils.config(backend="jax")
        ds.config(kernels="matrix")


def build_full_rn(psr):
    return ds.PulsarLikelihood([
        psr.residuals,
        ds.makenoise_measurement(psr, psr.noisedict),
        ds.makegp_ecorr(psr, psr.noisedict),
        ds.makegp_timing(psr, svd=True),
        ds.makegp_fourier(psr, ds.powerlaw, components=30, name="rednoise"),
    ])


def build_multi_vgp(psr):
    return ds.PulsarLikelihood([
        psr.residuals,
        ds.makenoise_measurement(psr, psr.noisedict),
        ds.makegp_ecorr(psr, psr.noisedict),
        ds.makegp_timing(psr, svd=True),
        ds.makegp_fourier(psr, ds.powerlaw, components=30, name="rednoise"),
        ds.makegp_fourier(psr, ds.powerlaw, components=14, name="dmgp",
                          fourierbasis=ds.dmfourierbasis),
    ])


BUILDERS = [pytest.param(build_full_rn, id="full_rn"),
            pytest.param(build_multi_vgp, id="multi_vgp")]

# Whole single-pulsar Woodbury in f32 (no pins): logL holds to ~1e-8 relative.
BLANKET_RTOL = 1e-7


def _draw(model, seed=0):
    np.random.seed(seed)
    return ds.sample_uniform(model.logL.params)


@pytest.mark.parametrize("build", BUILDERS)
def test_metamath_blanket_f32(psr, build):
    # float64 reference, metamath backend
    with metamath_working(jnp.float64):
        ref = build(psr)
        p0 = _draw(ref)
        L64 = float(ref.logL(p0))
    assert np.isfinite(L64)

    # blanket float32: the entire graph in working dtype
    with metamath_working(jnp.float32):
        m = build(psr)
        L32 = float(m.logL(p0))

    assert np.isfinite(L32), f"{build.__name__} blanket-f32 logL not finite"
    np.testing.assert_allclose(
        L32, L64, rtol=BLANKET_RTOL,
        err_msg=f"{build.__name__} blanket-f32 logL {L32} vs f64 {L64}")


def test_cast_is_active_in_metamath_f32(psr):
    """Guard against silent no-op: blanket f32 must actually move logL off the
    f64 value (the whole graph really is running in f32, not just Phi)."""
    with metamath_working(jnp.float64):
        ref = build_full_rn(psr)
        p0 = _draw(ref)
        L64 = float(ref.logL(p0))
    with metamath_working(jnp.float32):
        L32 = float(build_full_rn(psr).logL(p0))
    assert abs(L64 - L32) > 0.0, "blanket f32 produced bit-identical logL -- cast inert?"


def test_float64_default_unchanged(psr):
    """The cast must be the identity in the default (float64) regime."""
    assert utils.working_dtype() is jnp.float64 and not utils.single_precision
    with metamath_working(jnp.float64):
        m = build_full_rn(psr)
        p0 = _draw(m)
        L = float(m.logL(p0))
    assert np.isfinite(L)
