#!/usr/bin/env python3
"""Regression tests for the optimal statistic (discovery.optimal).

These pin the OS outputs on a small, fixed 3-pulsar dataset so that
performance refactors (e.g. rewriting ``trace(A @ B)`` as ``sum(A * B)``
in the pairwise normalization) provably do not change results.

The golden values were generated from the implementation *before* the
trace->sum refactor, with the fixed ``PARAMS`` below. ``rtol=1e-8`` is far
looser than the ~1e-12 floating-point reassociation the refactor introduces,
but tight enough to catch any genuine change in the computation.
"""

from pathlib import Path
import pytest

import numpy as np

import jax
jax.config.update('jax_enable_x64', True)

import discovery as ds
from discovery import optimal


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PSR_FILES = [
    "v1p1_de440_pint_bipm2019-B1855+09.feather",
    "v1p1_de440_pint_bipm2019-J0023+0923.feather",
    "v1p1_de440_pint_bipm2019-J0030+0451.feather",
]

# fixed parameters used to generate the golden values
PARAMS = {
    'B1855+09_rednoise_gamma': 3.2,   'B1855+09_rednoise_log10_A': -14.5,
    'J0023+0923_rednoise_gamma': 3.2, 'J0023+0923_rednoise_log10_A': -14.5,
    'J0030+0451_rednoise_gamma': 3.2, 'J0030+0451_rednoise_log10_A': -14.5,
    'gw_gamma': 3.2,                  'gw_log10_A': -14.5,
}

# golden values (generated pre-refactor; see module docstring)
GOLDEN_OS = {
    'hd':     dict(os=8.3961773831359343e-29, os_sigma=3.3828949412484901e-29, snr=2.4819503794691431e+00),
    'mono':   dict(os=1.7353193676835731e-30, os_sigma=9.2639350397105645e-30, snr=1.8731989810431465e-01),
    'dipole': dict(os=3.7929188575728078e-29, os_sigma=1.8191630514555593e-29, snr=2.0849801531193126e+00),
}
GOLDEN_Q = dict(eig_min=-1.7242147637816876e-01, eig_max=2.0158562767707627e-01)
GOLDEN_CDF_X = np.array([0.0, 1.0, 2.0])
GOLDEN_CDF = np.array([5.1039176222643856e-01, 8.5178948105146968e-01, 9.7186069481346693e-01])

RTOL = 1e-8


@pytest.fixture(scope="module")
def os_obj():
    psrs = [ds.Pulsar.read_feather(DATA_DIR / f) for f in PSR_FILES]
    T = ds.getspan(psrs)
    gbl = ds.GlobalLikelihood([
        ds.PulsarLikelihood([
            psr.residuals,
            ds.makenoise_measurement(psr, psr.noisedict),
            ds.makegp_ecorr(psr, psr.noisedict),
            ds.makegp_timing(psr, svd=True),
            ds.makegp_fourier(psr, ds.powerlaw, 30, T=T, name='rednoise'),
            ds.makegp_fourier(psr, ds.powerlaw, 14, T=T,
                              common=['gw_log10_A', 'gw_gamma'], name='gw'),
        ]) for psr in psrs])
    return ds.OS(gbl)


@pytest.mark.parametrize("orfname,orf", [
    ('hd', optimal.hd_orfa),
    ('mono', optimal.monopole_orfa),
    ('dipole', optimal.dipole_orfa),
])
def test_os_regression(os_obj, orfname, orf):
    """OS point estimate / sigma / snr must match pinned golden values."""
    result = os_obj.os(PARAMS, orf)
    golden = GOLDEN_OS[orfname]
    for key in ('os', 'os_sigma', 'snr'):
        np.testing.assert_allclose(float(result[key]), golden[key], rtol=RTOL,
                                   err_msg=f"{orfname} {key} changed")


def test_Q_eigenvalues_regression(os_obj):
    """Eigenvalues of the OS quadratic-form matrix Q must be unchanged."""
    eigs = np.linalg.eigvalsh(np.asarray(os_obj.Q(PARAMS)))
    np.testing.assert_allclose(eigs.min(), GOLDEN_Q['eig_min'], rtol=RTOL)
    np.testing.assert_allclose(eigs.max(), GOLDEN_Q['eig_max'], rtol=RTOL)


def test_gx2cdf_regression(os_obj):
    """quadax-based gx2cdf must match pinned golden values."""
    cdf = np.asarray(os_obj.gx2cdf(PARAMS, GOLDEN_CDF_X))
    np.testing.assert_allclose(cdf, GOLDEN_CDF, rtol=RTOL)


def test_trace_sum_identity():
    """Document the refactor's justification: trace(A @ B) == sum(A * B)
    for symmetric A, B (so the OS pairwise normalization is unchanged)."""
    rng = np.random.default_rng(0)
    for m in (4, 28):
        A = rng.normal(size=(m, m)); A = A + A.T
        B = rng.normal(size=(m, m)); B = B + B.T
        np.testing.assert_allclose(np.trace(A @ B), np.sum(A * B), rtol=1e-12)
