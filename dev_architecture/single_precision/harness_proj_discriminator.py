"""Discriminator: is timing-model projection a *conditioning* tool?

Session 3 (HANDOFF_projection.md) found that turning projection ON did NOT improve
the float32 logL error on the standard array harness -- because that harness builds
the timing model with ``svd=True``, which orthonormalises the design M. With an
orthonormal M the 1e40 timing prior lands as a harmless 1e-40 block that flushes to
zero; the baseline Woodbury is already float32-safe, so projection has nothing to win.

The open question the handoff leaves: with ``svd=False`` (the *raw* timing design,
which spans many orders of magnitude -- quadratic spindown terms, etc.), the baseline
1e40 Woodbury should genuinely lose conditioning in float32, and projection (which
never forms the 1e40 block, and whitens before subtracting) should hold. If so,
projection is a *conditioning* tool worth keeping for raw-basis / harder datasets,
even though it is not a *precision* tool on the SVD harness.

This script runs the 2x2 sweep -- (svd, project) in {True, False}^2 -- and reports
the float32-vs-float64 abs logL error in each cell. It also exploits a correctness
cross-check unique to projection: because projection depends only on the *column
span* of M (it is an orthogonal projector onto that span), the project=True logL must
be identical -- to f64 tol -- whether M is the raw design (svd=False) or its
orthonormalised version (svd=True). The raw 1e40 Woodbury enjoys no such invariance.

Run:  python dev_architecture/single_precision/harness_proj_discriminator.py [N ...]
"""
import sys
import contextlib
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp

import discovery as ds
from discovery import utils

DATA = Path(__file__).resolve().parents[2] / "data"
PSR_FILES = sorted(p for p in DATA.glob("v1p1_de440_pint_bipm2019-*.feather"))
CHAIN_FILE = DATA / "NG15yr-m3a-chain.feather"   # per-pulsar red_noise + global HD gw


@contextlib.contextmanager
def _working(dtype):
    ds.config(kernels="metamath")
    utils.config(backend="jax", working=dtype)
    try:
        yield
    finally:
        utils.config(backend="jax")
        ds.config(kernels="matrix")


def load_psrs(n):
    if n > len(PSR_FILES):
        raise ValueError(f"requested {n} pulsars, only {len(PSR_FILES)} available")
    return [ds.Pulsar.read_feather(f) for f in PSR_FILES[:n]]


def build(psrs, svd, project):
    """Production fused-path array model, with the timing model built either as the
    raw design (svd=False) or orthonormalised (svd=True), and either projected
    (project=True, ADR 0004) or given the ordinary 1e40 Woodbury prior."""
    T = ds.getspan(psrs)
    psls = [ds.PulsarLikelihood([psr.residuals,
                                 ds.makenoise_measurement(psr, psr.noisedict),
                                 ds.makegp_ecorr(psr, psr.noisedict),
                                 ds.makegp_timing(psr, svd=svd, project=project)])
            for psr in psrs]
    return ds.ArrayLikelihood(
        psls,
        commongp=ds.makecommongp_fourier(psrs, ds.powerlaw, components=30, T=T,
                                         name="red_noise"),
        globalgp=ds.makeglobalgp_fourier(psrs, ds.powerlaw, ds.hd_orf,
                                         components=14, T=T, name="gw"))


def _draws(params, n_draws, seed=0):
    """Physically-realistic parameter points drawn from the NG15 m3a posterior
    chain (per-pulsar red_noise_{gamma,log10_A} + global gw_{gamma,log10_A}),
    instead of broad-prior `sample_uniform` draws that wander into extreme
    (efac~10) regions where the f32 Cholesky NaNs for reasons unrelated to the
    timing treatment. Model param names match the chain columns exactly."""
    import pandas as pd
    chain = pd.read_feather(CHAIN_FILE)
    missing = [p for p in params if p not in chain.columns]
    if missing:
        raise KeyError(f"chain {CHAIN_FILE.name} is missing model params: {missing[:5]}")
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, len(chain), size=n_draws)
    return [{p: float(chain[p].iloc[i]) for p in params} for i in rows]


def _eval(psrs, svd, project, pts, dtype):
    with _working(dtype):
        m = build(psrs, svd, project)
        return np.array([float(m.logL(p)) for p in pts])


def measure(n, n_draws=5, seed=0):
    psrs = load_psrs(n)
    # Shared parameter draws: the projected and unprojected models have the SAME
    # hyperparameters (only the timing marginalisation differs), so one draw set works.
    with _working(jnp.float64):
        m = build(psrs, svd=True, project=False)
        pts = _draws(m.logL.params, n_draws, seed)

    cells = {}
    f64 = {}
    for svd in (True, False):
        for project in (True, False):
            L64 = _eval(psrs, svd, project, pts, jnp.float64)
            L32 = _eval(psrs, svd, project, pts, jnp.float32)
            f64[(svd, project)] = L64
            ok = np.isfinite(L32)
            abs_err = np.abs(L32[ok] - L64[ok])
            cells[(svd, project)] = dict(
                absL=np.abs(L64).max(),
                # abs_err measured over the FINITE draws only; n_finite separates
                # "timing-block conditioning" from "occasional GP-block f32 NaN".
                abs_err=abs_err.max() if abs_err.size else np.nan,
                n_finite=int(ok.sum()), n_draws=len(pts))

    # basis-invariance cross-check: projection logL must match across svd True/False
    # (same column span). Raw 1e40 Woodbury need not.
    proj_invariance = np.abs(f64[(True, True)] - f64[(False, True)]).max()
    wood_invariance = np.abs(f64[(True, False)] - f64[(False, False)]).max()
    return dict(n=n, ntoa=sum(len(p.residuals) for p in psrs),
                cells=cells,
                proj_invariance=proj_invariance,
                wood_invariance=wood_invariance)


def run_sweep(ns, n_draws=5, seed=0):
    for n in ns:
        r = measure(n, n_draws, seed)
        print(f"\n=== Npsr={r['n']}  Ntoa={r['ntoa']} ===")
        print(f"{'svd':>5} {'project':>8} {'|logL|':>12} {'abs_err(f32)':>13} {'finite':>9}")
        for svd in (True, False):
            for project in (True, False):
                c = r['cells'][(svd, project)]
                print(f"{str(svd):>5} {str(project):>8} {c['absL']:>12.4e} "
                      f"{c['abs_err']:>13.3e} {c['n_finite']:>4}/{c['n_draws']:<4}")
        print(f"  basis-invariance (f64, svd T vs F):  "
              f"projection={r['proj_invariance']:.3e}   1e40-Woodbury={r['wood_invariance']:.3e}")


if __name__ == "__main__":
    ns = [int(a) for a in sys.argv[1:]] or [6, 24]
    run_sweep(ns)
