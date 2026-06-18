"""Real-model discriminator table for the BATCHED single-level reference+delta
graph ``mh.vectorwoodbury_refdelta`` -- the production CURN+IRN case.

Unlike ``harness_refdelta_table.py`` (which summed the *scalar* ``woodbury_refdelta``
per pulsar in a Python loop), this drives the actual ARRAY graphs that a no-HD
``ArrayLikelihood`` uses: ``vectorwoodbury`` (direct, Half-A combined) vs
``vectorwoodbury_refdelta`` (Half-B). Same NG15 data, same m3a chain draws.

Model: ``recipes.intrinsic_plus_crn`` -- per-pulsar intrinsic red noise + a common
(CURN) spectrum on one shared basis via ``make_combined_crn``. No Hellings-Downs,
so the commongp routes through ``VectorWoodburyKernel`` -> ``vectorwoodbury`` (single
level, batched over pulsars). The sampled GP is the per-pulsar red noise (+ shared
CRN params); that is the cancellation reference+delta targets.

Reference Phi_ref = the commongp prior evaluated at the chain MEDIAN, frozen f64
(ADR 0001). Test points theta drawn from the NG15 m3a chain.

Run:  python dev_architecture/single_precision/harness_vectorwoodbury_refdelta.py [N ...]
"""
import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp

import discovery as ds
from discovery import recipes, signals, utils, metamath as mh, metamatrix as mm

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data"
PSR_FILES = sorted(DATA.glob("v1p1_de440_pint_bipm2019-*.feather"))
CHAIN = DATA / "NG15yr-m3a-chain.feather"


def _model(psrs):
    """ArrayLikelihood, commongp only: per-pulsar IRN + shared CRN (make_combined_crn).
    name='red_noise', crn_prefix='gw_' so params match the m3a chain columns."""
    T = ds.getspan(psrs)
    combined, crn_params = ds.make_combined_crn(14, ds.powerlaw, ds.powerlaw, crn_prefix="gw_")
    return ds.ArrayLikelihood(
        [recipes._psl_skeleton(p) for p in psrs],
        commongp=ds.makecommongp_fourier(psrs, combined, components=30, T=T,
                                         name="red_noise", common=crn_params),
    )


def _draws(params, chain, n_draws, seed):
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, len(chain), n_draws)
    return [{p: float(chain[p].iloc[i]) for p in params} for i in rows]


def _median_point(params, chain):
    return {p: float(np.median(chain[p].to_numpy())) for p in params}


def measure(n, n_draws=5, seed=0):
    import pandas as pd
    chain = pd.read_feather(CHAIN)
    psrs = [ds.Pulsar.read_feather(f) for f in PSR_FILES[:n]]

    ds.config(kernels="metamath")
    utils.config(backend="jax", working=jnp.float64)

    al = _model(psrs)
    params = al.logL.params
    missing = [p for p in params if p not in chain.columns]
    if missing:
        raise SystemExit(f"params not in chain: {missing[:6]}")
    theta_med = _median_point(params, chain)
    pts = _draws(params, chain, n_draws, seed)

    # trigger build of the vectorized kernel, then grab its pieces
    _ = al.logL
    vsm = al.vsm
    ys = al.ys
    Nsolves = [N.make_solve for N in vsm.Ns]

    # reference Phi_ref = commongp prior at the chain median, frozen f64 (ADR 0001)
    phi_ref = jnp.asarray(np.asarray(vsm.P.getN(theta_med)))
    Pref = mh.NoiseMatrix(phi_ref)

    def _eval(dtype):
        # rebuild the funcs INSIDE the dtype context: the working dtype is baked in
        # at materialization (mm.func), so an f64-built callable will not run f32.
        utils.config(backend="jax", working=dtype)
        direct_f = mm.func(mh.vectorwoodbury(ys, Nsolves, vsm.Fs, vsm.P.make_inv))
        refd_f = mm.func(mh.vectorwoodbury_refdelta(ys, Nsolves, vsm.Fs, vsm.P.make_inv, Pref.make_inv))
        base = np.array([float(direct_f(params=pt)) for pt in pts])
        refd = np.array([float(refd_f(params=pt)) for pt in pts])
        utils.config(backend="jax", working=jnp.float64)
        return base, refd

    b64, r64 = _eval(jnp.float64)
    b32, r32 = _eval(jnp.float32)
    return dict(n=n, ntoa=sum(len(p.residuals) for p in psrs),
                absL=np.abs(b64).max(),
                base_err=np.abs(b32 - b64).max(),
                refd_err=np.abs(r32 - r64).max(),
                f64_gap=np.abs(r64 - b64).max())   # refdelta f64 must equal direct f64


def main(ns):
    print(f"{'Npsr':>5} {'Ntoa':>8} {'|logL|':>12} {'vecwood_f32':>13} "
          f"{'refdelta_f32':>13} {'gain':>7} {'f64 chk':>9}")
    for n in ns:
        r = measure(n)
        gain = r["base_err"] / r["refd_err"] if r["refd_err"] > 0 else float("inf")
        print(f"{r['n']:>5} {r['ntoa']:>8} {r['absL']:>12.3g} {r['base_err']:>13.3g} "
              f"{r['refd_err']:>13.3g} {gain:>6.1f}x {r['f64_gap']:>9.2g}")


if __name__ == "__main__":
    ns = [int(a) for a in sys.argv[1:]] or [3, 6, 12, 45, 67]
    main(ns)
