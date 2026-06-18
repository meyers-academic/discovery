"""Real-model discriminator table for the FUSED two-level reference+delta
(single-precision Half B, the Hellings-Downs case).

The HD analogue of ``harness_vectorwoodbury_refdelta.py`` (which covered the
no-HD single-level CURN path). Here the model has a per-pulsar intrinsic
red-noise (IRN) inner GP *and* a dense cross-pulsar HD GW GP, so the array logL
does NOT factorise: it routes through the fused two-level kernel
(``vectorwoodburyjointsolve`` -> ``globalwoodbury_fused``) and its refdelta twin
(``vectorwoodburyjointsolve_refdelta`` -> ``globalwoodbury_fused_refdelta``).

Crucially this drives the *production wiring*, not hand-assembled graph pieces:
``ArrayLikelihood(reference=theta_ref)`` freezes both prior covariances at the
reference and routes ``.logL`` to the fused twins. So the table is a direct test
of the end-state path a real HD analysis would use.

Model: per-pulsar IRN (powerlaw, 30 comp, name='red_noise') + HD global GW
(powerlaw, 14 comp, hd_orf, name='gw'). HD's ORF is parameter-free, so the
free params are the per-pulsar red_noise_* and the shared gw_log10_A/gw_gamma --
all present in the NG15 m3a chain, which supplies both the frozen reference
(chain median, ADR 0001) and the test draws.

Run:  python dev_architecture/single_precision/harness_fused_refdelta.py [N ...]
"""
import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp

import discovery as ds
from discovery import recipes, utils

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data"
PSR_FILES = sorted(DATA.glob("v1p1_de440_pint_bipm2019-*.feather"))
CHAIN = DATA / "NG15yr-m3a-chain.feather"


def _model(psrs, reference=None):
    """ArrayLikelihood: per-pulsar IRN (name='red_noise') + HD global GW
    (name='gw'). reference=None -> fused path; a dict -> fused refdelta twin."""
    T = ds.getspan(psrs)
    return ds.ArrayLikelihood(
        [recipes._psl_skeleton(p) for p in psrs],
        commongp=ds.makecommongp_fourier(psrs, ds.powerlaw, components=30, T=T,
                                         name="red_noise"),
        globalgp=ds.makeglobalgp_fourier(psrs, ds.powerlaw, ds.hd_orf,
                                         components=14, T=T, name="gw"),
        reference=reference,
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

    params = _model(psrs).logL.params
    missing = [p for p in params if p not in chain.columns]
    if missing:
        raise SystemExit(f"params not in chain: {missing[:6]}")
    theta_med = _median_point(params, chain)
    pts = _draws(params, chain, n_draws, seed)

    def _eval(dtype):
        # build the model INSIDE the dtype context: the working dtype is baked in
        # at materialization (cached .logL -> mm.func), so a fresh build per dtype
        # is required -- reusing an f64-built logL would silently run f64.
        utils.config(backend="jax", working=dtype)
        base = _model(psrs)                          # fused (Half-A combine)
        refd = _model(psrs, reference=theta_med)     # fused refdelta (Half-B)
        b = np.array([float(base.logL(pt)) for pt in pts])
        r = np.array([float(refd.logL(pt)) for pt in pts])
        utils.config(backend="jax", working=jnp.float64)
        return b, r

    b64, r64 = _eval(jnp.float64)
    b32, r32 = _eval(jnp.float32)
    return dict(n=n, ntoa=sum(len(p.residuals) for p in psrs),
                absL=np.abs(b64).max(),
                base_err=np.abs(b32 - b64).max(),
                refd_err=np.abs(r32 - r64).max(),
                f64_gap=np.abs(r64 - b64).max())   # refdelta f64 must equal fused f64


def main(ns):
    print(f"{'Npsr':>5} {'Ntoa':>8} {'|logL|':>12} {'fused_f32':>12} "
          f"{'refdelta_f32':>13} {'gain':>7} {'f64 chk':>9}")
    for n in ns:
        r = measure(n)
        gain = r["base_err"] / r["refd_err"] if r["refd_err"] > 0 else float("inf")
        print(f"{r['n']:>5} {r['ntoa']:>8} {r['absL']:>12.3g} {r['base_err']:>12.3g} "
              f"{r['refd_err']:>13.3g} {gain:>6.1f}x {r['f64_gap']:>9.2g}")


if __name__ == "__main__":
    ns = [int(a) for a in sys.argv[1:]] or [3, 6, 12, 45, 67]
    main(ns)
