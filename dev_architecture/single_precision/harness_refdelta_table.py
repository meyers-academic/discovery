"""The discriminator table, now for reference+delta (Piece 2 'Half B').

`finding_projection_discriminator.md` measured the f32-vs-f64 abs logL error of the
DIRECT woodbury path as a function of array size, on realistic m3a posterior draws.
This harness builds the same kind of table for **`woodbury_refdelta`** and puts it
next to the direct baseline, so we can see how much the increment formula buys at
each scale.

Model / scope. `woodbury_refdelta` is the single-level (per-pulsar) graph; it is not
yet wired into the fused cross-pulsar (HD) path. But for a model with NO Hellings-Downs
coupling the array likelihood **factorises**, logL = sum_i logL_i, so summing the
per-pulsar single-level results IS the exact array logL. We therefore use the
per-pulsar noise model **white + ECORR + intrinsic red-noise GP** (timing omitted --
it is handled by projection, shown neutral in the discriminator; including it would
need a projection+refdelta composite and a 1e40-safe path). The sampled GP is the
per-pulsar red noise; that is exactly the cancellation refdelta targets.

Two ingredients per pulsar, both from the data:
  * the test parameter point theta -- drawn from the NG15 m3a posterior chain;
  * the reference covariance Phi_ref -- the red-noise prior evaluated at the chain
    MEDIAN (a "good spot", per Patrick), frozen as an f64 constant leaf.

Run:  python dev_architecture/single_precision/harness_refdelta_table.py [N ...]
"""
import sys
import contextlib
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp

import discovery as ds
from discovery import signals, utils, metamath as mh, metamatrix as mm

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data"
PSR_FILES = sorted(DATA.glob("v1p1_de440_pint_bipm2019-*.feather"))
CHAIN = DATA / "NG15yr-m3a-chain.feather"
RN_COMPONENTS = 30


@contextlib.contextmanager
def _working(dtype):
    ds.config(kernels="metamath")
    utils.config(backend="jax", working=dtype)
    try:
        yield
    finally:
        utils.config(backend="jax")
        ds.config(kernels="matrix")


def _kernel(psr):
    """Outermost WoodburyKernel for white+ECORR+red-noise (no timing). Its .N is
    the (white+ECORR) inner solve, .F the red-noise basis, .P the red-noise prior."""
    T = signals.getspan([psr])
    psl = ds.PulsarLikelihood([psr.residuals,
                               signals.makenoise_measurement(psr, psr.noisedict),
                               signals.makegp_ecorr(psr, psr.noisedict),
                               signals.makegp_fourier(psr, signals.powerlaw,
                                                      RN_COMPONENTS, T=T, name="red_noise")])
    return psl


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

    # param set + chain-median reference Phi arrays, computed ONCE in f64 (no
    # nested _working -- nesting collides on the global kernels/working config).
    with _working(jnp.float64):
        allparams = sorted(set().union(*[set(_kernel(p).logL.params) for p in psrs]))
        theta_med = _median_point(allparams, chain)
        phi_refs = [np.asarray(_kernel(p).N.P.N(theta_med)) for p in psrs]
    pts = _draws(allparams, chain, n_draws, seed)

    def _eval_all(dtype):
        """Return (baseline_totals, refdelta_totals) summed over pulsars, per draw."""
        base = np.zeros(n_draws); refd = np.zeros(n_draws)
        with _working(dtype):
            for psr, phi_ref in zip(psrs, phi_refs):
                psl = _kernel(psr)
                K = psl.N                         # red-noise WoodburyKernel
                y = psl.y
                Pref = mh.NoiseMatrix(jnp.asarray(phi_ref))   # frozen f64 reference
                fb = mm.func(K.make_kernelproduct(y))
                fr = mm.func(mh.woodbury_refdelta(y, K.N.make_solve, K.F,
                                                  K.P.make_inv, Pref.make_inv))
                for j, pt in enumerate(pts):
                    base[j] += float(fb(params=pt))
                    refd[j] += float(fr(params=pt))
        return base, refd

    b64, r64 = _eval_all(jnp.float64)
    b32, r32 = _eval_all(jnp.float32)
    return dict(n=n, ntoa=sum(len(p.residuals) for p in psrs),
                absL=np.abs(b64).max(),
                base_err=np.abs(b32 - b64).max(),
                refd_err=np.abs(r32 - r64).max(),
                # cross-check: refdelta f64 must equal baseline f64 (same logL)
                f64_gap=np.abs(r64 - b64).max())


def run_sweep(ns, n_draws=5, seed=0):
    print(f"{'Npsr':>5} {'Ntoa':>8} {'|logL|':>12} {'woodbury_f32':>13} "
          f"{'refdelta_f32':>13} {'gain':>7} {'f64 chk':>9}")
    for n in ns:
        r = measure(n, n_draws, seed)
        gain = r['base_err'] / r['refd_err'] if r['refd_err'] > 0 else np.inf
        print(f"{r['n']:>5} {r['ntoa']:>8} {r['absL']:>12.4e} {r['base_err']:>13.3e} "
              f"{r['refd_err']:>13.3e} {gain:>7.1f} {r['f64_gap']:>9.1e}")


if __name__ == "__main__":
    ns = [int(a) for a in sys.argv[1:]] or [3, 6, 12, 45, 67]
    run_sweep(ns)
