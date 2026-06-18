"""Large-array float32-vs-float64 validation harness (Piece 2, roadmap step 2).

The measuring stick for everything downstream. Builds the *production* fused-path
array model (per-pulsar white+ECORR+timing noise, a common red-noise GP, and a
global Hellings--Downs GP) at a chosen number of pulsars, evaluates logL in float64
and float32 at the SAME parameter draws, and reports:

  * |logL|                         -- the scale the f32 ulp must sit under
  * abs/rel error on the absolute logL          (f32 vs f64)
  * **ΔlogL error**                -- the sampling-relevant quantity: the error on
    logL(θ_a) − logL(θ_b) between draws, which is what an MCMC actually sees.

What this measures depends on what's implemented in the kernel:
  * With **Half A** (the f64 final combination, now in metamath.py via
    `combine_logp_f64`), `working=float32` gives a *float64* logL computed from
    *float32 components* (mu from the f32 Cholesky). So this harness measures the
    quantity that actually matters -- "does f64-logL-with-f32-components equal
    all-f64 logL?" -- NOT the strawman of storing a 1e6 logL in float32.
  * The residual ΔlogL error it reports is therefore the f32-*component* error
    (mu born from the f32 Cholesky; plus, in the fused path, the inner ytNmy_proj
    f32 cancellation that cannot be pinned). That is what **reference+delta**
    (Half B) must remove -- re-run this harness after Half B and the residual
    should drop to ~1e-6-relative.

The headline question: as the array grows, does that residual stay below the ~1e-2
precision we sample to? (Empirically, no -- it sits at ~0.06 already at 3-6 pulsars,
which is the motivation for Half B.)

Run:  python dev_architecture/single_precision/harness_f32_array.py [N ...]
Import:  from harness_f32_array import run_sweep; run_sweep([3, 10, 20], n_draws=5)

Uses fixed (noisedict) white noise on purpose: the f64 pins then fold to constants,
and we isolate the GP-driven cancellation. (Sampled WN can overflow the f32 Cholesky
on physically extreme sample_uniform draws -- a separate, known conditioning issue.)

NOTE: with Half A in place, `working=float32` already returns a float64 logL, so
`abs_err` dropping *below* `f32_ulp` is expected and confirms the f64 combine; the
remaining `dlogL_err` is the component error Half B targets.
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
# the v1p1 single-pulsar feathers (exclude the NG15yr chain files)
PSR_FILES = sorted(p for p in DATA.glob("v1p1_de440_pint_bipm2019-*.feather"))


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


def build(psrs):
    """The production fused-path array model (commongp red noise + globalgp HD).
    Must be called inside a `_working(dtype)` block so the graph materializes in
    that dtype."""
    T = ds.getspan(psrs)
    psls = [ds.PulsarLikelihood([psr.residuals,
                                 ds.makenoise_measurement(psr, psr.noisedict),
                                 ds.makegp_ecorr(psr, psr.noisedict),
                                 ds.makegp_timing(psr, svd=True)]) for psr in psrs]
    return ds.ArrayLikelihood(
        psls,
        commongp=ds.makecommongp_fourier(psrs, ds.powerlaw, components=30, T=T,
                                         name="rednoise"),
        globalgp=ds.makeglobalgp_fourier(psrs, ds.powerlaw, ds.hd_orf,
                                         components=14, T=T, name="gw"))


def _draws(params, n_draws, seed=0):
    np.random.seed(seed)
    return [ds.sample_uniform(params) for _ in range(n_draws)]


def measure(n, n_draws=5, seed=0):
    """Build at f64 and f32, evaluate the same draws, return a metrics dict."""
    psrs = load_psrs(n)

    with _working(jnp.float64):
        m64 = build(psrs)
        params = m64.logL.params
        pts = _draws(params, n_draws, seed)
        L64 = np.array([float(m64.logL(p)) for p in pts])

    with _working(jnp.float32):
        m32 = build(psrs)
        L32 = np.array([float(m32.logL(p)) for p in pts])

    abs_err = np.abs(L32 - L64)
    # sampling-relevant: error on every pairwise difference logL(a)-logL(b)
    dd = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d64, d32 = L64[i] - L64[j], L32[i] - L32[j]
            dd.append(abs(d32 - d64))
    dlogL_err = np.array(dd) if dd else np.array([np.nan])
    return dict(n=n, ntoa=sum(len(p.residuals) for p in psrs),
                absL=np.abs(L64).max(), abs_err=abs_err.max(),
                rel_err=(abs_err / np.abs(L64)).max(),
                dlogL_err=dlogL_err.max(),
                f32_ulp=np.abs(L64).max() * np.finfo(np.float32).eps,
                finite=bool(np.isfinite(L32).all()))


def run_sweep(ns, n_draws=5, seed=0):
    print(f"{'Npsr':>5} {'Ntoa':>7} {'|logL|':>12} {'absLerr':>11} "
          f"{'relLerr':>10} {'dlogLerr':>11} {'f32 ulp':>11} {'finite':>7}")
    rows = []
    for n in ns:
        r = measure(n, n_draws, seed)
        rows.append(r)
        print(f"{r['n']:>5} {r['ntoa']:>7} {r['absL']:>12.4e} {r['abs_err']:>11.3e} "
              f"{r['rel_err']:>10.2e} {r['dlogL_err']:>11.3e} {r['f32_ulp']:>11.3e} "
              f"{str(r['finite']):>7}")
    return rows


if __name__ == "__main__":
    ns = [int(a) for a in sys.argv[1:]] or [3, 6, 12, 24]
    run_sweep(ns)
