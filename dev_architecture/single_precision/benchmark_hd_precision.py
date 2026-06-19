"""Benchmark: legacy matrix.py vs metamath, single- vs double-precision, on the
same Hellings-Downs array model.

Three configs, all on the *same* real NG15 HD model, so timings/memory/accuracy
are directly comparable:

  * matrix_f64    -- legacy closure path (likelihood.py), float64, reference=None.
                     The trusted production baseline; used as the accuracy truth.
  * metamath_f64  -- graph path (likelihood_metamath.py), NORMAL fused kernel,
                     float64, reference=None.
  * metamath_f32  -- graph path, REFERENCE+DELTA fused kernel, float32,
                     reference=theta_ref (chain median). The single-precision win.

For each (pulsar count N, config) it reports, per logL call (after JIT warmup):

    ms/call  -- median wall-clock of one logL(params) call, ms
    cur_MB   -- device memory currently in use, MB
    peak_MB  -- device peak bytes in use, MB (the headline GPU footprint)
    acc      -- max |logL_config - logL_matrix_f64| over the draws (the three
                f64 configs should agree to ~1e-9; f32 shows its floor)

So one run answers: is metamath actually slower than matrix.py? does f32-refdelta
on the GPU beat both? what does each cost in memory? and how accurate is f32?

Memory note: JAX's peak_bytes_in_use is cumulative for the process, so in the
default (all-configs) mode each peak includes the earlier configs' allocations.
For a clean per-config GPU footprint, run ONE config per process:

    python benchmark_hd_precision.py --only matrix_f64   --n 45
    python benchmark_hd_precision.py --only metamath_f64 --n 45
    python benchmark_hd_precision.py --only metamath_f32 --n 45

Model: per-pulsar IRN (powerlaw, 30 comp, name='red_noise') + HD global GW
(powerlaw, 14 comp, hd_orf, name='gw') -- identical to harness_fused_refdelta.py.

Run:
    python dev_architecture/single_precision/benchmark_hd_precision.py [N ...]
    python dev_architecture/single_precision/benchmark_hd_precision.py 6 12 45
    python dev_architecture/single_precision/benchmark_hd_precision.py --only metamath_f32 45
"""
import argparse
import time
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

# (name, kernels-backend, working dtype, uses-reference)
CONFIGS = [
    ("matrix_f64",   "matrix",   jnp.float64, False),
    ("metamath_f64", "metamath", jnp.float64, False),
    ("metamath_f32", "metamath", jnp.float32, True),
]


def _model(psrs, reference=None):
    """ArrayLikelihood (whichever backend ds.config currently selects): per-pulsar
    IRN (name='red_noise') + HD global GW (name='gw'). reference=None -> normal
    path; a dict -> metamath fused refdelta."""
    T = ds.getspan(psrs)
    kw = dict(
        commongp=ds.makecommongp_fourier(psrs, ds.powerlaw, components=30, T=T,
                                         name="red_noise"),
        globalgp=ds.makeglobalgp_fourier(psrs, ds.powerlaw, ds.hd_orf,
                                         components=14, T=T, name="gw"),
    )
    # `reference` is a metamath-only kwarg; the legacy matrix ArrayLikelihood does
    # not accept it, so only pass it when a reference is actually supplied.
    if reference is not None:
        kw["reference"] = reference
    return ds.ArrayLikelihood([recipes._psl_skeleton(p) for p in psrs], **kw)


def _draws(params, chain, n_draws, seed):
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, len(chain), n_draws)
    return [{p: float(chain[p].iloc[i]) for p in params} for i in rows]


def _median_point(params, chain):
    return {p: float(np.median(chain[p].to_numpy())) for p in params}


def _mem_mb():
    """(current, peak) device memory in MB, or (nan, nan) if unavailable (CPU)."""
    try:
        s = jax.local_devices()[0].memory_stats()
        return (s.get("bytes_in_use", 0) / 1e6,
                s.get("peak_bytes_in_use", 0) / 1e6)
    except Exception:
        return (float("nan"), float("nan"))


def _time_call(logL, pts, n_repeat=10):
    """Median ms per call. Warms up (JIT compile) on pts[0], then times each draw
    n_repeat times, blocking on the device so we time real compute."""
    jax.block_until_ready(logL(pts[0]))
    ts = []
    for pt in pts:
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            jax.block_until_ready(logL(pt))
            ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1e3)


def _build_eval(psrs, pts, kernels, dtype, reference):
    """Build the model INSIDE the (kernels, dtype) context -- both the backend and
    the working dtype are baked in at construction (mm.func materialization), so a
    model built under one and reused under another would silently run the wrong
    thing. Returns (logL values, ms/call)."""
    ds.config(kernels=kernels)
    utils.config(backend="jax", working=dtype)
    model = _model(psrs, reference=reference)
    logL = model.logL
    vals = np.array([float(logL(pt)) for pt in pts])
    ms = _time_call(logL, pts)
    return vals, ms


def measure(n, only=None, n_draws=5, seed=0):
    import pandas as pd
    chain = pd.read_feather(CHAIN)
    psrs = [ds.Pulsar.read_feather(f) for f in PSR_FILES[:n]]

    # params + reference point (build a throwaway matrix model just for names)
    ds.config(kernels="matrix")
    utils.config(backend="jax", working=jnp.float64)
    params = _model(psrs).logL.params
    missing = [p for p in params if p not in chain.columns]
    if missing:
        raise SystemExit(f"params not in chain: {missing[:6]}")
    theta_med = _median_point(params, chain)
    pts = _draws(params, chain, n_draws, seed)
    ntoa = sum(len(p.residuals) for p in psrs)

    truth = None
    rows = []
    for name, kernels, dtype, uses_ref in CONFIGS:
        if only is not None and name != only:
            continue
        reference = theta_med if uses_ref else None
        vals, ms = _build_eval(psrs, pts, kernels, dtype, reference)
        cur, peak = _mem_mb()
        if name == "matrix_f64":
            truth = vals
        acc = np.abs(vals - truth).max() if truth is not None else float("nan")
        rows.append(dict(n=n, ntoa=ntoa, config=name, ms=ms,
                         cur=cur, peak=peak, acc=acc, absL=np.abs(vals).max()))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ns", nargs="*", type=int, default=[3, 6, 12, 45, 67])
    ap.add_argument("--only", choices=[c[0] for c in CONFIGS], default=None,
                    help="measure a single config (clean per-config GPU peak)")
    ap.add_argument("--n", type=int, default=None,
                    help="single pulsar count (shorthand for one N)")
    args = ap.parse_args()
    ns = [args.n] if args.n is not None else args.ns

    dev = jax.local_devices()[0]
    print(f"device: {dev.platform} {dev.device_kind}")
    print(f"{'Npsr':>5} {'Ntoa':>8} {'config':>14} | {'ms/call':>9} "
          f"{'cur_MB':>8} {'peak_MB':>8} | {'acc_vs_matrix':>13}")
    for n in ns:
        for r in measure(n, only=args.only):
            print(f"{r['n']:>5} {r['ntoa']:>8} {r['config']:>14} | {r['ms']:>9.3f} "
                  f"{r['cur']:>8.1f} {r['peak']:>8.1f} | {r['acc']:>13.3g}")
        print()


if __name__ == "__main__":
    main()
