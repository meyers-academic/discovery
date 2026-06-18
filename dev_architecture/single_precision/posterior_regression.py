"""f32-vs-f64 POSTERIOR regression — does single precision move the posterior?

Sets "the bar" from data, not from the absolute logL error. The discriminator
(`finding_projection_discriminator.md`) showed the f32 logL has a ~f32-eps *relative*
floor (~0.75 abs ΔlogL at NG15 scale); the open question (ADR 0003) is whether that
floor actually moves the posterior a sampler would see.

Plan (Patrick, this session):
  * Model = NG15 **m2a / CURN** (per-pulsar red noise + one common uncorrelated
    process, NO Hellings-Downs) over the full **67 pulsars** — the model the
    `NG15yr-m2a-chain.feather` (f64, production) was run with.
  * Sample it in **float32** via the metamath kernel, **timing-model projection ON**
    (`project=True`), with NUTS: 200 warmup + 200 samples, `max_tree_depth=6`.
  * Compare the f32 posterior to the f64 m2a chain via **1-D Jensen-Shannon divergence**
    per parameter. Bar to clear first: JS < 1e-2 (and we expect to do far better).

The model's common process is named `crn_*`; the chain calls it `gw_*` (same physical
quantity) — mapped at comparison time. Per-pulsar `{psr}_red_noise_*` names match directly.

Run (background):  python dev_architecture/single_precision/posterior_regression.py
Outputs:           posterior_regression_f32.feather (samples) + printed JS table.
"""
import sys
import contextlib
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)   # x64 on; working=float32 makes the SOLVE f32
import numpy as np
import jax.numpy as jnp

import discovery as ds
from discovery import signals, utils, prior
from discovery.samplers import numpyro as dsnumpyro

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data"
PSR_FILES = sorted(DATA.glob("v1p1_de440_pint_bipm2019-*.feather"))
M2A_CHAIN = DATA / "NG15yr-m2a-chain.feather"
OUT = HERE / "posterior_regression_f32.feather"


@contextlib.contextmanager
def _working(dtype):
    ds.config(kernels="metamath")
    utils.config(backend="jax", working=dtype)
    try:
        yield
    finally:
        utils.config(backend="jax")
        ds.config(kernels="matrix")


def build_curn_proj(psrs, rn_components=30, crn_components=14):
    """NG15 m2a/CURN model, metamath path, timing-model projection ON.

    Mirrors discovery.models.nanograv.makemodel_curn but (a) routes through the
    config-swapped metamath ds.* classes and (b) marks the timing GP project=True.
    Must be built inside a _working(dtype) block."""
    T = signals.getspan(psrs)
    psls = [ds.PulsarLikelihood([psr.residuals,
                                 signals.makenoise_measurement(psr, psr.noisedict),
                                 signals.makegp_ecorr(psr, psr.noisedict),
                                 signals.makegp_timing(psr, svd=True, project=True)])
            for psr in psrs]
    curngp = signals.makecommongp_fourier(
        psrs, signals.makepowerlaw_crn(crn_components, crn_gamma='variable'),
        rn_components, T=T, common=['crn_log10_A', 'crn_gamma'], name='red_noise')
    return ds.ArrayLikelihood(psls, commongp=curngp)


def run(n_warmup=200, n_samples=200, max_tree_depth=6, seed=0):
    psrs = [ds.Pulsar.read_feather(f) for f in PSR_FILES]
    print(f"loaded {len(psrs)} pulsars, {sum(len(p.residuals) for p in psrs)} TOAs")
    with _working(jnp.float32):
        like = build_curn_proj(psrs)
        print(f"model params: {len(like.logL.params)}")
        model = dsnumpyro.makemodel_transformed(like.logL)
        sampler = dsnumpyro.makesampler_nuts(
            model, num_warmup=n_warmup, num_samples=n_samples,
            max_tree_depth=max_tree_depth)
        sampler.run(jax.random.PRNGKey(seed))
        df = sampler.to_df()
    df.to_feather(OUT)
    print(f"wrote {len(df)} f32 samples -> {OUT.name}")
    return df


# ----- comparison -----------------------------------------------------------
def _js_1d(a, b, bins=60):
    """Jensen-Shannon divergence (base-2, in [0,1]) between two 1-D samples on a
    shared histogram support."""
    from scipy.spatial.distance import jensenshannon
    lo = min(np.min(a), np.min(b)); hi = max(np.max(a), np.max(b))
    if hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    pa, _ = np.histogram(a, edges, density=True)
    pb, _ = np.histogram(b, edges, density=True)
    pa += 1e-12; pb += 1e-12
    return float(jensenshannon(pa, pb, base=2) ** 2)   # ^2 = divergence


def _chain_col(col):
    return col.replace('crn_', 'gw_') if col.startswith('crn_') else col


def compare(f32_df=None, n_ctrl=5, seed=0):
    """JS divergence f32-vs-chain, against the SAMPLING-NOISE FLOOR.

    With only ~200 f32 samples, a 60-bin JS vs a ~100k-sample chain is dominated
    by finite-sample noise even for two identical distributions. So we also
    measure the floor = JS(f64 chain subsampled to N, full chain), and flag a
    parameter only if its f32 JS EXCEEDS that floor by more than the bar -- i.e.
    if f32 differs from f64 by more than f64-vs-f64 sampling noise would. We also
    report robust moments (standardised mean shift, std ratio)."""
    import pandas as pd
    if f32_df is None:
        f32_df = pd.read_feather(OUT)
    chain = pd.read_feather(M2A_CHAIN)
    rng = np.random.default_rng(seed)
    n = len(f32_df)
    cols = [c for c in f32_df.columns if _chain_col(c) in chain.columns]

    rows = []
    for col in cols:
        ch = chain[_chain_col(col)].to_numpy()
        a = f32_df[col].to_numpy()
        js = _js_1d(a, ch)
        # sampling-noise floor for THIS parameter
        floor = np.median([_js_1d(ch[rng.integers(0, len(ch), n)], ch)
                           for _ in range(n_ctrl)])
        dmean = abs(a.mean() - ch.mean()) / ch.std()
        rows.append((col, js, floor, js - floor, dmean, a.std() / ch.std()))
    rows.sort(key=lambda r: -r[3])   # sort by EXCESS over floor

    print(f"\n{'parameter':<38} {'JS':>9} {'floor':>9} {'excess':>10} "
          f"{'|dmu|/s':>8} {'s32/s':>7}")
    for name, js, floor, exc, dmu, sr in rows:
        flag = "  <-- real f32 effect" if exc > 1e-2 else ""
        print(f"{name:<38} {js:>9.3e} {floor:>9.3e} {exc:>+10.3e} "
              f"{dmu:>8.3f} {sr:>7.3f}{flag}")
    exc = np.array([r[3] for r in rows])
    js = np.array([r[1] for r in rows])
    print(f"\nJS median {np.median(js):.3e} | floor median "
          f"{np.median([r[2] for r in rows]):.3e} | "
          f"EXCESS median {np.median(exc):+.3e} max {exc.max():+.3e}  (bar 1e-2)")
    print(f"n_params={len(rows)}  params over bar (excess>1e-2): "
          f"{int((exc > 1e-2).sum())}")
    return rows


if __name__ == "__main__":
    if sys.argv[1:] == ["compare"]:
        compare()
    else:
        df = run()
        compare(df)
