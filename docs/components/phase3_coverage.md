# Phase 3 — parity coverage sweep

_The deletion gate: every kernel constructor that production code or the
example/tutorial notebooks actually trigger must be exercised by a parity
route, OR be pinned as a documented `xfail` we close in Phase 4._

Scope of "used": `signals.py` / `measurement_noise.py` / `deterministic.py`
call sites, plus the builders invoked by `examples/*.ipynb` and
`docs/tutorials/*.ipynb` (incl. `cw_extsignal_example.ipynb`).

## Coverage table

| Constructor | Emitted by | metamath-mapped? | Parity route | Status |
|---|---|---|---|---|
| `NoiseMatrix1D` (collapsed) | `makenoise_measurement*` | factory fn | `measurement_white`, … | ✅ |
| `NoiseMatrixSM` (collapsed) | `makenoise_measurement(ecorr=True)` | factory fn | `ecorr_sm` | ✅ |
| `NoiseMatrix1D_novar`/`_var` | `makegp_ecorr*`, `makegp_timing`, `makegp_improper` | ✅ | `ecorr_gp`, `timing`, … | ✅ |
| `NoiseMatrix12D_var` (1D) | `makegp_fourier` (1D PSD) | ✅→`NoiseMatrix12D` | `full_rn`, … | ✅ |
| `NoiseMatrix2D_var` (2D) | `makegp_fftcov`/`avgcov`/`intcov` | ✅ | **`fftcov_2d` (added)** | ✅ closed |
| `NoiseMatrix2D_novar` | `makegp_fourier_variance` (fixed) | ✅ (added) | **`fourier_variance_fixed`** | ✅ closed (Phase 4a) |
| `VectorNoiseMatrix12D_var` | `makecommongp_fourier` | ✅ | `common_rn` | ✅ (closed Phase 1) |
| `CompoundDelay` | `makedelay`/`makedelay_binary` | ✅ | **`delay` (added)** | ✅ closed |
| `WoodburyKernel`, `CompoundGP`, `VectorCompoundGP`, `VectorWoodburyKernel_varP` | likelihood layer | ✅ | all model tests | ✅ |
| `CompoundGlobalGP` | globalgp-as-list (CURN) | ✅ (Phase 4b) | **`global_compound`** | ✅ closed (Phase 4b) |

PSD/ORF helpers (`powerlaw`, `freespectrum`, `brokenpowerlaw`,
`make_combined_crn`, `makepowerlaw_crn`, `hd_orf`, …) are prior functions, not
kernel constructors — they ride on the GP builders above.

## Gaps found and disposition

- **A — 2D variable noise (`NoiseMatrix2D_var`).** Used for real by
  `makegp_fftcov` in `os_example` / `numpyro_example`; no 1D-PSD parity row
  reached it. **Closed:** added `test_pulsar.py::_fftcov_2d` (passes matrix /
  mh_patched / mh_native).
- **B — delay (`CompoundDelay`).** Used by the CW examples via `makedelay`; no
  parity row touched a delay. **Closed:** added `test_pulsar.py::_delay`
  (parameter-free deterministic delay; passes all routes).
- **C — fixed 2D GP prior (`NoiseMatrix2D_novar`).** Emitted by
  `makegp_fourier_variance` when the variance matrix is supplied (no notebook
  uses this builder). The **kernel** maps (`NoiseMatrix2D_novar → mh.NoiseMatrix2D`),
  but the **likelihood path** is unsupported in metamath: an all-constant 2D GP
  prior reaches `metamath.CompoundGP._build_mixed_logprior`, which requires
  `gp.index` — a marginalized constant GP has none.
  **Closed in Phase 4a:** `metamath.CompoundGP` now builds a real combined dense
  `Phi` (block-diagonal, promoting 1D blocks) for a mixed but *marginalized*
  compound, gating the coefficient-log-prior branch to the vector/decentered
  path. `test_pulsar.py::test_logL[fourier_variance_fixed]` passes (xfail removed).
- **D — `CompoundGlobalGP`.** The Phase-2 factory fallthrough (globalgp passed
  as a *list*, e.g. HD + monopole). Originally lived only in `matrix.py` and
  built `matrix.NoiseMatrix1D_var` directly, so it couldn't consume metamath GP
  priors. **Closed in Phase 4b:** relocated to `signals.CompoundGlobalGP`,
  backend-agnostic (factory + `utils.GlobalVariableGP`, reading only
  mode-neutral `gp.Phi.getN`/`.getN.params`/`gp.Phi_inv`). Both `likelihood.py`
  and `likelihood_metamath.py` now call it; `matrix.CompoundGlobalGP` is dead
  code that goes with `matrix.py` in Phase 5. Routed by `test_global.py`'s
  `global_compound` (logL + conditional).

## Integration check — `cw_extsignal_example.ipynb`

Ran the notebook's model build + `clogL`/`logL` at its fixed test point (5
pulsars, `n_cw=60`, CW + HD global GP + decentered common red noise) under both
kernel modes via `ds.config(kernels=...)`:

```
clogL  matrix=840535.522183  metamath=840535.522183  diff=5.8e-10
logL   matrix=840636.442313  metamath=840636.442313  diff=5.8e-10
```

Agreement at ~1e-9 relative. The CW/ExtSignal + global-HD + decenter path is
sound in both modes at real scale. (The notebook's NUTS sampling cells were not
run — out of scope for the kernel-parity check.)

## Phase 3 result

All production / notebook-driven kernel constructors are now parity-routed.
The two carry-overs (C `NoiseMatrix2D_novar`, D `CompoundGlobalGP`) were closed
in **Phase 4** — see the updated dispositions above; there are no remaining
`xfail`s. Every constructor the matrix oracle covers is now matched by the
metamath path, so the oracle may be deleted in Phase 5.
