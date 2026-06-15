# `metamatrix-meyers` vs. upstream `nanograv/discovery@metamatrix`

_Generated 2026-06-15. Compares local branch `metamatrix-meyers` against
`upstream/metamatrix` (nanograv)._

## Topology

- Merge base is **the tip of `upstream/metamatrix`** — i.e. our branch is a
  strict fast-forward superset. Upstream has **no commits we don't have**.
- We are **7 commits ahead**:

  | SHA | Summary |
  |---|---|
  | `c468668` | Merge matrix.py-based decentering + deterministic CW that works with `ExtSignal`; new `ExtSignal` class (Fourier deterministic signal carrying its own basis) |
  | `235e910` | Unit tests comparing global / array / single-pulsar likelihoods across old (`matrix.py`) vs new (`metamath`) paths |
  | `ae7d4d7` | Big `metamath` update: Sherman–Morrison, coefficient likelihoods, testing suites, `_patch.py` toggle between old `matrix.py` and new `metamath.py` classes |
  | `bb378da` | Relocate helper functions/classes so they can be used independently from `matrix.py` and `metamath.py` (toward independent v1.0 paths) |
  | `2fecd49` | Remove `basis.py` |
  | `246cc50` | Scaffold metamath-native likelihood path + 3-route parity harness |
  | `ad18fb4` | Migrate `likelihood_metamath` to compose metamath kernels directly |
  | `95d20b8` | Move GP/Kernel marker classes to `kernel_helpers` |

  _(c468668 + 235e910 carry across the merge base; the 7 listed under "ahead"
  are the ones not on upstream.)_

Committed diff: **19 files, +3014 / −333**.

## What changed, by theme

### 1. A second, graph-based likelihood path (`metamath`)
The headline change. There is now a parallel implementation of the three
top-level likelihoods built on the `metamath.py` graph machinery instead of
the `matrix.py` closure classes.

- **New `src/discovery/likelihood_metamath.py`** (+853): `PulsarLikelihood`,
  `GlobalLikelihood`, `ArrayLikelihood` reimplemented to compose metamath
  kernels directly.
- **`metamath.py`** (+527): grows the kernel/GP class layer — `WoodburyKernel`,
  `GlobalWoodburyKernel`, `VectorWoodburyKernel`, `NoiseMatrix`/`NoiseMatrix1D`/
  `NoiseMatrix2D`, and a Sherman–Morrison `NoiseMatrixSM` + `smsolve`.
  Adds coefficient-likelihood pieces (`gaussian_coefficient_logprior`,
  `vectorgpcomponent`, `woodburykernelsolve`) and mixed-prior support
  (`_build_mixed_logprior`).

### 2. Runtime switch between the two paths
- **`src/discovery/__init__.py`**: new `discovery.config(kernels=...)`
  accepting `'matrix'` (legacy, default) or `'metamath'`. It rebinds
  `PulsarLikelihood`/`GlobalLikelihood`/`ArrayLikelihood` to the chosen
  implementation. Distinct from `matrix.config(backend=...)` (numpy/jax).
- **New `src/discovery/_kernel_switch.py`** (+94): `apply_patches()` /
  `restore_patches()` / `patched_kernels()` implement the monkeypatch toggle.

### 3. Code relocation toward independent paths
- **New `src/discovery/kernel_helpers.py`** (+182): GP/Kernel marker classes
  (`Kernel`, `ConstantKernel`, `VariableKernel`, `ConstantMatrix`,
  `VariableMatrix`, `NoiseMatrix`, `GP` + variants), the `ExtSignal` class, and
  Sherman–Morrison index helpers (`make_uind`, `smup_ind`, `smdp_ind`,
  `smup_ind_correct`) — moved out of `matrix.py` so both paths can share them.
- **`matrix.py`** (−225 net on those classes): the relocated classes/helpers
  are deleted from here.
- **`basis.py` removed** (−207); `__init__` no longer does `from .basis import *`.

### 4. ExtSignal / deterministic CW
- **`signals.py`** (+87): `make_extsignal_fourier` — a deterministic Fourier
  signal that carries its own basis, folded into the likelihood via GP cross-terms.
- **`deterministic.py`** (+25): `makecw_extsignal` — continuous-wave signal on
  its own (higher-frequency) Fourier basis, a thin wrapper over
  `make_extsignal_fourier` + `makefourier_binary`. CW params never enter the GP prior.
- **`likelihood.py`** (+87): `make_kernelproduct_gpcomponent` gains an
  `extsignals=` argument (matched in `metamath.py`).

### 5. Parity test suite (`tests/metamatrix/`, all new, +1004)
A harness that runs each likelihood three ways and asserts numerical agreement
between the `matrix.py` oracle and the `metamath` rewrite:
- `_routes.py`, `_comparison.py`, `_patch.py`, `conftest.py` — harness plumbing.
- `test_pulsar.py`, `test_global.py`, `test_array.py`, `test_sm.py` — coverage
  for single-pulsar, global, array, and Sherman–Morrison paths.

## Uncommitted working-tree changes (not in any commit yet)
- `M examples/likelihood_example.ipynb`, `M examples/numpyro_example.ipynb`
- `?? docs/components/metamatrix_architecture.md` — design rationale / end-state
  ("matrix.py does not exist"; parity suite as the gate to deleting it)
- `?? docs/components/mixed_phi_compoundgp_plan.md`
- `?? docs/tutorials/cw_extsignal_example.ipynb`

## Where this leaves `matrix.py` / `likelihood.py` (can we delete them?)

Not yet — both are still live, for distinct reasons:

- **`likelihood.py`** is still the default path *and* the parity test oracle.
  It can't go until the metamath path is certified at parity on every
  production path the suite covers.
- **`matrix.py`** still has real dependents beyond being the oracle:
  - `signals.py` / `deterministic.py` call `matrix.X(...)`. The metamath path
    works only because `_kernel_switch._PATCHES` monkeypatches those names *on
    the `matrix` module* — so `matrix.py` must still exist for the shim to patch.
  - `likelihood_metamath.py` still imports `matrix` for non-kernel utilities:
    `matrix.ConstantGP`/`VariableGP` (isinstance markers), `CompoundGlobalGP`,
    `make_logdet_estimator`, `cgsolve`, and the `jnp`/`jsp` linalg aliases
    (lines 215, 750, 779–781).
  - `kernel_helpers.__getattr__` proxies the numerical-backend globals
    (`jnp`, `jsp`, …) **back from `matrix.py`** — i.e. the backend config still
    physically lives in `matrix.py`.
  - `__init__.py` still does `from .matrix import *`.

**The signals.py concern is resolved, not a blocker.** You do *not* have to
reimplement `signals.py` to make the metamath path work — that's the whole
point of the `_kernel_switch` shim. Migrating `signals.py` is the *cleanup*
step that lets the shim (and then `matrix.py`) be deleted, not a prerequisite
for the path to function.

### Exit criteria — what has to be true to delete both files
1. Backend/config extracted out of `matrix.py` into a path-neutral module, so
   `kernel_helpers` no longer reaches back into `matrix`.
2. `signals.py` constructs the surviving (metamath) kernels directly, retiring
   `_kernel_switch.py`.
3. `likelihood_metamath.py`'s remaining `matrix.*` utility imports rehomed.
4. Parity suite covers every kernel constructor any production/example code
   actually uses (the deletion gate).
5. `matrix.py` + `likelihood.py` deleted; `likelihood_metamath.py` promoted to
   `likelihood.py`.

See `exit_plan.md` for the phased plan against these criteria.

## Bottom line
This branch carries the entire `metamatrix` refactor forward into a working,
**toggleable two-path state**: the legacy `matrix.py` closure path (still the
default and the test oracle) and a new `metamath.py` graph path, selectable via
`discovery.config(kernels=...)`, with a parity suite certifying they agree.
Upstream `metamatrix` does not yet have any of the metamath likelihood path,
the kernel switch, the ExtSignal/CW work, or the parity tests.
