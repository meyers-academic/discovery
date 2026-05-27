"""Shared kernel primitives used by both `matrix.py` and `metamath.py`.

These are leaf numerical helpers — small, pure-JAX, no class hierarchy or
const/var bookkeeping. They live here so that both the legacy `matrix.py`
implementation and the graph-based `metamath.py` rewrite can reuse them
without one depending on the other. Eventual home for whichever subset
Discovery v1.0 still needs after `matrix.py` is removed.

Contents
--------
- `Kernel` — empty marker base class shared by all kernel/GP classes.
- `make_uind(U)` — pack a 0/1 exposure matrix into a (n_epoch, max_per_epoch+1)
  index array used by the indexed Sherman-Morrison primitives below.
- `smup_ind` / `vsmup_ind` — per-epoch SM rank-1 update factor.
- `smdp_ind` / `vsmdp_ind` — per-epoch SM logdet (log1p) contribution.
- `smup_ind_correct` / `vsmup_ind_correct` — apply the SM correction to a
  pre-padded residual (1D) or basis (2D, vmapped over the batch axis).
"""
import numpy as np

import jax
import jax.numpy as jnp


class Kernel:
    """Marker base class for kernel / GP / noise-matrix objects."""
    pass


class ConstantKernel(Kernel):
    pass


class VariableKernel(Kernel):
    pass


class ConstantMatrix:
    pass


class VariableMatrix:
    pass


class NoiseMatrix:
    """Base type for noise-kernel objects. Subclassed by `matrix.NoiseMatrix*`
    and `metamath.NoiseMatrix`; carried as a marker for isinstance dispatch."""
    pass


class GP:
    pass


class ConstantGP:
    """A Gaussian-process signal whose prior precision is fixed at trace time
    (no parameters). `Phi` is the prior covariance; `F` is the per-pulsar
    design matrix."""
    def __init__(self, Phi, F):
        self.Phi, self.F = Phi, F


class VariableGP:
    """A Gaussian-process signal whose prior precision depends on parameters."""
    def __init__(self, Phi, F):
        self.Phi, self.F = Phi, F


class GlobalVariableGP:
    """Like VariableGP, but with per-pulsar design matrices in a list `Fs`.
    Factories returning a GlobalVariableGP should set `.index` as a dict
    of component-vector-name -> slice within the stacked Fs."""
    def __init__(self, Phi, Fs):
        self.Phi, self.Fs = Phi, Fs
        self.Phi_inv = None


class ExtSignal:
    """A deterministic signal carried on its own (non-GP) Fourier-style basis.

    Used for signals (e.g. a continuous wave) whose basis must extend to
    higher frequencies than the GP bases reach. Unlike a GP it has no prior:
    its coefficients are a deterministic function of a handful of physical
    parameters, supplied by ``coeffs``.

    The object is purely declarative and noise-free; all noise-dependent
    linear algebra (cross-terms with the GP basis and the data) lives in
    the kernelproduct_gpcomponent path of whichever backend is in use.

    Parameters
    ----------
    Fs : list of array
        Per-pulsar design matrices, one ``(ntoa_i, n_ext)`` array per pulsar,
        in the same pulsar order as the likelihood.
    coeffs : callable
        ``coeffs(params) -> (npsr, n_ext)`` deterministic coefficient map,
        with a ``.params`` attribute listing its parameter names.
    name : str
        Identifier for the signal.
    """
    def __init__(self, Fs, coeffs, name='extsignal'):
        self.Fs, self.coeffs, self.name = Fs, coeffs, name

    @property
    def params(self):
        return self.coeffs.params


def make_uind(U):
    """Pack 0/1 exposure matrix U (n_toa, n_epoch) into the index encoding
    expected by the indexed Sherman-Morrison primitives.

    Returns Uind of shape (n_epoch, max_per_epoch + 1) with TOA indices
    shifted by +1 so that 0 acts as a sentinel (used together with a
    zero-prepended y / +inf-prepended N).
    """
    Uind = np.zeros((U.shape[1], jnp.max(jnp.sum(U, axis=0)) + 1), 'i')

    for i in range(U.shape[1]):
        ind = np.where(U[:, i])[0]
        Uind[i, 0:len(ind)] = ind + 1

    return Uind


def smup_ind(A, l, Amb, ind):
    Amu = 1.0 / A[ind]

    vtAmb = l * jnp.sum(Amb[ind])
    vtAmu = l * jnp.sum(Amu)

    return Amu * (vtAmb / (1.0 + vtAmu))


def smdp_ind(A, l, ind):
    Amu = 1.0 / A[ind]
    vtAmu = l * jnp.sum(Amu)

    return jnp.log1p(vtAmu)


vsmup_ind = jax.vmap(smup_ind, in_axes=(None, 0, None, 0))
vsmdp_ind = jax.vmap(smdp_ind, in_axes=(None, 0, 0))


def smup_ind_correct(yp, Np, Uind, P):
    corrections = vsmup_ind(Np, P, yp / Np, Uind)
    return (yp / Np).at[Uind.reshape(-1)].add(-corrections.reshape(-1))[1:]


vsmup_ind_correct = jax.vmap(smup_ind_correct, in_axes=(0, None, None, None))


# ---- Configurable aliases ----
# `matrix.config(backend=..., factor=...)` populates module-level names on
# `matrix.py` (`jnp`, `jsp`, `jnparray`, `jnpsplit`, `jnpnormal`,
# `matrix_factor`, `matrix_solve`, `matrix_norm`, `regularize_FtNmF`, ...) so
# downstream code can switch numpy vs jax, single vs double precision, and
# cholesky vs LU factorization through one entry point. To let both
# `matrix.py`-backed and `metamath`-backed code reach those same values
# without depending on `matrix` directly, we forward them via PEP 562
# module `__getattr__` here. Read `kh.jnp` and you get the current value of
# `matrix.jnp` at call time — no caching, no shadow copy.

_CONFIG_ALIASES = frozenset({
    "jnp", "jsp",
    "jnparray", "jnpzeros", "intarray",
    "jnpkey", "jnpsplit", "jnpnormal",
    "matrix_factor", "matrix_solve", "matrix_norm",
    "regularize_FtNmF", "SM_algorithm",
    "single_precision", "partial",
})


def __getattr__(name):
    if name in _CONFIG_ALIASES:
        # deferred import to break the kernel_helpers ↔ matrix cycle
        from . import matrix as _matrix
        return getattr(_matrix, name)
    raise AttributeError(
        f"module 'discovery.kernel_helpers' has no attribute {name!r}")
