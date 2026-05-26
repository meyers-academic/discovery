"""Monkeypatch context manager swapping matrix.py classes for metamath equivalents.

This is the "Tier 0/1 shortcut" — collapse the matrix.NoiseMatrix* lattice onto
mh.NoiseMatrix, and route Woodbury/Compound through metamath. signals.py and
likelihood.py both access matrix.X at call time, so models built inside the
context capture the metamath classes.
"""

from contextlib import contextmanager

import discovery as ds
from discovery import metamath as mh


_PATCHES = {
    # noise variants collapse to a single metamath NoiseMatrix; dispatch happens
    # at runtime via Sym.solve based on tensor ndim
    # Preserve 1D-vs-2D type discrimination so likelihood.py's isinstance
    # dispatch (e.g. likelihood.py:468 ndim selection) still works correctly.
    "NoiseMatrix1D_novar":     mh.NoiseMatrix1D,
    "NoiseMatrix1D_var":       mh.NoiseMatrix1D,
    "NoiseMatrix2D_var":       mh.NoiseMatrix2D,
    "NoiseMatrix12D_var":      mh.NoiseMatrix12D,
    "VectorNoiseMatrix1D_var": mh.NoiseMatrix1D,

    # ecorr Sherman-Morrison: dedicated indexed-SM graph
    "NoiseMatrixSM_novar":     mh.NoiseMatrixSM,
    "NoiseMatrixSM_var":       mh.NoiseMatrixSM,

    # Woodbury + compound
    "WoodburyKernel":            mh.WoodburyKernel,
    "VectorWoodburyKernel_varP": mh.VectorWoodburyKernel,
    "CompoundGP":                mh.CompoundGP,
    "VectorCompoundGP":          mh.CompoundGP,
    "CompoundDelay":             mh.CompoundDelay,
}


@contextmanager
def metamatrix_patch():
    """Swap matrix.X → metamath equivalents for the duration of the block.

    Build the model *inside* the context; once built, its closures reference
    metamath objects and the patch can be lifted safely.
    """
    saved = {}
    for name, replacement in _PATCHES.items():
        saved[name] = getattr(ds.matrix, name)
        setattr(ds.matrix, name, replacement)
    try:
        yield
    finally:
        for name, original in saved.items():
            setattr(ds.matrix, name, original)
