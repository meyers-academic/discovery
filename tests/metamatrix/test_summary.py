"""Model-summary snapshots (discovery.summary).

A built likelihood can describe itself via ``model.summary()`` /
``summary_frame()`` / ``_repr_html_``. These tests exercise every recipe under
both kernel backends and assert the one invariant that makes the summary
trustworthy: the free parameters it reports are a *superset* of
``logL.params`` -- it never hides a parameter the likelihood actually varies.
The only allowed extras are deterministic-signal and non-zero-mean parameters,
which some likelihood paths marginalize out of ``logL.params`` but which are
still genuine model structure worth showing.
"""

import pandas as pd
import pytest

import jax

import discovery as ds
import discovery.recipes as R
from discovery import summary as S


@pytest.fixture(params=["matrix", "metamath"])
def backend(request):
    """Run each test under both kernel backends, restoring matrix afterwards."""
    ds.config(kernels=request.param)
    yield request.param
    ds.config(kernels="matrix")


def _extra_allowed(model):
    """Params the summary may report beyond logL.params: extsignal coefficients
    and non-zero prior-mean amplitudes (handled outside the main param vector)."""
    extra = set()
    for ext in getattr(model, "extsignals", None) or []:
        extra.update(getattr(ext, "params", []))
    cg = getattr(model, "commongp", None)
    for g in (cg if isinstance(cg, list) else [cg]) if cg is not None else []:
        for attr in ("means", "mean"):
            fn = getattr(g, attr, None)
            if fn is not None:
                extra.update(getattr(fn, "params", []))
    return extra


def _summary_params(model):
    cols, com = S._collect(model)
    return set(S._totals(cols, com)["varying"])


SINGLE = [pytest.param(f, id=f.__name__) for f in R.SINGLE_PULSAR]
MULTI = [pytest.param(f, id=f.__name__) for f in (R.GLOBAL + R.ARRAY)]


@pytest.mark.parametrize("recipe", SINGLE)
def test_single_pulsar_summary(recipe, psr, backend):
    model = recipe(psr)

    # the reliability invariant: never hide a varied parameter
    assert set(model.logL.params) <= _summary_params(model)
    assert _summary_params(model) - set(model.logL.params) <= _extra_allowed(model)

    text = model.summary()
    assert isinstance(text, str) and psr.name in text
    assert "free params" in text

    frame = model.summary_frame()
    assert isinstance(frame, pd.DataFrame)
    assert {"signal", "kind", "basis", "n_free", "access"} <= set(frame.columns)
    assert len(frame) >= 1

    assert "<pre" in model._repr_html_()
    assert isinstance(repr(model), str) and type(model).__name__ in repr(model)

    # kernel tree (both renderings) names the pulsar and carries a handle
    comp = model.tree()
    assert isinstance(comp, str) and psr.name in comp and "signals[" in comp
    assert isinstance(model.tree(literal=True), str)

    # independent free/fixed toggles
    assert "(fixed)" in model.summary(show_fixed=True, show_free=False) or \
        frame["n_fixed"].sum() == 0
    assert "(fixed)" not in model.summary(show_fixed=False)


@pytest.mark.parametrize("recipe", MULTI)
def test_multi_pulsar_summary(recipe, psrs, backend):
    model = recipe(psrs)

    assert set(model.logL.params) <= _summary_params(model)
    assert _summary_params(model) - set(model.logL.params) <= _extra_allowed(model)

    text = model.summary()
    assert isinstance(text, str)
    for p in psrs:
        assert p.name in text

    frame = model.summary_frame()
    assert isinstance(frame, pd.DataFrame)
    # one collection per pulsar should appear
    assert set(p.name for p in psrs) <= set(frame["collection"])

    assert "<pre" in model._repr_html_()

    # kernel tree: every pulsar named, per-pulsar handles present
    comp = model.tree()
    for p in psrs:
        assert p.name in comp
    assert "psls[0].signals[" in comp
    assert isinstance(model.tree(literal=True), str)


def test_fixed_white_noise_is_reported(psr):
    """White noise pinned from the noise dictionary is invisible to logL.params
    but must still show up as fixed parameters in the summary."""
    ds.config(kernels="matrix")
    model = R.full_rn(psr)
    frame = model.summary_frame()
    meas = frame[frame["signal"] == "measurement"].iloc[0]
    assert meas["n_free"] == 0
    assert meas["n_fixed"] > 0          # efac / equad baked in from noisedict
    assert f"{psr.name}_" in meas["fixed_params"]


@pytest.fixture
def _matrix_backend():
    ds.config(kernels="matrix")
    yield
    ds.config(kernels="matrix")


def test_literal_tree_fuses_constant_gps(psr, _matrix_backend):
    """concat=True fuses ECORR+timing into one Woodbury layer; the literal tree
    must show their column slices summing to the live model.N.N.F width."""
    model = R.full_rn(psr)
    lit = model.tree(literal=True)
    assert "fused" in lit
    # ECORR (360) then timing (166) -> 526, matching the live fused basis
    assert model.N.N.F.shape[1] == 526
    assert "[:, 0:360]" in lit and "[:, 360:526]" in lit

    # concat=False chains them into separate layers (no fusion)
    assert "fused" not in R.full_rn_concat_false(psr).tree(literal=True)


def test_signal_object_reprs(psr, psrs, _matrix_backend):
    """signals[i] / commongp / globalgp print an informative repr, not <object>."""
    model = R.full_rn(psr)
    rn = model.signals[4]
    assert "rednoise" in repr(rn) and "VariableGP" in repr(rn)
    assert "fixed prior" in repr(model.signals[3])     # timing model (ConstantGP)
    assert "white noise" in repr(model.signals[1])     # measurement kernel

    g = R.intrinsic_rn_plus_global_hd(psrs)
    assert "GlobalVariableGP" in repr(g.globalgp) and "hd_orf" in repr(g.globalgp)
