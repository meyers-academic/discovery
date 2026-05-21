#!/usr/bin/env python3
"""Tests for discovery.prior array-valued parameter handling.

Covers the `array=True` flag on `sample_uniform` / `makelogprior_uniform`
and the scalar-reduction fix in the dict-based logprior. These are pure unit
tests -- they operate on hand-built parameter-name lists and need no data.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import jax
jax.config.update('jax_enable_x64', True)

import discovery as ds
from discovery.likelihood import ArrayLogL


# parameter-name lists chosen to match priordict_standard regexes
SCALAR_PARAMS = ['B1855+09_red_noise_log10_A', 'B1855+09_red_noise_gamma',
                 'crn_log10_A', 'crn_gamma']
VEC_PARAM = 'B1855+09_red_noise_log10_rho(30)'          # array-valued: 30 columns
MIXED_PARAMS = [VEC_PARAM, 'crn_log10_A', 'crn_gamma']  # flat size 30 + 1 + 1 = 32


@pytest.mark.unit
class TestPriorArrays:

    # --- sample_uniform(array=...) ------------------------------------------

    def test_default_return_is_dict(self):
        s = ds.sample_uniform(SCALAR_PARAMS)
        assert isinstance(s, dict) and set(s) == set(SCALAR_PARAMS)

    def test_array_scalar_params(self):
        x = ds.sample_uniform(SCALAR_PARAMS, array=True)
        assert x.shape == (len(SCALAR_PARAMS),)

    def test_array_expands_vector_params(self):
        # the (30) suffix becomes 30 flat columns; scalars one each
        x = ds.sample_uniform(MIXED_PARAMS, array=True)
        assert x.shape == (32,)

        # the dict form keeps the block as a (30,) array
        d = ds.sample_uniform(MIXED_PARAMS)
        assert np.shape(d[VEC_PARAM]) == (30,)
        assert np.ndim(d['crn_log10_A']) == 0

    def test_array_within_prior_bounds(self):
        x = np.asarray(ds.sample_uniform(MIXED_PARAMS, array=True))
        assert np.all((x[0:30] >= -9) & (x[0:30] <= -4))   # log10_rho
        assert -18 <= x[30] <= -11                          # crn_log10_A
        assert 0 <= x[31] <= 7                              # crn_gamma

    def test_array_batched(self):
        x = ds.sample_uniform(MIXED_PARAMS, n=5, array=True)
        assert x.shape == (5, 32)

    # --- makelogprior_uniform(array=...) -----------------------------------

    def test_array_logprior_in_range(self):
        x = ds.sample_uniform(MIXED_PARAMS, array=True)
        logp = ds.makelogprior_uniform(MIXED_PARAMS, array=True)
        # a draw from the prior is always inside its own support
        assert float(logp(x)) == 0.0

    def test_array_logprior_out_of_range(self):
        x = np.asarray(ds.sample_uniform(MIXED_PARAMS, array=True))
        logp = ds.makelogprior_uniform(MIXED_PARAMS, array=True)
        x[0] = 999.0                                        # one column out of bounds
        assert float(logp(x)) == -np.inf

    def test_array_logprior_unknown_param_raises(self):
        with pytest.raises(KeyError):
            ds.makelogprior_uniform(['totally_unknown_param'], array=True)

    # --- dict-based logprior stays scalar with array-valued params ----------

    def test_dict_logprior_scalar_with_vector_param(self):
        d = ds.sample_uniform([VEC_PARAM])                  # {VEC_PARAM: (30,)}
        val = ds.makelogprior_uniform([VEC_PARAM])(d)
        assert np.ndim(val) == 0                            # scalar, not a (30,) array
        assert float(val) == 0.0

    def test_dict_logprior_out_of_range_with_vector_param(self):
        d = ds.sample_uniform([VEC_PARAM])
        d[VEC_PARAM] = np.asarray(d[VEC_PARAM]).copy()
        d[VEC_PARAM][0] = 999.0
        assert float(ds.makelogprior_uniform([VEC_PARAM])(d)) == -np.inf

    # --- layout agreement across the array-native pieces -------------------

    def test_sample_and_logprior_layout_consistent(self):
        logp = ds.makelogprior_uniform(MIXED_PARAMS, array=True)
        for _ in range(5):
            x = ds.sample_uniform(MIXED_PARAMS, array=True)
            # a layout mismatch would land a value outside another param's
            # disjoint bounds and give -inf
            assert float(logp(x)) == 0.0

    def test_layout_matches_arraylogl(self):
        # sample_uniform's flat array must match ArrayLogL's column layout
        arr = ArrayLogL(SimpleNamespace(params=MIXED_PARAMS))
        x = ds.sample_uniform(MIXED_PARAMS, array=True)
        assert x.shape == (arr.size,)
