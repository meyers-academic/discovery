"""
Comparison tests between old matrix.py and new matrix_refactored.py.

These tests verify that the refactored graph-based implementation produces
identical results to the original specialized classes.

Test cases:
1. test_constant_case: WoodburyKernel_novar vs WoodburyKernel (all constant)
2. test_variable_p_case: WoodburyKernel_varP vs WoodburyKernel (P varies)
3. test_nested_case: Manual composition vs WoodburyKernel (nested structure)
4. test_nested_variable_case: Nested with variable outer P
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

# Import old implementation
from discovery.matrix import (
    NoiseMatrix1D_novar,
    NoiseMatrix1D_var,
    WoodburyKernel_novar,
    WoodburyKernel_varP,
)

# Import new implementation
from discovery.matrix_refactored import WoodburyKernel


def test_constant_case():
    """
    Compare WoodburyKernel_novar (old) vs WoodburyKernel (new) with all constant.

    Setup:
    - N: constant diagonal noise matrix
    - F: constant design matrix
    - P: constant prior covariance
    - y: constant data vector

    Expected: Identical results for kernel product (log-likelihood)
    """
    # Setup data
    np.random.seed(42)
    n_data, n_basis = 100, 10

    N_diag = np.ones(n_data) * 0.5
    F_matrix = np.random.randn(n_data, n_basis)
    P_diag = np.ones(n_basis) * 2.0
    y_data = np.random.randn(n_data)

    # OLD IMPLEMENTATION
    N_old = NoiseMatrix1D_novar(N_diag)
    P_old = NoiseMatrix1D_novar(P_diag)
    kernel_old = WoodburyKernel_novar(N_old, F_matrix, P_old)
    loglike_old = kernel_old.make_kernelproduct(y_data)
    ll_old = loglike_old({})  # No parameters

    # NEW IMPLEMENTATION
    kernel_new = WoodburyKernel(N_diag, F_matrix, P_diag)
    loglike_new = kernel_new.make_kernelproduct(y_data)
    ll_new = loglike_new({})  # No parameters

    # COMPARE
    assert np.allclose(ll_old, ll_new, rtol=1e-6, atol=1e-8), \
        f"Results differ: old={ll_old:.10f}, new={ll_new:.10f}"


def test_variable_p_case():
    """
    Compare WoodburyKernel_varP (old) vs WoodburyKernel (new) with variable P.

    Setup:
    - N: constant diagonal noise matrix
    - F: constant design matrix
    - P: variable prior covariance (depends on 'amplitude' parameter)
    - y: constant data vector

    Expected: Identical results for multiple parameter values
    """
    # Setup data
    np.random.seed(43)
    n_data, n_basis = 100, 10

    N_diag = np.ones(n_data) * 0.5
    F_matrix = np.random.randn(n_data, n_basis)
    y_data = np.random.randn(n_data)

    # OLD IMPLEMENTATION
    N_old = NoiseMatrix1D_novar(N_diag)

    # P function with .params attribute (works with both old and new)
    def P_func(params):
        return jnp.ones(n_basis) * params['amplitude']**2
    P_func.params = ['amplitude']

    P_old = NoiseMatrix1D_var(P_func)
    kernel_old = WoodburyKernel_varP(N_old, F_matrix, P_old)
    loglike_old = kernel_old.make_kernelproduct(y_data)

    # NEW IMPLEMENTATION
    # Same P function works directly
    kernel_new = WoodburyKernel(N_diag, F_matrix, P_func)
    loglike_new = kernel_new.make_kernelproduct(y_data)

    # COMPARE multiple parameter values
    test_amplitudes = [0.5, 1.0, 1.5, 2.0, 3.0]

    for amp in test_amplitudes:
        params = {'amplitude': amp}
        ll_old = loglike_old(params)
        ll_new = loglike_new(params)

        assert np.allclose(ll_old, ll_new, rtol=1e-6, atol=1e-8), \
            f"Results differ for amplitude={amp}: old={ll_old:.10f}, new={ll_new:.10f}"


def test_nested_case():
    """
    Compare manual composition (old approach) vs nested WoodburyKernel (new).

    Setup:
    - Inner structure: N_base + F_inner^T P_inner F_inner
    - Outer structure: inner + F_outer^T P_outer F_outer

    Old approach: Manually build the inner kernel first, then use as N
    New approach: Directly nest WoodburyKernel instances

    Expected: Both approaches produce identical results
    """
    # Setup data
    np.random.seed(44)
    n_data = 100
    n_inner, n_outer = 5, 8

    N_base = np.ones(n_data) * 0.1
    F_inner = np.random.randn(n_data, n_inner)
    F_outer = np.random.randn(n_data, n_outer)
    y_data = np.random.randn(n_data)

    # Fixed parameters for comparison
    inner_amp = 1.5
    outer_amp = 2.0

    # OLD IMPLEMENTATION (manual composition)
    N_base_old = NoiseMatrix1D_novar(N_base)
    P_inner_old = NoiseMatrix1D_novar(np.ones(n_inner) * inner_amp**2)
    inner_old = WoodburyKernel_novar(N_base_old, F_inner, P_inner_old)

    P_outer_old = NoiseMatrix1D_novar(np.ones(n_outer) * outer_amp**2)
    outer_old = WoodburyKernel_novar(inner_old, F_outer, P_outer_old)

    loglike_old = outer_old.make_kernelproduct(y_data)
    ll_old = loglike_old({})  # All constant for this test

    # NEW IMPLEMENTATION (direct nesting)
    inner_new = WoodburyKernel(
        N_base,
        F_inner,
        np.ones(n_inner) * inner_amp**2
    )

    outer_new = WoodburyKernel(
        inner_new,  # Directly nest!
        F_outer,
        np.ones(n_outer) * outer_amp**2
    )

    loglike_new = outer_new.make_kernelproduct(y_data)
    ll_new = loglike_new({})  # All constant for this test

    # COMPARE (slightly looser tolerance due to numerical recursion)
    assert np.allclose(ll_old, ll_new, rtol=1e-5, atol=1e-7), \
        f"Nested results differ: old={ll_old:.10f}, new={ll_new:.10f}"


def test_nested_variable_case():
    """
    Test nested structure with variable outer prior.

    This tests a case that would be complex with the old approach but is
    simple with the new approach.

    Setup:
    - Inner: N_base + F_inner^T P_inner F_inner (all constant)
    - Outer: inner + F_outer^T P_outer F_outer (P_outer varies)

    Expected: Both approaches produce identical results for multiple parameters
    """
    # Setup data
    np.random.seed(45)
    n_data = 100
    n_inner, n_outer = 5, 8

    N_base = np.ones(n_data) * 0.1
    F_inner = np.random.randn(n_data, n_inner)
    F_outer = np.random.randn(n_data, n_outer)
    y_data = np.random.randn(n_data)
    inner_amp = 1.5

    # OLD IMPLEMENTATION
    N_base_old = NoiseMatrix1D_novar(N_base)
    P_inner_old = NoiseMatrix1D_novar(np.ones(n_inner) * inner_amp**2)
    inner_old = WoodburyKernel_novar(N_base_old, F_inner, P_inner_old)

    # P_outer function with .params attribute (works with both old and new)
    def P_outer_func(params):
        return jnp.ones(n_outer) * params['outer_amp']**2
    P_outer_func.params = ['outer_amp']

    P_outer_old = NoiseMatrix1D_var(P_outer_func)
    outer_old = WoodburyKernel_varP(inner_old, F_outer, P_outer_old)
    loglike_old = outer_old.make_kernelproduct(y_data)

    # NEW IMPLEMENTATION
    inner_new = WoodburyKernel(
        N_base,
        F_inner,
        np.ones(n_inner) * inner_amp**2
    )

    # Same P_outer function works directly
    outer_new = WoodburyKernel(
        inner_new,  # Nest the inner kernel
        F_outer,
        P_outer_func  # Variable P
    )
    loglike_new = outer_new.make_kernelproduct(y_data)

    # COMPARE multiple parameter values
    test_amplitudes = [0.5, 1.0, 1.5, 2.0, 2.5]

    for amp in test_amplitudes:
        params = {'outer_amp': amp}
        ll_old = loglike_old(params)
        ll_new = loglike_new(params)

        assert np.allclose(ll_old, ll_new, rtol=1e-5, atol=1e-7), \
            f"Nested variable results differ for outer_amp={amp}: old={ll_old:.10f}, new={ll_new:.10f}"
