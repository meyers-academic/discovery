"""
Comparison tests between old matrix.py and new matrix_refactored.py.

These tests verify that the refactored graph-based implementation produces
identical results to the original specialized classes.

Test cases:
1. Simple constant case: WoodburyKernel_novar vs WoodburyKernel (all constant)
2. Variable P case: WoodburyKernel_varP vs WoodburyKernel (P varies)
3. Nested case: Manual composition vs WoodburyKernel (nested structure)
"""

import sys
import numpy as np
import jax
import jax.numpy as jnp

# Add src to path for imports
sys.path.insert(0, '/home/user/discovery/src')

# Import old implementation
from discovery.matrix import (
    NoiseMatrix1D_novar,
    NoiseMatrix1D_var,
    WoodburyKernel_novar,
    WoodburyKernel_varP,
)

# Import new implementation
from matrix_refactored import WoodburyKernel


def print_test_header(title):
    """Print a formatted test header."""
    print("\n" + "=" * 80)
    print(f"TEST: {title}")
    print("=" * 80)


def print_results(old_result, new_result, rtol=1e-6, atol=1e-8):
    """Print comparison results."""
    print(f"\n  Old implementation result: {old_result:.10f}")
    print(f"  New implementation result: {new_result:.10f}")
    print(f"  Absolute difference:       {abs(old_result - new_result):.2e}")
    print(f"  Relative difference:       {abs(old_result - new_result) / abs(old_result):.2e}")

    if np.allclose(old_result, new_result, rtol=rtol, atol=atol):
        print(f"  ✓ PASS (within rtol={rtol}, atol={atol})")
        return True
    else:
        print(f"  ✗ FAIL (exceeds tolerance)")
        return False


# ============================================================================
# Test 1: Simple Constant Case
# ============================================================================

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
    print_test_header("Simple Constant Case")

    # Setup data
    np.random.seed(42)
    n_data, n_basis = 100, 10

    N_diag = np.ones(n_data) * 0.5
    F_matrix = np.random.randn(n_data, n_basis)
    P_diag = np.ones(n_basis) * 2.0
    y_data = np.random.randn(n_data)

    print(f"\nSetup:")
    print(f"  n_data = {n_data}, n_basis = {n_basis}")
    print(f"  N: diagonal (constant)")
    print(f"  F: dense matrix (constant)")
    print(f"  P: diagonal (constant)")
    print(f"  y: data vector (constant)")

    # OLD IMPLEMENTATION
    print(f"\n  Creating OLD implementation (WoodburyKernel_novar)...")
    N_old = NoiseMatrix1D_novar(N_diag)
    P_old = NoiseMatrix1D_novar(P_diag)
    kernel_old = WoodburyKernel_novar(N_old, F_matrix, P_old)
    loglike_old = kernel_old.make_kernelproduct(y_data)
    ll_old = loglike_old({})  # No parameters

    # NEW IMPLEMENTATION
    print(f"  Creating NEW implementation (WoodburyKernel)...")
    kernel_new = WoodburyKernel(N_diag, F_matrix, P_diag)
    loglike_new = kernel_new.make_kernelproduct(y_data)
    ll_new = loglike_new({})  # No parameters

    # COMPARE
    print(f"\nResults:")
    passed = print_results(ll_old, ll_new)

    if passed:
        print(f"\n✓ Test PASSED: Constant case produces identical results!")
    else:
        print(f"\n✗ Test FAILED: Results differ beyond tolerance!")

    return passed


# ============================================================================
# Test 2: Variable P Case
# ============================================================================

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
    print_test_header("Variable P Case")

    # Setup data
    np.random.seed(43)
    n_data, n_basis = 100, 10

    N_diag = np.ones(n_data) * 0.5
    F_matrix = np.random.randn(n_data, n_basis)
    y_data = np.random.randn(n_data)

    print(f"\nSetup:")
    print(f"  n_data = {n_data}, n_basis = {n_basis}")
    print(f"  N: diagonal (constant)")
    print(f"  F: dense matrix (constant)")
    print(f"  P: diagonal (VARIABLE - depends on 'amplitude')")
    print(f"  y: data vector (constant)")

    # OLD IMPLEMENTATION
    print(f"\n  Creating OLD implementation (WoodburyKernel_varP)...")
    N_old = NoiseMatrix1D_novar(N_diag)

    # P_var uses the old API pattern
    class P_old_var(NoiseMatrix1D_var):
        def __init__(self, n_basis):
            self.n_basis = n_basis
            self.params = ['amplitude']

    P_old = P_old_var(n_basis)
    kernel_old = WoodburyKernel_varP(N_old, F_matrix, P_old)
    loglike_old = kernel_old.make_kernelproduct(y_data)

    # NEW IMPLEMENTATION
    print(f"  Creating NEW implementation (WoodburyKernel)...")

    # P_new uses lambda function
    P_new_func = lambda params: jnp.ones(n_basis) * params['amplitude']**2
    kernel_new = WoodburyKernel(N_diag, F_matrix, P_new_func)
    loglike_new = kernel_new.make_kernelproduct(y_data)

    # COMPARE multiple parameter values
    print(f"\nResults for different parameter values:")

    all_passed = True
    test_amplitudes = [0.5, 1.0, 1.5, 2.0, 3.0]

    for amp in test_amplitudes:
        params = {'amplitude': amp}

        ll_old = loglike_old(params)
        ll_new = loglike_new(params)

        print(f"\n  amplitude = {amp}:")
        passed = print_results(ll_old, ll_new)
        all_passed = all_passed and passed

    if all_passed:
        print(f"\n✓ Test PASSED: Variable P case produces identical results for all parameter values!")
    else:
        print(f"\n✗ Test FAILED: Some parameter values produced different results!")

    return all_passed


# ============================================================================
# Test 3: Nested Case
# ============================================================================

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
    print_test_header("Nested Case")

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

    print(f"\nSetup:")
    print(f"  n_data = {n_data}")
    print(f"  Inner: {n_inner} basis functions")
    print(f"  Outer: {n_outer} basis functions")
    print(f"  Structure: (N_base + F_inner^T P_inner F_inner) + F_outer^T P_outer F_outer")
    print(f"  Parameters: inner_amp={inner_amp}, outer_amp={outer_amp}")

    # OLD IMPLEMENTATION (manual composition)
    print(f"\n  Creating OLD implementation (manual composition)...")
    print(f"    Step 1: Build inner kernel")
    N_base_old = NoiseMatrix1D_novar(N_base)
    P_inner_old = NoiseMatrix1D_novar(np.ones(n_inner) * inner_amp**2)
    inner_old = WoodburyKernel_novar(N_base_old, F_inner, P_inner_old)

    print(f"    Step 2: Use inner as N for outer kernel")
    P_outer_old = NoiseMatrix1D_novar(np.ones(n_outer) * outer_amp**2)
    outer_old = WoodburyKernel_novar(inner_old, F_outer, P_outer_old)

    print(f"    Step 3: Create log-likelihood closure")
    loglike_old = outer_old.make_kernelproduct(y_data)
    ll_old = loglike_old({})  # All constant for this test

    # NEW IMPLEMENTATION (direct nesting)
    print(f"\n  Creating NEW implementation (direct nesting)...")
    print(f"    Step 1: Create inner WoodburyKernel")
    inner_new = WoodburyKernel(
        N_base,
        F_inner,
        np.ones(n_inner) * inner_amp**2
    )

    print(f"    Step 2: Create outer WoodburyKernel with inner as N")
    outer_new = WoodburyKernel(
        inner_new,  # Directly nest!
        F_outer,
        np.ones(n_outer) * outer_amp**2
    )

    print(f"    Step 3: Create log-likelihood closure")
    loglike_new = outer_new.make_kernelproduct(y_data)
    ll_new = loglike_new({})  # All constant for this test

    # COMPARE
    print(f"\nResults:")
    passed = print_results(ll_old, ll_new, rtol=1e-5, atol=1e-7)

    if passed:
        print(f"\n✓ Test PASSED: Nested structure produces identical results!")
        print(f"\nKey advantage of new approach:")
        print(f"  - Old: Required manual step-by-step composition")
        print(f"  - New: Direct nesting with WoodburyKernel(inner, F, P)")
    else:
        print(f"\n✗ Test FAILED: Nested results differ!")

    return passed


# ============================================================================
# Test 4: Nested with Variable P_outer
# ============================================================================

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
    print_test_header("Nested Case with Variable P_outer")

    # Setup data
    np.random.seed(45)
    n_data = 100
    n_inner, n_outer = 5, 8

    N_base = np.ones(n_data) * 0.1
    F_inner = np.random.randn(n_data, n_inner)
    F_outer = np.random.randn(n_data, n_outer)
    y_data = np.random.randn(n_data)
    inner_amp = 1.5

    print(f"\nSetup:")
    print(f"  n_data = {n_data}")
    print(f"  Inner: {n_inner} basis (constant)")
    print(f"  Outer: {n_outer} basis (P_outer VARIABLE)")
    print(f"  Structure: (N_base + F_inner^T P_inner F_inner) + F_outer^T P_outer F_outer")

    # OLD IMPLEMENTATION
    print(f"\n  Creating OLD implementation...")
    print(f"    Step 1: Build constant inner kernel")
    N_base_old = NoiseMatrix1D_novar(N_base)
    P_inner_old = NoiseMatrix1D_novar(np.ones(n_inner) * inner_amp**2)
    inner_old = WoodburyKernel_novar(N_base_old, F_inner, P_inner_old)

    print(f"    Step 2: Use inner as N, create variable P outer")

    class P_outer_var(NoiseMatrix1D_var):
        def __init__(self, n_basis):
            self.n_basis = n_basis
            self.params = ['outer_amp']

    P_outer_old = P_outer_var(n_outer)
    outer_old = WoodburyKernel_varP(inner_old, F_outer, P_outer_old)
    loglike_old = outer_old.make_kernelproduct(y_data)

    # NEW IMPLEMENTATION
    print(f"\n  Creating NEW implementation...")
    print(f"    Step 1: Create inner WoodburyKernel (constant)")
    inner_new = WoodburyKernel(
        N_base,
        F_inner,
        np.ones(n_inner) * inner_amp**2
    )

    print(f"    Step 2: Create outer with variable P_outer")
    P_outer_func = lambda params: jnp.ones(n_outer) * params['outer_amp']**2
    outer_new = WoodburyKernel(
        inner_new,  # Nest the inner kernel
        F_outer,
        P_outer_func  # Variable P
    )
    loglike_new = outer_new.make_kernelproduct(y_data)

    # COMPARE multiple parameter values
    print(f"\nResults for different outer_amp values:")

    all_passed = True
    test_amplitudes = [0.5, 1.0, 1.5, 2.0, 2.5]

    for amp in test_amplitudes:
        params = {'outer_amp': amp}

        ll_old = loglike_old(params)
        ll_new = loglike_new(params)

        print(f"\n  outer_amp = {amp}:")
        passed = print_results(ll_old, ll_new, rtol=1e-5, atol=1e-7)
        all_passed = all_passed and passed

    if all_passed:
        print(f"\n✓ Test PASSED: Nested variable case produces identical results!")
        print(f"\nKey advantage: New approach handles complex nesting naturally")
    else:
        print(f"\n✗ Test FAILED: Some parameter values produced different results!")

    return all_passed


# ============================================================================
# Run all tests
# ============================================================================

def run_all_tests():
    """Run all comparison tests."""
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 18 + "COMPARISON TEST SUITE: OLD vs NEW" + " " * 27 + "║")
    print("╚" + "═" * 78 + "╝")

    results = {}

    # Run tests
    results['constant'] = test_constant_case()
    results['variable_p'] = test_variable_p_case()
    results['nested'] = test_nested_case()
    results['nested_variable'] = test_nested_variable_case()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name:25s}: {status}")

    all_passed = all(results.values())

    print("\n" + "=" * 80)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("\nThe refactored graph-based implementation produces identical results")
        print("to the original specialized classes while eliminating code duplication.")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease review the failing tests above.")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
