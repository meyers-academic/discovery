# Comparison Tests: Old vs New Implementation

This document describes the comparison tests in `test_comparison.py` that verify the refactored graph-based implementation produces identical results to the original specialized classes.

## Requirements

To run these tests, you need:
- `numpy`
- `jax` and `jax.numpy`
- The original `discovery` package installed (`src/discovery/matrix.py`)

```bash
pip install numpy jax jaxlib
cd /home/user/discovery
python test_comparison.py
```

## Test Suite Overview

The test suite includes 4 comprehensive tests comparing the old and new implementations:

### Test 1: Simple Constant Case

**Purpose**: Verify that the new unified `WoodburyKernel` produces identical results to the old specialized `WoodburyKernel_novar` class when all components are constant.

**Setup**:
- `N`: Constant diagonal noise matrix (100 elements)
- `F`: Constant dense design matrix (100 × 10)
- `P`: Constant diagonal prior covariance (10 elements)
- `y`: Constant data vector (100 elements)

**Old Approach**:
```python
N_old = NoiseMatrix1D_novar(N_diag)
P_old = NoiseMatrix1D_novar(P_diag)
kernel_old = WoodburyKernel_novar(N_old, F_matrix, P_old)
loglike_old = kernel_old.make_kernelproduct(y_data)
ll = loglike_old({})  # No parameters needed
```

**New Approach**:
```python
kernel_new = WoodburyKernel(N_diag, F_matrix, P_diag)
loglike_new = kernel_new.make_kernelproduct(y_data)
ll = loglike_new({})  # Automatically detects all constant
```

**Verification**:
- Computes log-likelihood: `-0.5 * y^T (N + F^T P F)^{-1} y - 0.5 * log|N + F^T P F|`
- Checks results match within `rtol=1e-6, atol=1e-8`

**Key Advantage**:
- Old: Must manually choose `WoodburyKernel_novar` class
- New: Single `WoodburyKernel` class automatically optimizes

---

### Test 2: Variable P Case

**Purpose**: Verify that the new implementation correctly handles parameter-dependent components, matching the old `WoodburyKernel_varP` class.

**Setup**:
- `N`: Constant diagonal noise matrix
- `F`: Constant dense design matrix
- `P`: **Variable** prior depending on `'amplitude'` parameter
- `y`: Constant data vector

**Old Approach**:
```python
N_old = NoiseMatrix1D_novar(N_diag)
P_old = NoiseMatrix1D_var(...)  # Custom class with params attribute
kernel_old = WoodburyKernel_varP(N_old, F_matrix, P_old)
loglike_old = kernel_old.make_kernelproduct(y_data)
ll = loglike_old({'amplitude': 1.5})
```

**New Approach**:
```python
P_func = lambda params: jnp.ones(n_basis) * params['amplitude']**2
kernel_new = WoodburyKernel(N_diag, F_matrix, P_func)
loglike_new = kernel_new.make_kernelproduct(y_data)
ll = loglike_new({'amplitude': 1.5})  # Automatically detects parameter
```

**Verification**:
- Tests with multiple amplitude values: [0.5, 1.0, 1.5, 2.0, 3.0]
- All must match within tolerance
- Graph automatically caches constant components (N, F)

**Key Advantage**:
- Old: Must choose correct specialized class (`_varP`, `_varN`, `_varNP`, etc.)
- New: Single class automatically detects what varies

---

### Test 3: Nested Case (Constant)

**Purpose**: Verify that nested Woodbury structures work correctly. This tests:
```
W = (N_base + F_inner^T P_inner F_inner) + F_outer^T P_outer F_outer
```

**Setup**:
- Inner structure: `N_base + F_inner^T P_inner F_inner`
- Outer structure: `inner + F_outer^T P_outer F_outer`
- All components constant for this test

**Old Approach** (Manual Composition):
```python
# Step 1: Build inner kernel
N_base_old = NoiseMatrix1D_novar(N_base)
P_inner_old = NoiseMatrix1D_novar(P_inner)
inner_old = WoodburyKernel_novar(N_base_old, F_inner, P_inner_old)

# Step 2: Use inner as N for outer
P_outer_old = NoiseMatrix1D_novar(P_outer)
outer_old = WoodburyKernel_novar(inner_old, F_outer, P_outer_old)

# Step 3: Create closure
loglike_old = outer_old.make_kernelproduct(y_data)
```

**New Approach** (Direct Nesting):
```python
# Step 1: Create inner kernel
inner_new = WoodburyKernel(N_base, F_inner, P_inner)

# Step 2: Nest directly!
outer_new = WoodburyKernel(inner_new, F_outer, P_outer)

# Step 3: Create closure
loglike_new = outer_new.make_kernelproduct(y_data)
```

**Verification**:
- Both approaches compute the same nested structure
- Results match within `rtol=1e-5, atol=1e-7` (slightly looser due to numerical recursion)

**Key Advantage**:
- Old: Works but requires manual step-by-step composition
- New: Natural nesting, kernel implements Leaf interface (`solve()`, `compute_logdet()`)

---

### Test 4: Nested with Variable P_outer

**Purpose**: Test complex nested structure with parameter-dependent outer prior. This would require `WoodburyKernel_varP` with nested N in the old approach.

**Setup**:
- Inner: Constant Woodbury structure
- Outer: Uses inner as N, **variable** P_outer

**Old Approach**:
```python
# Build constant inner
inner_old = WoodburyKernel_novar(N_base_old, F_inner, P_inner_old)

# Use as N with variable P
P_outer_var = NoiseMatrix1D_var(...)  # Custom class
outer_old = WoodburyKernel_varP(inner_old, F_outer, P_outer_var)
```

**New Approach**:
```python
# Create inner
inner_new = WoodburyKernel(N_base, F_inner, P_inner)

# Nest with variable P
P_outer_func = lambda params: jnp.ones(n) * params['outer_amp']**2
outer_new = WoodburyKernel(inner_new, F_outer, P_outer_func)
```

**Verification**:
- Tests with multiple `outer_amp` values: [0.5, 1.0, 1.5, 2.0, 2.5]
- All must match within tolerance
- Graph automatically caches inner structure (doesn't depend on outer_amp)

**Key Advantage**:
- Old: Complex setup with multiple specialized classes
- New: Simple nesting, automatic optimization

---

## Expected Output

When tests pass, you should see:

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                  COMPARISON TEST SUITE: OLD vs NEW                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

================================================================================
TEST: Simple Constant Case
================================================================================
...
  ✓ PASS (within rtol=1e-06, atol=1e-08)

✓ Test PASSED: Constant case produces identical results!

================================================================================
TEST: Variable P Case
================================================================================
...
  ✓ PASS (within rtol=1e-06, atol=1e-08)

✓ Test PASSED: Variable P case produces identical results for all parameter values!

================================================================================
TEST: Nested Case
================================================================================
...
  ✓ PASS (within rtol=1e-05, atol=1e-07)

✓ Test PASSED: Nested structure produces identical results!

================================================================================
TEST: Nested Case with Variable P_outer
================================================================================
...
  ✓ PASS (within rtol=1e-05, atol=1e-07)

✓ Test PASSED: Nested variable case produces identical results!

================================================================================
SUMMARY
================================================================================
  constant                 : ✓ PASS
  variable_p               : ✓ PASS
  nested                   : ✓ PASS
  nested_variable          : ✓ PASS

================================================================================
✓ ALL TESTS PASSED

The refactored graph-based implementation produces identical results
to the original specialized classes while eliminating code duplication.
================================================================================
```

## What These Tests Prove

1. **Correctness**: The new implementation produces numerically identical results to the extensively tested old implementation

2. **Completeness**: All use cases from the old specialized classes are covered by the single unified class

3. **Compatibility**: The API is similar enough that migration is straightforward

4. **Nesting**: The new approach naturally supports recursive nesting that would be complex in the old approach

## Benefits Demonstrated

### Code Reduction
- **Old**: ~10 specialized classes (`WoodburyKernel_novar`, `_varP`, `_varN`, `_varNP`, `_varFP`, etc.)
- **New**: Single `WoodburyKernel` class

### Automatic Optimization
- **Old**: User must manually choose the right specialized class
- **New**: Graph automatically detects constant/variable components and optimizes

### Natural Nesting
- **Old**: Nesting works but requires understanding which specialized class to use
- **New**: Natural syntax: `WoodburyKernel(inner_kernel, F, P)`

### Maintainability
- **Old**: Bug fixes must be applied to each specialized class
- **New**: Single implementation, fix once

## Running Individual Tests

You can run individual tests by importing and calling them:

```python
from test_comparison import test_constant_case, test_variable_p_case

# Run just one test
test_constant_case()

# Or specific test
test_nested_case()
```

## Troubleshooting

### Import Errors
If you get `ModuleNotFoundError` for numpy or jax:
```bash
pip install numpy jax jaxlib
```

### Path Issues
If you get import errors for `discovery.matrix`:
```bash
export PYTHONPATH=/home/user/discovery/src:$PYTHONPATH
python test_comparison.py
```

### Tolerance Failures
If tests fail due to numerical differences:
- Check JAX version (different versions may have slightly different numerics)
- Nested tests use looser tolerance (`rtol=1e-5`) due to recursive operations
- Differences beyond `1e-5` relative error indicate a real problem

## Next Steps

After these tests pass:
1. Test with your actual production data
2. Migrate existing code incrementally
3. Replace old specialized classes one by one
4. Eventually deprecate old classes

## Contact

For issues with these tests, check:
- `woodbury_graph.py` - Core graph library
- `matrix_refactored.py` - Integration wrapper
- `REFACTORING_SUMMARY.md` - Full design documentation
