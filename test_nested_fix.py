"""
Quick test to verify nested WoodburyKernel works after the fix.
"""
import jax
import jax.numpy as jnp
from matrix_refactored import WoodburyKernel

# Setup
n_data = 50
n_inner, n_outer = 3, 5
key = jax.random.PRNGKey(0)

N_base = jnp.ones(n_data) * 0.1
F_inner = jax.random.normal(key, (n_data, n_inner))
F_outer = jax.random.normal(jax.random.PRNGKey(1), (n_data, n_outer))
y_data = jax.random.normal(jax.random.PRNGKey(2), (n_data,))

# Create nested structure
print("Creating nested WoodburyKernel...")
print(f"  Inner: N_base + F_inner^T P_inner F_inner")
print(f"  Outer: inner + F_outer^T P_outer F_outer")

# Inner kernel (constant P_inner for simplicity)
P_inner = jnp.ones(n_inner) * 2.0
inner = WoodburyKernel(N_base, F_inner, P_inner)

# Outer kernel with inner as N (variable P_outer)
P_outer_func = lambda params: jnp.ones(n_outer) * params['outer_amp']**2
outer = WoodburyKernel(inner, F_outer, P_outer_func)

print(f"\nInner is_constant: {inner.is_constant}")
print(f"Inner params: {inner.params}")
print(f"\nOuter is_constant: {outer.is_constant}")
print(f"Outer params: {outer.params}")

# Create log-likelihood
print("\nCreating kernel product closure...")
loglike = outer.make_kernelproduct(y_data)

print(f"Closure params: {loglike.params}")

# Evaluate
print("\nEvaluating with params...")
try:
    params = {'outer_amp': 1.5}
    ll = loglike(params)
    print(f"✓ SUCCESS! Log-likelihood: {ll:.4f}")

    # Test with different parameters
    print("\nTesting with multiple parameter values:")
    for amp in [0.5, 1.0, 2.0]:
        ll = loglike({'outer_amp': amp})
        print(f"  outer_amp={amp}: ll={ll:.4f}")

    print("\n✓ All tests passed!")

except Exception as e:
    print(f"✗ FAILED with error:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
