"""
matrix_refactored.py - Integration layer for the new graph-based approach.

This file shows how to integrate the graph library with your existing code.
It provides the WoodburyKernel class that wraps WoodburyGraph and provides
a compatible API with the current matrix.py.

STRUCTURE:
  - woodbury_graph.py: Core graph library (Leaf, OpNode, WoodburyGraph)
  - matrix_refactored.py: This file - integration wrapper (WoodburyKernel)
  - matrix.py: Your existing legacy code (keep separate for now)
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Set, Callable, Optional
import inspect

# Import core graph library
from .woodbury_graph import (
    Leaf,
    DataLeaf,
    ParameterLeaf,
    WoodburyGraph,
)


# ============================================================================
# Helper Leaf Classes
# ============================================================================


class FunctionLeaf(Leaf):
    """Leaf for functions of multiple parameters."""

    def __init__(self, func: Callable, param_names: list, name=None):
        self.func = func
        self._params = set(param_names)
        self.name = name or "func"

    @property
    def is_constant(self):
        return len(self._params) == 0

    @property
    def params(self):
        return self._params

    def eval(self, params):
        return self.func(params)

    def __repr__(self):
        return f"{self.name}[func]"


class OldNoiseMatrixWrapper(Leaf):
    """
    Wrapper for old-style NoiseMatrix objects.

    Old NoiseMatrix has:
    - solve_1d(y) -> returns (solution, logdet)
    - solve_2d(Y) -> returns (solution, logdet)
    - params attribute (optional)
    """

    def __init__(self, noise_matrix, name=None):
        self.noise_matrix = noise_matrix
        self.name = name or "N"

    @property
    def is_constant(self):
        # Old NoiseMatrix is constant if it has no params
        return not hasattr(self.noise_matrix, 'params') or len(self.noise_matrix.params) == 0

    @property
    def params(self):
        if hasattr(self.noise_matrix, 'params'):
            return set(self.noise_matrix.params)
        return set()

    def eval(self, params=None):
        # Can't materialize - this shouldn't be called
        raise NotImplementedError("OldNoiseMatrixWrapper doesn't support eval - use solve instead")

    def solve(self, b, params):
        """Use old solve interface."""
        if b.ndim == 1:
            solution, _ = self.noise_matrix.solve_1d(b)
        else:
            solution, _ = self.noise_matrix.solve_2d(b)
        return solution

    def __repr__(self):
        const_str = "const" if self.is_constant else "var"
        return f"{self.name}[OldNoiseMatrix,{const_str}]"


# ============================================================================
# WoodburyKernel - Integration Wrapper
# ============================================================================


class WoodburyKernel:
    """
    Unified Woodbury kernel using graph-based approach.

    This is the NEW API that replaces all these old classes:
    - WoodburyKernel_novar
    - WoodburyKernel_varP
    - WoodburyKernel_varN
    - WoodburyKernel_varNP
    - WoodburyKernel_varFP
    - etc.

    The graph automatically handles optimization based on what's constant/variable.

    Example:
        # All of these use the SAME class:
        kernel1 = WoodburyKernel(N_const, F_const, P_const)      # All constant
        kernel2 = WoodburyKernel(N_const, F_const, P_variable)   # Only P varies
        kernel3 = WoodburyKernel(N_var, F_const, P_var)          # N, P vary

        # Even nested structures:
        inner = WoodburyKernel(N_base, F_inner, P_inner)
        outer = WoodburyKernel(inner, F_outer, P_outer)  # Works!
    """

    def __init__(self, N, F, P):
        """
        Create Woodbury kernel.

        Args:
            N: Noise specification - can be:
                - numpy/jax array (constant diagonal)
                - Old NoiseMatrix object (has solve_1d, solve_2d methods)
                - Another WoodburyKernel (for nesting!)
                - Callable function (parameter-dependent)

            F: Design matrix - can be:
                - numpy/jax array (constant)
                - Callable function (parameter-dependent)

            P: Prior specification - can be:
                - numpy/jax array (constant)
                - Old NoiseMatrix object
                - Callable function (parameter-dependent)
        """
        self.N_spec = N
        self.F_spec = F
        self.P_spec = P
        self.name = "WoodburyKernel"

    # ========================================================================
    # Leaf-like interface (for nesting)
    # ========================================================================

    @property
    def is_constant(self):
        """Constant if all components are constant."""
        N_leaf = self._make_leaf(self.N_spec, "N")
        F_leaf = self._make_leaf(self.F_spec, "F")
        P_leaf = self._make_leaf(self.P_spec, "P")
        return N_leaf.is_constant and F_leaf.is_constant and P_leaf.is_constant

    @property
    def params(self):
        """All parameters needed."""
        N_leaf = self._make_leaf(self.N_spec, "N")
        F_leaf = self._make_leaf(self.F_spec, "F")
        P_leaf = self._make_leaf(self.P_spec, "P")
        return N_leaf.params | F_leaf.params | P_leaf.params

    def eval(self, params=None):
        """
        Evaluate to get the matrix N + F^T P F.

        NOTE: This should rarely be called! For nested structures, we use solve()
        and compute_logdet() instead to avoid forming the full matrix.

        This is only here for completeness of the Leaf interface.
        """
        raise NotImplementedError(
            "WoodburyKernel.eval() should not be called! "
            "For nested structures, use solve() or compute_logdet() methods instead. "
            "We never want to materialize the full matrix N + F^T P F."
        )

    def solve(self, b, params):
        """
        Solve (N + F^T P F)^{-1} b using Woodbury identity.

        This makes WoodburyKernel act like a Leaf for nesting.

        Args:
            b: RHS vector or matrix
            params: Parameter dictionary

        Returns:
            Solution to (N + F^T P F)^{-1} b
        """
        # Create leaves
        N_leaf = self._make_leaf(self.N_spec, "N")
        F_leaf = self._make_leaf(self.F_spec, "F")
        P_leaf = self._make_leaf(self.P_spec, "P")
        b_leaf = DataLeaf(b, name="b")

        # Build graph and solve
        graph = WoodburyGraph(N_leaf, F_leaf, P_leaf, b_leaf)
        return graph.solve(b, params)

    def compute_logdet(self, params):
        """
        Compute log|N + F^T P F| using Woodbury determinant identity.

        This makes WoodburyKernel work correctly in nested structures.

        Args:
            params: Parameter dictionary

        Returns:
            log|N + F^T P F| = log|N| + log|P| + log|S|
        """
        # Create leaves (need a dummy y for graph creation)
        N_leaf = self._make_leaf(self.N_spec, "N")
        F_leaf = self._make_leaf(self.F_spec, "F")
        P_leaf = self._make_leaf(self.P_spec, "P")

        # Create dummy y (won't be used for logdet calculation)
        y_dummy = DataLeaf(jnp.zeros(1), name="y_dummy")

        # Build graph
        graph = WoodburyGraph(N_leaf, F_leaf, P_leaf, y_dummy)

        # Use the graph's compute_logdet method
        return graph.compute_logdet(params)

    # ========================================================================
    # Main API Methods
    # ========================================================================

    def make_kernelproduct(self, y):
        """
        Create kernel product closure.

        This is the MAIN API method that matches the current interface exactly!

        Args:
            y: Data vector - can be:
                - numpy/jax array (constant)
                - Callable function (parameter-dependent)

        Returns:
            Closure that computes: -0.5 * y^T (N + F^T P F)^{-1} y - 0.5 * log|N + F^T P F|

        Example:
            kernel = WoodburyKernel(N, F, P)
            loglike = kernel.make_kernelproduct(y_data)
            ll = loglike({'amplitude': 1.5})
        """
        # Convert to leaves
        N_leaf = self._make_leaf(self.N_spec, "N")
        F_leaf = self._make_leaf(self.F_spec, "F")
        P_leaf = self._make_leaf(self.P_spec, "P")
        y_leaf = self._make_leaf(y, "y")

        # Build graph
        graph = WoodburyGraph(N_leaf, F_leaf, P_leaf, y_leaf)

        # Create closure
        return graph.make_kernelproduct_closure()

    def make_kernelsolve(self, y, T):
        """
        Create kernel solve closure for T^T (N + F^T P F)^{-1} y.

        This matches the current interface for conditional sampling.

        Args:
            y: Data vector
            T: Projection matrix

        Returns:
            Closure that computes T^T W^{-1} y and T^T W^{-1} T

        Example:
            kernel = WoodburyKernel(N, F, P)
            solver = kernel.make_kernelsolve(y_data, T_matrix)
            TtWy, TtWT = solver({'amplitude': 1.5})
        """
        # Import InnerProductOp from graph library
        from woodbury_graph import InnerProductOp

        # Convert to leaves
        N_leaf = self._make_leaf(self.N_spec, "N")
        F_leaf = self._make_leaf(self.F_spec, "F")
        P_leaf = self._make_leaf(self.P_spec, "P")
        y_leaf = self._make_leaf(y, "y")
        T_leaf = self._make_leaf(T, "T")

        # Build graphs
        graph_y = WoodburyGraph(N_leaf, F_leaf, P_leaf, y_leaf)
        graph_T = WoodburyGraph(N_leaf, F_leaf, P_leaf, T_leaf)

        # Create nodes for T^T W^{-1} y and T^T W^{-1} T
        # Note: graph_y.solution creates it lazily
        TtWy = InnerProductOp(T_leaf, graph_y.solution, name="TtWy")
        TtWT = InnerProductOp(T_leaf, graph_T.solution, name="TtWT")

        # Precompute constants
        for node in [TtWy, TtWT]:
            if hasattr(node, 'precompute'):
                node.precompute()

        def closure(params):
            return TtWy.eval(params), TtWT.eval(params)

        all_params = graph_y.params | graph_T.params | T_leaf.params
        closure.params = sorted(all_params)

        return closure

    # ========================================================================
    # Helper: Convert specs to leaves
    # ========================================================================

    def _make_leaf(self, spec, name):
        """
        Convert a specification to a Leaf node.

        This handles all the different input types:
        - Arrays -> DataLeaf
        - WoodburyKernel -> itself (for nesting)
        - Old NoiseMatrix -> OldNoiseMatrixWrapper
        - Callables -> FunctionLeaf
        """
        # 1. Constant array (numpy or jax)
        if isinstance(spec, (jax.Array, jnp.ndarray, np.ndarray)):
            return DataLeaf(spec, name=name)

        # 2. Another WoodburyKernel (for nesting)
        # MUST check this BEFORE callable since WoodburyKernel is callable via solve()
        elif isinstance(spec, WoodburyKernel):
            return spec

        # 3. WoodburyGraph or anything else with solve()
        elif hasattr(spec, 'solve') and not callable(spec):
            return spec

        # 4. Old-style NoiseMatrix (has solve_1d/solve_2d)
        elif hasattr(spec, 'solve_1d'):
            return OldNoiseMatrixWrapper(spec, name=name)

        # 5. Callable function
        elif callable(spec):
            if hasattr(spec, 'params'):
                # It's a parameterized function with .params attribute
                return FunctionLeaf(spec, list(spec.params), name=name)
            else:
                # Try to infer params from function signature
                try:
                    sig = inspect.signature(spec)
                    param_names = list(sig.parameters.keys())
                    if 'params' in param_names:
                        # It takes a params dict
                        return FunctionLeaf(spec, [], name=name)
                    else:
                        return FunctionLeaf(spec, param_names, name=name)
                except:
                    # Fallback: assume no params
                    return FunctionLeaf(spec, [], name=name)

        else:
            raise TypeError(f"Unknown specification type: {type(spec)}")


# ============================================================================
# EXAMPLES
# ============================================================================


def example_simple():
    """Simple example: constant N, F; variable P."""
    print("=" * 70)
    print("Example 1: Simple Case (N, F constant; P variable)")
    print("=" * 70)

    # Data
    n_data, n_basis = 100, 10
    key = jax.random.PRNGKey(0)

    N_data = jnp.ones(n_data) * 0.5
    F_matrix = jax.random.normal(key, (n_data, n_basis))
    y_data = jax.random.normal(jax.random.PRNGKey(1), (n_data,))

    # Variable prior
    class P_var:
        params = ['amplitude']
        def __call__(self, params):
            return jnp.ones(n_basis) * params['amplitude']**2

    # Create kernel
    kernel = WoodburyKernel(N_data, F_matrix, P_var())

    print(f"\nKernel properties:")
    print(f"  is_constant: {kernel.is_constant}")
    print(f"  params: {kernel.params}")

    # Create likelihood
    loglike = kernel.make_kernelproduct(y_data)

    print(f"\nClosure properties:")
    print(f"  params: {loglike.params}")

    # Evaluate
    ll = loglike({'amplitude': 1.5})
    print(f"\nLog-likelihood: {ll:.4f}")


def example_nested():
    """Nested example: inner + outer Woodbury."""
    print("\n\n" + "=" * 70)
    print("Example 2: Nested Woodbury")
    print("=" * 70)

    n_data = 100
    n_inner, n_outer = 5, 10
    key = jax.random.PRNGKey(0)

    N_base = jnp.ones(n_data) * 0.1
    F_inner = jax.random.normal(key, (n_data, n_inner))
    F_outer = jax.random.normal(jax.random.PRNGKey(1), (n_data, n_outer))
    y_data = jax.random.normal(jax.random.PRNGKey(2), (n_data,))

    # Variable priors
    class P_inner:
        params = ['inner_amp']
        def __call__(self, params):
            return jnp.ones(n_inner) * params['inner_amp']**2

    class P_outer:
        params = ['outer_amp']
        def __call__(self, params):
            return jnp.ones(n_outer) * params['outer_amp']**2

    # Inner Woodbury
    inner = WoodburyKernel(N_base, F_inner, P_inner())

    # Outer Woodbury - uses inner as N!
    outer = WoodburyKernel(inner, F_outer, P_outer())

    print(f"\nOuter kernel properties:")
    print(f"  is_constant: {outer.is_constant}")
    print(f"  params: {outer.params}")

    # Create likelihood
    loglike = outer.make_kernelproduct(y_data)

    print(f"\nClosure properties:")
    print(f"  params: {loglike.params}")

    # Evaluate
    ll = loglike({'inner_amp': 1.0, 'outer_amp': 2.0})
    print(f"\nLog-likelihood: {ll:.4f}")


def example_all_combinations():
    """Show that single class handles all constant/variable patterns."""
    print("\n\n" + "=" * 70)
    print("Example 3: All Constant/Variable Combinations")
    print("=" * 70)

    n_data, n_basis = 100, 10
    key = jax.random.PRNGKey(0)

    N_const = jnp.ones(n_data) * 0.5
    F_const = jax.random.normal(key, (n_data, n_basis))
    P_const = jnp.ones(n_basis) * 2.0
    y_const = jax.random.normal(jax.random.PRNGKey(1), (n_data,))

    class N_var:
        params = ['noise_level']
        def __call__(self, params):
            return jnp.ones(n_data) * params['noise_level']

    class P_var:
        params = ['amplitude']
        def __call__(self, params):
            return jnp.ones(n_basis) * params['amplitude']**2

    cases = [
        ("All constant", N_const, F_const, P_const, {}),
        ("Only P varies", N_const, F_const, P_var(), {'amplitude': 1.5}),
        ("Only N varies", N_var(), F_const, P_const, {'noise_level': 0.5}),
        ("N and P vary", N_var(), F_const, P_var(), {'noise_level': 0.5, 'amplitude': 1.5}),
    ]

    print("\nSingle WoodburyKernel class handles all cases:")
    print("-" * 70)

    for case_name, N, F, P, params in cases:
        kernel = WoodburyKernel(N, F, P)
        loglike = kernel.make_kernelproduct(y_const)
        ll = loglike(params)
        print(f"{case_name:20s}: params={sorted(loglike.params):30s} ll={ll:.4f}")


if __name__ == "__main__":
    example_simple()
    example_nested()
    example_all_combinations()

    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("File structure:")
    print("  woodbury_graph.py:     Core graph library (import this)")
    print("  matrix_refactored.py:  Integration wrapper (WoodburyKernel)")
    print("  matrix.py:             Your legacy code (keep separate)")
    print()
    print("To use in a notebook:")
    print("  from matrix_refactored import WoodburyKernel")
    print("  kernel = WoodburyKernel(N, F, P)")
    print("  loglike = kernel.make_kernelproduct(y)")
    print("  ll = loglike(params)")
    print("=" * 70)
