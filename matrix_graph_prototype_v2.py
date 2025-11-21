"""
Prototype v2: Focus on Woodbury solve operations, not matrix construction.

Key insight: We need (N + F^T P F)^{-1} y, not N + F^T P F.
The Woodbury identity lets us solve without forming the full matrix.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from typing import Any, Callable, Optional, Set, List, Union


# ============================================================================
# Leaf Nodes (Data)
# ============================================================================

class Node:
    """Base class for data nodes."""

    def __init__(self):
        self._is_constant = None
        self._params = None

    @property
    def is_constant(self) -> bool:
        raise NotImplementedError

    @property
    def params(self) -> Set[str]:
        raise NotImplementedError

    def eval(self, params):
        raise NotImplementedError

    def make_closure(self):
        raise NotImplementedError


class ConstantNode(Node):
    """Constant data (evaluated once)."""

    def __init__(self, value):
        super().__init__()
        self.value = jnp.array(value) if not isinstance(value, jax.Array) else value
        self._is_constant = True
        self._params = set()

    @property
    def is_constant(self):
        return True

    @property
    def params(self):
        return set()

    def eval(self, params=None):
        return self.value

    def make_closure(self):
        value = self.value

        def closure(params=None):
            return value

        closure.params = []
        return closure

    def __repr__(self):
        return f"Const[{self.value.shape}]"


class VariableNode(Node):
    """Variable data (depends on parameters)."""

    def __init__(self, param_name: str, transform: Optional[Callable] = None):
        super().__init__()
        self.param_name = param_name
        self.transform = transform or (lambda x: x)
        self._is_constant = False
        self._params = {param_name}

    @property
    def is_constant(self):
        return False

    @property
    def params(self):
        return {self.param_name}

    def eval(self, params):
        return self.transform(params[self.param_name])

    def make_closure(self):
        param_name = self.param_name
        transform = self.transform

        def closure(params):
            return transform(params[param_name])

        closure.params = [param_name]
        return closure

    def __repr__(self):
        return f"Var[{self.param_name}]"


# ============================================================================
# Woodbury Structure (Core Abstraction)
# ============================================================================


class WoodburyStructure:
    """
    Represents the structure (N, F, P) and can solve with (N + F^T P F).

    The key operation is solve(y, params) which returns:
        - solution = (N + F^T P F)^{-1} y
        - logdet = log|N + F^T P F|

    Uses Woodbury identity - never forms the full matrix!

    Supports nesting: N can itself be another WoodburyStructure.
    """

    def __init__(self, N: Union[Node, "WoodburyStructure"], F: Node, P: Node):
        """
        Args:
            N: Base noise matrix (Node) OR another WoodburyStructure for nesting
            F: Design matrix (n_data x n_basis)
            P: Prior covariance (n_basis x n_basis or n_basis diagonal)
        """
        self.N = N
        self.F = F
        self.P = P

    @property
    def is_nested(self) -> bool:
        """True if N is also a WoodburyStructure."""
        return isinstance(self.N, WoodburyStructure)

    def depth(self) -> int:
        """Nesting depth of Woodbury structures."""
        if isinstance(self.N, WoodburyStructure):
            return 1 + self.N.depth()
        return 1

    @property
    def params(self) -> Set[str]:
        """All parameters needed by this structure."""
        params = set()
        if isinstance(self.N, WoodburyStructure):
            params.update(self.N.params)
        elif hasattr(self.N, "params"):
            params.update(self.N.params)
        params.update(self.F.params)
        params.update(self.P.params)
        return params

    def solve(self, y: Node, params) -> tuple:
        """
        Solve (N + F^T P F)^{-1} y using Woodbury identity.

        Woodbury identity:
        (A + UCV)^{-1} b = A^{-1}b - A^{-1}U(C^{-1} + VA^{-1}U)^{-1}VA^{-1}b

        In our notation:
        (N + F^T P F)^{-1} y = N^{-1}y - N^{-1}F(P^{-1} + F^T N^{-1}F)^{-1}F^T N^{-1}y

        Returns:
            solution: (N + F^T P F)^{-1} y
            logdet: log|N + F^T P F|
        """
        # Evaluate data nodes
        F_val = self.F.eval(params)
        P_val = self.P.eval(params)
        y_val = y.eval(params)

        # Step 1: Solve with N (recursively if needed)
        if isinstance(self.N, WoodburyStructure):
            # Recursive case: N is also Woodbury
            Ninv_y, logdet_N = self.N.solve(y, params)

            # Need N^{-1} F (solve multiple RHS)
            Ninv_F_list = []
            for i in range(F_val.shape[1]):
                F_col_node = ConstantNode(F_val[:, i])
                Ninv_F_col, _ = self.N.solve(F_col_node, params)
                Ninv_F_list.append(Ninv_F_col)
            Ninv_F = jnp.column_stack(Ninv_F_list)
        else:
            # Base case: N is simple
            N_val = self.N.eval(params)
            Ninv_y, logdet_N = self._solve_base(N_val, y_val)
            Ninv_F, _ = self._solve_base(N_val, F_val)

        # Step 2: Form Schur complement S = P^{-1} + F^T N^{-1} F
        FtNinvF = F_val.T @ Ninv_F

        if P_val.ndim == 1:
            # Diagonal P
            S = jnp.diag(1.0 / P_val) + FtNinvF
            logdet_P = jnp.sum(jnp.log(P_val))
        else:
            # Dense P
            P_factor = jsp.linalg.cho_factor(P_val)
            P_inv = jsp.linalg.cho_solve(P_factor, jnp.eye(P_val.shape[0]))
            S = P_inv + FtNinvF
            logdet_P = 2.0 * jnp.sum(jnp.log(jnp.diag(P_factor[0])))

        # Step 3: Solve S α = F^T N^{-1} y
        S_factor = jsp.linalg.cho_factor(S)
        logdet_S = 2.0 * jnp.sum(jnp.log(jnp.diag(S_factor[0])))

        FtNinvy = F_val.T @ Ninv_y
        alpha = jsp.linalg.cho_solve(S_factor, FtNinvy)

        # Step 4: Woodbury correction
        solution = Ninv_y - Ninv_F @ alpha

        # Step 5: Total log-determinant
        # log|N + F^T P F| = log|N| - log|P| + log|P^{-1} + F^T N^{-1} F|
        logdet = logdet_N - logdet_P + logdet_S

        return solution, logdet

    def _solve_base(self, N, y):
        """Solve with base (non-Woodbury) N."""
        if N.ndim == 1:
            # Diagonal N
            if y.ndim == 1:
                solution = y / N
            else:
                solution = y / N[:, None]
            logdet = jnp.sum(jnp.log(N))
        else:
            # Dense N
            N_factor = jsp.linalg.cho_factor(N)
            solution = jsp.linalg.cho_solve(N_factor, y)
            logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(N_factor[0])))
        return solution, logdet

    def make_solve_closure(self, y: Node):
        """
        Create an optimized closure for solving.

        Optimization based on what's constant:
        - If F and P are constant, can precompute some structure
        - If everything is constant, can solve once
        """
        # Check what's constant
        all_constant = self.F.is_constant and self.P.is_constant and y.is_constant
        if not isinstance(self.N, WoodburyStructure):
            all_constant = all_constant and self.N.is_constant

        if all_constant:
            # Everything constant - solve once
            solution, logdet = self.solve(y, {})

            def closure(params=None):
                return solution, logdet

            closure.params = []
        else:
            # General case - solve each time
            def closure(params):
                return self.solve(y, params)

            closure.params = sorted(self.params | y.params)

        return closure

    def __repr__(self):
        nested_str = f", depth={self.depth()}" if self.is_nested else ""
        param_str = ",".join(sorted(self.params)) if self.params else "const"
        return f"Woodbury[{param_str}{nested_str}]"


# ============================================================================
# Kernel Product (Log-Likelihood)
# ============================================================================


class WoodburyKernelProduct:
    """
    Computes the Gaussian log-likelihood:
        -0.5 * y^T (N + F^T P F)^{-1} y - 0.5 * log|N + F^T P F|

    This is the core quantity for Gaussian process inference.
    """

    def __init__(self, woodbury: WoodburyStructure, y: Node):
        self.woodbury = woodbury
        self.y = y

    @property
    def params(self):
        return self.woodbury.params | self.y.params

    def eval(self, params):
        """Evaluate the kernel product."""
        solution, logdet = self.woodbury.solve(self.y, params)
        y_val = self.y.eval(params)
        return -0.5 * jnp.dot(y_val, solution) - 0.5 * logdet

    def make_closure(self):
        """Create optimized closure for log-likelihood."""
        solve_closure = self.woodbury.make_solve_closure(self.y)

        if self.y.is_constant:
            y_val = self.y.eval(None)

            def closure(params):
                solution, logdet = solve_closure(params)
                return -0.5 * jnp.dot(y_val, solution) - 0.5 * logdet
        else:
            y_closure = self.y.make_closure()

            def closure(params):
                solution, logdet = solve_closure(params)
                y_val = y_closure(params)
                return -0.5 * jnp.dot(y_val, solution) - 0.5 * logdet

        closure.params = sorted(self.params)
        return closure


# ============================================================================
# Examples
# ============================================================================


def example_simple_woodbury():
    """Simple Woodbury: (N + F^T P F)^{-1} y with variable P."""
    print("=" * 70)
    print("Example 1: Simple Woodbury Solve")
    print("=" * 70)

    # Create test data
    n_data, n_basis = 100, 10
    key = jax.random.PRNGKey(0)

    N_data = jnp.ones(n_data)
    F_matrix = jax.random.normal(key, (n_data, n_basis))
    y_data = jax.random.normal(jax.random.PRNGKey(1), (n_data,))

    # Build structure
    N_node = ConstantNode(N_data)
    F_node = ConstantNode(F_matrix)
    P_node = VariableNode("amplitude", transform=lambda a: jnp.ones(n_basis) * a**2)
    y_node = ConstantNode(y_data)

    woodbury = WoodburyStructure(N_node, F_node, P_node)

    print(f"Structure: {woodbury}")
    print(f"Parameters: {woodbury.params}")
    print(f"Is nested: {woodbury.is_nested}")
    print()

    # Solve with different amplitudes
    for amp in [0.5, 1.0, 2.0]:
        params = {"amplitude": amp}
        solution, logdet = woodbury.solve(y_node, params)
        print(f"  amplitude={amp}: ||solution||={jnp.linalg.norm(solution):.4f}, logdet={logdet:.2f}")

    # Create closure (can be JIT compiled!)
    solve_closure = woodbury.make_solve_closure(y_node)
    jit_solve = jax.jit(solve_closure)

    solution, logdet = jit_solve({"amplitude": 1.5})
    print(f"\nJIT compiled: ||solution||={jnp.linalg.norm(solution):.4f}, logdet={logdet:.2f}")
    print()


def example_nested_woodbury():
    """Nested Woodbury: ((N_base + F_inner^T P_inner F_inner) + F_outer^T P_outer F_outer)^{-1} y"""
    print("=" * 70)
    print("Example 2: Nested Woodbury Solve (Key Use Case!)")
    print("=" * 70)

    # Create test data
    n_data = 100
    n_inner, n_outer = 5, 10
    key = jax.random.PRNGKey(0)

    N_base = jnp.ones(n_data) * 0.1
    F_inner = jax.random.normal(key, (n_data, n_inner))
    F_outer = jax.random.normal(jax.random.PRNGKey(1), (n_data, n_outer))
    y_data = jax.random.normal(jax.random.PRNGKey(2), (n_data,))

    # Build nested structure
    # Inner: N = N_base + F_inner^T P_inner F_inner
    N_base_node = ConstantNode(N_base)
    F_inner_node = ConstantNode(F_inner)
    P_inner_node = VariableNode("inner_amp", transform=lambda a: jnp.ones(n_inner) * a**2)

    N_woodbury = WoodburyStructure(N_base_node, F_inner_node, P_inner_node)

    # Outer: full = N_woodbury + F_outer^T P_outer F_outer
    F_outer_node = ConstantNode(F_outer)
    P_outer_node = VariableNode("outer_amp", transform=lambda a: jnp.ones(n_outer) * a**2)

    full_woodbury = WoodburyStructure(N_woodbury, F_outer_node, P_outer_node)

    print(f"Structure: {full_woodbury}")
    print(f"Is nested: {full_woodbury.is_nested}")
    print(f"Depth: {full_woodbury.depth()}")
    print(f"Parameters: {full_woodbury.params}")
    print()

    # Solve - automatically handles recursion!
    y_node = ConstantNode(y_data)
    params = {"inner_amp": 1.0, "outer_amp": 2.0}

    solution, logdet = full_woodbury.solve(y_node, params)
    print(f"Solution: ||solution||={jnp.linalg.norm(solution):.4f}")
    print(f"Log-determinant: {logdet:.2f}")
    print()
    print("Note: This solve NEVER formed the full matrix!")
    print("It recursively applied Woodbury identity.")
    print()


def example_kernel_product():
    """Computing log-likelihood using kernel product."""
    print("=" * 70)
    print("Example 3: Kernel Product (Log-Likelihood)")
    print("=" * 70)

    # Create test data
    n_data, n_basis = 100, 10
    key = jax.random.PRNGKey(0)

    N_data = jnp.ones(n_data) * 0.5
    F_matrix = jax.random.normal(key, (n_data, n_basis))
    y_data = jax.random.normal(jax.random.PRNGKey(1), (n_data,))

    # Build structure
    N_node = ConstantNode(N_data)
    F_node = ConstantNode(F_matrix)
    P_node = VariableNode("amplitude", transform=lambda a: jnp.ones(n_basis) * a**2)
    y_node = ConstantNode(y_data)

    woodbury = WoodburyStructure(N_node, F_node, P_node)
    loglike = WoodburyKernelProduct(woodbury, y_node)

    print(f"Log-likelihood parameters: {loglike.params}")
    print()

    # Evaluate for different amplitudes
    print("Log-likelihood vs amplitude:")
    for amp in [0.1, 0.5, 1.0, 2.0, 5.0]:
        ll = loglike.eval({"amplitude": amp})
        print(f"  amplitude={amp:4.1f}: loglike={ll:8.2f}")

    # Create JIT-compiled closure
    loglike_closure = loglike.make_closure()
    jit_loglike = jax.jit(loglike_closure)

    # Can now use for optimization!
    ll = jit_loglike({"amplitude": 1.5})
    print(f"\nJIT compiled: loglike={ll:.2f}")
    print()


def example_comparison_with_direct():
    """Compare Woodbury solve with direct matrix inversion (for validation)."""
    print("=" * 70)
    print("Example 4: Validation vs Direct Matrix Inversion")
    print("=" * 70)

    # Small example so we can form the full matrix
    n_data, n_basis = 50, 5
    key = jax.random.PRNGKey(42)

    N_data = jnp.ones(n_data) * 0.5
    F_matrix = jax.random.normal(key, (n_data, n_basis))
    P_data = jnp.ones(n_basis) * 2.0
    y_data = jax.random.normal(jax.random.PRNGKey(1), (n_data,))

    # Woodbury approach (efficient, no full matrix)
    N_node = ConstantNode(N_data)
    F_node = ConstantNode(F_matrix)
    P_node = ConstantNode(P_data)
    y_node = ConstantNode(y_data)

    woodbury = WoodburyStructure(N_node, F_node, P_node)
    solution_woodbury, logdet_woodbury = woodbury.solve(y_node, {})

    # Direct approach (inefficient, forms full matrix)
    N_mat = jnp.diag(N_data)
    P_mat = jnp.diag(P_data)
    full_cov = N_mat + F_matrix.T @ P_mat @ F_matrix

    full_cov_factor = jsp.linalg.cho_factor(full_cov)
    solution_direct = jsp.linalg.cho_solve(full_cov_factor, y_data)
    logdet_direct = 2.0 * jnp.sum(jnp.log(jnp.diag(full_cov_factor[0])))

    # Compare
    solution_err = jnp.linalg.norm(solution_woodbury - solution_direct)
    logdet_err = abs(logdet_woodbury - logdet_direct)

    print(f"Solution error: {solution_err:.2e} (should be ~0)")
    print(f"Logdet error:   {logdet_err:.2e} (should be ~0)")
    print()
    print("✓ Woodbury identity is exact!" if solution_err < 1e-6 else "✗ Error!")
    print()


if __name__ == "__main__":
    example_simple_woodbury()
    example_nested_woodbury()
    example_kernel_product()
    example_comparison_with_direct()

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print("This design:")
    print("  1. Focuses on SOLVING, not building matrices")
    print("  2. Uses Woodbury identity - never forms N + F^T P F")
    print("  3. Handles nested Woodbury recursively")
    print("  4. Automatically optimizes based on constant/variable")
    print("  5. Can be JIT compiled")
    print("  6. Eliminates code duplication from current matrix.py")
    print("=" * 70)
