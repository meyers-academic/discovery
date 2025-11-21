"""
Prototype implementation of matrix graph structure.

This demonstrates the core concept for refactoring matrix.py to eliminate
code duplication between constant and variable cases.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from typing import Any, Callable, Optional, Set, List


class Node:
    """Base class for all computational graph nodes."""

    def __init__(self):
        self.dependencies: List[Node] = []
        self._is_constant: Optional[bool] = None
        self._params: Optional[Set[str]] = None

    @property
    def is_constant(self) -> bool:
        """True if this node and all dependencies are constant."""
        if self._is_constant is None:
            self._is_constant = all(dep.is_constant for dep in self.dependencies)
        return self._is_constant

    @property
    def params(self) -> Set[str]:
        """Set of all parameter names needed by this node."""
        if self._params is None:
            self._params = set()
            for dep in self.dependencies:
                self._params.update(dep.params)
        return self._params

    def eval(self, params=None):
        """Evaluate the node given parameters."""
        raise NotImplementedError

    def make_closure(self) -> Callable:
        """Create optimized closure based on constant/variable structure."""
        raise NotImplementedError

    def __repr__(self):
        const_str = "const" if self.is_constant else f"var({','.join(sorted(self.params))})"
        return f"{self.__class__.__name__}[{const_str}]"


# ============================================================================
# Leaf Nodes
# ============================================================================

class ConstantNode(Node):
    """Leaf node with constant value."""

    def __init__(self, value):
        super().__init__()
        self.value = jnp.array(value) if not isinstance(value, jax.Array) else value
        self._is_constant = True
        self._params = set()

    def eval(self, params=None):
        return self.value

    def make_closure(self):
        """Constant returns itself immediately."""
        value = self.value
        def closure(params=None):
            return value
        closure.params = []
        closure.is_constant = True
        return closure


class VariableNode(Node):
    """Leaf node with parameter-dependent value."""

    def __init__(self, param_name: str, transform: Optional[Callable] = None):
        super().__init__()
        self.param_name = param_name
        self.transform = transform or (lambda x: x)
        self._is_constant = False
        self._params = {param_name}

    def eval(self, params):
        value = params[self.param_name]
        return self.transform(value)

    def make_closure(self):
        """Variable looks up parameter on each call."""
        param_name = self.param_name
        transform = self.transform
        def closure(params):
            return transform(params[param_name])
        closure.params = [param_name]
        closure.is_constant = False
        return closure


class FunctionNode(Node):
    """Leaf node with custom function of multiple parameters."""

    def __init__(self, func: Callable, param_names: List[str]):
        super().__init__()
        self.func = func
        self._params = set(param_names)
        self._is_constant = len(param_names) == 0

    def eval(self, params=None):
        return self.func(params) if params else self.func({})

    def make_closure(self):
        """Return the function itself."""
        func = self.func
        def closure(params):
            return func(params)
        closure.params = sorted(self._params)
        closure.is_constant = self.is_constant
        return closure


# ============================================================================
# Operation Nodes
# ============================================================================

class TransposeNode(Node):
    """Transpose of a matrix node."""

    def __init__(self, matrix: Node):
        super().__init__()
        self.matrix = matrix
        self.dependencies = [matrix]

    def eval(self, params=None):
        return self.matrix.eval(params).T

    def make_closure(self):
        if self.is_constant:
            value = self.eval(None)
            def closure(params=None):
                return value
            closure.params = []
            closure.is_constant = True
        else:
            mat_closure = self.matrix.make_closure()
            def closure(params):
                return mat_closure(params).T
            closure.params = sorted(self.params)
            closure.is_constant = False
        return closure


class MatmulNode(Node):
    """Matrix multiplication of two nodes."""

    def __init__(self, left: Node, right: Node):
        super().__init__()
        self.left = left
        self.right = right
        self.dependencies = [left, right]

    def eval(self, params=None):
        return self.left.eval(params) @ self.right.eval(params)

    def make_closure(self):
        if self.is_constant:
            # Both constant: precompute
            value = self.eval(None)
            def closure(params=None):
                return value
            closure.params = []
            closure.is_constant = True
        elif self.left.is_constant:
            # Left constant, right variable
            left_val = self.left.eval(None)
            right_closure = self.right.make_closure()
            def closure(params):
                return left_val @ right_closure(params)
            closure.params = sorted(self.params)
            closure.is_constant = False
        elif self.right.is_constant:
            # Left variable, right constant
            left_closure = self.left.make_closure()
            right_val = self.right.eval(None)
            def closure(params):
                return left_closure(params) @ right_val
            closure.params = sorted(self.params)
            closure.is_constant = False
        else:
            # Both variable
            left_closure = self.left.make_closure()
            right_closure = self.right.make_closure()
            def closure(params):
                return left_closure(params) @ right_closure(params)
            closure.params = sorted(self.params)
            closure.is_constant = False
        return closure


class SumNode(Node):
    """Sum of multiple matrix nodes."""

    def __init__(self, *nodes: Node):
        super().__init__()
        self.nodes = nodes
        self.dependencies = list(nodes)

    def eval(self, params=None):
        result = self.nodes[0].eval(params)
        for node in self.nodes[1:]:
            result = result + node.eval(params)
        return result

    def make_closure(self):
        if self.is_constant:
            value = self.eval(None)
            def closure(params=None):
                return value
            closure.params = []
            closure.is_constant = True
        else:
            # Separate constant and variable nodes for optimization
            const_nodes = [n for n in self.nodes if n.is_constant]
            var_nodes = [n for n in self.nodes if not n.is_constant]

            if const_nodes:
                const_sum = const_nodes[0].eval(None)
                for node in const_nodes[1:]:
                    const_sum = const_sum + node.eval(None)
            else:
                const_sum = None

            var_closures = [node.make_closure() for node in var_nodes]

            if const_sum is not None and var_closures:
                def closure(params):
                    result = const_sum
                    for cl in var_closures:
                        result = result + cl(params)
                    return result
            elif var_closures:
                def closure(params):
                    result = var_closures[0](params)
                    for cl in var_closures[1:]:
                        result = result + cl(params)
                    return result
            else:
                # Should not happen, but handle gracefully
                def closure(params=None):
                    return const_sum

            closure.params = sorted(self.params)
            closure.is_constant = False
        return closure


class DiagNode(Node):
    """Create diagonal matrix from vector or extract diagonal from matrix."""

    def __init__(self, vector: Node, extract: bool = False):
        super().__init__()
        self.vector = vector
        self.extract = extract
        self.dependencies = [vector]

    def eval(self, params=None):
        v = self.vector.eval(params)
        if self.extract:
            return jnp.diag(v) if v.ndim == 2 else v
        else:
            return jnp.diag(v) if v.ndim == 1 else v

    def make_closure(self):
        if self.is_constant:
            value = self.eval(None)
            def closure(params=None):
                return value
            closure.params = []
            closure.is_constant = True
        else:
            vec_closure = self.vector.make_closure()
            extract = self.extract
            def closure(params):
                v = vec_closure(params)
                if extract:
                    return jnp.diag(v) if v.ndim == 2 else v
                else:
                    return jnp.diag(v) if v.ndim == 1 else v
            closure.params = sorted(self.params)
            closure.is_constant = False
        return closure


# ============================================================================
# Woodbury Structure Nodes
# ============================================================================

class WoodburyNode(Node):
    """Represents the matrix N + F^T P F.

    This is the core structure for Gaussian process covariance matrices.
    N can itself be another WoodburyNode, allowing recursive nesting.

    The Woodbury identity allows efficient inversion:
    (N + F^T P F)^{-1} = N^{-1} - N^{-1}F(P^{-1} + F^T N^{-1} F)^{-1}F^T N^{-1}
    """

    def __init__(self, N: Node, F: Node, P: Node):
        """
        Args:
            N: Base noise/covariance matrix (can be WoodburyNode for nesting)
            F: Design matrix (n_data x n_basis)
            P: Prior covariance matrix (n_basis x n_basis), can be diagonal
        """
        super().__init__()
        self.N = N
        self.F = F
        self.P = P
        self.dependencies = [N, F, P]

    @property
    def is_nested(self) -> bool:
        """True if N is also a WoodburyNode."""
        return isinstance(self.N, WoodburyNode)

    def depth(self) -> int:
        """Nesting depth of Woodbury structures."""
        if isinstance(self.N, WoodburyNode):
            return 1 + self.N.depth()
        return 1

    def eval(self, params=None):
        """Evaluate full matrix: N + F^T P F"""
        N_val = self.N.eval(params)
        F_val = self.F.eval(params)
        P_val = self.P.eval(params)

        # Handle 1D (diagonal) vs 2D cases
        N_mat = jnp.diag(N_val) if N_val.ndim == 1 else N_val
        P_mat = jnp.diag(P_val) if P_val.ndim == 1 else P_val

        return N_mat + F_val.T @ P_mat @ F_val

    def make_closure(self):
        """Create closure for computing N + F^T P F."""
        if self.is_constant:
            value = self.eval(None)
            def closure(params=None):
                return value
            closure.params = []
            closure.is_constant = True
            return closure

        N_closure = self.N.make_closure()
        F_closure = self.F.make_closure()
        P_closure = self.P.make_closure()

        # Optimize based on what's constant
        n_const = self.N.is_constant
        f_const = self.F.is_constant
        p_const = self.P.is_constant

        if f_const and p_const:
            # Most common optimization: F and P fixed, only N varies
            F_val = self.F.eval(None)
            P_val = self.P.eval(None)
            P_mat = jnp.diag(P_val) if P_val.ndim == 1 else P_val
            FtPF = F_val.T @ (P_mat @ F_val)

            def closure(params):
                N_val = N_closure(params)
                N_mat = jnp.diag(N_val) if N_val.ndim == 1 else N_val
                return N_mat + FtPF

        elif n_const and p_const:
            # N and P fixed, F varies
            N_val = self.N.eval(None)
            P_val = self.P.eval(None)
            N_mat = jnp.diag(N_val) if N_val.ndim == 1 else N_val
            P_mat = jnp.diag(P_val) if P_val.ndim == 1 else P_val

            def closure(params):
                F_val = F_closure(params)
                return N_mat + F_val.T @ (P_mat @ F_val)

        elif n_const and f_const:
            # N and F fixed, P varies
            N_val = self.N.eval(None)
            F_val = self.F.eval(None)
            N_mat = jnp.diag(N_val) if N_val.ndim == 1 else N_val

            def closure(params):
                P_val = P_closure(params)
                P_mat = jnp.diag(P_val) if P_val.ndim == 1 else P_mat
                return N_mat + F_val.T @ (P_mat @ F_val)

        else:
            # General case: multiple things vary
            def closure(params):
                N_val = N_closure(params)
                F_val = F_closure(params)
                P_val = P_closure(params)
                N_mat = jnp.diag(N_val) if N_val.ndim == 1 else N_val
                P_mat = jnp.diag(P_val) if P_val.ndim == 1 else P_val
                return N_mat + F_val.T @ (P_mat @ F_val)

        closure.params = sorted(self.params)
        closure.is_constant = False
        return closure

    def __repr__(self):
        const_str = "const" if self.is_constant else f"var({','.join(sorted(self.params))})"
        depth_str = f", depth={self.depth()}" if self.is_nested else ""
        return f"Woodbury[{const_str}{depth_str}]"


# ============================================================================
# Example Usage
# ============================================================================

def example_simple():
    """Simple Woodbury example: (N + F^T P F) with constant matrices."""
    print("=" * 70)
    print("Example 1: Simple constant Woodbury structure")
    print("=" * 70)

    # Create test data
    n_data, n_basis = 100, 10

    N_data = jnp.ones(n_data)  # White noise (diagonal)
    F_matrix = jnp.array(jax.random.normal(jax.random.PRNGKey(0), (n_data, n_basis)))
    P_prior = jnp.ones(n_basis) * 2.0  # Prior variance (diagonal)

    # Build graph
    N_node = ConstantNode(N_data)
    F_node = ConstantNode(F_matrix)
    P_node = ConstantNode(P_prior)

    woodbury = WoodburyNode(N_node, F_node, P_node)

    print(f"Woodbury structure: {woodbury}")
    print(f"Is constant: {woodbury.is_constant}")
    print(f"Is nested: {woodbury.is_nested}")
    print(f"Depth: {woodbury.depth()}")
    print(f"Parameters: {woodbury.params}")

    # Evaluate
    cov_matrix = woodbury.eval()
    print(f"Covariance matrix shape: {cov_matrix.shape}")

    # Make closure (should be optimized)
    closure = woodbury.make_closure()
    print(f"Closure params: {closure.params}")
    print(f"Closure is constant: {closure.is_constant}")
    print()


def example_variable_prior():
    """Example with variable prior amplitude."""
    print("=" * 70)
    print("Example 2: Variable prior amplitude")
    print("=" * 70)

    # Create test data
    n_data, n_basis = 100, 10

    N_data = jnp.ones(n_data)
    F_matrix = jnp.array(jax.random.normal(jax.random.PRNGKey(0), (n_data, n_basis)))

    # Build graph with variable P
    N_node = ConstantNode(N_data)
    F_node = ConstantNode(F_matrix)
    P_node = VariableNode('amplitude', transform=lambda a: jnp.ones(n_basis) * a**2)

    woodbury = WoodburyNode(N_node, F_node, P_node)

    print(f"Woodbury structure: {woodbury}")
    print(f"Is constant: {woodbury.is_constant}")
    print(f"Parameters: {woodbury.params}")

    # Evaluate with different amplitudes
    for amp in [0.5, 1.0, 2.0]:
        params = {'amplitude': amp}
        cov_matrix = woodbury.eval(params)
        print(f"  amplitude={amp}: trace = {jnp.trace(cov_matrix):.2f}")

    # Make closure
    closure = woodbury.make_closure()
    print(f"Closure params: {closure.params}")

    # Use closure (can be JIT compiled!)
    jit_closure = jax.jit(closure)
    result = jit_closure({'amplitude': 1.5})
    print(f"JIT compiled result shape: {result.shape}")
    print()


def example_nested():
    """Example with nested Woodbury structure."""
    print("=" * 70)
    print("Example 3: Nested Woodbury structure")
    print("=" * 70)

    # Create test data
    n_data = 100
    n_inner, n_outer = 5, 10

    N_base = jnp.ones(n_data) * 0.1  # Base white noise
    F_inner = jnp.array(jax.random.normal(jax.random.PRNGKey(0), (n_data, n_inner)))
    F_outer = jnp.array(jax.random.normal(jax.random.PRNGKey(1), (n_data, n_outer)))

    # Build nested structure
    # Inner: N = N_base + F_inner^T P_inner F_inner
    N_base_node = ConstantNode(N_base)
    F_inner_node = ConstantNode(F_inner)
    P_inner_node = VariableNode('inner_amp',
                                transform=lambda a: jnp.ones(n_inner) * a**2)

    N_node = WoodburyNode(N_base_node, F_inner_node, P_inner_node)

    # Outer: full = N + F_outer^T P_outer F_outer
    F_outer_node = ConstantNode(F_outer)
    P_outer_node = VariableNode('outer_amp',
                                transform=lambda a: jnp.ones(n_outer) * a**2)

    woodbury_nested = WoodburyNode(N_node, F_outer_node, P_outer_node)

    print(f"Nested Woodbury structure: {woodbury_nested}")
    print(f"Is nested: {woodbury_nested.is_nested}")
    print(f"Depth: {woodbury_nested.depth()}")
    print(f"Parameters: {woodbury_nested.params}")

    # Evaluate
    params = {'inner_amp': 1.0, 'outer_amp': 2.0}
    cov_matrix = woodbury_nested.eval(params)
    print(f"Full covariance matrix shape: {cov_matrix.shape}")
    print(f"Trace: {jnp.trace(cov_matrix):.2f}")

    # The nested structure is:
    # full = (N_base + F_inner^T diag(inner_amp^2) F_inner) + F_outer^T diag(outer_amp^2) F_outer

    # Make closure (automatically optimized for constant F matrices)
    closure = woodbury_nested.make_closure()
    print(f"Closure params: {closure.params}")
    print()


def example_multiple_components():
    """Example showing how to combine multiple variable priors."""
    print("=" * 70)
    print("Example 4: Multiple variable components (Red + DM + WN)")
    print("=" * 70)

    n_data = 100
    n_rn, n_dm = 30, 20  # red noise and DM variation frequencies

    N_data = jnp.ones(n_data) * 0.01  # White noise
    F_rn = jnp.array(jax.random.normal(jax.random.PRNGKey(0), (n_data, n_rn)))
    F_dm = jnp.array(jax.random.normal(jax.random.PRNGKey(1), (n_data, n_dm)))

    # Build using nested structure
    # Start with white noise
    N_node = ConstantNode(N_data)

    # Add red noise: N + F_rn^T P_rn F_rn
    F_rn_node = ConstantNode(F_rn)
    P_rn_node = VariableNode('rn_amp', transform=lambda a: jnp.ones(n_rn) * a**2)

    N_with_rn = WoodburyNode(N_node, F_rn_node, P_rn_node)

    # Add DM variation: (N + F_rn^T P_rn F_rn) + F_dm^T P_dm F_dm
    F_dm_node = ConstantNode(F_dm)
    P_dm_node = VariableNode('dm_amp', transform=lambda a: jnp.ones(n_dm) * a**2)

    full_model = WoodburyNode(N_with_rn, F_dm_node, P_dm_node)

    print(f"Full model: {full_model}")
    print(f"Depth: {full_model.depth()}")
    print(f"Parameters: {full_model.params}")

    # Evaluate with different parameter values
    test_params = [
        {'rn_amp': 1.0, 'dm_amp': 0.5},
        {'rn_amp': 2.0, 'dm_amp': 0.5},
        {'rn_amp': 1.0, 'dm_amp': 1.0},
    ]

    for params in test_params:
        cov = full_model.eval(params)
        print(f"  {params}: trace = {jnp.trace(cov):.2f}")

    print()


if __name__ == '__main__':
    example_simple()
    example_variable_prior()
    example_nested()
    example_multiple_components()

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print("The graph structure allows:")
    print("  1. Single implementation for constant/variable cases")
    print("  2. Automatic optimization based on graph structure")
    print("  3. Natural support for nested Woodbury structures")
    print("  4. Automatic parameter tracking")
    print("  5. JIT compilation support")
    print("=" * 70)
