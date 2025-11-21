"""
Final clean design addressing recursion and memory efficiency:

1. Lazy node creation - only build what's needed
2. Recursive support - WoodburyGraph can be nested
3. Minimal memory - don't create unused nodes

This is the version to implement in the actual codebase.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from typing import Set, Optional


# ============================================================================
# Leaves (Data)
# ============================================================================


class Leaf:
    """Base class for leaf nodes (data)."""

    @property
    def is_constant(self) -> bool:
        raise NotImplementedError

    @property
    def params(self) -> Set[str]:
        raise NotImplementedError

    def eval(self, params=None):
        raise NotImplementedError


class DataLeaf(Leaf):
    """Constant data."""

    def __init__(self, value, name=None):
        self.value = jnp.array(value) if not isinstance(value, jax.Array) else value
        self.name = name or "data"

    @property
    def is_constant(self):
        return True

    @property
    def params(self):
        return set()

    def eval(self, params=None):
        return self.value

    def __repr__(self):
        return f"{self.name}[const]"


class ParameterLeaf(Leaf):
    """Parameter-dependent data."""

    def __init__(self, param_name: str, transform=None):
        self.param_name = param_name
        self.transform = transform or (lambda x: x)
        self.name = param_name

    @property
    def is_constant(self):
        return False

    @property
    def params(self):
        return {self.param_name}

    def eval(self, params):
        return self.transform(params[self.param_name])

    def __repr__(self):
        return f"{self.name}[var]"


# ============================================================================
# Operation Nodes
# ============================================================================


class OpNode:
    """Base class for operation nodes with automatic caching."""

    def __init__(self, *inputs, name=None):
        self.inputs = inputs
        self.name = name or self.__class__.__name__
        self._cached_value = None
        self._is_evaluated = False

    @property
    def is_constant(self) -> bool:
        """Constant if all inputs are constant."""
        return all(inp.is_constant for inp in self.inputs)

    @property
    def params(self) -> Set[str]:
        """All parameters needed."""
        params = set()
        for inp in self.inputs:
            params.update(inp.params)
        return params

    def compute(self, *input_values):
        """Compute operation. Override in subclasses."""
        raise NotImplementedError

    def eval(self, params=None):
        """Evaluate, using cache if constant and already computed."""
        if self.is_constant and self._is_evaluated:
            return self._cached_value

        input_values = [inp.eval(params) for inp in self.inputs]
        result = self.compute(*input_values)

        if self.is_constant:
            self._cached_value = result
            self._is_evaluated = True

        return result

    def precompute(self):
        """Force precomputation if constant."""
        if self.is_constant and not self._is_evaluated:
            self.eval(None)

    def __repr__(self):
        const_str = "const" if self.is_constant else f"var"
        return f"{self.name}[{const_str}]"


class SolveOp(OpNode):
    """
    A^{-1} b - supports both matrix and WoodburyGraph inputs for A.
    """

    def __init__(self, A, b, name=None):
        super().__init__(A, b, name=name or f"{A.name}^-1*{b.name}")
        self.A = A
        self.b = b

    def compute(self, A_val, b_val):
        """Solve A x = b."""
        # Check if A_val is actually a special object with solve method
        # (This happens when A is a WoodburyGraph acting as a leaf)
        if hasattr(self.A, 'solve'):
            # Recursive case: A is a WoodburyGraph
            # We need params here, so we need to restructure this...
            # Actually, this won't work in compute(). Need different approach.
            raise NotImplementedError("Use WoodburyGraph.solve() directly for nested case")

        # Base case: A is a matrix
        if A_val.ndim == 1:
            if b_val.ndim == 1:
                return b_val / A_val
            else:
                return b_val / A_val[:, None]
        else:
            A_factor = jsp.linalg.cho_factor(A_val)
            return jsp.linalg.cho_solve(A_factor, b_val)


class InnerProductOp(OpNode):
    """A^T B"""

    def __init__(self, A, B, name=None):
        super().__init__(A, B, name=name or f"{A.name}^T*{B.name}")

    def compute(self, A_val, B_val):
        return A_val.T @ B_val


class DotProductOp(OpNode):
    """a^T b (scalar)"""

    def __init__(self, a, b, name=None):
        super().__init__(a, b, name=name or f"{a.name}^T*{b.name}")

    def compute(self, a_val, b_val):
        return jnp.dot(a_val, b_val)


class InvertOp(OpNode):
    """A^{-1}"""

    def __init__(self, A, name=None):
        super().__init__(A, name=name or f"{A.name}^-1")

    def compute(self, A_val):
        if A_val.ndim == 1:
            return 1.0 / A_val
        else:
            A_factor = jsp.linalg.cho_factor(A_val)
            return jsp.linalg.cho_solve(A_factor, jnp.eye(A_val.shape[0]))


class AddOp(OpNode):
    """A + B"""

    def __init__(self, A, B, name=None):
        super().__init__(A, B, name=name or f"{A.name}+{B.name}")

    def compute(self, A_val, B_val):
        if A_val.ndim == 1 and B_val.ndim == 2:
            return jnp.diag(A_val) + B_val
        elif A_val.ndim == 2 and B_val.ndim == 1:
            return A_val + jnp.diag(B_val)
        else:
            return A_val + B_val


class SubtractOp(OpNode):
    """A - B"""

    def __init__(self, A, B, name=None):
        super().__init__(A, B, name=name or f"{A.name}-{B.name}")

    def compute(self, A_val, B_val):
        return A_val - B_val


class MatmulOp(OpNode):
    """A @ B"""

    def __init__(self, A, B, name=None):
        super().__init__(A, B, name=name or f"{A.name}@{B.name}")

    def compute(self, A_val, B_val):
        return A_val @ B_val


class LogDetOp(OpNode):
    """log|A|"""

    def __init__(self, A, name=None):
        super().__init__(A, name=name or f"log|{A.name}|")

    def compute(self, A_val):
        if A_val.ndim == 1:
            return jnp.sum(jnp.log(A_val))
        else:
            A_factor = jsp.linalg.cho_factor(A_val)
            return 2.0 * jnp.sum(jnp.log(jnp.diag(A_factor[0])))


class ScalarOp(OpNode):
    """a * x + b"""

    def __init__(self, x, a=1.0, b=0.0, name=None):
        super().__init__(x, name=name or f"{a}*{x.name}+{b}")
        self.a = a
        self.b = b

    def compute(self, x_val):
        return self.a * x_val + self.b


# ============================================================================
# Woodbury Graph with Lazy Building
# ============================================================================


class WoodburyGraph:
    """
    Computation graph for Woodbury operations with lazy node creation.

    Key features:
    - Nodes created only when accessed (via @property)
    - Can act as a Leaf for recursive nesting
    - Minimal memory footprint
    """

    def __init__(self, N: Leaf, F: Leaf, P: Leaf, y: Leaf, name: str = "Woodbury"):
        """
        Create Woodbury graph.

        Args:
            N: Noise matrix (can be another WoodburyGraph for recursion!)
            F: Design matrix
            P: Prior covariance
            y: Data vector
            name: Optional name for this graph
        """
        self.N = N
        self.F = F
        self.P = P
        self.y = y
        self.name = name

        # Cache for lazily created nodes
        self._nodes = {}

    def _get_or_create(self, key, factory):
        """Get node from cache or create it."""
        if key not in self._nodes:
            self._nodes[key] = factory()
        return self._nodes[key]

    # Leaf-like interface (for recursion)
    @property
    def is_constant(self):
        """Constant if all inputs are constant."""
        return self.N.is_constant and self.F.is_constant and self.P.is_constant

    @property
    def params(self):
        """All parameters needed."""
        return self.N.params | self.F.params | self.P.params

    # Core operations (lazy)
    @property
    def Nmy(self):
        return self._get_or_create('Nmy',
            lambda: SolveOp(self.N, self.y, name='Nmy'))

    @property
    def NmF(self):
        return self._get_or_create('NmF',
            lambda: SolveOp(self.N, self.F, name='NmF'))

    @property
    def FtNmy(self):
        return self._get_or_create('FtNmy',
            lambda: InnerProductOp(self.F, self.Nmy, name='FtNmy'))

    @property
    def FtNmF(self):
        return self._get_or_create('FtNmF',
            lambda: InnerProductOp(self.F, self.NmF, name='FtNmF'))

    @property
    def ytNmy(self):
        return self._get_or_create('ytNmy',
            lambda: DotProductOp(self.y, self.Nmy, name='ytNmy'))

    @property
    def Pinv(self):
        return self._get_or_create('Pinv',
            lambda: InvertOp(self.P, name='Pinv'))

    @property
    def S(self):
        """Schur complement: P^{-1} + F^T N^{-1} F"""
        return self._get_or_create('S',
            lambda: AddOp(self.Pinv, self.FtNmF, name='S'))

    @property
    def SmFtNmy(self):
        return self._get_or_create('SmFtNmy',
            lambda: SolveOp(self.S, self.FtNmy, name='SmFtNmy'))

    @property
    def logdetN(self):
        return self._get_or_create('logdetN',
            lambda: LogDetOp(self.N, name='logdetN'))

    @property
    def logdetP(self):
        return self._get_or_create('logdetP',
            lambda: LogDetOp(self.P, name='logdetP'))

    @property
    def logdetS(self):
        return self._get_or_create('logdetS',
            lambda: LogDetOp(self.S, name='logdetS'))

    # Nodes for kernel product (lazy)
    @property
    def quad_correction(self):
        """FtNmy^T SmFtNmy"""
        return self._get_or_create('quad_correction',
            lambda: DotProductOp(self.FtNmy, self.SmFtNmy, name='quad_correction'))

    @property
    def ytWmy(self):
        """y^T (N + F^T P F)^{-1} y = ytNmy - FtNmy^T SmFtNmy"""
        return self._get_or_create('ytWmy',
            lambda: SubtractOp(self.ytNmy, self.quad_correction, name='ytWmy'))

    @property
    def logdet(self):
        """log|N + F^T P F| = log|N| - log|P| + log|S|"""
        return self._get_or_create('logdet',
            lambda: SubtractOp(
                AddOp(self.logdetN, self.logdetS, name='logdetN+logdetS'),
                self.logdetP,
                name='logdet'))

    @property
    def kernel_product(self):
        """-0.5 * ytWmy - 0.5 * logdet"""
        return self._get_or_create('kernel_product',
            lambda: SubtractOp(
                ScalarOp(self.ytWmy, a=-0.5, name='-0.5*ytWmy'),
                ScalarOp(self.logdet, a=0.5, name='0.5*logdet'),
                name='kernel_product'))

    # Nodes for solve (lazy - only created if needed!)
    @property
    def correction(self):
        """NmF @ SmFtNmy"""
        return self._get_or_create('correction',
            lambda: MatmulOp(self.NmF, self.SmFtNmy, name='correction'))

    @property
    def solution(self):
        """(N + F^T P F)^{-1} y = Nmy - correction"""
        return self._get_or_create('solution',
            lambda: SubtractOp(self.Nmy, self.correction, name='solution'))

    def make_kernelproduct_closure(self):
        """
        Create kernel product closure.

        Only creates nodes needed for kernel product.
        Does NOT create: solution, correction (saves memory!)
        """
        # Access triggers lazy creation of needed nodes only
        kernel_product = self.kernel_product

        # Precompute constants
        self._precompute_constants()

        def closure(params):
            return kernel_product.eval(params)

        closure.params = sorted(kernel_product.params)
        closure.created_nodes = list(self._nodes.keys())

        return closure

    def make_solve_closure(self):
        """
        Create solve closure.

        This DOES create solution and correction nodes.
        """
        solution = self.solution
        logdet = self.logdet

        self._precompute_constants()

        def closure(params):
            return solution.eval(params), logdet.eval(params)

        closure.params = sorted(solution.params | logdet.params)
        closure.created_nodes = list(self._nodes.keys())

        return closure

    def _precompute_constants(self):
        """Precompute all constant nodes that exist."""
        for node in self._nodes.values():
            if hasattr(node, 'is_constant') and node.is_constant:
                if hasattr(node, 'precompute'):
                    node.precompute()

    def print_created_nodes(self):
        """Print which nodes have been created."""
        print(f"\n{self.name} - Created nodes:")
        for name, node in self._nodes.items():
            print(f"  {node}")


# ============================================================================
# Examples
# ============================================================================


def example_lazy_building():
    """Show that only needed nodes are created."""
    print("=" * 70)
    print("Example: Lazy Node Creation")
    print("=" * 70)

    # Data
    n_data, n_basis = 100, 10
    key = jax.random.PRNGKey(0)

    N_data = jnp.ones(n_data) * 0.5
    F_matrix = jax.random.normal(key, (n_data, n_basis))
    y_data = jax.random.normal(jax.random.PRNGKey(1), (n_data,))

    # Create leaves
    N = DataLeaf(N_data, name="N")
    F = DataLeaf(F_matrix, name="F")
    P = ParameterLeaf("amplitude", transform=lambda a: jnp.ones(n_basis) * a**2)
    y = DataLeaf(y_data, name="y")

    # Create graph
    graph = WoodburyGraph(N, F, P, y, name="WoodburyGraph")

    print("\nAfter creating graph (no nodes built yet):")
    graph.print_created_nodes()

    # Create kernel product closure
    print("\n" + "-" * 70)
    print("Creating kernel product closure...")
    kernelproduct = graph.make_kernelproduct_closure()

    graph.print_created_nodes()

    print(f"\nClosure parameters: {kernelproduct.params}")
    print(f"Total nodes created: {len(kernelproduct.created_nodes)}")
    print("\nNote: 'solution' and 'correction' were NOT created!")
    print("      They're not needed for kernel product.")

    # Use it
    ll = kernelproduct({"amplitude": 1.5})
    print(f"\nKernel product: {ll:.2f}")


def example_solve_closure():
    """Show that solution/correction ARE created when needed."""
    print("\n\n" + "=" * 70)
    print("Example: Solve Closure (creates more nodes)")
    print("=" * 70)

    n_data, n_basis = 100, 10
    key = jax.random.PRNGKey(0)

    N_data = jnp.ones(n_data) * 0.5
    F_matrix = jax.random.normal(key, (n_data, n_basis))
    y_data = jax.random.normal(jax.random.PRNGKey(1), (n_data,))

    N = DataLeaf(N_data, name="N")
    F = DataLeaf(F_matrix, name="F")
    P = ParameterLeaf("amplitude", transform=lambda a: jnp.ones(n_basis) * a**2)
    y = DataLeaf(y_data, name="y")

    graph = WoodburyGraph(N, F, P, y, name="WoodburyGraph")

    print("\nCreating solve closure...")
    solve = graph.make_solve_closure()

    graph.print_created_nodes()

    print(f"\nTotal nodes created: {len(solve.created_nodes)}")
    print("\nNote: NOW 'solution' and 'correction' were created!")
    print("      They're needed for solve.")

    solution, logdet = solve({"amplitude": 1.5})
    print(f"\nSolution norm: {jnp.linalg.norm(solution):.4f}")
    print(f"Log determinant: {logdet:.2f}")


def example_nested_woodbury():
    """Show recursive nesting (N is another WoodburyGraph)."""
    print("\n\n" + "=" * 70)
    print("Example: Nested Woodbury (Recursion)")
    print("=" * 70)

    n_data = 100
    n_inner, n_outer = 5, 10
    key = jax.random.PRNGKey(0)

    N_base = jnp.ones(n_data) * 0.1
    F_inner = jax.random.normal(key, (n_data, n_inner))
    F_outer = jax.random.normal(jax.random.PRNGKey(1), (n_data, n_outer))
    y_data = jax.random.normal(jax.random.PRNGKey(2), (n_data,))

    # Inner Woodbury: N_inner = N_base + F_inner^T P_inner F_inner
    N_base_leaf = DataLeaf(N_base, name="N_base")
    F_inner_leaf = DataLeaf(F_inner, name="F_inner")
    P_inner_leaf = ParameterLeaf("inner_amp", transform=lambda a: jnp.ones(n_inner) * a**2)
    y_leaf = DataLeaf(y_data, name="y")

    inner_graph = WoodburyGraph(N_base_leaf, F_inner_leaf, P_inner_leaf, y_leaf, name="InnerWoodbury")

    print("Created inner Woodbury structure")
    print(f"  Inner graph is_constant: {inner_graph.is_constant}")
    print(f"  Inner graph params: {inner_graph.params}")

    # Outer Woodbury: full = inner_graph + F_outer^T P_outer F_outer
    # NOTE: inner_graph acts as the N leaf!
    F_outer_leaf = DataLeaf(F_outer, name="F_outer")
    P_outer_leaf = ParameterLeaf("outer_amp", transform=lambda a: jnp.ones(n_outer) * a**2)

    print("\nCreating outer Woodbury with inner_graph as N:")
    print("  outer_graph = WoodburyGraph(inner_graph, F_outer, P_outer, y)")

    # This demonstrates the concept, though SolveOp would need modification for full recursion
    # In practice, you'd implement a solve() method on WoodburyGraph
    outer_graph = WoodburyGraph(inner_graph, F_outer_leaf, P_outer_leaf, y_leaf, name="OuterWoodbury")

    print(f"\n  Outer graph is_constant: {outer_graph.is_constant}")
    print(f"  Outer graph params: {outer_graph.params}")
    print("\nNote: For full recursion, WoodburyGraph would need a solve() method")
    print("      that SolveOp can call instead of treating it as a matrix.")


if __name__ == "__main__":
    example_lazy_building()
    example_solve_closure()
    example_nested_woodbury()

    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("1. LAZY BUILDING")
    print("   - Nodes created only when accessed via @property")
    print("   - kernel_product closure: ~15 nodes")
    print("   - Does NOT create 'solution' or 'correction'")
    print("   - Saves memory!")
    print()
    print("2. MINIMAL FOOTPRINT")
    print("   - Only hold references to nodes in computation path")
    print("   - Unused nodes can be garbage collected")
    print("   - Critical for GPU memory")
    print()
    print("3. RECURSION READY")
    print("   - WoodburyGraph implements Leaf-like interface")
    print("   - Can nest: outer = WoodburyGraph(inner, F, P, y)")
    print("   - Need to add solve() method for full recursion")
    print()
    print("This is the design to implement in matrix.py!")
    print("=" * 70)
