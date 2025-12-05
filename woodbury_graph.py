"""
Corrected version with proper recursion support.

Key fix: SolveOp checks for custom solve() method BEFORE calling eval(),
and WoodburyGraph implements solve() for recursion.
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
    A^{-1} b - properly handles both matrix and WoodburyGraph inputs.

    FIXED: Checks for custom solve() method BEFORE calling eval().
    """

    def __init__(self, A, b, name=None):
        super().__init__(A, b, name=name or f"{A.name}^-1*{b.name}")
        self.A = A
        self.b = b

    def eval(self, params=None):
        """
        Override eval to handle WoodburyGraph recursion.

        Key: Check for solve() method BEFORE calling A.eval()!
        """
        # Check cache first
        if self.is_constant and self._is_evaluated:
            return self._cached_value

        # Evaluate RHS
        b_val = self.b.eval(params)

        # Check if A has custom solve method (e.g., WoodburyGraph)
        if hasattr(self.A, 'solve'):
            # Recursive case: A is WoodburyGraph
            # Call its solve() method directly - don't eval()!
            result = self.A.solve(b_val, params)
        else:
            # Base case: A is a matrix
            A_val = self.A.eval(params)
            result = self._solve_matrix(A_val, b_val)

        # Cache if constant
        if self.is_constant:
            self._cached_value = result
            self._is_evaluated = True

        return result

    def _solve_matrix(self, A_val, b_val):
        """Solve with a standard matrix."""
        if A_val.ndim == 1:
            # Diagonal matrix
            if b_val.ndim == 1:
                return b_val / A_val
            else:
                return b_val / A_val[:, None]
        else:
            # Dense matrix
            A_factor = jsp.linalg.cho_factor(A_val)
            return jsp.linalg.cho_solve(A_factor, b_val)

    def compute(self, A_val, b_val):
        """Fallback for standard eval path (not used for WoodburyGraph)."""
        return self._solve_matrix(A_val, b_val)


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
    """
    log|A| - properly handles both matrix and WoodburyGraph/WoodburyKernel inputs.

    Similar to SolveOp, checks for custom logdet() method BEFORE calling eval().
    """

    def __init__(self, A, name=None):
        super().__init__(A, name=name or f"log|{A.name}|")
        self.A = A

    def eval(self, params=None):
        """
        Override eval to handle WoodburyGraph/WoodburyKernel recursion.

        Key: Check for compute_logdet() method BEFORE calling A.eval()!
        """
        # Check cache first
        if self.is_constant and self._is_evaluated:
            return self._cached_value

        # Check if A has custom logdet method (e.g., WoodburyGraph, WoodburyKernel)
        if hasattr(self.A, 'compute_logdet'):
            # Recursive case: A is WoodburyGraph or WoodburyKernel
            # Call its compute_logdet() method directly - don't eval()!
            result = self.A.compute_logdet(params)
        else:
            # Base case: A is a matrix
            A_val = self.A.eval(params)
            result = self._compute_logdet(A_val)

        # Cache if constant
        if self.is_constant:
            self._cached_value = result
            self._is_evaluated = True

        return result

    def _compute_logdet(self, A_val):
        """Compute log-determinant of a standard matrix."""
        if A_val.ndim == 1:
            return jnp.sum(jnp.log(A_val))
        else:
            A_factor = jsp.linalg.cho_factor(A_val)
            return 2.0 * jnp.sum(jnp.log(jnp.diag(A_factor[0])))

    def compute(self, A_val):
        """Fallback for standard eval path (not used for WoodburyGraph/Kernel)."""
        return self._compute_logdet(A_val)


class ScalarOp(OpNode):
    """a * x + b"""

    def __init__(self, x, a=1.0, b=0.0, name=None):
        super().__init__(x, name=name or f"{a}*{x.name}+{b}")
        self.a = a
        self.b = b

    def compute(self, x_val):
        return self.a * x_val + self.b


# ============================================================================
# Woodbury Graph with Recursion Support
# ============================================================================


class WoodburyGraph:
    """
    Computation graph for Woodbury operations with lazy building and recursion.

    Can act as a Leaf for nesting (implements is_constant, params, solve).
    """

    def __init__(self, N: Leaf, F: Leaf, P: Leaf, y: Leaf, name: str = "Woodbury"):
        """
        Create Woodbury graph.

        Args:
            N: Noise matrix (can be another WoodburyGraph for recursion!)
            F: Design matrix
            P: Prior covariance
            y: Data vector
            name: Optional name
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

    # ========================================================================
    # Leaf-like interface (for recursion)
    # ========================================================================

    @property
    def is_constant(self):
        """Constant if all inputs are constant."""
        return self.N.is_constant and self.F.is_constant and self.P.is_constant

    @property
    def params(self):
        """All parameters needed."""
        return self.N.params | self.F.params | self.P.params

    def solve(self, b, params):
        """
        Solve (N + F^T P F)^{-1} b using Woodbury identity.

        This is the KEY METHOD for recursion!
        When SolveOp(WoodburyGraph, b) is created, it calls this method.

        Args:
            b: RHS vector or matrix (numpy/jax array)
            params: Parameter dictionary

        Returns:
            Solution to (N + F^T P F)^{-1} b
        """
        # Create a temporary leaf for b
        b_leaf = DataLeaf(b, name="b_temp")

        # Build solve graph for this specific b
        # Reuse cached nodes that don't depend on b (NmF, FtNmF, S, etc.)

        # Nmy = N^{-1} b (depends on b)
        Nmy_temp = SolveOp(self.N, b_leaf, name="Nmy_temp")

        # FtNmy = F^T N^{-1} b (depends on b via Nmy)
        FtNmy_temp = InnerProductOp(self.F, Nmy_temp, name="FtNmy_temp")

        # S = P^{-1} + F^T N^{-1} F (REUSE - doesn't depend on b!)
        S = self.S

        # SmFtNmy = S^{-1} (F^T N^{-1} b)
        SmFtNmy_temp = SolveOp(S, FtNmy_temp, name="SmFtNmy_temp")

        # NmF = N^{-1} F (REUSE - doesn't depend on b!)
        NmF = self.NmF

        # correction = N^{-1} F S^{-1} F^T N^{-1} b
        correction_temp = MatmulOp(NmF, SmFtNmy_temp, name="correction_temp")

        # solution = N^{-1} b - correction
        solution_temp = SubtractOp(Nmy_temp, correction_temp, name="solution_temp")

        # Evaluate and return
        return solution_temp.eval(params)

    def compute_logdet(self, params):
        """
        Compute log|N + F^T P F| using Woodbury determinant identity.

        This is the KEY METHOD for nested log-determinant calculations!
        When LogDetOp(WoodburyGraph) is created, it calls this method.

        Args:
            params: Parameter dictionary

        Returns:
            log|N + F^T P F| = log|N| + log|P| + log|S|
        """
        # Use the logdet property (which creates the LogDetOp node)
        return self.logdet.eval(params)

    # ========================================================================
    # Core operation nodes (lazy)
    # ========================================================================

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

    # ========================================================================
    # Nodes for kernel product (lazy)
    # ========================================================================

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
        """log|N + F^T P F| = log|N| + log|P| + log|S| where S = P^{-1} + F^T N^{-1} F"""
        return self._get_or_create('logdet',
            lambda: AddOp(
                AddOp(self.logdetN, self.logdetP, name='logdetN+logdetP'),
                self.logdetS,
                name='logdet'))

    @property
    def kernel_product(self):
        """-0.5 * ytWmy - 0.5 * logdet"""
        return self._get_or_create('kernel_product',
            lambda: SubtractOp(
                ScalarOp(self.ytWmy, a=-0.5, name='-0.5*ytWmy'),
                ScalarOp(self.logdet, a=0.5, name='0.5*logdet'),
                name='kernel_product'))

    # ========================================================================
    # Nodes for full solution (lazy - only created if needed!)
    # ========================================================================

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

    # ========================================================================
    # Closure creation
    # ========================================================================

    def make_kernelproduct_closure(self):
        """Create kernel product closure (doesn't create solution/correction)."""
        kernel_product = self.kernel_product
        self._precompute_constants()

        def closure(params):
            return kernel_product.eval(params)

        closure.params = sorted(kernel_product.params)
        closure.created_nodes = list(self._nodes.keys())

        return closure

    def make_solve_closure(self):
        """Create solve closure (does create solution/correction)."""
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


def example_nested_recursion():
    """Demonstrate recursive nesting - the key feature!"""
    print("=" * 70)
    print("Example: Nested Woodbury (RECURSION)")
    print("=" * 70)

    n_data = 100
    n_inner, n_outer = 5, 10
    key = jax.random.PRNGKey(0)

    N_base = jnp.ones(n_data) * 0.1
    F_inner = jax.random.normal(key, (n_data, n_inner))
    F_outer = jax.random.normal(jax.random.PRNGKey(1), (n_data, n_outer))
    y_data = jax.random.normal(jax.random.PRNGKey(2), (n_data,))

    # ========================================================================
    # Inner Woodbury: N_inner = N_base + F_inner^T P_inner F_inner
    # ========================================================================
    N_base_leaf = DataLeaf(N_base, name="N_base")
    F_inner_leaf = DataLeaf(F_inner, name="F_inner")
    P_inner_leaf = ParameterLeaf("inner_amp", transform=lambda a: jnp.ones(n_inner) * a**2)
    y_leaf = DataLeaf(y_data, name="y")

    inner_graph = WoodburyGraph(N_base_leaf, F_inner_leaf, P_inner_leaf, y_leaf, name="InnerWoodbury")

    print("\nInner Woodbury created:")
    print(f"  Represents: N_base + F_inner^T P_inner F_inner")
    print(f"  is_constant: {inner_graph.is_constant}")
    print(f"  params: {inner_graph.params}")

    # ========================================================================
    # Outer Woodbury: full = inner_graph + F_outer^T P_outer F_outer
    # ========================================================================
    F_outer_leaf = DataLeaf(F_outer, name="F_outer")
    P_outer_leaf = ParameterLeaf("outer_amp", transform=lambda a: jnp.ones(n_outer) * a**2)

    print("\nOuter Woodbury: Using inner_graph as N!")
    print("  outer = WoodburyGraph(inner_graph, F_outer, P_outer, y)")

    outer_graph = WoodburyGraph(inner_graph, F_outer_leaf, P_outer_leaf, y_leaf, name="OuterWoodbury")

    print(f"  Represents: (N_base + F_inner^T P_inner F_inner) + F_outer^T P_outer F_outer")
    print(f"  is_constant: {outer_graph.is_constant}")
    print(f"  params: {outer_graph.params}")

    # ========================================================================
    # Test recursion: When outer_graph computes Nmy, it calls inner_graph.solve()!
    # ========================================================================
    print("\n" + "-" * 70)
    print("Creating kernel product closure...")
    print("-" * 70)

    kernelproduct = outer_graph.make_kernelproduct_closure()

    outer_graph.print_created_nodes()

    print(f"\nClosure params: {kernelproduct.params}")

    # Evaluate
    params = {"inner_amp": 1.0, "outer_amp": 2.0}
    ll = kernelproduct(params)

    print(f"\nKernel product evaluated: {ll:.2f}")
    print("\nKey point: When computing outer Nmy = SolveOp(inner_graph, y):")
    print("  1. SolveOp.eval() detects inner_graph has solve() method")
    print("  2. Calls inner_graph.solve(y, params) directly")
    print("  3. Inner graph applies Woodbury identity recursively")
    print("  4. Never forms any full matrices!")

    # ========================================================================
    # Verify it works with different parameters
    # ========================================================================
    print("\n" + "-" * 70)
    print("Testing with different parameter values:")
    print("-" * 70)

    for inner_amp in [0.5, 1.0, 2.0]:
        for outer_amp in [0.5, 1.0, 2.0]:
            params = {"inner_amp": inner_amp, "outer_amp": outer_amp}
            ll = kernelproduct(params)
            print(f"  inner={inner_amp}, outer={outer_amp}: loglike={ll:.2f}")


def example_manual_solve():
    """Show that solve() method works correctly."""
    print("\n\n" + "=" * 70)
    print("Example: Manual solve() call")
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

    graph = WoodburyGraph(N, F, P, y)

    print("\nCalling solve() method directly:")
    print("  solution = graph.solve(y_data, {'amplitude': 1.5})")

    params = {"amplitude": 1.5}
    solution = graph.solve(y_data, params)

    print(f"\nSolution norm: {jnp.linalg.norm(solution):.4f}")

    # Compare with closure-based solve
    solve_closure = graph.make_solve_closure()
    solution2, _ = solve_closure(params)

    print(f"Closure solution norm: {jnp.linalg.norm(solution2):.4f}")
    print(f"Difference: {jnp.linalg.norm(solution - solution2):.2e}")


def example_memory_efficiency():
    """Show that only needed nodes are created."""
    print("\n\n" + "=" * 70)
    print("Example: Memory Efficiency")
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

    # Test 1: Kernel product
    graph1 = WoodburyGraph(N, F, P, y, name="Graph1_KernelProduct")
    kernelproduct = graph1.make_kernelproduct_closure()

    print("\nKernel product closure:")
    graph1.print_created_nodes()
    print(f"Total nodes: {len(kernelproduct.created_nodes)}")
    if 'solution' not in kernelproduct.created_nodes:
        print("✓ 'solution' NOT created (not needed)")
    if 'correction' not in kernelproduct.created_nodes:
        print("✓ 'correction' NOT created (not needed)")

    # Test 2: Solve
    graph2 = WoodburyGraph(N, F, P, y, name="Graph2_Solve")
    solve = graph2.make_solve_closure()

    print("\n\nSolve closure:")
    graph2.print_created_nodes()
    print(f"Total nodes: {len(solve.created_nodes)}")
    if 'solution' in solve.created_nodes:
        print("✓ 'solution' created (needed)")
    if 'correction' in solve.created_nodes:
        print("✓ 'correction' created (needed)")


if __name__ == "__main__":
    example_nested_recursion()
    example_manual_solve()
    example_memory_efficiency()

    print("\n\n" + "=" * 70)
    print("SUMMARY: CORRECTED VERSION")
    print("=" * 70)
    print("Key fixes:")
    print("  1. SolveOp.eval() checks for solve() BEFORE calling A.eval()")
    print("  2. WoodburyGraph.solve(b, params) implements Woodbury identity")
    print("  3. Recursion works: outer = WoodburyGraph(inner, F, P, y)")
    print()
    print("Features:")
    print("  ✓ Proper recursion support")
    print("  ✓ Lazy node creation (memory efficient)")
    print("  ✓ Zero code duplication")
    print("  ✓ Automatic constant detection and caching")
    print()
    print("This version is ready to implement in matrix.py!")
    print("=" * 70)
