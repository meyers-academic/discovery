"""
Clean graph design: Leaves are data (N, F, P, y), nodes are operations (FtNmF, etc.)

Key insight: Each operation knows how to compute itself and whether it can be precomputed.
No code duplication - the graph structure handles constant/variable optimization automatically.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from typing import Set, List


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
    """
    Base class for operation nodes.

    An operation node:
    - Has input nodes (dependencies)
    - Knows how to compute its value
    - Automatically detects if it's constant
    - Can precompute if constant
    """

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
        """All parameters needed by this operation."""
        params = set()
        for inp in self.inputs:
            params.update(inp.params)
        return params

    def compute(self, *input_values):
        """Compute the operation given input values. Override in subclasses."""
        raise NotImplementedError

    def eval(self, params=None):
        """
        Evaluate this operation.

        If constant and already evaluated, return cached value.
        Otherwise compute from inputs.
        """
        if self.is_constant and self._is_evaluated:
            return self._cached_value

        # Evaluate inputs
        input_values = [inp.eval(params) for inp in self.inputs]

        # Compute this operation
        result = self.compute(*input_values)

        # Cache if constant
        if self.is_constant:
            self._cached_value = result
            self._is_evaluated = True

        return result

    def precompute(self):
        """Force precomputation if constant."""
        if self.is_constant and not self._is_evaluated:
            self.eval(None)

    def __repr__(self):
        const_str = "const" if self.is_constant else f"var({','.join(sorted(self.params))})"
        return f"{self.name}[{const_str}]"


# ============================================================================
# Specific Operations
# ============================================================================


class SolveOp(OpNode):
    """
    A^{-1} b operation.

    Inputs: A (matrix), b (vector or matrix)
    Output: Solution to A x = b
    """

    def __init__(self, A, b, name=None):
        super().__init__(A, b, name=name or f"{A.name}^-1*{b.name}")
        self.A = A
        self.b = b

    def compute(self, A_val, b_val):
        """Solve A x = b."""
        if A_val.ndim == 1:
            # Diagonal A
            if b_val.ndim == 1:
                return b_val / A_val
            else:
                return b_val / A_val[:, None]
        else:
            # Dense A
            A_factor = jsp.linalg.cho_factor(A_val)
            return jsp.linalg.cho_solve(A_factor, b_val)


class InnerProductOp(OpNode):
    """
    A^T B operation (matrix transpose times matrix/vector).

    Inputs: A (matrix), B (vector or matrix)
    Output: A^T B
    """

    def __init__(self, A, B, name=None):
        super().__init__(A, B, name=name or f"{A.name}^T*{B.name}")
        self.A = A
        self.B = B

    def compute(self, A_val, B_val):
        """Compute A^T B."""
        return A_val.T @ B_val


class DotProductOp(OpNode):
    """
    a^T b operation (vector dot product / quadratic form).

    Inputs: a (vector), b (vector)
    Output: a^T b (scalar)
    """

    def __init__(self, a, b, name=None):
        super().__init__(a, b, name=name or f"{a.name}^T*{b.name}")
        self.a = a
        self.b = b

    def compute(self, a_val, b_val):
        """Compute a^T b."""
        return jnp.dot(a_val, b_val)


class InvertOp(OpNode):
    """
    A^{-1} operation (matrix inverse or diagonal reciprocal).

    Inputs: A (matrix or vector)
    Output: A^{-1}
    """

    def __init__(self, A, name=None):
        super().__init__(A, name=name or f"{A.name}^-1")
        self.A = A

    def compute(self, A_val):
        """Compute A^{-1}."""
        if A_val.ndim == 1:
            # Diagonal - just reciprocal
            return 1.0 / A_val
        else:
            # Dense - use Cholesky
            A_factor = jsp.linalg.cho_factor(A_val)
            return jsp.linalg.cho_solve(A_factor, jnp.eye(A_val.shape[0]))


class AddOp(OpNode):
    """
    A + B operation.

    Inputs: A, B (matrices or vectors)
    Output: A + B
    """

    def __init__(self, A, B, name=None):
        super().__init__(A, B, name=name or f"{A.name}+{B.name}")
        self.A = A
        self.B = B

    def compute(self, A_val, B_val):
        """Compute A + B."""
        # Handle diagonal + dense cases
        if A_val.ndim == 1 and B_val.ndim == 2:
            return jnp.diag(A_val) + B_val
        elif A_val.ndim == 2 and B_val.ndim == 1:
            return A_val + jnp.diag(B_val)
        else:
            return A_val + B_val


class SubtractOp(OpNode):
    """
    A - B operation.

    Inputs: A, B
    Output: A - B
    """

    def __init__(self, A, B, name=None):
        super().__init__(A, B, name=name or f"{A.name}-{B.name}")
        self.A = A
        self.B = B

    def compute(self, A_val, B_val):
        """Compute A - B."""
        return A_val - B_val


class MatmulOp(OpNode):
    """
    A @ B operation (matrix multiply).

    Inputs: A, B
    Output: A @ B
    """

    def __init__(self, A, B, name=None):
        super().__init__(A, B, name=name or f"{A.name}@{B.name}")
        self.A = A
        self.B = B

    def compute(self, A_val, B_val):
        """Compute A @ B."""
        return A_val @ B_val


class LogDetOp(OpNode):
    """
    log|A| operation (log determinant).

    Inputs: A (matrix or diagonal vector)
    Output: log|A| (scalar)
    """

    def __init__(self, A, name=None):
        super().__init__(A, name=name or f"log|{A.name}|")
        self.A = A

    def compute(self, A_val):
        """Compute log|A|."""
        if A_val.ndim == 1:
            # Diagonal
            return jnp.sum(jnp.log(A_val))
        else:
            # Dense
            A_factor = jsp.linalg.cho_factor(A_val)
            return 2.0 * jnp.sum(jnp.log(jnp.diag(A_factor[0])))


class ScalarOp(OpNode):
    """
    Scalar operation: a * x + b.

    Inputs: x
    Output: a * x + b
    """

    def __init__(self, x, a=1.0, b=0.0, name=None):
        super().__init__(x, name=name or f"{a}*{x.name}+{b}")
        self.x = x
        self.a = a
        self.b = b

    def compute(self, x_val):
        """Compute a * x + b."""
        return self.a * x_val + self.b


# ============================================================================
# Woodbury Graph Builder
# ============================================================================


class WoodburyGraph:
    """
    Build computation graph for Woodbury solve and kernel product.

    Given leaves N, F, P, y, constructs all intermediate operation nodes:
    - Nmy, NmF (solve ops)
    - FtNmy, FtNmF (inner products)
    - Pinv, S (Schur complement)
    - etc.

    Each operation node automatically handles constant/variable optimization.
    """

    def __init__(self, N: Leaf, F: Leaf, P: Leaf, y: Leaf):
        """
        Build Woodbury computation graph.

        Args:
            N: Noise matrix (diagonal or dense)
            F: Design matrix
            P: Prior covariance (diagonal or dense)
            y: Data vector
        """
        self.N = N
        self.F = F
        self.P = P
        self.y = y

        # Build all operation nodes
        self._build_graph()

    def _build_graph(self):
        """Construct all intermediate operation nodes."""
        # Level 1: Solve operations
        self.Nmy = SolveOp(self.N, self.y, name="Nmy")
        self.NmF = SolveOp(self.N, self.F, name="NmF")

        # Level 2: Inner products
        self.FtNmy = InnerProductOp(self.F, self.Nmy, name="FtNmy")
        self.FtNmF = InnerProductOp(self.F, self.NmF, name="FtNmF")

        # Level 2b: Quadratic form (for kernel product)
        self.ytNmy = DotProductOp(self.y, self.Nmy, name="ytNmy")

        # Level 3: Schur complement components
        self.Pinv = InvertOp(self.P, name="Pinv")
        self.S = AddOp(self.Pinv, self.FtNmF, name="S")

        # Level 4: Schur solve
        self.SmFtNmy = SolveOp(self.S, self.FtNmy, name="SmFtNmy")

        # Level 5: Woodbury correction
        self.correction = MatmulOp(self.NmF, self.SmFtNmy, name="correction")
        self.solution = SubtractOp(self.Nmy, self.correction, name="solution")

        # Log determinants
        self.logdetN = LogDetOp(self.N, name="logdetN")
        self.logdetP = LogDetOp(self.P, name="logdetP")
        self.logdetS = LogDetOp(self.S, name="logdetS")

        # Total log determinant: log|N + F^T P F| = log|N| - log|P| + log|S|
        self.logdet = SubtractOp(
            AddOp(self.logdetN, self.logdetS, name="logdetN+logdetS"),
            self.logdetP,
            name="logdet",
        )

        # Kernel product: -0.5 * (ytNmy - FtNmy^T SmFtNmy) - 0.5 * logdet
        self.quad_correction = DotProductOp(self.FtNmy, self.SmFtNmy, name="FtNmy^T*SmFtNmy")
        self.ytWmy = SubtractOp(self.ytNmy, self.quad_correction, name="ytWmy")

        self.kernel_product = SubtractOp(
            ScalarOp(self.ytWmy, a=-0.5, name="-0.5*ytWmy"),
            ScalarOp(self.logdet, a=0.5, name="0.5*logdet"),
            name="kernel_product",
        )

    def precompute_constants(self):
        """Force precomputation of all constant nodes."""
        all_nodes = self._get_all_nodes()
        for node in all_nodes:
            if isinstance(node, OpNode) and node.is_constant:
                node.precompute()

    def _get_all_nodes(self) -> List[OpNode]:
        """Get all operation nodes in the graph."""
        return [
            self.Nmy,
            self.NmF,
            self.FtNmy,
            self.FtNmF,
            self.ytNmy,
            self.Pinv,
            self.S,
            self.SmFtNmy,
            self.correction,
            self.solution,
            self.logdetN,
            self.logdetP,
            self.logdetS,
            self.logdet,
            self.quad_correction,
            self.ytWmy,
            self.kernel_product,
        ]

    def make_kernelproduct_closure(self):
        """
        Create optimized closure for kernel product.

        Returns a function that computes -0.5 y^T (N + F^T P F)^{-1} y - 0.5 log|N + F^T P F|.

        The closure automatically uses precomputed values for constant nodes.
        """
        # Precompute all constant nodes
        self.precompute_constants()

        # The kernel_product node handles everything
        kernel_product = self.kernel_product

        def closure(params):
            return kernel_product.eval(params)

        closure.params = sorted(kernel_product.params)

        # Report what was precomputed
        const_nodes = [node for node in self._get_all_nodes() if node.is_constant]
        closure.precomputed = [node.name for node in const_nodes]

        return closure

    def make_solve_closure(self):
        """
        Create optimized closure for solve operation.

        Returns a function that computes (N + F^T P F)^{-1} y and log|N + F^T P F|.
        """
        # Precompute all constant nodes
        self.precompute_constants()

        solution = self.solution
        logdet = self.logdet

        def closure(params):
            return solution.eval(params), logdet.eval(params)

        closure.params = sorted(solution.params | logdet.params)

        const_nodes = [node for node in self._get_all_nodes() if node.is_constant]
        closure.precomputed = [node.name for node in const_nodes]

        return closure

    def print_graph(self):
        """Print the computation graph."""
        print("\nWoodbury Computation Graph:")
        print("=" * 70)
        print("\nLeaves (data):")
        print(f"  N: {self.N}")
        print(f"  F: {self.F}")
        print(f"  P: {self.P}")
        print(f"  y: {self.y}")

        print("\nOperation nodes:")
        for node in self._get_all_nodes():
            print(f"  {node}")

        const_nodes = [node for node in self._get_all_nodes() if node.is_constant]
        var_nodes = [node for node in self._get_all_nodes() if not node.is_constant]

        print(f"\nSummary:")
        print(f"  Constant nodes (can precompute): {len(const_nodes)}")
        print(f"  Variable nodes (compute in closure): {len(var_nodes)}")


# ============================================================================
# Example
# ============================================================================


def example_clean_graph():
    """Demonstrate the clean graph design."""
    print("=" * 70)
    print("Clean Graph Design: Leaves are data, nodes are operations")
    print("=" * 70)

    # Create test data
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

    # Build graph
    print("\nBuilding graph...")
    graph = WoodburyGraph(N, F, P, y)

    # Print graph structure
    graph.print_graph()

    # Create kernel product closure
    print("\n" + "=" * 70)
    print("Creating kernel product closure...")
    print("=" * 70)

    kernelproduct = graph.make_kernelproduct_closure()

    print(f"\nClosure parameters: {kernelproduct.params}")
    print(f"Precomputed nodes ({len(kernelproduct.precomputed)}):")
    for name in kernelproduct.precomputed:
        print(f"  ✓ {name}")

    # Use the closure
    print("\n" + "=" * 70)
    print("Evaluating kernel product:")
    print("=" * 70)
    for amp in [0.5, 1.0, 2.0]:
        ll = kernelproduct({"amplitude": amp})
        print(f"  amplitude={amp}: loglike={ll:.2f}")

    # JIT compile
    print("\nJIT compiling...")
    jit_kernelproduct = jax.jit(kernelproduct)
    ll = jit_kernelproduct({"amplitude": 1.5})
    print(f"JIT result: loglike={ll:.2f}")


def example_variable_cases():
    """Show how the graph automatically handles different variable patterns."""
    print("\n\n" + "=" * 70)
    print("Automatic Handling of Different Variable Patterns")
    print("=" * 70)

    n_data, n_basis = 100, 10
    key = jax.random.PRNGKey(0)

    N_data = jnp.ones(n_data) * 0.5
    F_matrix = jax.random.normal(key, (n_data, n_basis))
    y_data = jax.random.normal(jax.random.PRNGKey(1), (n_data,))

    # Case 1: N, F, y const; P var (most common)
    print("\nCase 1: N, F, y constant; P variable")
    print("-" * 70)
    N1 = DataLeaf(N_data, name="N")
    F1 = DataLeaf(F_matrix, name="F")
    P1 = ParameterLeaf("amplitude", transform=lambda a: jnp.ones(n_basis) * a**2)
    y1 = DataLeaf(y_data, name="y")

    graph1 = WoodburyGraph(N1, F1, P1, y1)
    closure1 = graph1.make_kernelproduct_closure()
    print(f"Parameters: {closure1.params}")
    print(f"Precomputed: {len(closure1.precomputed)} nodes")

    # Case 2: N, F const; P, y var
    print("\nCase 2: N, F constant; P, y variable")
    print("-" * 70)
    N2 = DataLeaf(N_data, name="N")
    F2 = DataLeaf(F_matrix, name="F")
    P2 = ParameterLeaf("amplitude", transform=lambda a: jnp.ones(n_basis) * a**2)
    y2 = ParameterLeaf("y_data")

    graph2 = WoodburyGraph(N2, F2, P2, y2)
    closure2 = graph2.make_kernelproduct_closure()
    print(f"Parameters: {closure2.params}")
    print(f"Precomputed: {len(closure2.precomputed)} nodes")

    # Case 3: Only N const
    print("\nCase 3: N constant; F, P, y variable")
    print("-" * 70)
    N3 = DataLeaf(N_data, name="N")
    F3 = ParameterLeaf("F_matrix")
    P3 = ParameterLeaf("amplitude", transform=lambda a: jnp.ones(n_basis) * a**2)
    y3 = ParameterLeaf("y_data")

    graph3 = WoodburyGraph(N3, F3, P3, y3)
    closure3 = graph3.make_kernelproduct_closure()
    print(f"Parameters: {closure3.params}")
    print(f"Precomputed: {len(closure3.precomputed)} nodes")

    print("\n" + "=" * 70)
    print("Note: The SAME graph building code handles all cases!")
    print("No separate classes for varP, varN, varFP, etc.")
    print("=" * 70)


if __name__ == "__main__":
    example_clean_graph()
    example_variable_cases()

    print("\n\n" + "=" * 70)
    print("KEY BENEFITS")
    print("=" * 70)
    print("1. NO CODE DUPLICATION")
    print("   - Single implementation for all constant/variable combinations")
    print("   - No separate WoodburyKernel_varP, _varN, _varFP, etc. classes")
    print()
    print("2. AUTOMATIC OPTIMIZATION")
    print("   - Constant detection is automatic (node.is_constant)")
    print("   - Precomputation is automatic (node.precompute())")
    print("   - Caching is automatic (checked in node.eval())")
    print()
    print("3. COMPOSABLE")
    print("   - Easy to add new operations (just define new OpNode subclass)")
    print("   - Easy to nest (N can be another WoodburyGraph's output)")
    print()
    print("4. CORRECT ABSTRACTION")
    print("   - Leaves: N, F, P, y (the data)")
    print("   - Nodes: Nmy, FtNmF, etc. (the operations)")
    print("   - Graph handles optimization automatically")
    print("=" * 70)
