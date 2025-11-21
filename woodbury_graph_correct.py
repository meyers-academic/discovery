"""
Correct graph structure: Nodes represent intermediate quantities, not full solves.

Key insight: The fundamental operations are:
- Nmy = N^{-1} y
- NmF = N^{-1} F
- FtNmy = F^T N^{-1} y
- FtNmF = F^T N^{-1} F
- etc.

These can be precomputed if their inputs are constant.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from typing import Union, Set


# ============================================================================
# Base Node
# ============================================================================


class Node:
    """Base class for all nodes (data and operations)."""

    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.dependencies = []

    @property
    def is_constant(self) -> bool:
        """True if this node and all dependencies are constant."""
        raise NotImplementedError

    @property
    def params(self) -> Set[str]:
        """Parameters this node depends on."""
        raise NotImplementedError

    def eval(self, params=None):
        """Evaluate this node."""
        raise NotImplementedError

    def __repr__(self):
        const_str = "const" if self.is_constant else f"var({','.join(sorted(self.params))})"
        return f"{self.name}[{const_str}]"


# ============================================================================
# Leaf Nodes (Data)
# ============================================================================


class ConstantNode(Node):
    """Constant data."""

    def __init__(self, value, name=None):
        super().__init__(name)
        self.value = jnp.array(value) if not isinstance(value, jax.Array) else value

    @property
    def is_constant(self):
        return True

    @property
    def params(self):
        return set()

    def eval(self, params=None):
        return self.value


class VariableNode(Node):
    """Variable data (parameter-dependent)."""

    def __init__(self, param_name: str, transform=None, name=None):
        super().__init__(name or param_name)
        self.param_name = param_name
        self.transform = transform or (lambda x: x)

    @property
    def is_constant(self):
        return False

    @property
    def params(self):
        return {self.param_name}

    def eval(self, params):
        return self.transform(params[self.param_name])


# ============================================================================
# Intermediate Quantity Nodes
# ============================================================================


class SolveNode(Node):
    """
    Represents A^{-1} b (solve A with b).

    This is a fundamental operation: Nmy = N^{-1} y or NmF = N^{-1} F.
    Can be precomputed if A and b are both constant.
    """

    def __init__(self, A: Node, b: Node, name=None):
        super().__init__(name or f"{A.name}^{{-1}}{b.name}")
        self.A = A
        self.b = b
        self.dependencies = [A, b]

    @property
    def is_constant(self):
        return self.A.is_constant and self.b.is_constant

    @property
    def params(self):
        return self.A.params | self.b.params

    def eval(self, params=None):
        """Solve A x = b."""
        A_val = self.A.eval(params)
        b_val = self.b.eval(params)

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


class InnerProductNode(Node):
    """
    Represents A^T b (inner product).

    Used for FtNmy = F^T (N^{-1} y) or FtNmF = F^T (N^{-1} F).
    Can be precomputed if A and b are constant.
    """

    def __init__(self, A: Node, b: Node, name=None):
        super().__init__(name or f"{A.name}^T {b.name}")
        self.A = A
        self.b = b
        self.dependencies = [A, b]

    @property
    def is_constant(self):
        return self.A.is_constant and self.b.is_constant

    @property
    def params(self):
        return self.A.params | self.b.params

    def eval(self, params=None):
        """Compute A^T b."""
        A_val = self.A.eval(params)
        b_val = self.b.eval(params)
        return A_val.T @ b_val


class QuadFormNode(Node):
    """
    Represents a^T b (quadratic form / dot product).

    Used for ytNmy = y^T (N^{-1} y).
    Can be precomputed if a and b are constant.
    """

    def __init__(self, a: Node, b: Node, name=None):
        super().__init__(name or f"{a.name}^T {b.name}")
        self.a = a
        self.b = b
        self.dependencies = [a, b]

    @property
    def is_constant(self):
        return self.a.is_constant and self.b.is_constant

    @property
    def params(self):
        return self.a.params | self.b.params

    def eval(self, params=None):
        """Compute a^T b."""
        a_val = self.a.eval(params)
        b_val = self.b.eval(params)
        return jnp.dot(a_val, b_val)


class InvertNode(Node):
    """
    Represents A^{-1} (matrix inverse).

    Used for Pinv = P^{-1}.
    Can be precomputed if A is constant.
    """

    def __init__(self, A: Node, name=None):
        super().__init__(name or f"{A.name}^{{-1}}")
        self.A = A
        self.dependencies = [A]

    @property
    def is_constant(self):
        return self.A.is_constant

    @property
    def params(self):
        return self.A.params

    def eval(self, params=None):
        """Compute A^{-1}."""
        A_val = self.A.eval(params)
        if A_val.ndim == 1:
            # Diagonal
            return 1.0 / A_val
        else:
            # Dense (for now, just use identity solve)
            A_factor = jsp.linalg.cho_factor(A_val)
            return jsp.linalg.cho_solve(A_factor, jnp.eye(A_val.shape[0]))


class SumNode(Node):
    """
    Represents A + B.

    Used for S = Pinv + FtNmF.
    Can be precomputed if A and B are constant.
    """

    def __init__(self, A: Node, B: Node, name=None):
        super().__init__(name or f"{A.name}+{B.name}")
        self.A = A
        self.B = B
        self.dependencies = [A, B]

    @property
    def is_constant(self):
        return self.A.is_constant and self.B.is_constant

    @property
    def params(self):
        return self.A.params | self.B.params

    def eval(self, params=None):
        """Compute A + B."""
        A_val = self.A.eval(params)
        B_val = self.B.eval(params)
        # Handle diagonal + dense
        if A_val.ndim == 1 and B_val.ndim == 2:
            return jnp.diag(A_val) + B_val
        elif A_val.ndim == 2 and B_val.ndim == 1:
            return A_val + jnp.diag(B_val)
        else:
            return A_val + B_val


class MatmulNode(Node):
    """
    Represents A @ b (matrix-vector or matrix-matrix multiply).

    Used for correction = NmF @ SmFtNmy.
    Can be precomputed if A and b are constant.
    """

    def __init__(self, A: Node, b: Node, name=None):
        super().__init__(name or f"{A.name}@{b.name}")
        self.A = A
        self.b = b
        self.dependencies = [A, b]

    @property
    def is_constant(self):
        return self.A.is_constant and self.b.is_constant

    @property
    def params(self):
        return self.A.params | self.b.params

    def eval(self, params=None):
        """Compute A @ b."""
        A_val = self.A.eval(params)
        b_val = self.b.eval(params)
        return A_val @ b_val


class SubtractNode(Node):
    """
    Represents A - B.

    Used for solution = Nmy - correction.
    Can be precomputed if A and B are constant.
    """

    def __init__(self, A: Node, B: Node, name=None):
        super().__init__(name or f"{A.name}-{B.name}")
        self.A = A
        self.B = B
        self.dependencies = [A, B]

    @property
    def is_constant(self):
        return self.A.is_constant and self.B.is_constant

    @property
    def params(self):
        return self.A.params | self.B.params

    def eval(self, params=None):
        """Compute A - B."""
        A_val = self.A.eval(params)
        B_val = self.B.eval(params)
        return A_val - B_val


class LogDetNode(Node):
    """
    Represents log|A| (log determinant).

    Can be precomputed if A is constant.
    """

    def __init__(self, A: Node, name=None):
        super().__init__(name or f"log|{A.name}|")
        self.A = A
        self.dependencies = [A]

    @property
    def is_constant(self):
        return self.A.is_constant

    @property
    def params(self):
        return self.A.params

    def eval(self, params=None):
        """Compute log|A|."""
        A_val = self.A.eval(params)
        if A_val.ndim == 1:
            # Diagonal
            return jnp.sum(jnp.log(A_val))
        else:
            # Dense
            A_factor = jsp.linalg.cho_factor(A_val)
            return 2.0 * jnp.sum(jnp.log(jnp.diag(A_factor[0])))


# ============================================================================
# Woodbury Graph Builder
# ============================================================================


class WoodburyGraph:
    """
    Builds the computation graph for Woodbury solve.

    Creates nodes for all intermediate quantities:
    - Nmy, NmF (solve operations)
    - FtNmy, FtNmF (inner products)
    - Pinv, S (Schur complement)
    - SmFtNmy (Schur solve)
    - correction, solution (final result)
    - Log determinants
    """

    def __init__(self, N: Node, F: Node, P: Node, y: Node):
        """
        Build graph for solving (N + F^T P F)^{-1} y.

        Args:
            N: Noise matrix (or another WoodburyGraph for nesting)
            F: Design matrix
            P: Prior covariance
            y: Data vector
        """
        self.N = N
        self.F = F
        self.P = P
        self.y = y

        # Build the graph
        self._build_graph()

    def _build_graph(self):
        """Build all intermediate nodes."""
        # Level 1: Solve with N
        self.Nmy = SolveNode(self.N, self.y, name="Nmy")
        self.NmF = SolveNode(self.N, self.F, name="NmF")

        # Level 2: Inner products
        self.FtNmy = InnerProductNode(self.F, self.Nmy, name="FtNmy")
        self.FtNmF = InnerProductNode(self.F, self.NmF, name="FtNmF")

        # Level 3: Schur complement
        self.Pinv = InvertNode(self.P, name="Pinv")
        self.S = SumNode(self.Pinv, self.FtNmF, name="S")

        # Level 4: Schur solve
        self.SmFtNmy = SolveNode(self.S, self.FtNmy, name="SmFtNmy")

        # Level 5: Final solution
        self.correction = MatmulNode(self.NmF, self.SmFtNmy, name="correction")
        self.solution = SubtractNode(self.Nmy, self.correction, name="solution")

        # Log determinants
        self.logdetN = LogDetNode(self.N, name="logdetN")
        self.logdetP = LogDetNode(self.P, name="logdetP")
        self.logdetS = LogDetNode(self.S, name="logdetS")

        # Combined log determinant: log|N + F^T P F| = log|N| - log|P| + log|S|
        # For now, compute this in eval

    @property
    def params(self):
        """All parameters needed."""
        return self.solution.params

    def eval(self, params=None):
        """Evaluate the full solve."""
        solution = self.solution.eval(params)
        logdetN = self.logdetN.eval(params)
        logdetP = self.logdetP.eval(params)
        logdetS = self.logdetS.eval(params)
        logdet = logdetN - logdetP + logdetS
        return solution, logdet

    def make_solve_closure(self):
        """
        Create optimized closure for solving.

        This examines which nodes are constant and precomputes them.
        """
        # Identify which intermediate quantities are constant
        precompute = {}

        if self.Nmy.is_constant:
            precompute["Nmy"] = self.Nmy.eval(None)
        if self.NmF.is_constant:
            precompute["NmF"] = self.NmF.eval(None)
        if self.FtNmy.is_constant:
            precompute["FtNmy"] = self.FtNmy.eval(None)
        if self.FtNmF.is_constant:
            precompute["FtNmF"] = self.FtNmF.eval(None)
        if self.Pinv.is_constant:
            precompute["Pinv"] = self.Pinv.eval(None)
        if self.logdetN.is_constant:
            precompute["logdetN"] = self.logdetN.eval(None)
        if self.logdetP.is_constant:
            precompute["logdetP"] = self.logdetP.eval(None)

        # Most common case: N, F, y constant; only P varies
        if self.Nmy.is_constant and self.NmF.is_constant and self.FtNmy.is_constant and self.FtNmF.is_constant:
            # Everything involving N, F, y is precomputed
            Nmy = precompute["Nmy"]
            NmF = precompute["NmF"]
            FtNmy = precompute["FtNmy"]
            FtNmF = precompute["FtNmF"]
            logdetN = precompute["logdetN"]

            P_node = self.P

            def closure(params):
                # Only compute things involving P
                P_val = P_node.eval(params)

                # Pinv
                if P_val.ndim == 1:
                    Pinv = 1.0 / P_val
                    logdetP = jnp.sum(jnp.log(P_val))
                else:
                    P_factor = jsp.linalg.cho_factor(P_val)
                    Pinv = jsp.linalg.cho_solve(P_factor, jnp.eye(P_val.shape[0]))
                    logdetP = 2.0 * jnp.sum(jnp.log(jnp.diag(P_factor[0])))

                # S = Pinv + FtNmF (FtNmF precomputed!)
                if Pinv.ndim == 1:
                    S = jnp.diag(Pinv) + FtNmF
                else:
                    S = Pinv + FtNmF

                # Solve S
                S_factor = jsp.linalg.cho_factor(S)
                SmFtNmy = jsp.linalg.cho_solve(S_factor, FtNmy)  # FtNmy precomputed!
                logdetS = 2.0 * jnp.sum(jnp.log(jnp.diag(S_factor[0])))

                # Final solution
                correction = NmF @ SmFtNmy  # NmF precomputed!
                solution = Nmy - correction  # Nmy precomputed!

                # Log determinant
                logdet = logdetN - logdetP + logdetS  # logdetN precomputed!

                return solution, logdet

            closure.params = sorted(self.P.params)
            closure.precomputed = list(precompute.keys())

        else:
            # General case: compute everything
            def closure(params):
                return self.eval(params)

            closure.params = sorted(self.params)
            closure.precomputed = []

        return closure

    def print_graph(self):
        """Print the computation graph showing what's constant."""
        print(f"\nWoodbury Graph:")
        print(f"  Data:")
        print(f"    N: {self.N}")
        print(f"    F: {self.F}")
        print(f"    P: {self.P}")
        print(f"    y: {self.y}")
        print(f"  Intermediate quantities:")
        print(f"    Nmy:     {self.Nmy}")
        print(f"    NmF:     {self.NmF}")
        print(f"    FtNmy:   {self.FtNmy}")
        print(f"    FtNmF:   {self.FtNmF}")
        print(f"    Pinv:    {self.Pinv}")
        print(f"    S:       {self.S}")
        print(f"    SmFtNmy: {self.SmFtNmy}")
        print(f"  Results:")
        print(f"    solution: {self.solution}")
        print(f"    logdet:   log|N| - log|P| + log|S|")
        print(f"              {self.logdetN} - {self.logdetP} + {self.logdetS}")


# ============================================================================
# Example
# ============================================================================


def example_optimal_precomputation():
    """Show optimal precomputation when N, F, y constant, P varies."""
    print("=" * 70)
    print("Example: Optimal Precomputation (N, F, y const; P var)")
    print("=" * 70)

    # Data
    n_data, n_basis = 100, 10
    key = jax.random.PRNGKey(0)

    N_data = jnp.ones(n_data) * 0.5
    F_matrix = jax.random.normal(key, (n_data, n_basis))
    y_data = jax.random.normal(jax.random.PRNGKey(1), (n_data,))

    # Build nodes
    N_node = ConstantNode(N_data, name="N")
    F_node = ConstantNode(F_matrix, name="F")
    P_node = VariableNode("amplitude", transform=lambda a: jnp.ones(n_basis) * a**2, name="P")
    y_node = ConstantNode(y_data, name="y")

    # Build graph
    graph = WoodburyGraph(N_node, F_node, P_node, y_node)

    # Print graph structure
    graph.print_graph()

    # Make closure
    print("\n" + "=" * 70)
    print("Creating optimized closure...")
    closure = graph.make_solve_closure()

    print(f"\nClosure parameters: {closure.params}")
    print(f"Precomputed quantities: {closure.precomputed}")
    print("\nPrecomputed:")
    for name in closure.precomputed:
        print(f"  ✓ {name}")
    print("\nComputed in closure:")
    for name in ["Pinv", "S", "SmFtNmy", "correction", "solution", "logdetP", "logdetS"]:
        if name not in closure.precomputed:
            print(f"  • {name}")

    # Use closure
    print("\n" + "=" * 70)
    print("Evaluating for different amplitudes:")
    for amp in [0.5, 1.0, 2.0]:
        solution, logdet = closure({"amplitude": amp})
        print(f"  amp={amp}: ||solution||={jnp.linalg.norm(solution):.4f}, logdet={logdet:.2f}")

    # JIT compile
    jit_closure = jax.jit(closure)
    solution, logdet = jit_closure({"amplitude": 1.5})
    print(f"\nJIT compiled: ||solution||={jnp.linalg.norm(solution):.4f}, logdet={logdet:.2f}")


def example_kernel_product():
    """Show kernel product with optimal precomputation."""
    print("\n\n" + "=" * 70)
    print("Example: Kernel Product with Precomputation")
    print("=" * 70)

    # Data
    n_data, n_basis = 100, 10
    key = jax.random.PRNGKey(0)

    N_data = jnp.ones(n_data) * 0.5
    F_matrix = jax.random.normal(key, (n_data, n_basis))
    y_data = jax.random.normal(jax.random.PRNGKey(1), (n_data,))

    # Build nodes
    N_node = ConstantNode(N_data, name="N")
    F_node = ConstantNode(F_matrix, name="F")
    P_node = VariableNode("amplitude", transform=lambda a: jnp.ones(n_basis) * a**2, name="P")
    y_node = ConstantNode(y_data, name="y")

    # Build graph
    graph = WoodburyGraph(N_node, F_node, P_node, y_node)

    # For kernel product, we also need ytNmy = y^T N^{-1} y
    ytNmy_node = QuadFormNode(y_node, graph.Nmy, name="ytNmy")

    print(f"ytNmy node: {ytNmy_node}")
    print(f"  This is CONSTANT because y and Nmy are both constant!")

    # Precompute what we can
    ytNmy = ytNmy_node.eval(None)
    FtNmy = graph.FtNmy.eval(None)
    FtNmF = graph.FtNmF.eval(None)
    NmF = graph.NmF.eval(None)
    logdetN = graph.logdetN.eval(None)

    print(f"\nPrecomputed: ytNmy, FtNmy, FtNmF, NmF, logdetN")

    # Create kernel product closure
    def kernel_product_closure(params):
        P_val = P_node.eval(params)

        # Pinv
        if P_val.ndim == 1:
            Pinv = 1.0 / P_val
            logdetP = jnp.sum(jnp.log(P_val))
        else:
            P_factor = jsp.linalg.cho_factor(P_val)
            Pinv = jsp.linalg.cho_solve(P_factor, jnp.eye(P_val.shape[0]))
            logdetP = 2.0 * jnp.sum(jnp.log(jnp.diag(P_factor[0])))

        # S = Pinv + FtNmF (FtNmF precomputed!)
        if Pinv.ndim == 1:
            S = jnp.diag(Pinv) + FtNmF
        else:
            S = Pinv + FtNmF

        # Solve S
        S_factor = jsp.linalg.cho_factor(S)
        SmFtNmy = jsp.linalg.cho_solve(S_factor, FtNmy)  # FtNmy precomputed!
        logdetS = 2.0 * jnp.sum(jnp.log(jnp.diag(S_factor[0])))

        # Quadratic form: y^T (N + F^T P F)^{-1} y = ytNmy - FtNmy^T SmFtNmy
        ytWmy = ytNmy - FtNmy @ SmFtNmy  # Both ytNmy and FtNmy precomputed!

        # Log determinant
        logdet = logdetN - logdetP + logdetS  # logdetN precomputed!

        # Kernel product
        return -0.5 * ytWmy - 0.5 * logdet

    # Evaluate
    print("\nKernel product (log-likelihood):")
    for amp in [0.1, 0.5, 1.0, 2.0]:
        ll = kernel_product_closure({"amplitude": amp})
        print(f"  amp={amp:4.1f}: loglike={ll:8.2f}")


if __name__ == "__main__":
    example_optimal_precomputation()
    example_kernel_product()

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("Key points:")
    print("  1. Nodes represent INTERMEDIATE QUANTITIES (Nmy, FtNmy, etc.)")
    print("  2. Each node knows if it's constant (can be precomputed)")
    print("  3. Closure generation identifies precomputable quantities")
    print("  4. For common case (N,F,y const; P var), we precompute:")
    print("     Nmy, NmF, FtNmy, FtNmF, ytNmy, logdetN")
    print("  5. This is the CORRECT way to build the graph!")
    print("=" * 70)
