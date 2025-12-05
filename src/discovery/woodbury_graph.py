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
# Factorization Operations (for efficient reuse)
# ============================================================================


class CholeskyFactorOp(OpNode):
    """
    Compute and cache Cholesky factorization for reuse in multiple operations.

    Returns a tuple ('diag', data) or ('cholesky', factor) so consumers
    can handle both diagonal and dense cases.
    """

    def compute(self, A_val):
        if A_val.ndim == 1:
            # Diagonal matrix - return as-is (no factorization needed)
            return ('diag', A_val)
        else:
            # Dense matrix - compute Cholesky factorization
            factor = jsp.linalg.cho_factor(A_val)
            return ('cholesky', factor)


class SolveWithFactorOp(OpNode):
    """
    Solve A^{-1} b using pre-computed Cholesky factorization.

    First input should be a CholeskyFactorOp node.
    """

    def __init__(self, factor_node, b, name=None):
        super().__init__(factor_node, b, name=name)
        self.factor_node = factor_node
        self.b = b

    def compute(self, factor_val, b_val):
        factor_type, factor_data = factor_val

        if factor_type == 'diag':
            # Diagonal solve
            if b_val.ndim == 1:
                return b_val / factor_data
            else:
                return b_val / factor_data[:, None]
        else:
            # Dense solve using Cholesky factorization
            return jsp.linalg.cho_solve(factor_data, b_val)


class LogDetFromFactorOp(OpNode):
    """
    Compute log|A| from pre-computed Cholesky factorization.

    Input should be a CholeskyFactorOp node.
    """

    def compute(self, factor_val):
        factor_type, factor_data = factor_val

        if factor_type == 'diag':
            return jnp.sum(jnp.log(factor_data))
        else:
            # Log determinant from Cholesky: log|A| = 2 * sum(log(diag(L)))
            return 2.0 * jnp.sum(jnp.log(jnp.diag(factor_data[0])))


class InvertFromFactorOp(OpNode):
    """
    Compute A^{-1} from pre-computed Cholesky factorization.

    Input should be a CholeskyFactorOp node.
    """

    def compute(self, factor_val):
        factor_type, factor_data = factor_val

        if factor_type == 'diag':
            return 1.0 / factor_data
        else:
            # Compute full inverse by solving A x = I
            n = factor_data[0].shape[0]
            return jsp.linalg.cho_solve(factor_data, jnp.eye(n))


class SolveAndLogDetOp(OpNode):
    """
    Compute both A^{-1}b and log|A| using a single Cholesky factorization.

    Returns a tuple (solution, logdet) for extraction via IndexOp.
    This is more efficient than computing them separately when both are needed.
    """

    def __init__(self, A, b, name=None):
        super().__init__(A, b, name=name)
        self.A = A
        self.b = b

    def compute(self, A_val, b_val):
        if A_val.ndim == 1:
            # Diagonal case
            solution = b_val / A_val
            logdet = jnp.sum(jnp.log(A_val))
        else:
            # Dense case - factor once, use twice
            factor = jsp.linalg.cho_factor(A_val)
            solution = jsp.linalg.cho_solve(factor, b_val)
            logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(factor[0])))

        return (solution, logdet)


class IndexOp(OpNode):
    """
    Extract element from a tuple-valued node.

    IMPORTANT: Does NOT cache the extracted value to avoid duplication.
    Only the parent node caches the full tuple.
    """

    def __init__(self, tuple_node, index, name=None):
        super().__init__(tuple_node, name=name)
        self.index = index

    def eval(self, params=None):
        """
        Override eval to NOT cache - just extract from parent's cached tuple.
        This avoids storing the same data twice (once in tuple, once extracted).
        """
        # Get the cached tuple from parent (parent handles caching)
        tuple_val = self.inputs[0].eval(params)
        # Extract and return (O(1), no additional storage)
        return tuple_val[self.index]

    @property
    def is_constant(self):
        """Inherit constantness from parent."""
        return self.inputs[0].is_constant

    def compute(self, tuple_val):
        """Fallback (not used since we override eval)."""
        return tuple_val[self.index]


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

    # Factorizations (compute once, reuse multiple times)
    # Only create factor nodes for actual matrices, not nested structures
    @property
    def N_factor(self):
        """Cholesky factorization of N (only if N is a matrix, not nested)"""
        # Don't factor nested structures - they have their own solve() method
        if hasattr(self.N, 'solve'):
            return None
        return self._get_or_create('N_factor',
            lambda: CholeskyFactorOp(self.N, name='N_factor'))

    @property
    def P_factor(self):
        """Cholesky factorization of P (only if P is a matrix, not nested)"""
        # Don't factor nested structures
        if hasattr(self.P, 'solve'):
            return None
        return self._get_or_create('P_factor',
            lambda: CholeskyFactorOp(self.P, name='P_factor'))

    # Operations using N - conditional on whether N is nested
    @property
    def Nmy(self):
        """N^{-1} y - uses factorization for matrices, SolveOp for nested"""
        if hasattr(self.N, 'solve'):
            # Nested case - use SolveOp which calls N.solve()
            return self._get_or_create('Nmy',
                lambda: SolveOp(self.N, self.y, name='Nmy'))
        else:
            # Matrix case - use factorization
            return self._get_or_create('Nmy',
                lambda: SolveWithFactorOp(self.N_factor, self.y, name='Nmy'))

    @property
    def NmF(self):
        """N^{-1} F - uses factorization for matrices, SolveOp for nested"""
        if hasattr(self.N, 'solve'):
            # Nested case - use SolveOp which calls N.solve()
            return self._get_or_create('NmF',
                lambda: SolveOp(self.N, self.F, name='NmF'))
        else:
            # Matrix case - use factorization
            return self._get_or_create('NmF',
                lambda: SolveWithFactorOp(self.N_factor, self.F, name='NmF'))

    @property
    def logdetN(self):
        """log|N| - uses factorization for matrices, LogDetOp for nested"""
        if hasattr(self.N, 'compute_logdet'):
            # Nested case - use LogDetOp which calls N.compute_logdet()
            return self._get_or_create('logdetN',
                lambda: LogDetOp(self.N, name='logdetN'))
        else:
            # Matrix case - use factorization
            return self._get_or_create('logdetN',
                lambda: LogDetFromFactorOp(self.N_factor, name='logdetN'))

    # Operations using P - conditional on whether P is nested
    @property
    def Pinv(self):
        """P^{-1} - uses factorization for matrices, InvertOp for nested"""
        if hasattr(self.P, 'solve'):
            # Nested case - can't efficiently invert nested structure
            # Use old InvertOp which will try to materialize (not ideal but rare)
            return self._get_or_create('Pinv',
                lambda: InvertOp(self.P, name='Pinv'))
        else:
            # Matrix case - use factorization
            return self._get_or_create('Pinv',
                lambda: InvertFromFactorOp(self.P_factor, name='Pinv'))

    @property
    def logdetP(self):
        """log|P| - uses factorization for matrices, LogDetOp for nested"""
        if hasattr(self.P, 'compute_logdet'):
            # Nested case - use LogDetOp which calls P.compute_logdet()
            return self._get_or_create('logdetP',
                lambda: LogDetOp(self.P, name='logdetP'))
        else:
            # Matrix case - use factorization
            return self._get_or_create('logdetP',
                lambda: LogDetFromFactorOp(self.P_factor, name='logdetP'))

    # Intermediate computations
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
    def S(self):
        """Schur complement: P^{-1} + F^T N^{-1} F"""
        return self._get_or_create('S',
            lambda: AddOp(self.Pinv, self.FtNmF, name='S'))

    # Operations on S (combined solve and logdet)
    @property
    def SmFtNmy_and_logdetS(self):
        """
        Combined operation: compute both S^{-1}(F^T N^{-1} y) and log|S|
        using a single Cholesky factorization.
        """
        return self._get_or_create('SmFtNmy_and_logdetS',
            lambda: SolveAndLogDetOp(self.S, self.FtNmy, name='SmFtNmy_and_logdetS'))

    @property
    def SmFtNmy(self):
        """S^{-1}(F^T N^{-1} y) - extracted from combined op (no additional storage)"""
        return self._get_or_create('SmFtNmy',
            lambda: IndexOp(self.SmFtNmy_and_logdetS, 0, name='SmFtNmy'))

    @property
    def logdetS(self):
        """log|S| - extracted from combined op (no additional storage)"""
        return self._get_or_create('logdetS',
            lambda: IndexOp(self.SmFtNmy_and_logdetS, 1, name='logdetS'))

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