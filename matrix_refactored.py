"""
matrix_refactored.py - Refactored version showing graph-based approach

This shows how the current matrix.py classes would be replaced with
the graph-based design, maintaining the same API for backward compatibility.

Key changes:
  OLD: Separate classes WoodburyKernel_varP, WoodburyKernel_varN, etc.
  NEW: Single WoodburyKernel class that uses graph internally

  OLD: Manual optimization in each class
  NEW: Automatic optimization via graph structure
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from typing import Set, Optional, Callable


# ============================================================================
# Import from corrected graph implementation
# (In actual implementation, this would be in the same file or imported)
# ============================================================================

# For now, we'll inline the necessary classes from woodbury_graph_corrected.py

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


class FunctionLeaf(Leaf):
    """Function of multiple parameters."""

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


# OpNode and specific operations (abbreviated - see woodbury_graph_corrected.py for full)
class OpNode:
    def __init__(self, *inputs, name=None):
        self.inputs = inputs
        self.name = name or self.__class__.__name__
        self._cached_value = None
        self._is_evaluated = False

    @property
    def is_constant(self):
        return all(inp.is_constant for inp in self.inputs)

    @property
    def params(self):
        params = set()
        for inp in self.inputs:
            params.update(inp.params)
        return params

    def eval(self, params=None):
        if self.is_constant and self._is_evaluated:
            return self._cached_value
        input_values = [inp.eval(params) for inp in self.inputs]
        result = self.compute(*input_values)
        if self.is_constant:
            self._cached_value = result
            self._is_evaluated = True
        return result

    def precompute(self):
        if self.is_constant and not self._is_evaluated:
            self.eval(None)

    def compute(self, *input_values):
        raise NotImplementedError


class SolveOp(OpNode):
    def __init__(self, A, b, name=None):
        super().__init__(A, b, name=name)
        self.A = A
        self.b = b

    def eval(self, params=None):
        if self.is_constant and self._is_evaluated:
            return self._cached_value

        b_val = self.b.eval(params)

        if hasattr(self.A, 'solve'):
            result = self.A.solve(b_val, params)
        else:
            A_val = self.A.eval(params)
            result = self._solve_matrix(A_val, b_val)

        if self.is_constant:
            self._cached_value = result
            self._is_evaluated = True

        return result

    def _solve_matrix(self, A_val, b_val):
        if A_val.ndim == 1:
            return b_val / A_val if b_val.ndim == 1 else b_val / A_val[:, None]
        else:
            A_factor = jsp.linalg.cho_factor(A_val)
            return jsp.linalg.cho_solve(A_factor, b_val)

    def compute(self, A_val, b_val):
        return self._solve_matrix(A_val, b_val)


class InnerProductOp(OpNode):
    def compute(self, A_val, B_val):
        return A_val.T @ B_val


class DotProductOp(OpNode):
    def compute(self, a_val, b_val):
        return jnp.dot(a_val, b_val)


class InvertOp(OpNode):
    def compute(self, A_val):
        if A_val.ndim == 1:
            return 1.0 / A_val
        else:
            A_factor = jsp.linalg.cho_factor(A_val)
            return jsp.linalg.cho_solve(A_factor, jnp.eye(A_val.shape[0]))


class AddOp(OpNode):
    def compute(self, A_val, B_val):
        if A_val.ndim == 1 and B_val.ndim == 2:
            return jnp.diag(A_val) + B_val
        elif A_val.ndim == 2 and B_val.ndim == 1:
            return A_val + jnp.diag(B_val)
        else:
            return A_val + B_val


class SubtractOp(OpNode):
    def compute(self, A_val, B_val):
        return A_val - B_val


class MatmulOp(OpNode):
    def compute(self, A_val, B_val):
        return A_val @ B_val


class LogDetOp(OpNode):
    def compute(self, A_val):
        if A_val.ndim == 1:
            return jnp.sum(jnp.log(A_val))
        else:
            A_factor = jsp.linalg.cho_factor(A_val)
            return 2.0 * jnp.sum(jnp.log(jnp.diag(A_factor[0])))


class ScalarOp(OpNode):
    def __init__(self, x, a=1.0, b=0.0, name=None):
        super().__init__(x, name=name)
        self.a = a
        self.b = b

    def compute(self, x_val):
        return self.a * x_val + self.b


# ============================================================================
# WoodburyGraph (simplified version)
# ============================================================================


class WoodburyGraph:
    """Computation graph for Woodbury operations."""

    def __init__(self, N: Leaf, F: Leaf, P: Leaf, y: Leaf, name: str = "Woodbury"):
        self.N = N
        self.F = F
        self.P = P
        self.y = y
        self.name = name
        self._nodes = {}

    def _get_or_create(self, key, factory):
        if key not in self._nodes:
            self._nodes[key] = factory()
        return self._nodes[key]

    @property
    def is_constant(self):
        return self.N.is_constant and self.F.is_constant and self.P.is_constant

    @property
    def params(self):
        return self.N.params | self.F.params | self.P.params

    def solve(self, b, params):
        """Solve (N + F^T P F)^{-1} b"""
        b_leaf = DataLeaf(b, name="b_temp")
        Nmy_temp = SolveOp(self.N, b_leaf, name="Nmy_temp")
        FtNmy_temp = InnerProductOp(self.F, Nmy_temp, name="FtNmy_temp")
        SmFtNmy_temp = SolveOp(self.S, FtNmy_temp, name="SmFtNmy_temp")
        correction_temp = MatmulOp(self.NmF, SmFtNmy_temp, name="correction_temp")
        solution_temp = SubtractOp(Nmy_temp, correction_temp, name="solution_temp")
        return solution_temp.eval(params)

    # Lazy properties
    @property
    def Nmy(self):
        return self._get_or_create('Nmy', lambda: SolveOp(self.N, self.y, name='Nmy'))

    @property
    def NmF(self):
        return self._get_or_create('NmF', lambda: SolveOp(self.N, self.F, name='NmF'))

    @property
    def FtNmy(self):
        return self._get_or_create('FtNmy', lambda: InnerProductOp(self.F, self.Nmy, name='FtNmy'))

    @property
    def FtNmF(self):
        return self._get_or_create('FtNmF', lambda: InnerProductOp(self.F, self.NmF, name='FtNmF'))

    @property
    def ytNmy(self):
        return self._get_or_create('ytNmy', lambda: DotProductOp(self.y, self.Nmy, name='ytNmy'))

    @property
    def Pinv(self):
        return self._get_or_create('Pinv', lambda: InvertOp(self.P, name='Pinv'))

    @property
    def S(self):
        return self._get_or_create('S', lambda: AddOp(self.Pinv, self.FtNmF, name='S'))

    @property
    def SmFtNmy(self):
        return self._get_or_create('SmFtNmy', lambda: SolveOp(self.S, self.FtNmy, name='SmFtNmy'))

    @property
    def logdetN(self):
        return self._get_or_create('logdetN', lambda: LogDetOp(self.N, name='logdetN'))

    @property
    def logdetP(self):
        return self._get_or_create('logdetP', lambda: LogDetOp(self.P, name='logdetP'))

    @property
    def logdetS(self):
        return self._get_or_create('logdetS', lambda: LogDetOp(self.S, name='logdetS'))

    @property
    def quad_correction(self):
        return self._get_or_create('quad_correction',
            lambda: DotProductOp(self.FtNmy, self.SmFtNmy, name='quad_correction'))

    @property
    def ytWmy(self):
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
        return self._get_or_create('kernel_product',
            lambda: SubtractOp(
                ScalarOp(self.ytWmy, a=-0.5, name='-0.5*ytWmy'),
                ScalarOp(self.logdet, a=0.5, name='0.5*logdet'),
                name='kernel_product'))

    def make_kernelproduct_closure(self):
        """Create kernel product closure."""
        kernel_product = self.kernel_product
        self._precompute_constants()

        def closure(params):
            return kernel_product.eval(params)

        closure.params = sorted(kernel_product.params)
        return closure

    def _precompute_constants(self):
        """Precompute all constant nodes."""
        for node in self._nodes.values():
            if hasattr(node, 'is_constant') and node.is_constant:
                if hasattr(node, 'precompute'):
                    node.precompute()


# ============================================================================
# NEW API: Single WoodburyKernel class
# ============================================================================


class WoodburyKernel:
    """
    Unified Woodbury kernel using graph-based approach.

    Replaces:
    - WoodburyKernel_novar
    - WoodburyKernel_varP
    - WoodburyKernel_varN
    - WoodburyKernel_varNP
    - WoodburyKernel_varFP
    - etc.

    The graph automatically handles optimization based on what's constant/variable.
    """

    def __init__(self, N, F, P):
        """
        Create Woodbury kernel.

        Args:
            N: Noise specification - can be:
                - numpy array (constant diagonal)
                - NoiseMatrix object (has solve_1d, solve_2d methods)
                - WoodburyKernel (for nesting!)
                - Callable (parameter-dependent)

            F: Design matrix - can be:
                - numpy array (constant)
                - Callable (parameter-dependent)

            P: Prior specification - can be:
                - numpy array (constant)
                - NoiseMatrix object
                - Callable (parameter-dependent)
        """
        self.N_spec = N
        self.F_spec = F
        self.P_spec = P
        self._graph = None  # Cache the graph
        self.name = "WoodburyKernel"

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
        # Create a temporary graph with b as y
        N_leaf = self._make_leaf(self.N_spec, "N")
        F_leaf = self._make_leaf(self.F_spec, "F")
        P_leaf = self._make_leaf(self.P_spec, "P")
        b_leaf = DataLeaf(b, name="b")

        graph = WoodburyGraph(N_leaf, F_leaf, P_leaf, b_leaf)
        return graph.solve(b, params)

    def _make_leaf(self, spec, name):
        """Convert specification to a Leaf."""
        if isinstance(spec, (jax.Array, jnp.ndarray)):
            # Constant array
            return DataLeaf(spec, name=name)
        elif isinstance(spec, WoodburyKernel):
            # Another WoodburyKernel - acts as a leaf for nesting
            # Check this BEFORE callable to avoid issues
            return spec
        elif hasattr(spec, 'solve'):
            # WoodburyGraph or similar - acts as a leaf
            return spec
        elif hasattr(spec, 'solve_1d'):
            # Old-style NoiseMatrix - wrap it
            return OldNoiseMatrixWrapper(spec, name=name)
        elif callable(spec):
            # Could be a parameter function or a NoiseMatrix
            if hasattr(spec, 'params'):
                # It's a parameterized function with .params attribute
                return FunctionLeaf(spec, spec.params, name=name)
            else:
                # Assume it's a simple callable
                # Try to infer params from function signature
                import inspect
                sig = inspect.signature(spec)
                param_names = list(sig.parameters.keys())
                if 'params' in param_names:
                    # It takes a params dict
                    return FunctionLeaf(spec, [], name=name)
                else:
                    return FunctionLeaf(spec, param_names, name=name)
        else:
            raise TypeError(f"Unknown specification type: {type(spec)}")

    def make_kernelproduct(self, y):
        """
        Create kernel product closure.

        This is the MAIN API method - matches current interface exactly!

        Args:
            y: Data vector - can be:
                - numpy array (constant)
                - Callable (parameter-dependent)

        Returns:
            Closure that computes -0.5 y^T (N + F^T P F)^{-1} y - 0.5 log|N + F^T P F|
        """
        # Convert specs to leaves
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

        Args:
            y: Data vector
            T: Projection matrix

        Returns:
            Closure that computes T^T W^{-1} y and T^T W^{-1} T
        """
        # Convert to leaves
        N_leaf = self._make_leaf(self.N_spec, "N")
        F_leaf = self._make_leaf(self.F_spec, "F")
        P_leaf = self._make_leaf(self.P_spec, "P")
        y_leaf = self._make_leaf(y, "y")
        T_leaf = self._make_leaf(T, "T")

        # Build graph for solve
        graph_y = WoodburyGraph(N_leaf, F_leaf, P_leaf, y_leaf)
        graph_T = WoodburyGraph(N_leaf, F_leaf, P_leaf, T_leaf)

        # Create nodes for T^T W^{-1} y and T^T W^{-1} T
        TtWy = InnerProductOp(T_leaf, graph_y.solution, name="TtWy")
        TtWT = InnerProductOp(T_leaf, graph_T.solution, name="TtWT")

        # Precompute
        for node in [graph_y.solution, graph_T.solution, TtWy, TtWT]:
            if hasattr(node, 'precompute'):
                node.precompute()

        def closure(params):
            return TtWy.eval(params), TtWT.eval(params)

        all_params = graph_y.params | graph_T.params | T_leaf.params
        closure.params = sorted(all_params)

        return closure


class OldNoiseMatrixWrapper(Leaf):
    """
    Wrapper for old-style NoiseMatrix objects.

    Old NoiseMatrix has:
    - solve_1d(y) -> returns (solution, logdet)
    - solve_2d(Y) -> returns (solution, logdet)
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
        # Can't really materialize - this shouldn't be called
        raise NotImplementedError("OldNoiseMatrixWrapper doesn't support eval - use solve instead")

    def solve(self, b, params):
        """Use old solve interface."""
        if b.ndim == 1:
            solution, _ = self.noise_matrix.solve_1d(b)
        else:
            solution, _ = self.noise_matrix.solve_2d(b)
        return solution


# ============================================================================
# MIGRATION EXAMPLES
# ============================================================================


def example_migration_simple():
    """
    Show how OLD code would look vs NEW code.
    """
    print("=" * 70)
    print("Example 1: Simple Migration (constant N, F; variable P)")
    print("=" * 70)

    # Sample data
    n_data, n_basis = 100, 10
    key = jax.random.PRNGKey(0)

    N_data = jnp.ones(n_data) * 0.5
    F_matrix = jax.random.normal(key, (n_data, n_basis))
    y_data = jax.random.normal(jax.random.PRNGKey(1), (n_data,))

    # Define variable prior
    class P_var:
        params = ['amplitude']

        def __call__(self, params):
            return jnp.ones(n_basis) * params['amplitude']**2

        def make_inv(self):
            def inv(params):
                P = self(params)
                return 1.0 / P, jnp.sum(jnp.log(P))
            inv.params = self.params
            return inv

    P_variable = P_var()

    print("\n" + "-" * 70)
    print("OLD CODE (current matrix.py):")
    print("-" * 70)
    print("""
    from discovery.matrix import WoodburyKernel_varP, NoiseMatrix1D_novar

    # Have to choose the right class for constant/variable pattern
    N_noise = NoiseMatrix1D_novar(N_data)
    kernel = WoodburyKernel_varP(N_noise, F_matrix, P_variable)

    # Create closure
    loglike = kernel.make_kernelproduct(y_data)

    # Evaluate
    ll = loglike({'amplitude': 1.5})
    """)

    print("\n" + "-" * 70)
    print("NEW CODE (refactored):")
    print("-" * 70)
    print("""
    from discovery.matrix import WoodburyKernel

    # Same initialization - graph figures out optimization automatically!
    kernel = WoodburyKernel(N_data, F_matrix, P_variable)

    # Same API!
    loglike = kernel.make_kernelproduct(y_data)

    # Same usage!
    ll = loglike({'amplitude': 1.5})
    """)

    # Actually run the new code
    kernel_new = WoodburyKernel(N_data, F_matrix, P_variable)
    loglike_new = kernel_new.make_kernelproduct(y_data)

    ll = loglike_new({'amplitude': 1.5})
    print(f"\nResult: {ll:.4f}")

    print("\nKey difference:")
    print("  OLD: Had to choose WoodburyKernel_varP class manually")
    print("  NEW: Single WoodburyKernel class, automatic optimization")


def example_migration_nested():
    """Show nested Woodbury structures."""
    print("\n\n" + "=" * 70)
    print("Example 2: Nested Woodbury (inner + outer)")
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

    print("\nOLD CODE: Would need custom implementation for nesting")
    print("  (or manually expand to full Woodbury structure)")

    print("\nNEW CODE: Natural nesting")
    print("-" * 70)

    # Inner Woodbury
    inner_kernel = WoodburyKernel(N_base, F_inner, P_inner())

    # Outer Woodbury - uses inner_kernel as N!
    outer_kernel = WoodburyKernel(inner_kernel, F_outer, P_outer())

    # Create likelihood
    loglike = outer_kernel.make_kernelproduct(y_data)

    print(f"Closure parameters: {loglike.params}")

    # Evaluate
    ll = loglike({'inner_amp': 1.0, 'outer_amp': 2.0})
    print(f"Result: {ll:.4f}")

    print("\nNesting works automatically with the graph!")


def example_comparison_all_cases():
    """Show that single class handles all constant/variable combinations."""
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

    print("\nOLD CODE would need:")
    print("  WoodburyKernel_novar")
    print("  WoodburyKernel_varP")
    print("  WoodburyKernel_varN")
    print("  WoodburyKernel_varNP")
    print("  ... etc")

    print("\nNEW CODE uses single WoodburyKernel for all:")
    print("-" * 70)

    for case_name, N, F, P, params in cases:
        kernel = WoodburyKernel(N, F, P)
        loglike = kernel.make_kernelproduct(y_const)
        ll = loglike(params)
        print(f"{case_name:20s}: params={loglike.params}, ll={ll:.4f}")

    print("\nSame class, automatic optimization for each case!")


if __name__ == "__main__":
    example_migration_simple()
    example_migration_nested()
    example_comparison_all_cases()

    print("\n\n" + "=" * 70)
    print("MIGRATION SUMMARY")
    print("=" * 70)
    print("Benefits:")
    print("  1. Single WoodburyKernel class replaces ~10 classes")
    print("  2. Same API: kernel.make_kernelproduct(y)")
    print("  3. Automatic optimization based on graph")
    print("  4. Natural nesting support")
    print("  5. Zero code duplication")
    print()
    print("Migration path:")
    print("  1. Add new WoodburyKernel class to matrix.py")
    print("  2. Mark old classes as deprecated")
    print("  3. Update examples to use new class")
    print("  4. Eventually remove old classes")
    print("=" * 70)
