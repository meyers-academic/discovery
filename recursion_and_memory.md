"""
Addressing two key questions:

1. Recursion: What if N is itself a WoodburyGraph?
2. Memory: Do we keep unused nodes (like 'solution') when we only need kernel product?
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from typing import Set


# ============================================================================
# Question 1: Recursion - N as a WoodburyGraph
# ============================================================================

print("=" * 70)
print("QUESTION 1: Recursion")
print("=" * 70)

print("""
If N is itself a WoodburyGraph, we need it to act like a Leaf that can be solved with.

Solution: WoodburyGraph implements the same interface as Leaf:
  - is_constant property
  - params property
  - eval(params) method -> but returns a solver, not a matrix!

SolveOp needs to detect if input has custom solve:
  - If input has .solve() method, use it
  - Otherwise, use standard matrix solve

This way, WoodburyGraph acts as a "composite leaf" that can be nested.
""")

# Example structure:

class SolveOp_Recursive:
    """
    Solve operation that handles both matrix and WoodburyGraph inputs.
    """

    def __init__(self, A, b, name=None):
        self.A = A
        self.b = b
        self.name = name or f"{A.name}^-1*{b.name}"

    def compute(self, params):
        """Solve A x = b, handling both matrix and WoodburyGraph."""
        b_val = self.b.eval(params)

        # Check if A has custom solve method (e.g., it's a WoodburyGraph)
        if hasattr(self.A, 'solve'):
            # Recursive case: A is a WoodburyGraph
            solution, _ = self.A.solve(b_val, params)
            return solution
        else:
            # Base case: A is a matrix
            A_val = self.A.eval(params)
            if A_val.ndim == 1:
                if b_val.ndim == 1:
                    return b_val / A_val
                else:
                    return b_val / A_val[:, None]
            else:
                A_factor = jsp.linalg.cho_factor(A_val)
                return jsp.linalg.cho_solve(A_factor, b_val)

print("""
Example nested structure:

    # Inner Woodbury: N_inner = N_base + F_inner^T P_inner F_inner
    inner_graph = WoodburyGraph(N_base, F_inner, P_inner, y)

    # Outer Woodbury: full = N_inner + F_outer^T P_outer F_outer
    # N is now a WoodburyGraph!
    outer_graph = WoodburyGraph(inner_graph, F_outer, P_outer, y)

    # When outer_graph computes Nmy = SolveOp(inner_graph, y):
    # - SolveOp detects inner_graph has .solve() method
    # - Calls inner_graph.solve(y, params) recursively
    # - Never forms any matrices!

Key: WoodburyGraph needs to implement:
    - solve(b, params) -> returns solution (for SolveOp to use)
    - is_constant property -> for optimization
    - params property -> for closure generation
""")

# ============================================================================
# Question 2: Memory - Unused Nodes
# ============================================================================

print("\n\n" + "=" * 70)
print("QUESTION 2: Memory and Unused Nodes")
print("=" * 70)

print("""
Problem: In _build_graph(), we create ALL nodes including:
  - solution (line 402)
  - correction (line 401)

But for kernel_product, we only use:
  - kernel_product node
    -> depends on ytWmy and logdet
    -> ytWmy depends on ytNmy and quad_correction
    -> quad_correction depends on FtNmy and SmFtNmy
    -> logdet depends on logdetN, logdetP, logdetS

We NEVER use solution or correction for kernel product!

Two approaches:

A) LAZY BUILDING: Only build nodes when requested
   - Don't create solution/correction unless make_solve_closure() is called
   - Only create kernel_product if make_kernelproduct_closure() is called

B) PRUNING: Build everything, but closure only captures what it uses
   - Python GC will clean up unreferenced nodes eventually
   - Closure only holds references to nodes in its computation path

Approach A is cleaner and more efficient.
""")

# Example of lazy building:

class WoodburyGraph_Lazy:
    """
    Lazy graph building: only create nodes when needed.
    """

    def __init__(self, N, F, P, y):
        self.N = N
        self.F = F
        self.P = P
        self.y = y

        # Don't build anything yet!
        self._nodes = {}

    def _get_or_create(self, name, factory):
        """Get node if exists, otherwise create it."""
        if name not in self._nodes:
            self._nodes[name] = factory()
        return self._nodes[name]

    @property
    def Nmy(self):
        """Lazy: create only when accessed."""
        return self._get_or_create('Nmy',
            lambda: SolveOp(self.N, self.y, name='Nmy'))

    @property
    def NmF(self):
        """Lazy: create only when accessed."""
        return self._get_or_create('NmF',
            lambda: SolveOp(self.N, self.F, name='NmF'))

    @property
    def FtNmy(self):
        """Lazy: depends on Nmy."""
        return self._get_or_create('FtNmy',
            lambda: InnerProductOp(self.F, self.Nmy, name='FtNmy'))

    @property
    def FtNmF(self):
        """Lazy: depends on NmF."""
        return self._get_or_create('FtNmF',
            lambda: InnerProductOp(self.F, self.NmF, name='FtNmF'))

    @property
    def ytNmy(self):
        """Lazy: depends on Nmy."""
        return self._get_or_create('ytNmy',
            lambda: DotProductOp(self.y, self.Nmy, name='ytNmy'))

    @property
    def Pinv(self):
        return self._get_or_create('Pinv',
            lambda: InvertOp(self.P, name='Pinv'))

    @property
    def S(self):
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

    @property
    def quad_correction(self):
        """Only created if kernel product is requested."""
        return self._get_or_create('quad_correction',
            lambda: DotProductOp(self.FtNmy, self.SmFtNmy, name='quad_correction'))

    @property
    def ytWmy(self):
        """Only created if kernel product is requested."""
        return self._get_or_create('ytWmy',
            lambda: SubtractOp(self.ytNmy, self.quad_correction, name='ytWmy'))

    @property
    def logdet(self):
        """Only created if needed."""
        return self._get_or_create('logdet',
            lambda: SubtractOp(
                AddOp(self.logdetN, self.logdetS, name='logdetN+logdetS'),
                self.logdetP,
                name='logdet'))

    @property
    def kernel_product(self):
        """Only created if make_kernelproduct_closure() is called."""
        return self._get_or_create('kernel_product',
            lambda: SubtractOp(
                ScalarOp(self.ytWmy, a=-0.5, name='-0.5*ytWmy'),
                ScalarOp(self.logdet, a=0.5, name='0.5*logdet'),
                name='kernel_product'))

    # THESE ARE ONLY CREATED IF NEEDED:

    @property
    def correction(self):
        """Only created if make_solve_closure() is called."""
        return self._get_or_create('correction',
            lambda: MatmulOp(self.NmF, self.SmFtNmy, name='correction'))

    @property
    def solution(self):
        """Only created if make_solve_closure() is called."""
        return self._get_or_create('solution',
            lambda: SubtractOp(self.Nmy, self.correction, name='solution'))

    def make_kernelproduct_closure(self):
        """
        Create kernel product closure.

        Only creates nodes needed for kernel product:
        - kernel_product -> ytWmy, logdet
        - ytWmy -> ytNmy, quad_correction
        - quad_correction -> FtNmy, SmFtNmy
        - SmFtNmy -> S, FtNmy
        - S -> Pinv, FtNmF
        - FtNmF -> F, NmF
        - NmF -> N, F
        - FtNmy -> F, Nmy
        - Nmy -> N, y
        - ytNmy -> y, Nmy
        - logdet -> logdetN, logdetP, logdetS

        NEVER creates: solution, correction
        """
        # Access kernel_product triggers creation of only needed nodes
        kernel_product = self.kernel_product

        # Precompute constants
        self._precompute_constants()

        def closure(params):
            return kernel_product.eval(params)

        closure.params = sorted(kernel_product.params)

        # Report what was created
        closure.created_nodes = list(self._nodes.keys())

        return closure

    def make_solve_closure(self):
        """
        Create solve closure.

        This WILL create solution and correction nodes because we need them.
        """
        # Access solution triggers creation
        solution = self.solution
        logdet = self.logdet

        self._precompute_constants()

        def closure(params):
            return solution.eval(params), logdet.eval(params)

        closure.params = sorted(solution.params | logdet.params)
        closure.created_nodes = list(self._nodes.keys())

        return closure

    def _precompute_constants(self):
        """Precompute all constant nodes that have been created."""
        for node in self._nodes.values():
            if hasattr(node, 'is_constant') and node.is_constant:
                if hasattr(node, 'precompute'):
                    node.precompute()


print("""
Example usage:

    graph = WoodburyGraph_Lazy(N, F, P, y)

    # Case 1: Only need kernel product
    kernelproduct = graph.make_kernelproduct_closure()
    print(f"Created nodes: {kernelproduct.created_nodes}")
    # Output: ['Nmy', 'NmF', 'FtNmy', 'FtNmF', 'ytNmy', 'Pinv', 'S',
    #          'SmFtNmy', 'logdetN', 'logdetP', 'logdetS', 'quad_correction',
    #          'ytWmy', 'logdet', 'kernel_product']
    # Note: NO 'solution' or 'correction' created!

    # Case 2: Need solve
    solve = graph.make_solve_closure()
    print(f"Created nodes: {solve.created_nodes}")
    # Output: Now includes 'solution' and 'correction'

Benefits:
  1. Only create what you need
  2. Saves memory (especially important for GPU)
  3. Faster graph building
  4. Still zero code duplication
""")

# ============================================================================
# Summary
# ============================================================================

print("\n\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("""
Question 1: Recursion (N as WoodburyGraph)
-------------------------------------------
Solution: WoodburyGraph implements Leaf-like interface:
  - Add solve(b, params) method
  - SolveOp checks if input has custom solve() method
  - Enables natural nesting: outer_graph = WoodburyGraph(inner_graph, F, P, y)

Question 2: Unused Nodes
-------------------------
Solution: Lazy building with @property
  - Nodes created only when accessed
  - make_kernelproduct_closure() never creates solution/correction
  - make_solve_closure() creates them only when needed
  - Saves memory, especially critical for GPU

Implementation Strategy:
  1. Change _build_graph() to lazy properties
  2. Add solve() method to WoodburyGraph for recursion
  3. Update SolveOp to detect and use custom solve()

This gives us:
  ✓ Zero code duplication
  ✓ Automatic optimization
  ✓ Recursive nesting
  ✓ Minimal memory footprint
  ✓ Only compute what's needed
""")

print("=" * 70)
