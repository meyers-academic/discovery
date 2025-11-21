"""
Demonstrate proper closure creation where y is supplied upfront.

Key insight: When creating a closure, we SUPPLY y as data (not a parameter).
All y-dependent quantities get precomputed and baked into the closure.

This matches how the current code works with make_kernelproduct(y).
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp


# ============================================================================
# Example: Current Pattern in matrix.py
# ============================================================================


def show_current_pattern():
    """Show how current code works with make_kernelproduct(y)."""
    print("=" * 70)
    print("Current Pattern: make_kernelproduct(y)")
    print("=" * 70)

    print("""
Current code pattern:

class WoodburyKernel_varP:
    def __init__(self, N, F, P):
        self.N = N        # constant data
        self.F = F        # constant data
        self.P = P        # parameter function

    def make_kernelproduct(self, y):
        '''Supply y and get back a closure that only depends on P parameters.'''

        # Precompute everything involving N, F, y (ALL CONSTANT)
        Nmy = solve(N, y)           # ← computed ONCE
        NmF = solve(N, F)           # ← computed ONCE
        FtNmy = F.T @ Nmy           # ← computed ONCE
        FtNmF = F.T @ NmF           # ← computed ONCE
        ytNmy = y @ Nmy             # ← computed ONCE
        logdetN = logdet(N)         # ← computed ONCE

        # Return closure that uses precomputed values
        def kernelproduct(params):
            # Only compute things involving P (VARIABLE)
            P_val = self.P(params)
            Pinv = 1.0 / P_val
            S = diag(Pinv) + FtNmF  # ← uses precomputed FtNmF

            SmFtNmy = solve(S, FtNmy)  # ← uses precomputed FtNmy
            ytWmy = ytNmy - FtNmy @ SmFtNmy  # ← uses precomputed ytNmy, FtNmy

            logdetP = sum(log(P_val))
            logdetS = logdet(S)
            logdet = logdetN - logdetP + logdetS  # ← uses precomputed logdetN

            return -0.5 * ytWmy - 0.5 * logdet

        return kernelproduct

This is EXACTLY the right pattern!
y is supplied to make_kernelproduct(), not passed as a parameter.
All y-dependent quantities are precomputed in the closure.
""")


# ============================================================================
# Graph-Based Version
# ============================================================================


class WoodburyGraphClosure:
    """
    Graph-based version that matches the current pattern.

    Key: y is supplied when creating the closure, not when calling it.
    """

    def __init__(self, N_node, F_node, P_node):
        """
        Create Woodbury structure with N, F, P.

        Args:
            N_node: Noise matrix (constant or variable)
            F_node: Design matrix (constant or variable)
            P_node: Prior covariance (constant or variable)
        """
        self.N = N_node
        self.F = F_node
        self.P = P_node

    def make_kernelproduct(self, y_node):
        """
        Create kernel product closure given y.

        This is analogous to current make_kernelproduct(y).
        All y-dependent quantities are precomputed here.

        Args:
            y_node: Data vector (constant or variable)

        Returns:
            Closure that computes -0.5 y^T (N + F^T P F)^{-1} y - 0.5 log|N + F^T P F|
        """
        # Determine what can be precomputed
        N_const = self.N.is_constant
        F_const = self.F.is_constant
        P_const = self.P.is_constant
        y_const = y_node.is_constant

        # Case 1: N, F, y all constant; only P varies (MOST COMMON!)
        if N_const and F_const and y_const:
            return self._make_closure_NFy_const(y_node)

        # Case 2: N, F constant; P, y variable
        elif N_const and F_const:
            return self._make_closure_NF_const(y_node)

        # Case 3: General (all variable)
        else:
            return self._make_closure_general(y_node)

    def _make_closure_NFy_const(self, y_node):
        """
        Optimal case: N, F, y constant; only P varies.

        Precompute: Nmy, NmF, FtNmy, FtNmF, ytNmy, logdetN
        """
        # Evaluate constant nodes (no params needed)
        N_val = self.N.eval()
        F_val = self.F.eval()
        y_val = y_node.eval()

        # Precompute solve operations
        Nmy = self._solve(N_val, y_val)
        NmF = self._solve(N_val, F_val)

        # Precompute inner products
        FtNmy = F_val.T @ Nmy
        FtNmF = F_val.T @ NmF

        # Precompute quadratic form
        ytNmy = y_val @ Nmy

        # Precompute log determinant
        logdetN = self._logdet(N_val)

        # Capture P_node for closure
        P_node = self.P

        print(f"Precomputed for closure:")
        print(f"  ✓ Nmy     (shape {Nmy.shape})")
        print(f"  ✓ NmF     (shape {NmF.shape})")
        print(f"  ✓ FtNmy   (shape {FtNmy.shape})")
        print(f"  ✓ FtNmF   (shape {FtNmF.shape})")
        print(f"  ✓ ytNmy   (scalar: {ytNmy:.4f})")
        print(f"  ✓ logdetN (scalar: {logdetN:.4f})")

        # Create closure
        def kernelproduct(params):
            """Kernel product closure - only depends on P parameters."""
            # Evaluate P (the only variable)
            P_val = P_node.eval(params)

            # Compute Pinv
            if P_val.ndim == 1:
                Pinv = 1.0 / P_val
                logdetP = jnp.sum(jnp.log(P_val))
            else:
                P_factor = jsp.linalg.cho_factor(P_val)
                Pinv = jsp.linalg.cho_solve(P_factor, jnp.eye(P_val.shape[0]))
                logdetP = 2.0 * jnp.sum(jnp.log(jnp.diag(P_factor[0])))

            # Form Schur complement (uses precomputed FtNmF)
            if Pinv.ndim == 1:
                S = jnp.diag(Pinv) + FtNmF
            else:
                S = Pinv + FtNmF

            # Solve Schur (uses precomputed FtNmy)
            S_factor = jsp.linalg.cho_factor(S)
            SmFtNmy = jsp.linalg.cho_solve(S_factor, FtNmy)
            logdetS = 2.0 * jnp.sum(jnp.log(jnp.diag(S_factor[0])))

            # Quadratic form (uses precomputed ytNmy and FtNmy)
            ytWmy = ytNmy - FtNmy @ SmFtNmy

            # Log determinant (uses precomputed logdetN)
            logdet = logdetN - logdetP + logdetS

            # Kernel product
            return -0.5 * ytWmy - 0.5 * logdet

        kernelproduct.params = sorted(self.P.params)
        kernelproduct.precomputed = ["Nmy", "NmF", "FtNmy", "FtNmF", "ytNmy", "logdetN"]

        return kernelproduct

    def _make_closure_NF_const(self, y_node):
        """
        Case: N, F constant; P, y variable.

        Precompute: NmF, FtNmF, logdetN
        Compute in closure: Nmy, FtNmy, ytNmy (depend on y)
        """
        # Precompute N, F dependent quantities
        N_val = self.N.eval()
        F_val = self.F.eval()

        NmF = self._solve(N_val, F_val)
        FtNmF = F_val.T @ NmF
        logdetN = self._logdet(N_val)

        P_node = self.P
        y_node_closure = y_node

        print(f"Precomputed for closure:")
        print(f"  ✓ NmF     (shape {NmF.shape})")
        print(f"  ✓ FtNmF   (shape {FtNmF.shape})")
        print(f"  ✓ logdetN (scalar: {logdetN:.4f})")

        def kernelproduct(params):
            """Kernel product - depends on P and y parameters."""
            # Evaluate P and y
            P_val = P_node.eval(params)
            y_val = y_node_closure.eval(params)

            # Compute y-dependent quantities
            Nmy = self._solve(N_val, y_val)
            FtNmy = F_val.T @ Nmy
            ytNmy = y_val @ Nmy

            # Compute Pinv
            if P_val.ndim == 1:
                Pinv = 1.0 / P_val
                logdetP = jnp.sum(jnp.log(P_val))
            else:
                P_factor = jsp.linalg.cho_factor(P_val)
                Pinv = jsp.linalg.cho_solve(P_factor, jnp.eye(P_val.shape[0]))
                logdetP = 2.0 * jnp.sum(jnp.log(jnp.diag(P_factor[0])))

            # Schur (uses precomputed FtNmF)
            if Pinv.ndim == 1:
                S = jnp.diag(Pinv) + FtNmF
            else:
                S = Pinv + FtNmF

            S_factor = jsp.linalg.cho_factor(S)
            SmFtNmy = jsp.linalg.cho_solve(S_factor, FtNmy)
            logdetS = 2.0 * jnp.sum(jnp.log(jnp.diag(S_factor[0])))

            # Quadratic and logdet
            ytWmy = ytNmy - FtNmy @ SmFtNmy
            logdet = logdetN - logdetP + logdetS

            return -0.5 * ytWmy - 0.5 * logdet

        kernelproduct.params = sorted(self.P.params | y_node.params)
        kernelproduct.precomputed = ["NmF", "FtNmF", "logdetN"]

        return kernelproduct

    def _make_closure_general(self, y_node):
        """General case: compute everything in closure."""

        def kernelproduct(params):
            """Kernel product - depends on all parameters."""
            N_val = self.N.eval(params)
            F_val = self.F.eval(params)
            P_val = self.P.eval(params)
            y_val = y_node.eval(params)

            Nmy = self._solve(N_val, y_val)
            NmF = self._solve(N_val, F_val)
            FtNmy = F_val.T @ Nmy
            FtNmF = F_val.T @ NmF
            ytNmy = y_val @ Nmy

            if P_val.ndim == 1:
                Pinv = 1.0 / P_val
                logdetP = jnp.sum(jnp.log(P_val))
            else:
                P_factor = jsp.linalg.cho_factor(P_val)
                Pinv = jsp.linalg.cho_solve(P_factor, jnp.eye(P_val.shape[0]))
                logdetP = 2.0 * jnp.sum(jnp.log(jnp.diag(P_factor[0])))

            if Pinv.ndim == 1:
                S = jnp.diag(Pinv) + FtNmF
            else:
                S = Pinv + FtNmF

            S_factor = jsp.linalg.cho_factor(S)
            SmFtNmy = jsp.linalg.cho_solve(S_factor, FtNmy)
            logdetS = 2.0 * jnp.sum(jnp.log(jnp.diag(S_factor[0])))

            ytWmy = ytNmy - FtNmy @ SmFtNmy
            logdetN = self._logdet(N_val)
            logdet = logdetN - logdetP + logdetS

            return -0.5 * ytWmy - 0.5 * logdet

        kernelproduct.params = sorted(self.N.params | self.F.params | self.P.params | y_node.params)
        kernelproduct.precomputed = []

        return kernelproduct

    def _solve(self, A, b):
        """Solve A x = b."""
        if A.ndim == 1:
            if b.ndim == 1:
                return b / A
            else:
                return b / A[:, None]
        else:
            A_factor = jsp.linalg.cho_factor(A)
            return jsp.linalg.cho_solve(A_factor, b)

    def _logdet(self, A):
        """Compute log|A|."""
        if A.ndim == 1:
            return jnp.sum(jnp.log(A))
        else:
            A_factor = jsp.linalg.cho_factor(A)
            return 2.0 * jnp.sum(jnp.log(jnp.diag(A_factor[0])))


# ============================================================================
# Node classes for completeness
# ============================================================================


class ConstantNode:
    """Constant data node."""

    def __init__(self, value):
        self.value = jnp.array(value)
        self.is_constant = True
        self.params = set()

    def eval(self, params=None):
        return self.value


class VariableNode:
    """Variable data node."""

    def __init__(self, param_name, transform=None):
        self.param_name = param_name
        self.transform = transform or (lambda x: x)
        self.is_constant = False
        self.params = {param_name}

    def eval(self, params):
        return self.transform(params[self.param_name])


# ============================================================================
# Example
# ============================================================================


def example_closure_with_y():
    """Demonstrate closure creation with y supplied upfront."""
    print("\n" + "=" * 70)
    print("Example: Creating Closure with y Supplied")
    print("=" * 70)

    # Data
    n_data, n_basis = 100, 10
    key = jax.random.PRNGKey(0)

    N_data = jnp.ones(n_data) * 0.5
    F_matrix = jax.random.normal(key, (n_data, n_basis))
    y_data = jax.random.normal(jax.random.PRNGKey(1), (n_data,))

    # Create nodes
    N_node = ConstantNode(N_data)
    F_node = ConstantNode(F_matrix)
    P_node = VariableNode("amplitude", transform=lambda a: jnp.ones(n_basis) * a**2)

    # Create Woodbury structure
    woodbury = WoodburyGraphClosure(N_node, F_node, P_node)

    # Supply y to create closure
    print("\nSupplying y to create kernel product closure...")
    y_node = ConstantNode(y_data)
    kernelproduct = woodbury.make_kernelproduct(y_node)

    print(f"\nClosure parameters: {kernelproduct.params}")
    print(f"Precomputed quantities: {kernelproduct.precomputed}")

    # Use the closure
    print("\n" + "=" * 70)
    print("Using the closure:")
    print("=" * 70)
    for amp in [0.5, 1.0, 2.0]:
        ll = kernelproduct({"amplitude": amp})
        print(f"  amplitude={amp}: loglike={ll:.2f}")

    # JIT compile!
    print("\nJIT compiling...")
    jit_kernelproduct = jax.jit(kernelproduct)
    ll = jit_kernelproduct({"amplitude": 1.5})
    print(f"  JIT result: loglike={ll:.2f}")

    print("\n" + "=" * 70)
    print("Key insight:")
    print("  • y is supplied when CREATING the closure")
    print("  • All y-dependent quantities (Nmy, FtNmy, ytNmy) are precomputed")
    print("  • The closure only depends on P parameters")
    print("  • This matches the current make_kernelproduct(y) pattern!")
    print("=" * 70)


def example_variable_y():
    """Show case where y is variable (less common but possible)."""
    print("\n\n" + "=" * 70)
    print("Example: Variable y (less common)")
    print("=" * 70)

    # Data
    n_data, n_basis = 100, 10
    key = jax.random.PRNGKey(0)

    N_data = jnp.ones(n_data) * 0.5
    F_matrix = jax.random.normal(key, (n_data, n_basis))

    # Create nodes
    N_node = ConstantNode(N_data)
    F_node = ConstantNode(F_matrix)
    P_node = VariableNode("amplitude", transform=lambda a: jnp.ones(n_basis) * a**2)

    # y is variable this time
    def y_transform(params):
        return params["y_data"]

    y_node = VariableNode("y_data", transform=y_transform)

    # Create Woodbury structure
    woodbury = WoodburyGraphClosure(N_node, F_node, P_node)

    # Create closure (y is variable, so less can be precomputed)
    print("\nSupplying variable y to create closure...")
    kernelproduct = woodbury.make_kernelproduct(y_node)

    print(f"\nClosure parameters: {kernelproduct.params}")
    print(f"Precomputed quantities: {kernelproduct.precomputed}")

    print("\nNote: Fewer quantities can be precomputed when y is variable!")
    print("  Precomputed: NmF, FtNmF, logdetN")
    print("  Computed in closure: Nmy, FtNmy, ytNmy (depend on y)")


if __name__ == "__main__":
    show_current_pattern()
    example_closure_with_y()
    example_variable_y()

    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("Closure creation pattern:")
    print("  1. Supply y when creating the closure (not when calling it)")
    print("  2. Precompute all y-dependent quantities if y is constant")
    print("  3. Bake precomputed values into the closure")
    print("  4. Resulting closure only depends on remaining parameters")
    print("\nThis is EXACTLY how current code works with make_kernelproduct(y)!")
    print("=" * 70)
