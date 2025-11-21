# Matrix Graph Refactoring Design

## Overview
Design for refactoring matrix.py to use a computational graph structure that handles constant/variable components and nested Woodbury structures.

## Mathematical Structure

We need to solve: `(N + F^T P F)^{-1}` where:
- N can itself be composite: `N = N_2 + F_2^T P_2 F_2`
- Full nested form: `(N_2 + F_2^T P_2 F_2 + F^T P F)^{-1}`
- Solved via recursive Woodbury identity

## Core Design

### 1. Base Node Classes

```python
class Node:
    """Base class for all computational graph nodes."""

    def __init__(self):
        self.dependencies = []
        self._is_constant = None
        self._params = None

    @property
    def is_constant(self):
        """True if this node and all dependencies are constant."""
        if self._is_constant is None:
            self._is_constant = all(dep.is_constant for dep in self.dependencies)
        return self._is_constant

    @property
    def params(self):
        """Set of all parameter names needed by this node."""
        if self._params is None:
            self._params = set()
            for dep in self.dependencies:
                self._params.update(dep.params)
        return self._params

    def eval(self, params=None):
        """Evaluate the node given parameters."""
        raise NotImplementedError

    def make_closure(self):
        """Create optimized closure based on constant/variable structure."""
        raise NotImplementedError


class ConstantNode(Node):
    """Leaf node with constant value."""

    def __init__(self, value):
        super().__init__()
        self.value = jnparray(value)
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
        return closure


class VariableNode(Node):
    """Leaf node with parameter-dependent value."""

    def __init__(self, param_name, transform=None):
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
        return closure


class FunctionNode(Node):
    """Leaf node with custom function of multiple parameters."""

    def __init__(self, func, param_names):
        super().__init__()
        self.func = func
        self._params = set(param_names)
        self._is_constant = False

    def eval(self, params):
        return self.func(params)

    def make_closure(self):
        """Return the function itself."""
        func = self.func
        def closure(params):
            return func(params)
        closure.params = sorted(self._params)
        return closure
```

### 2. Operation Nodes

```python
class TransposeNode(Node):
    """Transpose of a matrix node."""

    def __init__(self, matrix):
        super().__init__()
        self.matrix = matrix
        self.dependencies = [matrix]

    def eval(self, params=None):
        return self.matrix.eval(params).T

    def make_closure(self):
        if self.is_constant:
            value = self.eval(None).T
            def closure(params=None):
                return value
            closure.params = []
        else:
            mat_closure = self.matrix.make_closure()
            def closure(params):
                return mat_closure(params).T
            closure.params = sorted(self.params)
        return closure


class MatmulNode(Node):
    """Matrix multiplication of two nodes."""

    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right
        self.dependencies = [left, right]

    def eval(self, params=None):
        return self.left.eval(params) @ self.right.eval(params)

    def make_closure(self):
        if self.is_constant:
            value = self.eval(None)
            def closure(params=None):
                return value
            closure.params = []
        elif self.left.is_constant:
            left_val = self.left.eval(None)
            right_closure = self.right.make_closure()
            def closure(params):
                return left_val @ right_closure(params)
            closure.params = sorted(self.params)
        elif self.right.is_constant:
            left_closure = self.left.make_closure()
            right_val = self.right.eval(None)
            def closure(params):
                return left_closure(params) @ right_val
            closure.params = sorted(self.params)
        else:
            left_closure = self.left.make_closure()
            right_closure = self.right.make_closure()
            def closure(params):
                return left_closure(params) @ right_closure(params)
            closure.params = sorted(self.params)
        return closure


class SumNode(Node):
    """Sum of multiple matrix nodes."""

    def __init__(self, *nodes):
        super().__init__()
        self.nodes = nodes
        self.dependencies = list(nodes)

    def eval(self, params=None):
        return sum(node.eval(params) for node in self.nodes)

    def make_closure(self):
        if self.is_constant:
            value = self.eval(None)
            def closure(params=None):
                return value
            closure.params = []
        else:
            # Separate constant and variable nodes
            const_sum = sum(node.eval(None) for node in self.nodes if node.is_constant)
            var_closures = [node.make_closure() for node in self.nodes if not node.is_constant]

            if const_sum != 0:
                def closure(params):
                    return const_sum + sum(cl(params) for cl in var_closures)
            else:
                def closure(params):
                    return sum(cl(params) for cl in var_closures)
            closure.params = sorted(self.params)
        return closure


class DiagNode(Node):
    """Diagonal matrix from vector or extract diagonal."""

    def __init__(self, vector):
        super().__init__()
        self.vector = vector
        self.dependencies = [vector]

    def eval(self, params=None):
        v = self.vector.eval(params)
        return jnp.diag(v) if v.ndim == 1 else jnp.diag(v)

    def make_closure(self):
        if self.is_constant:
            value = self.eval(None)
            def closure(params=None):
                return value
            closure.params = []
        else:
            vec_closure = self.vector.make_closure()
            def closure(params):
                v = vec_closure(params)
                return jnp.diag(v) if v.ndim == 1 else jnp.diag(v)
            closure.params = sorted(self.params)
        return closure
```

### 3. Woodbury Structure Node

```python
class WoodburyNode(Node):
    """Represents N + F^T P F structure.

    This is the core structure for Gaussian process covariance matrices.
    N can itself be another WoodburyNode, allowing nesting.
    """

    def __init__(self, N, F, P):
        """
        Args:
            N: Base noise/covariance matrix (can be WoodburyNode for nesting)
            F: Design matrix
            P: Prior covariance matrix
        """
        super().__init__()
        self.N = N
        self.F = F
        self.P = P
        self.dependencies = [N, F, P]

    def eval(self, params=None):
        """Evaluate full matrix: N + F^T P F"""
        N_val = self.N.eval(params)
        F_val = self.F.eval(params)
        P_val = self.P.eval(params)

        # Handle 1D (diagonal) vs 2D cases
        if N_val.ndim == 1:
            N_mat = jnp.diag(N_val)
        else:
            N_mat = N_val

        if P_val.ndim == 1:
            P_mat = jnp.diag(P_val)
        else:
            P_mat = P_val

        return N_mat + F_val.T @ P_mat @ F_val

    def make_closure(self):
        """Create closure for computing N + F^T P F."""
        if self.is_constant:
            value = self.eval(None)
            def closure(params=None):
                return value
            closure.params = []
        else:
            N_closure = self.N.make_closure()
            F_closure = self.F.make_closure()
            P_closure = self.P.make_closure()

            # Optimize based on what's constant
            n_const, f_const, p_const = self.N.is_constant, self.F.is_constant, self.P.is_constant

            if f_const and p_const:
                F_val = self.F.eval(None)
                P_val = self.P.eval(None)
                FtPF = F_val.T @ (P_val @ F_val if P_val.ndim == 2 else jnp.diag(P_val) @ F_val)
                def closure(params):
                    return N_closure(params) + FtPF
            elif n_const and p_const:
                N_val = self.N.eval(None)
                P_val = self.P.eval(None)
                P_mat = P_val if P_val.ndim == 2 else jnp.diag(P_val)
                def closure(params):
                    F_val = F_closure(params)
                    return N_val + F_val.T @ (P_mat @ F_val)
            else:
                def closure(params):
                    N_val = N_closure(params)
                    F_val = F_closure(params)
                    P_val = P_closure(params)
                    P_mat = P_val if P_val.ndim == 2 else jnp.diag(P_val)
                    return N_val + F_val.T @ (P_mat @ F_val)

            closure.params = sorted(self.params)
        return closure

    @property
    def is_nested(self):
        """True if N is also a WoodburyNode."""
        return isinstance(self.N, WoodburyNode)

    def depth(self):
        """Nesting depth of Woodbury structures."""
        if isinstance(self.N, WoodburyNode):
            return 1 + self.N.depth()
        return 1
```

### 4. Woodbury Solver Node

```python
class WoodburySolveNode(Node):
    """Solves (N + F^T P F)^{-1} y using Woodbury identity.

    Woodbury: (A + UCV)^{-1} = A^{-1} - A^{-1}U(C^{-1} + VA^{-1}U)^{-1}VA^{-1}
    Our case: (N + F^T P F)^{-1} = N^{-1} - N^{-1}F^T(P^{-1} + F N^{-1} F^T)^{-1} F N^{-1}

    If N itself is Woodbury, we solve recursively.
    """

    def __init__(self, woodbury, rhs=None):
        """
        Args:
            woodbury: WoodburyNode representing the matrix
            rhs: Right-hand side (y). If None, computes inverse properties only
        """
        super().__init__()
        self.woodbury = woodbury
        self.rhs = rhs
        self.dependencies = [woodbury] + ([rhs] if rhs else [])

    def eval(self, params=None):
        """Solve the system (returns solution and log-determinant)."""
        return self._solve(params)

    def _solve(self, params):
        """Recursive Woodbury solver."""
        N = self.woodbury.N
        F = self.woodbury.F.eval(params)
        P = self.woodbury.P.eval(params)
        y = self.rhs.eval(params) if self.rhs else None

        # Base case: N is simple (diagonal or dense)
        if not isinstance(N, WoodburyNode):
            N_val = N.eval(params)
            return self._solve_base(N_val, F, P, y)

        # Recursive case: N is itself Woodbury
        # Create a solver for N
        N_solver = WoodburySolveNode(N, self.rhs)
        N_inv_y, N_logdet = N_solver.eval(params)

        # Now apply Woodbury with N^{-1} computed recursively
        return self._solve_with_N_inv(N_solver, F, P, y, params)

    def _solve_base(self, N, F, P, y):
        """Base Woodbury solver when N is simple."""
        # Handle diagonal N
        if N.ndim == 1:
            N_inv_y = y / N if y is not None else None
            N_inv_F = F / N[:, None]
            N_logdet = jnp.sum(jnp.log(N))
        else:
            N_factor = matrix_factor(N)
            N_inv_y = matrix_solve(N_factor, y) if y is not None else None
            N_inv_F = matrix_solve(N_factor, F)
            N_logdet = jnp.sum(jnp.log(jnp.abs(jnp.diag(N_factor[0]))))

        # Schur complement: S = P^{-1} + F^T N^{-1} F
        FtNinvF = F.T @ N_inv_F

        if P.ndim == 1:
            S = jnp.diag(1.0 / P) + FtNinvF
            P_logdet = jnp.sum(jnp.log(P))
        else:
            P_factor = matrix_factor(P)
            P_inv = matrix_solve(P_factor, jnp.eye(P.shape[0]))
            S = P_inv + FtNinvF
            P_logdet = jnp.sum(jnp.log(jnp.abs(jnp.diag(P_factor[0]))))

        S_factor = matrix_factor(S)
        S_logdet = jnp.sum(jnp.log(jnp.abs(jnp.diag(S_factor[0]))))

        # Total log-determinant
        total_logdet = N_logdet - P_logdet + S_logdet

        if y is None:
            return None, total_logdet

        # Woodbury correction: y_solve = N^{-1}y - N^{-1}F S^{-1} F^T N^{-1}y
        FtNinvy = F.T @ N_inv_y
        correction = matrix_solve(S_factor, FtNinvy)
        y_solve = N_inv_y - N_inv_F @ correction

        return y_solve, total_logdet

    def _solve_with_N_inv(self, N_solver, F, P, y, params):
        """Apply Woodbury when we have N^{-1} from recursive solver."""
        # This delegates to the recursive N_solver
        # Implementation would be similar to _solve_base but using N_solver
        pass

    def make_closure(self):
        """Create optimized closure for solving."""
        if self.is_constant:
            solution, logdet = self.eval(None)
            def closure(params=None):
                return solution, logdet
            closure.params = []
        else:
            # Create closures for components
            # The actual implementation would dispatch to optimized
            # solver based on constant/variable structure
            def closure(params):
                return self._solve(params)
            closure.params = sorted(self.params)
        return closure


class WoodburyKernelProductNode(Node):
    """Computes -0.5 * y^T (N + F^T P F)^{-1} y - 0.5 * log|N + F^T P F|."""

    def __init__(self, woodbury, y):
        super().__init__()
        self.woodbury = woodbury
        self.y = y
        self.solver = WoodburySolveNode(woodbury, y)
        self.dependencies = [woodbury, y]

    def eval(self, params=None):
        """Compute kernel product (log-likelihood contribution)."""
        y_val = self.y.eval(params)
        solution, logdet = self.solver.eval(params)

        return -0.5 * jnp.dot(y_val, solution) - 0.5 * logdet

    def make_closure(self):
        """Create closure for kernel product."""
        if self.is_constant:
            value = self.eval(None)
            def closure(params=None):
                return value
            closure.params = []
        else:
            solver_closure = self.solver.make_closure()
            y_closure = self.y.make_closure()

            def closure(params):
                y_val = y_closure(params)
                solution, logdet = solver_closure(params)
                return -0.5 * jnp.dot(y_val, solution) - 0.5 * logdet

            closure.params = sorted(self.params)
        return closure
```

## Usage Examples

### Example 1: Simple Woodbury Structure

```python
# Simple case: (N + F^T P F)^{-1} where all are constant
N_node = ConstantNode(N_data)  # diagonal noise
F_node = ConstantNode(F_matrix)  # design matrix
P_node = ConstantNode(P_prior)  # prior covariance

woodbury = WoodburyNode(N_node, F_node, P_node)
y_node = ConstantNode(y_data)

# Create solver
solver = WoodburySolveNode(woodbury, y_node)
solution, logdet = solver.eval()

# Create kernel product (log-likelihood)
kernel_product = WoodburyKernelProductNode(woodbury, y_node)
loglike = kernel_product.eval()
```

### Example 2: Variable Prior

```python
# N and F constant, but P depends on parameters
N_node = ConstantNode(N_data)
F_node = ConstantNode(F_matrix)

# P is a function of amplitude parameters
def P_func(params):
    amplitudes = jnp.array([params[f'amp_{i}'] for i in range(n_components)])
    return jnp.diag(amplitudes**2)
P_node = FunctionNode(P_func, [f'amp_{i}' for i in range(n_components)])

woodbury = WoodburyNode(N_node, F_node, P_node)
y_node = ConstantNode(y_data)

# Create likelihood function
kernel_product = WoodburyKernelProductNode(woodbury, y_node)
loglike_closure = kernel_product.make_closure()

# Use in inference
params = {'amp_0': 1.0, 'amp_1': 2.0, 'amp_2': 0.5}
loglike_value = loglike_closure(params)
```

### Example 3: Nested Woodbury (N itself is Woodbury)

```python
# Inner Woodbury: N = N_base + F_inner^T P_inner F_inner
N_base_node = ConstantNode(N_base)  # base white noise
F_inner_node = ConstantNode(F_inner)  # inner design matrix
P_inner_node = VariableNode('inner_amp', transform=lambda a: jnp.diag(jnp.full(n_inner, a**2)))

N_node = WoodburyNode(N_base_node, F_inner_node, P_inner_node)

# Outer Woodbury: full = N + F_outer^T P_outer F_outer
F_outer_node = ConstantNode(F_outer)
P_outer_node = VariableNode('outer_amp', transform=lambda a: jnp.diag(jnp.full(n_outer, a**2)))

woodbury_nested = WoodburyNode(N_node, F_outer_node, P_outer_node)

# Check nesting
print(f"Is nested: {woodbury_nested.is_nested}")  # True
print(f"Depth: {woodbury_nested.depth()}")  # 2

# Solver automatically handles recursion
y_node = ConstantNode(y_data)
solver = WoodburySolveNode(woodbury_nested, y_node)
solution, logdet = solver.eval({'inner_amp': 1.5, 'outer_amp': 2.0})
```

### Example 4: Multiple Pulsars (Vector Case)

```python
# Each pulsar has its own data and design matrix
n_pulsars = 10

N_nodes = [ConstantNode(N_data[i]) for i in range(n_pulsars)]
F_nodes = [ConstantNode(F_matrices[i]) for i in range(n_pulsars)]
y_nodes = [ConstantNode(y_data[i]) for i in range(n_pulsars)]

# Shared prior (e.g., common red noise)
P_node = VariableNode('rn_amp', transform=lambda a: jnp.diag(jnp.full(n_freqs, a**2)))

# Create Woodbury for each pulsar
woodburys = [WoodburyNode(N_nodes[i], F_nodes[i], P_node) for i in range(n_pulsars)]

# Total log-likelihood is sum over pulsars
kernel_products = [WoodburyKernelProductNode(woodburys[i], y_nodes[i])
                   for i in range(n_pulsars)]

# Combine into single likelihood function
class SumNode(Node):
    def __init__(self, nodes):
        super().__init__()
        self.nodes = nodes
        self.dependencies = nodes

    def eval(self, params=None):
        return sum(node.eval(params) for node in self.nodes)

total_loglike_node = SumNode(kernel_products)
loglike_closure = total_loglike_node.make_closure()

# Evaluate
params = {'rn_amp': 1.5}
total_loglike = loglike_closure(params)
```

## Migration Strategy

1. **Phase 1**: Implement core node classes (Constant, Variable, Function)
2. **Phase 2**: Implement operation nodes (Transpose, Matmul, Sum, Diag)
3. **Phase 3**: Implement WoodburyNode and base solver (non-nested)
4. **Phase 4**: Implement recursive solver for nested structures
5. **Phase 5**: Add specialized optimizations for common patterns
6. **Phase 6**: Migrate existing code to use graph structure
7. **Phase 7**: Deprecate old classes once migration complete

## Benefits

1. **Eliminates duplication**: Single implementation handles constant/variable cases
2. **Automatic optimization**: Closures are optimized based on graph structure
3. **Composability**: Easy to build complex nested structures
4. **Parameter tracking**: Automatic detection of required parameters
5. **Recursive handling**: Natural support for nested Woodbury structures
6. **Testability**: Each node type can be tested independently
7. **Extensibility**: Easy to add new operation types

## Next Steps

1. Implement prototype of core classes
2. Test on simple examples from current codebase
3. Benchmark performance vs current implementation
4. Iterate on design based on findings
5. Begin gradual migration
