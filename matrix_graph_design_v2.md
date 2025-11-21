# Matrix Graph Refactoring Design v2: Woodbury Solve Operations

## Overview
Redesigned to focus on what we actually need: **solving systems** using the Woodbury identity, not building matrices.

## Key Insight

We don't need to compute N + F^T P F. We need:
1. **Solve**: (N + F^T P F)^{-1} y
2. **Log-determinant**: log|N + F^T P F|

The Woodbury identity gives us these **without forming the full matrix**:

```
(N + F^T P F)^{-1} y = N^{-1} y - N^{-1} F (S^{-1}) F^T N^{-1} y

where S = P^{-1} + F^T N^{-1} F  (Schur complement)

log|N + F^T P F| = log|N| - log|P| + log|S|
```

When N is itself Woodbury, we **recursively** apply the identity.

## Revised Design

### Core Abstraction: WoodburyStructure

A WoodburyStructure represents the **structure** (N, F, P), not the matrix itself.
Its main operation is `solve(y, params)` which returns the solution and log-determinant.

```python
class WoodburyStructure:
    """Represents the structure (N, F, P) and can solve with (N + F^T P F)."""

    def __init__(self, N, F, P):
        """
        Args:
            N: Either a base matrix node OR another WoodburyStructure (for nesting)
            F: Design matrix node
            P: Prior matrix node
        """
        self.N = N
        self.F = F
        self.P = P

    @property
    def is_nested(self):
        """True if N is also a WoodburyStructure."""
        return isinstance(self.N, WoodburyStructure)

    def depth(self):
        """Nesting depth."""
        if isinstance(self.N, WoodburyStructure):
            return 1 + self.N.depth()
        return 1

    def solve(self, y, params):
        """
        Solve (N + F^T P F)^{-1} y using Woodbury identity.

        Returns:
            solution: (N + F^T P F)^{-1} y
            logdet: log|N + F^T P F|
        """
        # Evaluate nodes to get actual matrices
        F_val = self.F.eval(params)
        P_val = self.P.eval(params)

        # Step 1: Solve with N (recursively if needed)
        if isinstance(self.N, WoodburyStructure):
            # Recursive case: N is also Woodbury
            Ninv_y, logdet_N = self.N.solve(y.eval(params), params)
            Ninv_F = self.N.solve_multi(F_val, params)  # N^{-1} F for multiple RHS
        else:
            # Base case: N is simple
            N_val = self.N.eval(params)
            Ninv_y, logdet_N = self._solve_base(N_val, y.eval(params))
            Ninv_F, _ = self._solve_base(N_val, F_val)

        # Step 2: Form Schur complement S = P^{-1} + F^T N^{-1} F
        FtNinvF = F_val.T @ Ninv_F

        if P_val.ndim == 1:
            # Diagonal P
            S = jnp.diag(1.0 / P_val) + FtNinvF
            logdet_P = jnp.sum(jnp.log(P_val))
        else:
            # Dense P
            P_factor = jsp.linalg.cho_factor(P_val)
            P_inv = jsp.linalg.cho_solve(P_factor, jnp.eye(P_val.shape[0]))
            S = P_inv + FtNinvF
            logdet_P = 2.0 * jnp.sum(jnp.log(jnp.diag(P_factor[0])))

        # Step 3: Solve S α = F^T N^{-1} y
        S_factor = jsp.linalg.cho_factor(S)
        logdet_S = 2.0 * jnp.sum(jnp.log(jnp.diag(S_factor[0])))

        FtNinvy = F_val.T @ Ninv_y
        alpha = jsp.linalg.cho_solve(S_factor, FtNinvy)

        # Step 4: Woodbury correction
        solution = Ninv_y - Ninv_F @ alpha

        # Step 5: Total log-determinant
        logdet = logdet_N - logdet_P + logdet_S

        return solution, logdet

    def _solve_base(self, N, y):
        """Solve with base (non-Woodbury) N."""
        if N.ndim == 1:
            # Diagonal
            solution = y / N
            logdet = jnp.sum(jnp.log(N))
        else:
            # Dense
            N_factor = jsp.linalg.cho_factor(N)
            solution = jsp.linalg.cho_solve(N_factor, y)
            logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(N_factor[0])))
        return solution, logdet

    def make_solve_closure(self, y):
        """
        Create an optimized closure for solving.

        This is where optimization happens based on constant/variable structure.
        """
        # Determine what's constant
        N_const = getattr(self.N, 'is_constant', False) if not isinstance(self.N, WoodburyStructure) else False
        F_const = self.F.is_constant
        P_const = self.P.is_constant
        y_const = y.is_constant

        # Pre-evaluate constant parts
        if F_const:
            F_val = self.F.eval(None)
        if P_const:
            P_val = self.P.eval(None)

        # Create optimized closure based on what varies
        if F_const and P_const:
            # Most common: F and P fixed, N and y vary
            # Can precompute some structure
            def closure(params):
                return self._solve_optimized_FP_const(y, params, F_val, P_val)
        else:
            # General case
            def closure(params):
                return self.solve(y, params)

        closure.params = sorted(self.params)
        return closure

    @property
    def params(self):
        """All parameters needed by this structure."""
        params = set()
        if isinstance(self.N, WoodburyStructure):
            params.update(self.N.params)
        elif hasattr(self.N, 'params'):
            params.update(self.N.params)
        params.update(self.F.params)
        params.update(self.P.params)
        return params
```

### Graph Pruning

We need to track which computations are actually used and eliminate unnecessary ones.

```python
class ComputationGraph:
    """Manages the computational graph and performs optimization."""

    def __init__(self):
        self.nodes = []
        self.dependencies = {}

    def add_node(self, node):
        """Add a node to the graph."""
        self.nodes.append(node)
        return node

    def mark_output(self, node):
        """Mark a node as an output (must be computed)."""
        node._is_output = True

    def prune(self):
        """
        Remove nodes that don't contribute to any output.

        This is a simple reachability analysis:
        1. Start from output nodes
        2. Mark all dependencies as needed
        3. Remove unmarked nodes
        """
        needed = set()

        # Find all output nodes
        outputs = [n for n in self.nodes if getattr(n, '_is_output', False)]

        # DFS to mark needed nodes
        def mark_needed(node):
            if node in needed:
                return
            needed.add(node)
            for dep in getattr(node, 'dependencies', []):
                mark_needed(dep)

        for output in outputs:
            mark_needed(output)

        # Remove unneeded nodes
        self.nodes = [n for n in self.nodes if n in needed]

        return len(self.nodes)

    def optimize_constants(self):
        """
        Pre-evaluate constant subgraphs.

        If a node and all its dependencies are constant, replace it
        with a ConstantNode containing the pre-evaluated result.
        """
        optimized_count = 0

        for i, node in enumerate(self.nodes):
            if isinstance(node, ConstantNode):
                continue

            if node.is_constant and not getattr(node, '_is_output', False):
                # Pre-evaluate and replace with constant
                value = node.eval(None)
                self.nodes[i] = ConstantNode(value)
                optimized_count += 1

        return optimized_count
```

## Revised Usage Examples

### Example 1: Simple Woodbury Solve

```python
# Data
N_data = jnp.ones(100)  # white noise
F_matrix = jnp.array(jax.random.normal(jax.random.PRNGKey(0), (100, 10)))
y_data = jnp.array(jax.random.normal(jax.random.PRNGKey(1), (100,)))

# Build structure
N_node = ConstantNode(N_data)
F_node = ConstantNode(F_matrix)
P_node = VariableNode('amplitude', transform=lambda a: jnp.ones(10) * a**2)
y_node = ConstantNode(y_data)

woodbury = WoodburyStructure(N_node, F_node, P_node)

# Solve
params = {'amplitude': 1.5}
solution, logdet = woodbury.solve(y_node, params)

# Or create a closure (can be JIT compiled)
solve_closure = woodbury.make_solve_closure(y_node)
jit_solve = jax.jit(solve_closure)
solution, logdet = jit_solve(params)
```

### Example 2: Nested Woodbury (Key Use Case!)

```python
# This is the critical case: (N_base + F_inner^T P_inner F_inner) + F_outer^T P_outer F_outer

# Base white noise
N_base_node = ConstantNode(jnp.ones(100) * 0.1)

# Inner Woodbury: N = N_base + F_inner^T P_inner F_inner
F_inner_node = ConstantNode(F_inner)
P_inner_node = VariableNode('inner_amp', transform=lambda a: jnp.ones(5) * a**2)
N_woodbury = WoodburyStructure(N_base_node, F_inner_node, P_inner_node)

# Outer Woodbury: full = N_woodbury + F_outer^T P_outer F_outer
F_outer_node = ConstantNode(F_outer)
P_outer_node = VariableNode('outer_amp', transform=lambda a: jnp.ones(10) * a**2)
full_woodbury = WoodburyStructure(N_woodbury, F_outer_node, P_outer_node)

print(f"Nesting depth: {full_woodbury.depth()}")  # 2
print(f"Is nested: {full_woodbury.is_nested}")  # True

# Solve - automatically handles recursive Woodbury
y_node = ConstantNode(y_data)
params = {'inner_amp': 1.0, 'outer_amp': 2.0}
solution, logdet = full_woodbury.solve(y_node, params)

# The solve() method automatically:
# 1. Solves with inner Woodbury recursively
# 2. Uses that to solve the outer Woodbury
# 3. Never forms the full matrix!
```

### Example 3: Kernel Product (Log-Likelihood)

```python
class WoodburyKernelProduct:
    """Computes -0.5 y^T (N + F^T P F)^{-1} y - 0.5 log|N + F^T P F|."""

    def __init__(self, woodbury_structure, y):
        self.woodbury = woodbury_structure
        self.y = y

    def eval(self, params):
        """Evaluate the kernel product."""
        solution, logdet = self.woodbury.solve(self.y, params)
        y_val = self.y.eval(params)
        return -0.5 * jnp.dot(y_val, solution) - 0.5 * logdet

    def make_closure(self):
        """Create optimized closure."""
        solve_closure = self.woodbury.make_solve_closure(self.y)
        y_closure = self.y.make_closure() if not self.y.is_constant else None

        if self.y.is_constant:
            y_val = self.y.eval(None)
            def closure(params):
                solution, logdet = solve_closure(params)
                return -0.5 * jnp.dot(y_val, solution) - 0.5 * logdet
        else:
            def closure(params):
                solution, logdet = solve_closure(params)
                y_val = y_closure(params)
                return -0.5 * jnp.dot(y_val, solution) - 0.5 * logdet

        closure.params = sorted(self.woodbury.params | self.y.params)
        return closure


# Usage
woodbury = WoodburyStructure(N_node, F_node, P_node)
y_node = ConstantNode(y_data)

loglike = WoodburyKernelProduct(woodbury, y_node)
loglike_closure = loglike.make_closure()

# Can be JIT compiled
jit_loglike = jax.jit(loglike_closure)
value = jit_loglike({'amplitude': 1.5})
```

### Example 4: Graph Optimization

```python
# Build a complex structure
graph = ComputationGraph()

# ... add many nodes ...

# Mark what we actually need
graph.mark_output(loglike_node)

# Optimize
print(f"Nodes before pruning: {len(graph.nodes)}")
graph.prune()
print(f"Nodes after pruning: {len(graph.nodes)}")

print(f"Constants before optimization: {sum(isinstance(n, ConstantNode) for n in graph.nodes)}")
graph.optimize_constants()
print(f"Constants after optimization: {sum(isinstance(n, ConstantNode) for n in graph.nodes)}")
```

## Key Differences from v1

1. **Focus on solve, not matrix construction**: WoodburyStructure has a `solve()` method, not an `eval()` that builds the matrix

2. **Simpler node types**: We still have ConstantNode, VariableNode for data, but the main abstraction is WoodburyStructure

3. **Automatic recursion**: When N is a WoodburyStructure, solve() automatically recurses

4. **Graph pruning**: Explicitly handle removing unused intermediate computations

5. **Clearer optimization**: Pre-evaluate constant subexpressions, optimize based on what's variable

## Benefits

1. **Never form full matrices**: Use Woodbury identity throughout
2. **Natural recursion**: Nested structures handled automatically
3. **Cleaner abstraction**: Focus on the operation (solve) not the object (matrix)
4. **Easier optimization**: Prune unused nodes, pre-evaluate constants
5. **Still eliminates duplication**: One WoodburyStructure replaces many _var variants

## Implementation Strategy

1. **Phase 1**: Implement WoodburyStructure with solve() for non-nested case
2. **Phase 2**: Add recursive solving for nested structures
3. **Phase 3**: Add optimization (closure generation based on constant/variable)
4. **Phase 4**: Add graph pruning and constant folding
5. **Phase 5**: Migrate existing code progressively
6. **Phase 6**: Add specialized fast paths (Sherman-Morrison, etc.)

This design is much more aligned with what the code actually needs!
