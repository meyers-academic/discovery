# Woodbury Operations: What We Actually Need to Compute

## The Woodbury Identity

To solve `(N + F^T P F)^{-1} y`, we use:

```
(N + F^T P F)^{-1} y = N^{-1} y - N^{-1} F (P^{-1} + F^T N^{-1} F)^{-1} F^T N^{-1} y
                     = Nmy - NmF S^{-1} FtNmy
```

where `S = P^{-1} + F^T N^{-1} F` is the Schur complement.

## Fundamental Intermediate Quantities

These are the **atomic operations** that should be the graph nodes:

### 1. Solve Operations (N^{-1} applied to vectors/matrices)

- **`Nmy = N^{-1} y`** - Solve N with vector y
  - Constant if: N constant AND y constant
  - Used in: final solution, kernel product

- **`NmF = N^{-1} F`** - Solve N with matrix F (multiple RHS)
  - Constant if: N constant AND F constant
  - Used in: Schur complement, final correction

### 2. Inner Products (F^T applied to solved quantities)

- **`FtNmy = F^T N^{-1} y`** - Inner product of F with solved y
  - Constant if: F constant AND Nmy constant (i.e., N, F, y all constant)
  - Used in: Schur solve, kernel product

- **`FtNmF = F^T N^{-1} F`** - Gram matrix of F under N^{-1}
  - Constant if: F constant AND NmF constant (i.e., N, F both constant)
  - Used in: Schur complement

### 3. Quadratic Forms (for log-likelihood)

- **`ytNmy = y^T N^{-1} y`** - Quadratic form of y under N^{-1}
  - Constant if: y constant AND Nmy constant (i.e., N, y both constant)
  - Used in: kernel product (log-likelihood)

### 4. Schur Complement Operations

- **`Pinv = P^{-1}`** - Inverse of prior
  - Constant if: P constant
  - Used in: Schur complement

- **`S = P^{-1} + F^T N^{-1} F = Pinv + FtNmF`** - Schur complement
  - Constant if: Pinv constant AND FtNmF constant
  - Used in: Schur solve, determinant

- **`Sm(FtNmy) = S^{-1} (F^T N^{-1} y)`** - Solve Schur with FtNmy
  - Used in: final correction

### 5. Final Correction

- **`correction = N^{-1} F S^{-1} F^T N^{-1} y = NmF @ Sm(FtNmy)`**
  - Used in: Woodbury solution

### 6. Log Determinants

- **`logdetN = log|N|`** - Log determinant of N
  - Constant if: N constant

- **`logdetP = log|P|`** - Log determinant of P
  - Constant if: P constant

- **`logdetS = log|S|`** - Log determinant of Schur
  - Depends on: S

- **`logdet = log|N + F^T P F| = logdetN - logdetP + logdetS`**

## Graph Structure

The graph nodes should represent these intermediate quantities, not the full solve operation.

```python
# Leaf nodes (data)
N_node = ConstantNode(N_data)      # or VariableNode
F_node = ConstantNode(F_matrix)    # or VariableNode
P_node = VariableNode('amplitude') # or ConstantNode
y_node = ConstantNode(y_data)      # or VariableNode

# Level 1: Solve operations
Nmy_node = SolveNode(N_node, y_node)          # N^{-1} y
NmF_node = SolveNode(N_node, F_node)          # N^{-1} F

# Level 2: Inner products
FtNmy_node = InnerProductNode(F_node, Nmy_node)    # F^T N^{-1} y
FtNmF_node = InnerProductNode(F_node, NmF_node)    # F^T N^{-1} F
ytNmy_node = QuadFormNode(y_node, Nmy_node)        # y^T N^{-1} y

# Level 3: Schur complement
Pinv_node = InvertNode(P_node)                     # P^{-1}
S_node = SumNode(Pinv_node, FtNmF_node)            # P^{-1} + F^T N^{-1} F

# Level 4: Schur solve
SmFtNmy_node = SolveNode(S_node, FtNmy_node)       # S^{-1} F^T N^{-1} y

# Level 5: Final quantities
correction_node = MatmulNode(NmF_node, SmFtNmy_node)    # N^{-1} F S^{-1} F^T N^{-1} y
solution_node = SubtractNode(Nmy_node, correction_node)  # N^{-1} y - correction

# Kernel product
logdetN_node = LogDetNode(N_node)
logdetP_node = LogDetNode(P_node)
logdetS_node = LogDetNode(S_node)
logdet_node = CombineLogDetNode([logdetN_node, logdetP_node, logdetS_node])

quad_node = InnerProductNode(y_node, solution_node)     # y^T solution
kernel_product = KernelProductNode(quad_node, logdet_node)  # -0.5 * (quad + logdet)
```

## Optimization Based on Constant/Variable Structure

### Case 1: N, F, y all constant, only P varies

**Precompute outside closure:**
- Nmy (constant)
- NmF (constant)
- FtNmy (constant)
- FtNmF (constant)
- ytNmy (constant)
- logdetN (constant)

**Compute inside closure (depends on P):**
- Pinv
- S = Pinv + FtNmF  (only addition with precomputed FtNmF)
- SmFtNmy = S^{-1} FtNmy  (solve with precomputed FtNmy)
- correction = NmF @ SmFtNmy  (matmul with precomputed NmF)
- solution = Nmy - correction  (subtract from precomputed Nmy)
- logdetP
- logdetS
- logdet = logdetN - logdetP + logdetS

This is the **most common case** in GP inference!

### Case 2: N, F constant, P, y variable

**Precompute outside closure:**
- NmF (constant)
- FtNmF (constant)
- logdetN (constant)

**Compute inside closure:**
- Nmy (depends on y)
- FtNmy (depends on y)
- ytNmy (depends on y)
- Pinv (depends on P)
- S (depends on P)
- SmFtNmy (depends on P, y)
- correction (depends on P, y)
- solution (depends on P, y)
- logdetP (depends on P)
- logdetS (depends on P)

### Case 3: Only N constant

**Precompute outside closure:**
- logdetN (constant)

**Compute inside closure:**
- Everything else (all depend on F, P, or y)

### Case 4: All variable

**Precompute outside closure:**
- Nothing

**Compute inside closure:**
- Everything

## Nested Woodbury

When N itself is Woodbury: `N = N_base + F_inner^T P_inner F_inner`

Then `N^{-1} y` is itself a Woodbury solve, which recursively computes its own intermediate quantities.

The graph becomes:
```
# Inner Woodbury (for N)
N_base_node = ...
F_inner_node = ...
P_inner_node = ...

# These create a sub-graph for solving with N
Nmy_subgraph = WoodburySolveGraph(N_base, F_inner, P_inner, y)
# This expands to its own Nmy, NmF, FtNmy, etc. for the inner solve

# Outer Woodbury uses the inner solve
Nmy_node = Nmy_subgraph.solution
NmF_node = WoodburySolveGraph(N_base, F_inner, P_inner, F_outer).solution

# Rest proceeds as before with outer F, P
...
```

## Key Design Principles

1. **Nodes represent intermediate quantities**, not full solve operations
2. **Each node knows when it can be precomputed** (all inputs constant)
3. **Graph structure makes dependencies explicit**
4. **Pruning removes unused intermediates** (e.g., ytNmy not needed for solve, only for kernel product)
5. **Nested structures create sub-graphs** that are recursively optimized

## Example: Optimal Closure Generation

```python
def make_kernel_product_closure(N, F, P, y):
    """Generate optimized closure for kernel product."""

    # Determine what's constant
    N_const = N.is_constant
    F_const = F.is_constant
    P_const = P.is_constant
    y_const = y.is_constant

    if N_const and F_const and y_const:
        # Most common: only P varies
        # Precompute everything involving N, F, y
        Nmy = solve(N.eval(), y.eval())
        NmF = solve(N.eval(), F.eval())
        FtNmy = F.eval().T @ Nmy
        FtNmF = F.eval().T @ NmF
        ytNmy = y.eval() @ Nmy
        logdetN = logdet(N.eval())

        def closure(params):
            P_val = P.eval(params)
            Pinv = invert(P_val)
            S = Pinv + FtNmF  # precomputed FtNmF
            SmFtNmy = solve(S, FtNmy)  # precomputed FtNmy
            correction = NmF @ SmFtNmy  # precomputed NmF
            ytWmy = ytNmy - FtNmy @ SmFtNmy  # precomputed ytNmy, FtNmy

            logdetP = logdet(P_val)
            logdetS = logdet(S)
            logdetW = logdetN - logdetP + logdetS  # precomputed logdetN

            return -0.5 * ytWmy - 0.5 * logdetW

        closure.params = P.params

    elif N_const and F_const:
        # N, F constant; P, y variable
        NmF = solve(N.eval(), F.eval())
        FtNmF = F.eval().T @ NmF
        logdetN = logdet(N.eval())

        def closure(params):
            y_val = y.eval(params)
            P_val = P.eval(params)

            Nmy = solve(N.eval(), y_val)
            FtNmy = F.eval().T @ Nmy
            ytNmy = y_val @ Nmy

            Pinv = invert(P_val)
            S = Pinv + FtNmF  # precomputed FtNmF
            SmFtNmy = solve(S, FtNmy)
            correction = NmF @ SmFtNmy  # precomputed NmF
            ytWmy = ytNmy - FtNmy @ SmFtNmy

            logdetP = logdet(P_val)
            logdetS = logdet(S)
            logdetW = logdetN - logdetP + logdetS  # precomputed logdetN

            return -0.5 * ytWmy - 0.5 * logdetW

        closure.params = sorted(P.params | y.params)

    else:
        # General case: compute everything
        def closure(params):
            N_val = N.eval(params)
            F_val = F.eval(params)
            P_val = P.eval(params)
            y_val = y.eval(params)

            Nmy = solve(N_val, y_val)
            NmF = solve(N_val, F_val)
            FtNmy = F_val.T @ Nmy
            FtNmF = F_val.T @ NmF
            ytNmy = y_val @ Nmy

            Pinv = invert(P_val)
            S = Pinv + FtNmF
            SmFtNmy = solve(S, FtNmy)
            correction = NmF @ SmFtNmy
            ytWmy = ytNmy - FtNmy @ SmFtNmy

            logdetN = logdet(N_val)
            logdetP = logdet(P_val)
            logdetS = logdet(S)
            logdetW = logdetN - logdetP + logdetS

            return -0.5 * ytWmy - 0.5 * logdetW

        closure.params = sorted(N.params | F.params | P.params | y.params)

    return closure
```

This approach **explicitly shows which quantities can be precomputed** based on the constant/variable structure!
