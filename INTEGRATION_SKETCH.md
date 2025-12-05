# Integration Sketch: likelihood_refactored.py

## Current Architecture (from likelihood.py)

```python
# PulsarLikelihood builds nested WoodburyKernel:
noise = makenoise_measurement(psr, noisedict)  # Returns NoiseMatrix1D_novar or NoiseMatrix1D_var
cgp = makegp_timing(psr)                       # Returns ConstantGP(NoiseMatrix, F)
vgp = makegp_fourier(psr, powerlaw, 10)       # Returns VariableGP(NoiseMatrix, F)

# Nesting happens in PulsarLikelihood.__init__:
csm = matrix.WoodburyKernel(noise, cgp.F, cgp.Phi)   # Old WoodburyKernel auto-selects variant
vsm = matrix.WoodburyKernel(csm, vgp.F, vgp.Phi)     # Nested structure

# Usage:
logL = vsm.make_kernelproduct(y)  # Returns closure
ll = logL(params)                   # Evaluate
```

## Key Observations

1. **NoiseMatrix objects** are passed to WoodburyKernel as N and P arguments
   - `NoiseMatrix1D_novar(noise_array)` - constant diagonal
   - `NoiseMatrix1D_var(getnoise_func)` - variable diagonal
   - `NoiseMatrixSM_novar(N, F, Phi)` - constant Sherman-Morrison
   - etc.

2. **F matrices** from GPs can be:
   - Constant numpy arrays
   - Functions that return arrays: `fmatfunc(params)`

3. **Phi (P) matrices** from GPs are NoiseMatrix objects:
   - `NoiseMatrix12D_var(priorfunc)` - variable diagonal
   - Can also be constant NoiseMatrix

4. **The old WoodburyKernel** dispatcher:
   - Looks at types of N, F, P
   - Selects appropriate variant (WoodburyKernel_novar, _varP, _varN, etc.)
   - Returns the specialized class

## Integration Challenge

The **new unified WoodburyKernel** needs to:
1. Accept the same inputs (NoiseMatrix objects)
2. Handle the same F types (arrays or functions)
3. Provide the same API (`make_kernelproduct`, `make_kernelsolve`, etc.)
4. Work with existing signal creation functions in `signals.py`

## Proposed Approach: Wrapper Strategy

**Option A: Replace old dispatcher with new one**

Keep old WoodburyKernel as a dispatcher that creates the new unified version:

```python
# In matrix.py (or matrix_refactored.py)
def WoodburyKernel(N, F, P):
    """
    Dispatcher that creates unified WoodburyKernel from refactored code.

    Maintains backward compatibility with existing signal creation.
    """
    from .matrix_refactored import WoodburyKernel as UnifiedWoodburyKernel
    return UnifiedWoodburyKernel(N, F, P)
```

Problem: This changes existing matrix.py which may break things.

**Option B: Create parallel likelihood_refactored.py**

```python
# likelihood_refactored.py
from . import matrix_refactored as mr
from . import signals
from . import matrix

class PulsarLikelihood:
    """Refactored PulsarLikelihood using new WoodburyKernel."""

    def __init__(self, *args, concat=True):
        # Same parsing logic as original
        y = [arg for arg in args if isinstance(arg, np.ndarray)]
        noise = [arg for arg in args if isinstance(arg, matrix.Kernel)]
        cgps = [arg for arg in args if isinstance(arg, matrix.ConstantGP)]
        vgps = [arg for arg in args if isinstance(arg, matrix.VariableGP)]

        # ... same setup logic ...

        # KEY DIFFERENCE: Use new unified WoodburyKernel
        if cgps:
            if len(cgps) > 1 and concat:
                cgp = matrix.CompoundGP(cgps)
                csm = mr.WoodburyKernel(noise, cgp.F, cgp.Phi)  # NEW!
            else:
                csm = noise
                for cgp in cgps:
                    csm = mr.WoodburyKernel(csm, cgp.F, cgp.Phi)  # NEW!
        else:
            csm = noise

        # ... rest same ...
```

Problem: Need to ensure `mr.WoodburyKernel` handles NoiseMatrix objects correctly.

## The Core Issue: NoiseMatrix Compatibility

**CRITICAL FINDING:** Variable NoiseMatrix objects don't have `solve_1d` attribute!

```python
# NoiseMatrix1D_novar:
#   - Has solve_1d(y) method that returns (solution, logdet)

# NoiseMatrix1D_var:
#   - Does NOT have solve_1d
#   - Has make_solve_1d() that returns a closure
#   - Has make_kernelproduct(y) that returns a closure
```

**Current _make_leaf detection:**
```python
elif hasattr(spec, 'solve_1d'):
    return OldNoiseMatrixWrapper(spec, name=name)
```

This only catches **constant** NoiseMatrix, not **variable** ones!

**Proposed Fix:**

Update _make_leaf to detect NoiseMatrix objects more broadly:

```python
# Check for NoiseMatrix objects (both constant and variable)
elif hasattr(spec, 'make_kernelproduct'):
    # Both constant and variable NoiseMatrix have this
    return OldNoiseMatrixWrapper(spec, name=name)
```

And update OldNoiseMatrixWrapper to handle both:

```python
class OldNoiseMatrixWrapper(Leaf):
    def __init__(self, noise_matrix, name=None):
        self.noise_matrix = noise_matrix
        self.name = name or "N"

    @property
    def is_constant(self):
        return not hasattr(self.noise_matrix, 'params') or len(self.noise_matrix.params) == 0

    @property
    def params(self):
        if hasattr(self.noise_matrix, 'params'):
            return set(self.noise_matrix.params)
        return set()

    def eval(self, params=None):
        # For constant: can materialize if it's diagonal
        if self.is_constant:
            if hasattr(self.noise_matrix, 'N'):
                return self.noise_matrix.N  # Direct access for constant
        # For variable: cannot materialize (depends on params)
        raise NotImplementedError("Cannot eval() variable NoiseMatrix")

    def solve(self, b, params):
        """Use solve interface - handles both constant and variable."""
        if hasattr(self.noise_matrix, 'solve_1d'):
            # Constant case - direct solve
            solution, _ = self.noise_matrix.solve_1d(b)
            return solution
        elif hasattr(self.noise_matrix, 'make_solve_1d'):
            # Variable case - create solver and use it
            solver = self.noise_matrix.make_solve_1d()
            solution, _ = solver(params)(b)  # Call with params
            return solution
        else:
            raise NotImplementedError(f"NoiseMatrix {type(self.noise_matrix)} has no solve method")
```

Wait, that's not quite right either. Let me think...

## Proposed Implementation Plan

### Phase 1: Verify NoiseMatrix Compatibility
1. Check that `OldNoiseMatrixWrapper` correctly handles:
   - NoiseMatrix1D_novar (has solve_1d)
   - NoiseMatrix1D_var (has make_solve_1d but not solve_1d?)
   - NoiseMatrixSM_novar

2. Add tests: create WoodburyKernel with real NoiseMatrix objects

### Phase 2: Create Simple PulsarLikelihood_Refactored
1. Copy PulsarLikelihood structure
2. Replace `matrix.WoodburyKernel` with `mr.WoodburyKernel`
3. Keep everything else the same
4. Test with simple case (noise + one constant GP)

### Phase 3: Handle Edge Cases
1. Test with variable GPs
2. Test with nested structures
3. Test with delays
4. Verify all methods work: `logL`, `conditional`, `sample`, `clogL`

### Phase 4: Integration with existing signals
1. Verify all `makegp_*` functions work with new likelihood
2. Verify all `makenoise_*` functions work
3. Run full integration tests

## Open Questions

1. **Does NoiseMatrix1D_var have solve_1d?**
   - If not, need to update _make_leaf detection logic

2. **What about NoiseMatrix.make_solve_1d vs solve_1d?**
   - Variable noise has make_solve_1d which returns a closure
   - OldNoiseMatrixWrapper might need to handle this

3. **Do we need to support make_kernelsolve_simple?**
   - Used in `conditional` property
   - Might not exist in new WoodburyKernel yet

4. **Do we need to support make_kernelproduct_gpcomponent?**
   - Used in `clogL` property
   - Might not exist in new WoodburyKernel yet

## Recommendation

**Start minimal:**
1. First verify that `mr.WoodburyKernel` can accept NoiseMatrix objects
2. Create simple test: just noise, no GPs
3. Then add one constant GP
4. Then add one variable GP
5. Then test nesting

**Don't try to implement full PulsarLikelihood_Refactored yet** - just verify the building blocks work.

Would you like me to:
A) First investigate NoiseMatrix compatibility with _make_leaf
B) Create a minimal test showing WoodburyKernel with NoiseMatrix inputs
C) Something else?

## Final Analysis

**The compatibility issue is BIGGER than expected:**

The current `OldNoiseMatrixWrapper.solve()` assumes NoiseMatrix has `solve_1d(b)` method. This works for:
- ✓ NoiseMatrix1D_novar (has solve_1d)
- ✗ NoiseMatrix1D_var (only has make_solve_1d, not solve_1d)

**Current code in matrix_refactored.py:**
```python
def solve(self, b, params):
    """Use old solve interface."""
    if b.ndim == 1:
        solution, _ = self.noise_matrix.solve_1d(b)  # FAILS for variable!
    else:
        solution, _ = self.noise_matrix.solve_2d(b)
    return solution
```

**This WILL FAIL for variable NoiseMatrix** because they don't have `solve_1d` attribute.

## Recommendation: Start Even Simpler

Given these complications, I recommend:

**Step 1: Just test with constant NoiseMatrix**
- Use NoiseMatrix1D_novar only
- Verify that WoodburyKernel(NoiseMatrix1D_novar, F, P) works
- Get one simple likelihood working

**Step 2: Fix variable NoiseMatrix support**  
- Update _make_leaf to detect variable NoiseMatrix differently
- Maybe variable NoiseMatrix shouldn't be wrapped at all?
- Or create a different wrapper class?

**Step 3: Only then create likelihood_refactored.py**

## My Recommendation to You

Before proceeding with likelihood_refactored.py implementation, please decide:

1. Should I first fix OldNoiseMatrixWrapper to handle variable NoiseMatrix?
2. Or should we start with a test using only constant noise?
3. Or take a different approach entirely?

The integration is more complex than I initially thought because of the variable vs constant NoiseMatrix distinction.
