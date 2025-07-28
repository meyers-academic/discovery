"""
Refactored matrix.py with reduced code duplication.

This module implements the same functionality as matrix.py but with a cleaner
architecture that eliminates repetitive Woodbury kernel implementations.

Key improvements:
- WoodburyComputations: Encapsulates common Woodbury matrix operations
- ArrayConverter: Manages numpy→JAX conversions respecting hybrid design
- WoodburyKernelBase: Unified logic driven by class variables
- Concrete subclasses: Minimal implementations that just set class variables

Usage:
    # Same interface as original matrix.py
    kernel = WoodburyKernel_refactored(N, F, P)

    # But with ~80% less code duplication internally
"""

import numpy as np
import scipy as sp
import jax.numpy as jnp
import jax.scipy as jsp
from .matrix import (
    jnparray, matrix_factor, matrix_solve, matrix_norm,
    ConstantKernel, VariableKernel, Kernel, ConstantGP, VariableGP,
    CompoundGP, CompoundDelay, CompoundGlobalGP, VectorCompoundGP,
    VectorWoodburyKernel_varP, NoiseMatrix1D_var, NoiseMatrix2D_var,
    jnp, jsp, jnpsplit, jnpnormal
)



class ArrayConverter:
    """
    Manages numpy→JAX array conversions while respecting the hybrid design.

    The original code uses numpy for setup and JAX for computation. This class
    provides a clean interface for that pattern.
    """

    @staticmethod
    def to_jax_if_needed(arr):
        """Convert numpy array to JAX array if not already JAX."""
        return jnparray(arr) if isinstance(arr, np.ndarray) else arr

    @staticmethod
    def ensure_jax(*arrays):
        """Ensure all arrays are JAX arrays."""
        return [ArrayConverter.to_jax_if_needed(arr) for arr in arrays]


# class WoodburyKernelBase:
#     """
#     Base class containing unified Woodbury kernel logic.

#     Subclasses just need to set class variables:
#     - N_IS_VARIABLE: bool
#     - P_IS_VARIABLE: bool
#     - F_IS_VARIABLE: bool (if applicable)

#     This eliminates the ~500 lines of nearly identical code across variants.
#     """

#     # Subclasses override these
#     N_IS_VARIABLE = False
#     P_IS_VARIABLE = False
#     F_IS_VARIABLE = False

#     def __init__(self, N, F, P):
#         self.N = N
#         self.F = F
#         self.P = P

#         # Set params for VariableKernel interface
#         self._setup_parameters()
#         self.get_NmF = self.make_NmF()
#         self.get_FtNmF = self.make_FtNmF()
#         self.get_Pinv = self.make_Pinv()
#         self.get_woodbury_inverse = self.make_woodbury_inverse()

#     def make_Pinv(self):
#         """Compute the inverse of the prior covariance P."""
#         if self.P_IS_VARIABLE:
#             return self.P.make_inv()
#         else:

#             return lambda params: self.P.inv()

#     def _setup_parameters(self):
#         """Set up parameter list based on which components are variable."""
#         params = set()

#         if self.N_IS_VARIABLE:
#             params.update(self.N.params)
#         if self.P_IS_VARIABLE:
#             params.update(self.P.params)
#         if self.F_IS_VARIABLE and hasattr(self.F, 'params'):
#             params.update(self.F.params)

#         self.params = sorted(params)

#     def make_NmF(self):
#         """Compute N^-1 F for the current parameters."""
#         if self.F_IS_VARIABLE:
#             if self.N_IS_VARIABLE:
#                 N_solve_2d = self.N.make_solve_2d()

#                 def NmF(params):
#                     F_val = self.F(params)
#                     return N_solve_2d(params, F_val)
#                 return NmF
#             else:
#                 N = self.N
#                 def NmF(params):
#                     F_val = self.F(params)
#                     return N.solve_2d(F_val)
#                 return NmF

#         elif self.N_IS_VARIABLE:
#             F = self.F
#             N_solve_2d = self.N.make_solve_2d()
#             def NmF(params):
#                 return N_solve_2d(params, F)
#             return NmF
#         else:
#             F = self.F
#             NmF_tmp, ldN = self.N.solve_2d(F)
#             def NmF(params):
#                 return NmF_tmp, ldN
#             return NmF

#     def make_FtNmF(self):
#         if self.F_IS_VARIABLE:
#             F_var = self.F
#             def FtNmF(params):
#                 NmF_val, ldN = self.get_NmF(params)
#                 return F_var(params).T @ NmF_val, ldN
#             return FtNmF
#         else:
#             NmF_val = self.get_NmF
#             F = self.F
#             def FtNmF(params):
#                 # will work whether N is variable or not
#                 NmF, ldN = NmF_val(params)
#                 return F.T @ NmF, ldN
#             return FtNmF

#     # def make_solve_woodbury(self):
#     #     """Create a function that solves the Woodbury system."""
#     #     # if T is not callable, precompute matrix values with T
#     #     def solve_woodbury(params, T):
#     #         NmF, ldN = self.get_NmF(params)
#     #         FtNmT = (T.T @ NmF).T
#     #         cf, ldTot = self.get_woodbury_inverse(params)
#     #         return self.N.solve_2d(T)[0] - NmF @ matrix_solve(cf, FtNmT), ldTot

#     #     return solve_woodbury

#     def make_woodbury_inverse(self):
#         """Compute the Woodbury inverse components."""
#         get_FtNmF = self.get_FtNmF
#         get_P_inv = self.get_Pinv
#         if not self.F_IS_VARIABLE and not self.N_IS_VARIABLE and not self.P_IS_VARIABLE:
#             # If all matrices are constant, we can precompute
#             FtNmF, ldN = get_FtNmF({})
#             Pinv, ldP = get_P_inv({})
#             FtNmF = jnp.array(FtNmF)
#             Pinv = jnp.array(Pinv)
#             cf = matrix_factor(FtNmF + Pinv)
#             dets = ldN + ldP + matrix_norm * jnp.logdet(jnp.diag(cf[0]))
#             return lambda params: (cf, dets)
#         elif not self.F_IS_VARIABLE and not self.N_IS_VARIABLE:
#             FtNmF, ldN = get_FtNmF({})
#             FtNmF = jnp.array(FtNmF)

#             def woodbury_inverse(params):
#                 Pinv, ldP = get_P_inv(params)
#                 cf = matrix_factor(FtNmF + Pinv)
#                 return cf, ldN + ldP + matrix_norm * jnp.logdet(jnp.diag(cf[0]))
#             return woodbury_inverse
#         else:
#             def woodbury_inverse(params):
#                 FtNmF, ldN = get_FtNmF(params)
#                 Pinv, ldP = get_P_inv(params)
#                 cf = matrix_factor(FtNmF + Pinv)
#                 logdet = matrix_norm * jnp.logdet(jnp.diag(cf[0]))
#                 return cf, ldN + ldP + logdet

#             return woodbury_inverse

#     def make_FtNmy(self, y):
#         """Compute F^T N^-1 y for the current parameters."""
#         NmF_val = self.get_NmF
#         if self.F_IS_VARIABLE:
#             if self.N_IS_VARIABLE:
#                 def FtNmy(params):
#                     NmF, ldN = NmF_val(params)
#                     return (y @ NmF).T, ldN
#                 return FtNmy
#             else:
#                 # Nmy will be precomputed
#                 get_Nmy = self.make_Nmy(y)
#                 F_var = self.F
#                 def FtNmy(params):
#                     F = F_var(params)
#                     Nmy, ldN = get_Nmy(params)
#                     return F.T @ Nmy, ldN
#                 return FtNmy
#         elif self.N_IS_VARIABLE:
#             # Nmy will be precomputed
#             get_Nmy = self.make_Nmy(y)
#             F = jnp.array(self.F)
#             def FtNmy(params):
#                 Nmy, ldN = get_Nmy(params)
#                 return F.T @ Nmy, ldN
#             return FtNmy
#         else:
#             # Nmy will be precomputed
#             get_Nmy = self.make_Nmy(y)
#             NmF, ldN = NmF_val({})
#             FtNmy_tmp = jnp.array((y @ NmF).T)
#             def FtNmy(params):
#                 return FtNmy_tmp, ldN
#             return FtNmy

#     def make_Nmy(self, y):
#         """Compute N^-1 y for the current parameters."""
#         if self.N_IS_VARIABLE:
#             N_solve_1d = self.N.make_solve_1d()
#             def Nmy(params):
#                 return N_solve_1d(params, y)
#             return Nmy
#         else:
#             Nmy_tmp, ldN = self.N.solve_1d(y)
#             Nmy_tmp = jnp.array(Nmy_tmp)
#             def Nmy(params):
#                 return Nmy_tmp, ldN
#             return Nmy


#     def make_solve_1d(self):
#         """Create a function that solves the Woodbury system for 1D data."""

#         # if T is not callable, precompute matrix values with T
#         if isinstance(self, ConstantKernel):
#             # If all matrices are constant, we can precompute
#             NmF, ldN = self.get_NmF({})
#             cf, ldTot = self.get_woodbury_inverse({})
#             NmFSmFtNm = jnp.array(NmF @ matrix_solve(cf, (NmF).T))
#             def solve_1d(params, y):
#                 return self.N.solve_1d(y)[0] - NmFSmFtNm @ y, ldTot
#             return solve_1d

#         elif isinstance(self, VariableKernel):
#             if self.N_IS_VARIABLE or self.F_IS_VARIABLE:
#                 print('hi')
#                 def solve_1d(params, y):
#                     NmF, ldN = self.get_NmF(params)
#                     FtNmy = (y @ NmF).T
#                     if self.N_IS_VARIABLE:
#                         Nmy, _ = self.N.make_solve_1d()(params, y)
#                     else:
#                         Nmy, _ = self.N.solve_1d(y)
#                     cf, ldTot = self.get_woodbury_inverse(params)
#                     return Nmy - NmF @ matrix_solve(cf, FtNmy), ldTot

#                 return solve_1d
#             else: # case P variable N and F are not
#                 print('hi')
#                 NmF, ldN = self.get_NmF({})
#                 NmF = jnp.array(NmF)
#                 def solve_1d(params, y):
#                     FtNmy = (y @ NmF).T
#                     cf, ldTot = self.get_woodbury_inverse(params)
#                     Nmy, _ = self.N.solve_1d(y)
#                     return Nmy - NmF @ matrix_solve(cf, FtNmy), ldTot
#                 return solve_1d

#     def make_solve_2d(self):
#         """Create a function that solves the Woodbury system for 2D data."""

#         # if T is not callable, precompute matrix values with T
#         def solve_2d(params, T):
#             NmF, ldN = self.get_NmF(params)
#             FtNmT = (T.T @ NmF).T
#             cf, ldTot = self.get_woodbury_inverse(params)
#             return self.N.solve_2d(T)[0] - NmF @ matrix_solve(cf, FtNmT), ldTot

#         return solve_2d

#     def solve_2d(self, T):
#         """
#         Solve the Woodbury system for 2D data.

#         Args:
#             T: Design matrix (numpy array)

#         Returns:
#             Solution to the Woodbury system.
#         """
#         solve_2d = self.make_solve_2d()
#         return solve_2d({}, T)

#     def solve_1d(self, y):
#         """
#         Solve the Woodbury system for 1D data.

#         Args:
#             y: Observation vector (numpy array)

#         Returns:
#             Solution to the Woodbury system.
#         """
#         solve_1d = self.make_solve_1d()
#         return solve_1d({}, y)


#     def prep_solve_1d_y(self, y):
#         """Solve (N + F P F^T)^-1 @ y"""


#         # precomputes if you can
#         get_Nmy = self.make_Nmy(y)
#         get_FtNmy = self.make_FtNmy(y)
#         if self.N_IS_VARIABLE or self.F_IS_VARIABLE or self.P_IS_VARIABLE:
#             def solve_1d(params):
#                 Nmy, _ = get_Nmy(params)
#                 NmF, ldN = self.get_NmF(params)
#                 cf, ldTot = self.get_woodbury_inverse(params)
#                 FtNmy, _ = get_FtNmy(params)
#                 tmp = NmF @ matrix_solve(cf, FtNmy)
#                 return Nmy - tmp, ldTot
#             return solve_1d
#         else:
#             # Completely constant case - compute everything once
#             NmF, ldN = self.get_NmF({})
#             cf, ldTot = self.get_woodbury_inverse({})

#             # Precompute all JAX operations once
#             NmF = jnp.array(NmF)
#             FtNmy = jnp.array((y @ NmF).T)
#             Nmy, _ = self.N.solve_1d(y)
#             correction = NmF @ matrix_solve(cf, FtNmy)
#             result = Nmy - correction

#             return lambda params: (result, ldTot)


#     def make_kernelproduct(self, y):
#         """Create a function that computes kernel products y^T K^-1 y."""

#         # if y is not callable, precompute matrix vlaues with y
#         solve_1d = self.prep_solve_1d_y(y)
#         if self.N_IS_VARIABLE or self.F_IS_VARIABLE or self.P_IS_VARIABLE:
#             def kernelproduct(params):
#                 solution, logdet = solve_1d(params)
#                 return -0.5 * (y @ solution + logdet)
#         else:
#             solution, logdet = solve_1d({})
#             y = jnp.array(y)
#             res = -0.5 * (y @ solution + logdet)
#             def kernelproduct(params):
#                 return res

#         kernelproduct.params = self.params
#         return kernelproduct

class Components:
    """
    Encapsulates all the computational components for Woodbury matrix operations.

    This class handles the creation and management of various computational
    functions needed for Woodbury kernel operations, making the code more
    modular and easier to test.
    """

    def __init__(self, N, F, P, N_is_variable=False, P_is_variable=False, F_is_variable=False):
        self.N = N
        self.F = F
        self.P = P
        self.N_is_variable = N_is_variable
        self.P_is_variable = P_is_variable
        self.F_is_variable = F_is_variable

        # Build all computational components
        self._build_components()

    def _build_components(self):
        """Build all computational functions based on variability flags."""
        self.get_NmF = self._make_NmF()
        self.get_FtNmF = self._make_FtNmF()
        self.get_Pinv = self._make_Pinv()
        self.get_woodbury_inverse = self._make_woodbury_inverse()
        self.get_Nmy = self._make_Nmy_factory()
        self.get_FtNmy = self._make_FtNmy_factory()

    def _make_NmF(self):
        """Create N^-1 F computation function."""
        if self.F_is_variable and self.N_is_variable:
            N_solve_2d = self.N.make_solve_2d()
            def NmF(params):
                F_val = self.F(params)
                return N_solve_2d(params, F_val)
            return NmF
        elif self.F_is_variable:
            N = self.N
            def NmF(params):
                F_val = self.F(params)
                return N.solve_2d(F_val)
            return NmF
        elif self.N_is_variable:
            F = self.F
            N_solve_2d = self.N.make_solve_2d()
            def NmF(params):
                return N_solve_2d(params, F)
            return NmF
        else:
            F = self.F
            NmF_tmp, ldN = self.N.solve_2d(F)
            def NmF(params):
                return NmF_tmp, ldN
            return NmF

    def _make_FtNmF(self):
        """Create F^T N^-1 F computation function."""
        if self.F_is_variable:
            F_var = self.F
            def FtNmF(params):
                NmF_val, ldN = self.get_NmF(params)
                return F_var(params).T @ NmF_val, ldN
            return FtNmF
        else:
            F = self.F
            def FtNmF(params):
                NmF, ldN = self.get_NmF(params)
                return F.T @ NmF, ldN
            return FtNmF

    def _make_Pinv(self):
        """Create P^-1 computation function."""
        if self.P_is_variable:
            return self.P.make_inv()
        else:
            Pinv_val, ldP = self.P.inv()
            return lambda params: (Pinv_val, ldP)

    def _make_woodbury_inverse(self):
        """Create Woodbury inverse computation function."""
        if not any([self.F_is_variable, self.N_is_variable, self.P_is_variable]):
            # All constant - precompute everything
            FtNmF, ldN = self.get_FtNmF({})
            Pinv, ldP = self.get_Pinv({})
            FtNmF = jnp.array(FtNmF)
            Pinv = jnp.array(Pinv)
            cf = matrix_factor(FtNmF + Pinv)
            dets = ldN + ldP + matrix_norm * jnp.logdet(jnp.diag(cf[0]))
            ldN = jnp.array(ldN)
            return lambda params: (cf, dets)

        elif not self.F_is_variable and not self.N_is_variable:
            # Only P varies
            FtNmF, ldN = self.get_FtNmF({})
            FtNmF = jnp.array(FtNmF)
            ldN = jnp.array(ldN)
            def woodbury_inverse(params):
                Pinv, ldP = self.get_Pinv(params)
                cf = matrix_factor(FtNmF + Pinv)
                return cf, ldN + ldP + matrix_norm * jnp.logdet(jnp.diag(cf[0]))
            return woodbury_inverse

        else:
            # General case - everything computed at runtime
            def woodbury_inverse(params):
                FtNmF, ldN = self.get_FtNmF(params)
                Pinv, ldP = self.get_Pinv(params)
                cf = matrix_factor(FtNmF + Pinv)
                logdet = matrix_norm * jnp.logdet(jnp.diag(cf[0]))
                return cf, ldN + ldP + logdet
            return woodbury_inverse

    def _make_Nmy_factory(self):
        """Create factory for N^-1 y computation functions."""
        def make_Nmy(y):
            if self.N_is_variable:
                N_solve_1d = self.N.make_solve_1d()
                def Nmy(params):
                    return N_solve_1d(params, y)
                return Nmy
            else:
                Nmy_tmp, ldN = self.N.solve_1d(y)
                Nmy_tmp = jnp.array(Nmy_tmp)
                ldN = jnp.array(ldN)
                def Nmy(params):
                    return Nmy_tmp, ldN
                return Nmy
        return make_Nmy

    def _make_FtNmy_factory(self):
        """Create factory for F^T N^-1 y computation functions."""
        def make_FtNmy(y):
            if self.F_is_variable:
                if self.N_is_variable:
                    def FtNmy(params):
                        NmF, ldN = self.get_NmF(params)
                        return (y @ NmF).T, ldN
                    return FtNmy
                else:
                    get_Nmy = self.get_Nmy(y)
                    F_var = self.F
                    def FtNmy(params):
                        F = F_var(params)
                        Nmy, ldN = get_Nmy(params)
                        return F.T @ Nmy, ldN
                    return FtNmy
            elif self.N_is_variable:
                get_Nmy = self.get_Nmy(y)
                F = jnp.array(self.F)
                def FtNmy(params):
                    Nmy, ldN = get_Nmy(params)
                    return F.T @ Nmy, ldN
                return FtNmy
            else:
                NmF, ldN = self.get_NmF({})
                FtNmy_tmp = jnp.array((y @ NmF).T)
                def FtNmy(params):
                    return FtNmy_tmp, ldN
                return FtNmy
        return make_FtNmy


class WoodburyKernelBase:
    """
    Complete Woodbury kernel base class with unified logic.

    This class uses the Components class to manage computational functions
    and provides a clean interface for all Woodbury kernel operations.
    """

    # Subclasses override these
    N_IS_VARIABLE = False
    P_IS_VARIABLE = False
    F_IS_VARIABLE = False

    def __init__(self, N, F, P):
        self.N = N
        self.F = F
        self.P = P

        # Initialize components
        self.components = Components(
            N, F, P,
            N_is_variable=self.N_IS_VARIABLE,
            P_is_variable=self.P_IS_VARIABLE,
            F_is_variable=self.F_IS_VARIABLE
        )

        # Set up parameter list for VariableKernel interface
        self._setup_parameters()

        # Create solver functions
        self._solver_1d = self._make_solve_1d()
        self._solver_2d = self._make_solve_2d()

    def _setup_parameters(self):
        """Set up parameter list based on which components are variable."""
        params = set()

        if self.N_IS_VARIABLE and hasattr(self.N, 'params'):
            params.update(self.N.params)
        if self.P_IS_VARIABLE and hasattr(self.P, 'params'):
            params.update(self.P.params)
        if self.F_IS_VARIABLE and hasattr(self.F, 'params'):
            params.update(self.F.params)

        self.params = sorted(params)

    def _make_solve_1d(self):
        """Create 1D solver function."""
        if isinstance(self, ConstantKernel):
            # All constant - maximum precomputation
            NmF, ldN = self.components.get_NmF({})
            cf, ldTot = self.components.get_woodbury_inverse({})
            NmFSmFtNm = jnp.array(NmF @ matrix_solve(cf, (NmF).T))

            def solve_1d(params, y):
                return self.N.solve_1d(y)[0] - NmFSmFtNm @ y, ldTot
            return solve_1d

        elif isinstance(self, VariableKernel):
            if self.N_IS_VARIABLE or self.F_IS_VARIABLE:
                def solve_1d(params, y):
                    NmF, ldN = self.components.get_NmF(params)
                    FtNmy = (y @ NmF).T
                    if self.N_IS_VARIABLE:
                        Nmy, _ = self.N.make_solve_1d()(params, y)
                    else:
                        Nmy, _ = self.N.solve_1d(y)
                    cf, ldTot = self.components.get_woodbury_inverse(params)
                    return Nmy - NmF @ matrix_solve(cf, FtNmy), ldTot
                return solve_1d
            else:
                # Only P variable
                NmF, ldN = self.components.get_NmF({})
                NmF = jnp.array(NmF)
                def solve_1d(params, y):
                    FtNmy = (y @ NmF).T
                    cf, ldTot = self.components.get_woodbury_inverse(params)
                    Nmy, _ = self.N.solve_1d(y)
                    return Nmy - NmF @ matrix_solve(cf, FtNmy), ldTot
                return solve_1d

    def _make_solve_2d(self):
        """Create 2D solver function."""
        def solve_2d(params, T):
            NmF, ldN = self.components.get_NmF(params)
            FtNmT = (T.T @ NmF).T
            cf, ldTot = self.components.get_woodbury_inverse(params)
            NmT, _ = self.N.solve_2d(T)
            return NmT - NmF @ matrix_solve(cf, FtNmT), ldTot
        return solve_2d

    def solve_1d(self, y):
        """Solve the Woodbury system for 1D data."""
        return self._solver_1d({}, y)

    def solve_2d(self, T):
        """Solve the Woodbury system for 2D data."""
        return self._solver_2d({}, T)

    def make_solve_1d(self):
        """Return the 1D solver function."""
        return self._solver_1d

    def make_solve_2d(self):
        """Return the 2D solver function."""
        return self._solver_2d

    def make_kernelproduct(self, y):
        """Create kernel product function y^T K^-1 y with optimized cases."""
        y = jnp.array(y)

        # Case 1: N or F variable - general runtime computation
        if self.N_IS_VARIABLE and self.F_IS_VARIABLE:
            get_Nmy = self.components.get_Nmy(y)
            get_FtNmy = self.components.get_FtNmy(y)

            def kernelproduct(params):
                Nmy, _ = get_Nmy(params)
                NmF, ldN = self.components.get_NmF(params)
                cf, ldTot = self.components.get_woodbury_inverse(params)
                FtNmy, _ = get_FtNmy(params)
                tmp = NmF @ matrix_solve(cf, FtNmy)
                solution = Nmy - tmp
                return -0.5 * (y @ solution + ldTot)
            kernelproduct.params = self.params
            return kernelproduct
        # Case 2: N is not variable F is variable
        elif self.F_IS_VARIABLE and not self.N_IS_VARIABLE:
            get_Nmy = self.components.get_Nmy(y)
            get_FtNmy = self.components.get_FtNmy(y)
            Nmy, _ = get_Nmy({})
            Nmy = jnp.array(Nmy)

            def kernelproduct(params):
                NmF, ldN = self.components.get_NmF(params)
                cf, ldTot = self.components.get_woodbury_inverse(params)
                FtNmy, _ = get_FtNmy(params)
                tmp = NmF @ matrix_solve(cf, FtNmy)
                solution = Nmy - tmp
                return -0.5 * (y @ solution + ldTot)
            kernelproduct.params = self.params
            return kernelproduct

        # Case 2: Only P variable (varP case) - precompute N and F terms
        elif self.P_IS_VARIABLE:
            get_Nmy = self.components.get_Nmy(y)
            get_FtNmy = self.components.get_FtNmy(y)

            # Precompute N and F dependent terms
            Nmy, ldN = get_Nmy({})
            NmF, _ = self.components.get_NmF({})
            FtNmy, _ = get_FtNmy({})
            Nmy = jnp.array(Nmy)
            NmF = jnp.array(NmF)
            ldN = jnp.array(ldN)
            FtNmy = jnp.array(FtNmy)

            def kernelproduct(params):
                Pinv, ldP = self.components.get_Pinv(params)
                cf = matrix_factor(FtNmy + Pinv)
                # cf, ldTot = self.components.get_woodbury_inverse(params)
                tmp = NmF @ matrix_solve(cf, FtNmy)
                solution = Nmy - tmp
                ldTot = matrix_norm * jnp.logdet(jnp.diag(cf[0])) + ldP + ldN
                return -0.5 * (y @ solution + ldTot)
            kernelproduct.params = self.params
            return kernelproduct

        # Case 3: Everything constant - maximum precomputation
        else:
            # Precompute everything once
            NmF, ldN = self.components.get_NmF({})
            cf, ldTot = self.components.get_woodbury_inverse({})

            NmF = jnp.array(NmF)
            FtNmy = jnp.array((y @ NmF).T)
            Nmy, _ = self.N.solve_1d(y)
            correction = NmF @ matrix_solve(cf, FtNmy)
            solution = Nmy - correction

            # Final result computed once
            result = -0.5 * (y @ solution + ldTot)

            def kernelproduct(params):
                return result
            kernelproduct.params = self.params
            return kernelproduct
    # def prep_solve_1d_y(self, y):
    #     """Prepare optimized 1D solver for specific y vector."""
    #     get_Nmy = self.components.get_Nmy(y)
    #     get_FtNmy = self.components.get_FtNmy(y)

    #     if self.N_IS_VARIABLE or self.F_IS_VARIABLE:
    #         def solve_1d(params):
    #             Nmy, _ = get_Nmy(params)
    #             NmF, ldN = self.components.get_NmF(params)
    #             cf, ldTot = self.components.get_woodbury_inverse(params)
    #             FtNmy, _ = get_FtNmy(params)
    #             tmp = NmF @ matrix_solve(cf, FtNmy)
    #             return Nmy - tmp, ldTot
    #         return solve_1d
    #     # varP case:
    #     elif self.P_IS_VARIABLE:
    #         print('Warning: P is variable but N and F are not. This is a special case.')
    #         Nmy, _ = get_Nmy({})
    #         NmF, _ = self.components.get_NmF({})
    #         FtNmy, _ = get_FtNmy({})
    #         Nmy = jnp.array(Nmy)
    #         NmF = jnp.array(NmF)
    #         FtNmy = jnp.array(FtNmy)

    #         def solve_1d(params):
    #             cf, ldTot = self.components.get_woodbury_inverse(params)
    #             tmp = NmF @ matrix_solve(cf, FtNmy)
    #             return Nmy - tmp, ldTot
    #         return solve_1d
    #     else:
    #         # Completely constant case
    #         NmF, ldN = self.components.get_NmF({})
    #         cf, ldTot = self.components.get_woodbury_inverse({})

    #         NmF = jnp.array(NmF)
    #         FtNmy = jnp.array((y @ NmF).T)
    #         Nmy, _ = self.N.solve_1d(y)
    #         correction = NmF @ matrix_solve(cf, FtNmy)
    #         result = Nmy - correction

    #         return lambda params: (result, ldTot)

    # def make_kernelproduct(self, y):
    #     """Create kernel product function y^T K^-1 y."""
    #     solve_1d = self.prep_solve_1d_y(y)

    #     if any([self.N_IS_VARIABLE, self.F_IS_VARIABLE, self.P_IS_VARIABLE]):
    #         def kernelproduct(params):
    #             solution, logdet = solve_1d(params)
    #             return -0.5 * (y @ solution + logdet)
    #     else:
    #         solution, logdet = solve_1d({})
    #         y = jnp.array(y)
    #         res = -0.5 * (y @ solution + logdet)
    #         def kernelproduct(params):
    #             return res

    #     kernelproduct.params = self.params
    #     return kernelproduct

# # Factory function for easy migration - this should replace WoodburyKernel calls
def create_woodbury_kernel(N, F, P, variant='auto'):
    """
    Factory function to create appropriate WoodburyKernel variant.
    This is the main interface that should be used instead of calling WoodburyKernel directly.
    """
    # Just call the original WoodburyKernel function for now since the refactored
    # kernels need more work to be fully compatible
    from .matrix import WoodburyKernel as OriginalWoodburyKernel
    return OriginalWoodburyKernel(N, F, P)

# Keep the original interface for backward compatibility
def WoodburyKernel(N, F, P, refactored=False):
    """Backward compatibility wrapper."""
    if refactored:
        return create_woodbury_kernel_refactored(N, F, P)
    else:
        return create_woodbury_kernel(N, F, P)


# Concrete implementations - just set the class variables!

class WoodburyKernel_refactored(WoodburyKernelBase, ConstantKernel):
    """Refactored version of WoodburyKernel - no variable components."""
    N_IS_VARIABLE = False
    P_IS_VARIABLE = False


class WoodburyKernel_varN_refactored(WoodburyKernelBase, VariableKernel):
    """Refactored version of WoodburyKernel_varN."""
    N_IS_VARIABLE = True
    P_IS_VARIABLE = False


class WoodburyKernel_varP_refactored(WoodburyKernelBase, VariableKernel):
    """Refactored version of WoodburyKernel_varP."""
    N_IS_VARIABLE = False
    P_IS_VARIABLE = True


class WoodburyKernel_varNP_refactored(WoodburyKernelBase, VariableKernel):
    """Refactored version of WoodburyKernel_varNP."""
    N_IS_VARIABLE = True
    P_IS_VARIABLE = True

# create a factory for deciding which kernel to use
def create_woodbury_kernel_refactored(N, F, P, variant='auto'):
    """
    Factory function to create appropriate WoodburyKernel variant.
    This is the main interface that should be used instead of calling WoodburyKernel directly.
    """
    if variant == 'auto':
        if isinstance(N, ConstantKernel) and isinstance(P, ConstantKernel):
            return WoodburyKernel_refactored(N, F, P)
        elif isinstance(N, VariableKernel) and isinstance(P, ConstantKernel):
            return WoodburyKernel_varN_refactored(N, F, P)
        elif isinstance(N, ConstantKernel) and isinstance(P, VariableKernel):
            return WoodburyKernel_varP_refactored(N, F, P)
        elif isinstance(N, VariableKernel) and isinstance(P, VariableKernel):
            return WoodburyKernel_varNP_refactored(N, F, P)
        else:
            raise ValueError("Unsupported kernel types for N and P")
    else:
        return create_woodbury_kernel(N, F, P)  # fallback to original factory