#!/usr/bin/env python3
"""
Implements some rudimentary time domain kernels,
similar to enterprise extensions:
https://github.com/nanograv/enterprise_extensions/blob/master/enterprise_extensions/gp_kernels.py

THESE ARE CURRENTLY ONLY IMPLEMENTED TO BE USED FOR SINGLE PULSARS
(i.e. they can't be used to define a time-domain kernel that implements
correlations between pulsars).
"""
import numpy as np
from . import signals
from . import matrix
import inspect


def linear_interp_basis(toas, dt=30 * 86400):
    """
    COPIED FROM ENTERPRISE
    Provides a basis for linear interpolation.

    :param toas: Pulsar TOAs in seconds
    :param dt: Linear interpolation step size in seconds.

    :returns: Linear interpolation basis and nodes
    """

    # evenly spaced points
    x = np.arange(toas.min(), toas.max() + dt, dt)
    M = np.zeros((len(toas), len(x)))

    # make linear interpolation basis
    for ii in range(len(x) - 1):
        idx = np.logical_and(toas >= x[ii], toas <= x[ii + 1])
        M[idx, ii] = (toas[idx] - x[ii + 1]) / (x[ii] - x[ii + 1])
        M[idx, ii + 1] = (toas[idx] - x[ii]) / (x[ii + 1] - x[ii])

    # only return non-zero columns
    idx = M.sum(axis=0) != 0

    return M[:, idx], x[idx]

def linear_interp_basis_dm(toas, freqs, dt=30*86400):
    """
    Linear interpolation basis for dispersion measure

    Parameters:
    -----------
    toas : array-like
        Pulsar TOAs in seconds
    freqs: array-like
        Radio frequency associated with each toa
    dt : float, optional, default=30 days
        time over which to coarse-grain time-domain kernel

    Returns:
    --------
    Udm : array-like
        interpolation basis matrix, scaled by radio frequencies
    avetoas : array-like
        times at which interpolation basis matrix is evaluated
        (evenly spaced from first toa to last toa, incremented by dt)
    """

    # get linear interpolation basis in time
    U, avetoas = linear_interp_basis(toas, dt=dt)

    # scale with radio frequency
    Dm = (1400/freqs)**2

    return U * Dm[:, None], avetoas


def makegp_time_domain_all_toas(psr, kernelfunc, name='td_gp_full'):
    """
    """
    DM = (1400/psr.freqs)**2
    Fmat = jnp.eye(psr.toas.size) *  DM[:, None]# design matrix
    argspec = inspect.getfullargspec(kernelfunc)
    argmap = [f'{psr.name}_{name}_{arg}' for arg in argspec.args if arg not in ['avetoas', 'freqs']]
    toas = matrix.jnparray(psr.toas)
    def getphi(params):
        return kernelfunc(toas, *[params[arg] for arg in argmap])
    getphi.params = argmap
    gp = matrix.VariableGP(matrix.NoiseMatrix2D_var(getphi), Fmat)
    gp.name = psr.name
    gp.pos = psr.pos
    gp.gpname = 'td'
    return gp

def makegp_time_domain_linear_interplation(psr, kernelfunc,
                                           name='td_gp_lin_interp',
                                           dt=86400 * 30):
    """
    Create a linear interpolation time domain GP, similar to
    how it is defined in enterprise.

    Parameters:
    -----------
    psr: discovery.Pulsar
        pulsar object
    kernelfunc : Callable
        prior funciton that defines the kernel. Should be a Callable
        with inputs given by a list of times to use to calculate delays
        at which to evaluate kernel.
    name : str
        name of this model
    dt : float, optional, default=30 days
        coarse-graining for linear interpolation model.

    """
    Umat, x = linear_interp_basis_dm(psr.toas, psr.freqs, dt=dt)

    argspec = inspect.getfullargspec(kernelfunc)
    argmap = [f'{psr.name}_{name}_{arg}' for arg in argspec.args if arg not in ['avetoas', 'freqs']]
    x = matrix.jnparray(x)
    def getphi(params):
        return kernelfunc(x, *[params[arg] for arg in argmap])
    getphi.params = argmap
    gp = matrix.VariableGP(matrix.NoiseMatrix2D_var(getphi), Umat)
    gp.name = psr.name
    gp.pos = psr.pos
    gp.gpname = 'td'
    return gp


# TODO: I assume these can be optimized so they don't
# have to create new "d" matrices every time.

def periodic_kernel(avetoas, log10_sigma=-7., log10_ell=2,
                    log10_gam_p=0, log10_p=0):
    """Quasi-periodic kernel for DM"""
    r = jnp.abs(avetoas[None, :] - avetoas[:, None])

    # convert units to seconds
    sigma = 10**log10_sigma
    l = 10**log10_ell * 86400
    p = 10**log10_p * 3.16e7
    gam_p = 10**log10_gam_p
    d = jnp.eye(r.shape[0]) * (sigma/500)**2
    K = sigma**2 * jnp.exp(-r**2/2/l**2 - gam_p*jnp.sin(jnp.pi*r/p)**2) + d
    return K

def se_kernel(avefreqs, log10_sigma=-7, log10_lam=3):
    """Squared-exponential kernel for FD"""
    tm = jnp.abs(avefreqs[None, :] - avefreqs[:, None])

    lam = 10**log10_lam
    sigma = 10**log10_sigma
    d = jnp.eye(tm.shape[0]) * (sigma/500)**2
    return sigma**2 * jnp.exp(-tm**2/2/lam) + d

def se_dm_kernel(avetoas, log10_sigma=-7, log10_ell=2):
    """Squared-exponential kernel for DM"""
    r = jnp.abs(avetoas[None, :] - avetoas[:, None])

    # Convert everything into seconds
    l = 10**log10_ell * 86400
    sigma = 10**log10_sigma
    d = jnp.eye(r.shape[0]) * (sigma/500)**2
    K = sigma**2 * np.exp(-r**2/2/l**2) + d
    return K

def dmx_ridge_prior(avetoas, log10_sigma=-7):
    """DMX-like signal with Gaussian prior"""
    sigma = 10**log10_sigma
    return sigma**2 * jnp.ones_like(avetoas)
