import numpy as np
import jax.numpy as jnp

import discovery
from .. import signals
from .. import utils
from .. import _kernels as kernels

# Likelihood classes are looked up on `discovery` at call time, so they follow
# ds.config(kernels=...).

priordict_standard = {
    "(.*_)?efac": [0.9, 1.1],
    "(.*_)?t2equad": [-8.5, -5],
    "(.*_)?tnequad": [-8.5, -5],
    "(.*_)?log10_ecorr": [-8.5, -5],
    "(.*_)?rednoise_log10_A.*": [-18, -10],
    "(.*_)?rednoise_gamma.*": [0, 7],
    "(.*_)?red_noise_log10_A.*": [-18, -10],  # deprecated
    "(.*_)?red_noise_gamma.*": [0, 7],  # deprecated
    "crn_log10_A.*": [-18, -11],
    "crn_gamma.*": [0, 7],
    "gw_(.*_)?log10_A": [-18, -11],
    "gw_(.*_)?gamma": [0, 7],
    "(.*_)?gp_log10_A": [-18, -10],
    "(.*_)?gp_gamma": [0, 7],
    "(.*_)?gp_alpha": [1, 3],
    "crn_log10_rho": [-9, -4],
    "gw_(.*_)?log10_rho": [-9, -4],
    r"(.*_)?red_noise_log10_rho\(([0-9]*)\)": [-9, -4],
    r"(.*_)?red_noise_crn_log10_rho\(([0-9]*)\)": [-9, -4],
    "(.*_)?log10_Amp": [-10, -2],
    "(.*_)?log10_tau": [0, 2.5],
    "(.*_)?2_t0": [54650, 54850],
    "(.*_)?1_t0": [57490, 57530]
}

# merge per-pulsar GPs into a single CommonGP; takes one list of GPs per pulsar,
# stacked along the coefficient axis (padded to the widest pulsar)
# experimental, will be moved to signals.py
def gps2commongp(gplists):
    priors = [[gp.Phi.getN for gp in gplist] for gplist in gplists]
    n_psr = len(gplists)

    Fs = [np.hstack([gp.F for gp in gplist]) for gplist in gplists]
    n_coeffs = [F.shape[1] for F in Fs]
    n_coeff_max = max(n_coeffs)

    def prior(params):
        yp = jnp.full((n_psr, n_coeff_max), 1e-40)

        for psr_index, psr_priors in enumerate(priors):
            phi = jnp.concatenate([getN(params) for getN in psr_priors])
            yp = yp.at[psr_index, :n_coeffs[psr_index]].set(phi)

        return yp
    prior.params = sorted(set(par for psr_priors in priors for getN in psr_priors for par in getN.params))

    Fs = [np.pad(F, [(0, 0), (0, n_coeff_max - F.shape[1])]) for F in Fs]

    return utils.VariableGP(kernels.VectorNoiseMatrix1D_var(prior), Fs)

# standard EPTA single-pulsar GPs
def _makegps(psr, Tred):
    return (([signals.makegp_fourier(psr, signals.powerlaw,
                                     components=psr.noisedict[psr.name + '_dm_gp_components'], T=signals.getspan(psr),
                                     fourierbasis=signals.make_fourierbasis_dm(alpha=2.0, tndm=False), name='dm_gp')]
             if psr.noisedict[psr.name + '_dm_gp_components'] else []) +
            ([signals.makegp_fourier(psr, signals.powerlaw,
                                     components=psr.noisedict[psr.name + '_chrom_components'], T=signals.getspan(psr),
                                     fourierbasis=signals.make_fourierbasis_chrom(alpha=4.0, tndm=False), name='chrom_gp')]
             if psr.noisedict[psr.name + '_chrom_components'] else []) + 
            ([signals.makegp_fourier(psr, signals.powerlaw,
                                     components=psr.noisedict[psr.name + '_red_components'], T=Tred, name='red_noise')]
             if psr.noisedict[psr.name + '_red_components'] else []))

# single-pulsar noise analysis for EPTA DR2new+. No exponential dips
def makemodel_singlepulsar(psr):
    return discovery.PulsarLikelihood([psr.residuals,
                                        signals.makenoise_measurement(psr, psr.noisedict),
                                        signals.makegp_timing(psr)] + _makegps(psr, Tred=signals.getspan(psr)))

# CURN model for EPTA DR2new+. No exponential dips
def makemodel_curn(psrs, crn_components=30, array=False):
    tspan = signals.getspan(psrs)

    if array:
        pslmodels = [discovery.PulsarLikelihood([psr.residuals,
                                                  signals.makenoise_measurement(psr, psr.noisedict),
                                                  signals.makegp_timing(psr)])
                     for psr in psrs]

        cgp = gps2commongp([_makegps(psr, tspan) +
                            [signals.makegp_fourier(psr, signals.powerlaw, crn_components, tspan,
                                                    common=['gw_crn_log10_A', 'gw_crn_gamma'], name='gw_crn')]
                            for psr in psrs])

        return discovery.ArrayLikelihood(pslmodels, commongp=cgp)
    else:
        psls = [discovery.PulsarLikelihood([psr.residuals,
                                             signals.makenoise_measurement(psr, psr.noisedict),
                                             signals.makegp_timing(psr)] + _makegps(psr, tspan) +
                                            [signals.makegp_fourier(psr, signals.powerlaw, crn_components, tspan,
                                                                    common=['gw_crn_log10_A', 'gw_crn_gamma'], name='gw_crn')])
                for psr in psrs]

        return discovery.GlobalLikelihood(psls)

# HD model for EPTA DR2new+. No exponential dips
def makemodel_hd(psrs, gw_components=30, array=False):
    tspan = signals.getspan(psrs)

    if array:
        psls = [discovery.PulsarLikelihood([psr.residuals,
                                            signals.makenoise_measurement(psr, psr.noisedict),
                                            signals.makegp_timing(psr)])
                for psr in psrs]

        cgp = gps2commongp([_makegps(psr, tspan) for psr in psrs])
        ggp = signals.makeglobalgp_fourier(psrs, signals.powerlaw, signals.hd_orf, components=gw_components, T=tspan, name='gw_hd')

        return discovery.ArrayLikelihood(psls, commongp=cgp, globalgp=ggp)        
    else:
        psls = [discovery.PulsarLikelihood([psr.residuals,
                                            signals.makenoise_measurement(psr, psr.noisedict),
                                            signals.makegp_timing(psr)] + _makegps(psr, tspan))
                for psr in psrs]

        ggp = signals.makeglobalgp_fourier(psrs, signals.powerlaw, signals.hd_orf, components=gw_components, T=tspan, name='gw_hd')

        return discovery.GlobalLikelihood(psls, globalgp=ggp)

