#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
from pathlib import Path
import pickle
import discovery as ds
from loguru import logger
import jax.numpy as jnp
import numpy as np
import jax
import inspect
import discovery.matrix as matrix
import typing
import numpyro
import numpyro.distributions as dist

def pickle_to_feather(psr_pickle, output_feather_directory, tag='ipta_edr3'):
    """convert from enterprise pickled pulsar objects to feather files"""
    # load pickle
    edr3_psrs = pickle.load(open(psr_pickle, 'rb'))
    feather_dir = Path(output_feather_directory)
    # make feather directory and parents. Do nothing if it exists
    feather_dir.mkdir(parents=True, exist_ok=True)
    for psr in edr3_psrs:
        # reset dmx
        psr._dmx = None
        # save to feather
        ds.Pulsar.save_feather(psr, feather_dir.joinpath(f'{psr.name}_{tag}.feather'))

def makeselection_ipta(excludepta=[]):
    """
    This is for setting white noise flags properly.

    For EFAC and EQUAD excludepta should be an empty list.

    For ECORR excludepta should be 'EPTA'.

    """
    def selection_ipta(psr):
        ptas = sorted(set(psr.flags['pta']))

        selection = np.full(len(psr.toas), fill_value='', dtype=object)
        for pta in ptas:
            if pta in excludepta:
                continue

            mask = psr.flags['pta'] == pta

            if pta in ['NANOGrav', 'MPTA']:
                selection[mask] = psr.flags['f'][mask]
            else:
                selection[mask] = psr.flags['group'][mask]

        return selection

    return selection_ipta


def makecommongp_fourier_general(psrs, prior, components, T, fourierbasis=ds.fourierbasis, means=None, common=[], vector=False, name='fourierCommonGP', exclude=['f', 'df']):
    """

    Hack until we get this into discovery itself.
    Allows one to supply different bases to ArrayLikelihood, as long as
    the basis size is the same for all pulsars. Needed for DMGP.

    What this does is sets the Fourier basis for each pulsar to start from 1/T, but
    uses the same number of Fourier coefficients for each pulsar.

    Michele has a separate funcion somewhere that changes the Fourier basis sizes
    based on pulsar like the EPTA does, so we should combine these at some point.
    """
    argspec = inspect.getfullargspec(prior)

    if vector:
        argmap = [arg if arg in common else f'{name}_{arg}' if f'{name}_{arg}' in common else
                  f'{name}_{arg}({len(psrs)})' for arg in argspec.args if arg not in exclude]
    else:
        argmaps = [[(arg if arg in common else f'{name}_{arg}' if f'{name}_{arg}' in common else f'{psr.name}_{name}_{arg}') +
                    (f'({components[arg] if isinstance(components, dict) else components})' if argspec.annotations.get(arg) == typing.Sequence else '') for psr in psrs]
                   for arg in argspec.args if arg not in exclude]

    # we'll create frequency bases using the longest vector parameter (e.g., for makefreespectrum_crn)
    if isinstance(components, dict):
        components = max(components.values())

    if isinstance(T, list):
        # separate time spans for each pulsar
        fs, dfs, fmats = zip(*[fourierbasis(psr, components, t) for t, psr in zip(T, psrs)])

        # need to vmap across fs and dfs.
        # note that all of them have to be the same size.
        vprior = jax.vmap(prior, in_axes=[0, 0] +
                                 [0 if isinstance(argmap, list) else None for argmap in argmaps])
        def priorfunc(params):
            vpars = [matrix.jnparray([params[arg] for arg in argmap]) if isinstance(argmap, list) else params[argmap]
                    for argmap in argmaps]
            return vprior(matrix.jnparray(fs), matrix.jnparray(dfs), *vpars)

        priorfunc.params = sorted(set(sum([argmap if isinstance(argmap, list) else [argmap] for argmap in argmaps], [])))
    else:
        # use same span for each pulsar
        fs, dfs, fmats = zip(*[fourierbasis(psr, components, T) for psr in psrs])
        f, df = fs[0], dfs[0]

        # f and df is the same, otherwise vmap
        vprior = jax.vmap(prior, in_axes=[None, None] +
                                         [0 if isinstance(argmap, list) else None for argmap in argmaps])

        def priorfunc(params):
            vpars = [matrix.jnparray([params[arg] for arg in argmap]) if isinstance(argmap, list) else params[argmap]
                    for argmap in argmaps]
            return vprior(f, df, *vpars)

        priorfunc.params = sorted(set(sum([argmap if isinstance(argmap, list) else [argmap] for argmap in argmaps], [])))

    gp = matrix.VariableGP(matrix.VectorNoiseMatrix1D_var(priorfunc), fmats)
    gp.index = {f'{psr.name}_{name}_coefficients({2*components})': slice(2*components*i,2*components*(i+1))
                for i, psr in enumerate(psrs)}
    if means is not None:
        margspec = inspect.getfullargspec(means)
        margs = margspec.args + [arg for arg in margspec.kwonlyargs if arg not in margspec.kwonlydefaults]

        # parameters carried by the pulsar objects (e.g., pos), should be at the beginning of function
        psrpars = [{arg: getattr(psr, arg) for arg in margspec.args if hasattr(psrs[0], arg) and arg not in exclude}
                   for psr in psrs]

        # other means parameters, either common or pulsar-specific
        margmaps = [{arg: f'{meansname}_{arg}' if (f'{meansname}_{arg}' in common or arg in common) else f'{psr.name}_{meansname}_{arg}'
                     for arg in margs if not hasattr(psr, arg) and arg not in exclude} for psr in psrs]

        def meanfunc(params):
            return matrix.jnparray([means(f, df, *psrpar.values(), **{arg: params[argname] for arg, argname in margmap.items()})
                                    for psrpar, margmap in zip(psrpars, margmaps)])
        meanfunc.params = sorted(set.union(*[set(margmap.values()) for margmap in margmaps]))

        gp.means = meanfunc

    return gp

def curn_model_ipta(psrs, wn_dict, num_red_noise_frequencies=30, num_dmgp_frequencies=100, num_curn_frequencies=14):
    """
    Common Uncorrelated Red Noise (CURN) model for IPTA with DMGP
    """
    # combines curn and red noise
    crn_rn_combined_powerlaw = ds.makepowerlaw_crn(num_curn_frequencies)

    tspan = ds.getspan(psrs)
    tspan_per_psr = [ds.getspan([psr]) for psr in psrs]
    curn_like = ds.ArrayLikelihood([ds.PulsarLikelihood([psr.residuals,
                                                         ds.makenoise_measurement(psr, selection=makeselection_ipta(),
                                                                                  noisedict=wn_dict, tnequad=True),
                                                         ds.makegp_timing(psr, svd=True),
                                                         ds.makegp_ecorr(psr,
                                                                         selection=makeselection_ipta(excludepta=['EPTA']), noisedict=wn_dict),
                                                         ])
                                    for psr in psrs],
                                   commongp=[ds.makecommongp_fourier(psrs, crn_rn_combined_powerlaw,  num_red_noise_frequencies,
                                                                     T=tspan, name='red_noise',
                                                                     common=['crn_log10_A', 'crn_gamma']),
                                             makecommongp_fourier_general(psrs, ds.powerlaw, num_dmgp_frequencies,
                                                                          fourierbasis=ds.dmfourierbasis,
                                                                          name='dmgp', T=tspan_per_psr)])
    return curn_like
def hd_model_ipta(psrs, wn_dict, num_red_noise_frequencies=30, num_dmgp_frequencies=100, num_hd_frequencies=14):
    """
    Hellings-Downs model with DMGP.
    Has not been tested yet.
    """
    # for RN and GW
    tspan = ds.getspan(psrs)
    # for DMGP
    tspan_per_psr = [ds.getspan([psr]) for psr in psrs]

    hd_like = ds.ArrayLikelihood([ds.PulsarLikelihood([psr.residuals,
                                                       ds.makenoise_measurement(psr, selection=makeselection_ipta(),
                                                                                noisedict=wn_dict, tnequad=True),
                                                       ds.makegp_timing(psr, svd=True),
                                                       ds.makegp_ecorr(psr,
                                                                       selection=makeselection_ipta(excludepta=['EPTA']),
                                                                       noisedict=wn_dict)])
                                  for psr in psrs],
                                 commongp=[makecommongp_fourier_general(psrs,
                                                                        ds.powerlaw, num_dmgp_frequencies,
                                                                        fourierbasis=ds.dmfourierbasis,
                                                                        name='dmgp', T=tspan_per_psr),
                                           ds.makecommongp_fourier(psrs, ds.powerlaw,
                                                                   num_red_noise_frequencies, T=tspan,
                                                                   name='red_noise')],
                                 globalgp=ds.makegp_fourier_global(psrs, ds.powerlaw,
                                                                   ds.hd_orf, num_hd_frequencies,
                                                                   tspan, name='gw'))
    return hd_like


def simple_dict_transformation(func):
    """change from dictionary as input to list of arrays as input

    Parameters
    ----------
    func : discovery likelihood
        discovery likelihood function
    """
    def to_dict(ys):
        xs = [y for y in ys.T]
        return dict(zip(func.params, jnp.array(xs).T))
    def transformed(ys):
        return func(to_dict(ys))
    return transformed

def load_wn_dict(wn_dict_fname):
    wn_dict = json.load(open(wn_dict_fname, "r"))

    # fix negative infinities
    # that are in a few pulsars
    for key, val in wn_dict.items():
        if np.isinf(val):
            wn_dict[key] = -30
    return wn_dict


def create_ipta_curn_numpyro_model(psrs, wn_dict, num_red_noise_frequencies=30,
                         num_dmgp_frequencies=100,
                         num_curn_frequencies=14):
    """
    create numpyro curn model
    for ipta analysis.

    using the `curn_model_ipta` along with
    the numpyro_curn_model function guarantees
    that parameters are added in the correct order.
    So this has been created to avoid confusion.
    """
    discovery_model = curn_model_ipta(psrs, wn_dict, num_red_noise_frequencies, num_dmgp_frequencies,
                                      num_curn_frequencies)
    logL = simple_dict_transformation(discovery_model.logL)
    npsrs = len(psrs)

    def numpyro_curn_model(rng_key=None):
        """
        numpyro model for curn analysis
        assumes

        Parameters
        ----------
        prior_dict : dictionary, optional
            prior dictionary for lncass parameters, by default PRIOR_DICT
        """

        # change

        # one extra for the gws
        # log10_rho_gw = numpyro.sample("log10_rho_gw", dist.Uniform(-15, -4).expand([n_rn_frequencies]), rng_key=rng_key)
        # rn A and Gamma
        log10_A_rn = numpyro.sample("red_noise_log10_A", dist.Uniform(-20, -11).expand([npsrs]), rng_key=rng_key)
        gamma_rn = numpyro.sample("red_noise_gamma", dist.Uniform(0, 7).expand([npsrs]), rng_key=rng_key)
        log10_A_dmgp = numpyro.sample("dmgp_log10_A", dist.Uniform(-20, -11).expand([npsrs]), rng_key=rng_key)
        gamma_dmgp = numpyro.sample("dmgp_gamma", dist.Uniform(0, 7).expand([npsrs]), rng_key=rng_key)
        params = jnp.atleast_2d(jnp.array([[gd, ad, g, a] for gd, ad, g,a in zip(gamma_dmgp, log10_A_dmgp, gamma_rn, log10_A_rn)]).flatten()).T
        params = jnp.append(params, numpyro.sample("crn_gamma", dist.Uniform(0, 7), rng_key=rng_key))
        params = jnp.append(params, numpyro.sample("crn_log10_A", dist.Uniform(-20, -11), rng_key=rng_key))
        ll = logL(params)
        # keep track of these, for resampling later if we need
        numpyro.deterministic('loglike', ll)
        numpyro.factor("ll", ll)
    return numpyro_curn_model
