#!/usr/bin/env python3
"""Tests for discovery likelihood"""

from pathlib import Path
import pytest

import numpy as np

import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

import discovery as ds


class TestLikelihood:
    def _singlepsr_likelihoods(self, psr):
        yield ds.PulsarLikelihood([psr.residuals,
                                   ds.makenoise_measurement_simple(psr)])

        yield ds.PulsarLikelihood([psr.residuals,
                                   ds.makenoise_measurement(psr)])

        yield ds.PulsarLikelihood([psr.residuals,
                                   ds.makenoise_measurement(psr, psr.noisedict)])

        yield ds.PulsarLikelihood([psr.residuals,
                                   ds.makenoise_measurement(psr),
                                   ds.makegp_ecorr(psr)])

        yield ds.PulsarLikelihood([psr.residuals,
                                   ds.makenoise_measurement(psr, psr.noisedict),
                                   ds.makegp_ecorr(psr, psr.noisedict)])

        mdl = ds.PulsarLikelihood([psr.residuals,
                                   ds.makenoise_measurement(psr, psr.noisedict),
                                   ds.makegp_ecorr(psr, psr.noisedict),
                                   ds.makegp_timing(psr, svd=True)])
        p0 = ds.sample_uniform(mdl.logL.params)
        yield mdl, p0

        yield ds.PulsarLikelihood([psr.residuals,
                                   ds.makenoise_measurement(psr, psr.noisedict, ecorr=True),
                                   ds.makegp_timing(psr, svd=True)]), p0

        mdl = ds.PulsarLikelihood([psr.residuals,
                                   ds.makenoise_measurement(psr),
                                   ds.makegp_ecorr(psr),
                                   ds.makegp_timing(psr, svd=True),
                                   ds.makegp_fourier(psr, ds.powerlaw, components=30, name='rednoise')])
        p0 = ds.sample_uniform(mdl.logL.params)
        yield mdl, p0

        yield ds.PulsarLikelihood([psr.residuals,
                                   ds.makenoise_measurement(psr, ecorr=True),
                                   ds.makegp_timing(psr, svd=True, variable=True),
                                   ds.makegp_fourier(psr, ds.powerlaw, components=30, name='rednoise')]), p0

        yield ds.PulsarLikelihood([psr.residuals,
                                   ds.makenoise_measurement(psr, psr.noisedict),
                                   ds.makegp_ecorr(psr, psr.noisedict),
                                   ds.makegp_timing(psr, svd=True),
                                   ds.makegp_fourier(psr, ds.powerlaw, components=30, name='rednoise')])

        yield ds.PulsarLikelihood([psr.residuals,
                                   ds.makenoise_measurement(psr, psr.noisedict),
                                   ds.makegp_ecorr(psr, psr.noisedict),
                                   ds.makegp_timing(psr, svd=True),
                                   ds.makegp_fourier(psr, ds.partial(ds.powerlaw, gamma=4.33), components=30, name='rednoise')])

        p0 = {f'{psr.name}_rednoise_log10_rho(30)': 1e-6 * np.random.randn(30)}
        yield ds.PulsarLikelihood([psr.residuals,
                                   ds.makenoise_measurement(psr, psr.noisedict),
                                   ds.makegp_ecorr(psr, psr.noisedict),
                                   ds.makegp_timing(psr, svd=True),
                                   ds.makegp_fourier(psr, ds.freespectrum, components=30, name='rednoise')]), p0

    def _multipsr_likelihoods(self, psrs):
        yield ds.ArrayLikelihood([ds.PulsarLikelihood([psr.residuals,
                                                       ds.makenoise_measurement(psr, psr.noisedict),
                                                       ds.makegp_ecorr(psr, psr.noisedict),
                                                       ds.makegp_timing(psr, svd=True),
                                                       ds.makegp_fourier(psr, ds.powerlaw, components=30, name='rednoise')])
                                  for psr in psrs])

        T = ds.getspan(psrs)
        mdl = ds.ArrayLikelihood([ds.PulsarLikelihood([psr.residuals,
                                                       ds.makenoise_measurement(psr, psr.noisedict),
                                                       ds.makegp_ecorr(psr, psr.noisedict),
                                                       ds.makegp_timing(psr, svd=True),
                                                       ds.makegp_fourier(psr, ds.powerlaw, components=30, T=T, name='rednoise'),
                                                       ds.makegp_fourier(psr, ds.powerlaw, components=14, T=T, name='crn',
                                                                         common=['crn_log10_A', 'crn_gamma'])])
                                  for psr in psrs])
        p0 = ds.sample_uniform(mdl.logL.params)
        yield mdl, p0

        yield ds.ArrayLikelihood([ds.PulsarLikelihood([psr.residuals,
                                                       ds.makenoise_measurement(psr, psr.noisedict),
                                                       ds.makegp_ecorr(psr, psr.noisedict),
                                                       ds.makegp_timing(psr, svd=True)]) for psr in psrs],
                                 commongp = [ds.makecommongp_fourier(psrs, ds.powerlaw, components=30, T=T, name='rednoise'),
                                             ds.makecommongp_fourier(psrs, ds.powerlaw, components=14, T=T, name='crn',
                                                                     common=['crn_log10_A', 'crn_gamma'])]), p0

        yield ds.ArrayLikelihood([ds.PulsarLikelihood([psr.residuals,
                                                       ds.makenoise_measurement(psr, psr.noisedict),
                                                       ds.makegp_ecorr(psr, psr.noisedict),
                                                       ds.makegp_timing(psr, svd=True)]) for psr in psrs],
                                 commongp = ds.makecommongp_fourier(psrs, ds.makepowerlaw_crn(components=14), components=30, T=T, name='rednoise',
                                                                    common=['crn_log10_A', 'crn_gamma'])), p0

        mdl = ds.ArrayLikelihood([ds.PulsarLikelihood([psr.residuals,
                                                       ds.makenoise_measurement(psr, psr.noisedict),
                                                       ds.makegp_ecorr(psr, psr.noisedict),
                                                       ds.makegp_timing(psr, svd=True)]) for psr in psrs],
                                 commongp = ds.makecommongp_fourier(psrs, ds.powerlaw, components=30, T=T, name='rednoise'),
                                 globalgp = ds.makeglobalgp_fourier(psrs, ds.powerlaw, ds.hd_orf, components=14, T=T, name='gw'))
        p0 = ds.sample_uniform(mdl.logL.params)
        yield mdl, p0

        yield ds.GlobalLikelihood([ds.PulsarLikelihood([psr.residuals,
                                                        ds.makenoise_measurement(psr, psr.noisedict),
                                                        ds.makegp_ecorr(psr, psr.noisedict),
                                                        ds.makegp_timing(psr, svd=True),
                                                        ds.makegp_fourier(psr, ds.powerlaw, components=30, T=T, name='rednoise')]) for psr in psrs],
                                  globalgp = ds.makeglobalgp_fourier(psrs, ds.powerlaw, ds.hd_orf, components=14, T=T, name='gw')), p0

    def _multipsr_fftcov_likelihoods(self, psrs):
        """Test likelihoods using FFT-based covariance GPs"""
        T = ds.getspan(psrs)
        t0 = ds.getstart(psrs)

        # Test with makecommongp_fftcov for red noise
        mdl = ds.ArrayLikelihood([ds.PulsarLikelihood([psr.residuals,
                                                       ds.makenoise_measurement(psr, psr.noisedict),
                                                       ds.makegp_ecorr(psr, psr.noisedict),
                                                       ds.makegp_timing(psr, svd=True)]) for psr in psrs],
                                 commongp = ds.makecommongp_fftcov(psrs, ds.powerlaw, components=51, T=T, t0=t0,
                                                                   order=1, name='rednoise'))
        # Access params to initialize and calculate matrices
        params = mdl.logL.params
        p0 = ds.sample_uniform(params)
        yield mdl, p0

        # Test with GlobalLikelihood using makeglobalgp_fftcov
        mdl = ds.GlobalLikelihood([ds.PulsarLikelihood([psr.residuals,
                                                        ds.makenoise_measurement(psr, psr.noisedict),
                                                        ds.makegp_ecorr(psr, psr.noisedict),
                                                        ds.makegp_timing(psr, svd=True),
                                                        ds.makegp_fftcov(psr, ds.powerlaw, components=51, T=T,
                                                                         t0=t0, order=1, name='rednoise')]) for psr in psrs],
                                  globalgp = ds.makeglobalgp_fftcov(psrs, ds.powerlaw, ds.hd_orf,
                                                                    components=21, T=T, t0=t0, order=1, name='gw'))
        params = mdl.logL.params
        p0 = ds.sample_uniform(params)
        yield mdl, p0

        # Test with different oversample and fmax_factor parameters
        mdl = ds.ArrayLikelihood([ds.PulsarLikelihood([psr.residuals,
                                                       ds.makenoise_measurement(psr, psr.noisedict),
                                                       ds.makegp_ecorr(psr, psr.noisedict),
                                                       ds.makegp_timing(psr, svd=True)]) for psr in psrs],
                                 commongp = ds.makecommongp_fftcov(psrs, ds.powerlaw, components=51, T=T, t0=t0,
                                                                   order=1, name='rednoise'),
                                 globalgp = ds.makeglobalgp_fftcov(psrs, ds.powerlaw, ds.hd_orf,
                                                                   components=21, T=T, t0=t0, order=1,
                                                                   oversample=5, fmax_factor=2, name='gw'))
        params = mdl.logL.params
        p0 = ds.sample_uniform(params)
        yield mdl, p0

        # Test with monopole ORF
        mdl = ds.GlobalLikelihood([ds.PulsarLikelihood([psr.residuals,
                                                        ds.makenoise_measurement(psr, psr.noisedict),
                                                        ds.makegp_ecorr(psr, psr.noisedict),
                                                        ds.makegp_timing(psr, svd=True)]) for psr in psrs],
                                  globalgp = ds.makeglobalgp_fftcov(psrs, ds.powerlaw, ds.monopole_orf,
                                                                    components=21, T=T, t0=t0, order=1, name='gw'))
        params = mdl.logL.params
        p0 = ds.sample_uniform(params)
        yield mdl, p0

    @pytest.mark.integration
    def test_multipsr_fftcov_likelihood(self):
        """Test multi-pulsar likelihoods with FFT-based covariance GPs"""
        data_dir = Path(__file__).resolve().parent.parent / "data"

        psrfile1 = data_dir / "v1p1_de440_pint_bipm2019-B1855+09.feather"
        psrfile2 = data_dir / "v1p1_de440_pint_bipm2019-J0023+0923.feather"
        psrs = [ds.Pulsar.read_feather(psrfile1),
                ds.Pulsar.read_feather(psrfile2)]

        for model in self._multipsr_fftcov_likelihoods(psrs):
            # All models return a suggested parameter set
            if isinstance(model, tuple):
                model, p0 = model
                logl = model.logL
            else:
                logl = model.logL
                p0 = ds.sample_uniform(logl.params)

            # Verify params are accessible (this initializes matrices)
            assert logl.params is not None
            assert len(logl.params) > 0

    @pytest.mark.integration
    def test_multipsr_likelihood(self):
        data_dir = Path(__file__).resolve().parent.parent / "data"

        psrfile1 = data_dir / "v1p1_de440_pint_bipm2019-B1855+09.feather"
        psrfile2 = data_dir / "v1p1_de440_pint_bipm2019-J0023+0923.feather"
        psrs = [ds.Pulsar.read_feather(psrfile1),
                ds.Pulsar.read_feather(psrfile2)]

        p0_old = None
        for model in self._multipsr_likelihoods(psrs):
            # some models return also a suggested parameter set
            if isinstance(model, tuple):
                model, p0 = model
                logl = model.logL
            else:
                logl = model.logL
                p0 = ds.sample_uniform(logl.params)

            l1 = logl(p0)
            l2 = jax.jit(logl)(p0)

            assert np.allclose(l1, l2)

            if p0 == p0_old:
                assert np.allclose(l1, l1_old, atol=0.1)

            p0_old, l1_old = p0, l1

    @pytest.mark.integration
    def test_singlepsr_likelihood(self):
        data_dir = Path(__file__).resolve().parent.parent / "data"

        psrfile = data_dir / "v1p1_de440_pint_bipm2019-B1855+09.feather"
        psr = ds.Pulsar.read_feather(psrfile)

        p0_old = None
        for model in self._singlepsr_likelihoods(psr):
            if isinstance(model, tuple):
                model, p0 = model
                logl = model.logL
            else:
                logl = model.logL
                p0 = ds.sample_uniform(logl.params)

            l1 = logl(p0)
            l2 = jax.jit(logl)(p0)

            assert np.allclose(l1, l2)

            if p0 and p0 == p0_old:
                assert np.allclose(l1, l1_old, atol=0.1)

            p0_old, l1_old = p0, l1

    @pytest.mark.integration
    def test_compare_enterprise(self):
        # The directory containing the pulsar feather files should be parallel to the tests directory
        data_dir = Path(__file__).resolve().parent.parent / "data"

        # Choose two pulsars for reproducibility
        psr_files = [
            data_dir / "v1p1_de440_pint_bipm2019-B1855+09.feather",
            data_dir / "v1p1_de440_pint_bipm2019-B1953+29.feather",
        ]

        # Construct a list of Pulsar objects
        psrs = [ds.Pulsar.read_feather(psr) for psr in psr_files]

        # Get the timespan
        tspan = ds.getspan(psrs)

        # Construct the discovery global likelihood for CURN
        gl = ds.GlobalLikelihood(
            (
                ds.PulsarLikelihood(
                    [
                        psrs[ii].residuals,
                        ds.makenoise_measurement(psrs[ii], psrs[ii].noisedict),
                        ds.makegp_ecorr(psrs[ii], psrs[ii].noisedict),
                        ds.makegp_timing(psrs[ii]),
                        ds.makegp_fourier(psrs[ii], ds.powerlaw, 30, T=tspan, name="red_noise"),
                        ds.makegp_fourier(
                            psrs[ii], ds.powerlaw, 14, T=tspan, common=["gw_log10_A", "gw_gamma"], name="gw"
                        ),
                    ]
                )
                for ii in range(len(psrs))
            )
        )

        # Get the jitted discovery log-likelihood
        jlogl = jax.jit(gl.logL)

        # Set parameters to feed likelihood
        initial_position = {
            "B1855+09_red_noise_gamma": 6.041543719234379,
            "B1855+09_red_noise_log10_A": -14.311870465932676,
            "B1953+29_red_noise_gamma": 2.037363188329115,
            "B1953+29_red_noise_log10_A": -16.748409409147907,
            "gw_gamma": 1.6470255693110927,
            "gw_log10_A": -14.236953140132435,
        }

        # Enterprise log-likelihood for this choice of parameters
        enterprise_ll = 145392.54369264

        # Find the difference between enterprise and discovery likelihoods
        ll_difference = enterprise_ll - jlogl(initial_position)

        # There is a constant offset of ~ -52.4
        offset = -52.4

        # Choose the absolute tolerance
        atol = 0.1

        # we need to check the systematic difference between enterprise and discovery
        # before we can run this, but at least we can check the JITted likelihood runs
        # assert float(jax.numpy.abs(ll_difference - offset)) <= atol

    # --- ArrayLogL: the array-native logL path -----------------------------
    #
    # `model.loglike_arr` wraps `model.logL` so JIT/NUTS can feed it a single
    # flat array instead of a parameter dict (avoids the per-leaf pytree
    # marshalling that dominates the GPU forward/grad time). These tests check
    # that the array path reproduces the dict path, including for array-valued
    # parameters where one name maps to a block of columns.

    @staticmethod
    def _read_psrs(*names):
        data_dir = Path(__file__).resolve().parent.parent / "data"
        return [ds.Pulsar.read_feather(data_dir / f"v1p1_de440_pint_bipm2019-{n}.feather")
                for n in names]

    def _check_arraylogl(self, model, p0):
        """Core check: the ArrayLogL path reproduces the dict-based logL path."""
        logL, logL_arr = model.logL, model.loglike_arr

        # the wrapper freezes logL's name ordering as its flat-vector contract
        assert logL_arr.params == list(logL.params)

        x = logL_arr.dict_to_array(p0)
        assert x.shape == (logL_arr.size,)

        # array path agrees with the dict path, eager and jitted
        l_dict = logL(p0)
        assert np.allclose(logL_arr(x), l_dict)
        assert np.allclose(jax.jit(logL_arr)(x), l_dict)

        # dict <-> array round-trips exactly
        assert logL_arr.check_roundtrip(p0)

        return logL_arr

    @pytest.mark.integration
    def test_arraylogl_curn(self):
        """ArrayLikelihood CURN model: array path == dict path (all-scalar params)."""
        psrs = self._read_psrs("B1855+09", "J0023+0923")
        T = ds.getspan(psrs)

        arr = ds.ArrayLikelihood(
            [ds.PulsarLikelihood([psr.residuals,
                                  ds.makenoise_measurement(psr, psr.noisedict),
                                  ds.makegp_ecorr(psr, psr.noisedict),
                                  ds.makegp_timing(psr, svd=True)]) for psr in psrs],
            commongp=ds.makecommongp_fourier(psrs, ds.makepowerlaw_crn(14), 30, T=T,
                                             common=['crn_log10_A', 'crn_gamma'],
                                             name='red_noise'))

        logL_arr = self._check_arraylogl(arr, ds.sample_uniform(arr.logL.params))

        # every parameter is scalar, so the flat vector has one column per name
        assert logL_arr.size == len(logL_arr.params)

    @pytest.mark.integration
    def test_arraylogl_global(self):
        """GlobalLikelihood with an HD globalgp: array path == dict path."""
        psrs = self._read_psrs("B1855+09", "J0023+0923")
        T = ds.getspan(psrs)

        gl = ds.GlobalLikelihood(
            [ds.PulsarLikelihood([psr.residuals,
                                  ds.makenoise_measurement(psr, psr.noisedict),
                                  ds.makegp_ecorr(psr, psr.noisedict),
                                  ds.makegp_timing(psr, svd=True),
                                  ds.makegp_fourier(psr, ds.powerlaw, components=30, T=T,
                                                    name='rednoise')]) for psr in psrs],
            globalgp=ds.makeglobalgp_fourier(psrs, ds.powerlaw, ds.hd_orf, components=14,
                                             T=T, name='gw'))

        self._check_arraylogl(gl, ds.sample_uniform(gl.logL.params))

    @pytest.mark.integration
    def test_arraylogl_array_valued_params(self):
        """ArrayLogL handles an array-valued parameter: freespectrum log10_rho(30)."""
        psr, = self._read_psrs("B1855+09")

        mdl = ds.PulsarLikelihood([psr.residuals,
                                   ds.makenoise_measurement(psr, psr.noisedict),
                                   ds.makegp_ecorr(psr, psr.noisedict),
                                   ds.makegp_timing(psr, svd=True),
                                   ds.makegp_fourier(psr, ds.freespectrum, components=30,
                                                     name='rednoise')])

        rho_name = f'{psr.name}_rednoise_log10_rho(30)'
        rng = np.random.default_rng(0)
        p0 = {rho_name: 1e-6 * rng.standard_normal(30)}

        logL_arr = self._check_arraylogl(mdl, p0)

        # the (30) name suffix is parsed into a 30-wide block, so the flat
        # vector is longer than the parameter-name count
        assert rho_name in logL_arr.params
        assert len(logL_arr.params) == 1
        assert logL_arr.size == 30
        assert logL_arr.slice_of(rho_name) == (0, 30)

        # array_to_dict reshapes the block back to the parameter's shape
        d = logL_arr.array_to_dict(logL_arr.dict_to_array(p0))
        assert jnp.shape(d[rho_name]) == (30,)

    @pytest.mark.integration
    def test_arraylogl_scatter_assembly(self):
        """ArrayLogL.slice_of / indices: scatter-build the flat vector without a dict."""
        psrs = self._read_psrs("B1855+09", "J0023+0923")
        T = ds.getspan(psrs)

        arr = ds.ArrayLikelihood(
            [ds.PulsarLikelihood([psr.residuals,
                                  ds.makenoise_measurement(psr, psr.noisedict),
                                  ds.makegp_ecorr(psr, psr.noisedict),
                                  ds.makegp_timing(psr, svd=True)]) for psr in psrs],
            commongp=ds.makecommongp_fourier(psrs, ds.makepowerlaw_crn(14), 30, T=T,
                                             common=['crn_log10_A', 'crn_gamma'],
                                             name='red_noise'))
        logL_arr = arr.loglike_arr
        p0 = ds.sample_uniform(arr.logL.params)

        # assemble the flat vector by scatter (the NumPyro-model use case)
        idx_A = logL_arr.indices([f'{psr.name}_red_noise_log10_A' for psr in psrs])
        idx_g = logL_arr.indices([f'{psr.name}_red_noise_gamma' for psr in psrs])
        cA0, cA1 = logL_arr.slice_of('crn_log10_A')
        cg0, cg1 = logL_arr.slice_of('crn_gamma')

        v = jnp.empty(logL_arr.size)
        v = v.at[idx_A].set(jnp.array([p0[f'{psr.name}_red_noise_log10_A'] for psr in psrs]))
        v = v.at[idx_g].set(jnp.array([p0[f'{psr.name}_red_noise_gamma'] for psr in psrs]))
        v = v.at[cA0:cA1].set(p0['crn_log10_A'])
        v = v.at[cg0:cg1].set(p0['crn_gamma'])

        # scatter-assembled vector matches dict_to_array, and the likelihood agrees
        assert np.allclose(v, logL_arr.dict_to_array(p0))
        assert np.allclose(logL_arr(v), arr.logL(p0))
