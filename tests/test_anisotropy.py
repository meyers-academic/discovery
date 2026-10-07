#!/usr/bin/env python3
"""Tests for anisotropic / basis-valued / parameter-dependent ORFs in makeglobalgp_fourier.

Three routes through `makeglobalgp_fourier` are exercised here:

  * a scalar, parameter-free ORF (hd_orf) -- Phi = Gamma (x) S, inverted via the
    build-time Kronecker shortcut;
  * a basis-valued ("pixel") ORF, and sums of several ORF terms -- Phi is block
    diagonal in frequency with a different (npsr x npsr) block per mode, inverted
    one mode at a time;
  * a parameter-dependent ORF, where Gamma itself is sampled.

The recurring assertion is that the structure-exploiting `Phi_inv` agrees with the
generic dense `Phi.make_inv()`, because it is meant to be the same operation and
not an approximation.

No healpy: the sky directions are a fixed deterministic set so the suite stays
dependency-clean.
"""

from pathlib import Path

import numpy as np
import pytest

import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

import discovery as ds


NFREQ = 5
NMODE = 2 * NFREQ
NPIX = 12


def _pulsars(n=3):
    data_dir = Path(__file__).resolve().parent.parent / "data"
    names = ["v1p1_de440_pint_bipm2019-B1855+09.feather",
             "v1p1_de440_pint_bipm2019-J0023+0923.feather",
             "v1p1_de440_pint_bipm2019-J0030+0451.feather",
             "v1p1_de440_pint_bipm2019-B1937+21.feather"]
    return [ds.Pulsar.read_feather(data_dir / nm) for nm in names[:n]]


def _directions(npix=NPIX):
    """A fixed, reproducible set of unit vectors standing in for a HEALPix grid."""
    v = np.random.default_rng(42).normal(size=(npix, 3))
    return v / np.linalg.norm(v, axis=1)[:, None]


_VECS = _directions()


def _fpc(pos, dirs):
    """Fplus/Fcross for one pulsar against every sky direction (numpy, build-time)."""
    x, y, z = pos
    theta = np.arccos(np.clip(dirs[:, 2], -1.0, 1.0))
    phi = np.arctan2(dirs[:, 1], dirs[:, 0])
    sp, cp, st, ct = np.sin(phi), np.cos(phi), np.sin(theta), np.cos(theta)
    m = sp * x - cp * y
    n = -ct * cp * x - ct * sp * y + st * z
    om = -st * cp * x - st * sp * y - ct * z
    return 0.5 * (m ** 2 - n ** 2) / (1 + om), (m * n) / (1 + om)


def _fibonacci(n):
    """Near-equal-area points on the sphere -- a healpy-free stand-in for a fine grid."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


def make_pixel_orf(vecs):
    """Per-pixel power basis: R_p = (Fp1 Fp2 + Fc1 Fc2) * 3 / (2 npix).

    Rank-one per pixel, so summing over pixels with unit coefficients is positive
    definite and converges to HD. The diagonal is doubled for the pulsar term.
    """
    npix = len(vecs)

    def pixel_orf(pos1, pos2):
        fp1, fc1 = _fpc(np.asarray(pos1), vecs)
        fp2, fc2 = _fpc(np.asarray(pos2), vecs)
        r = (fp1 * fp2 + fc1 * fc2) * 3.0 / (2.0 * npix)
        return 2.0 * r if np.all(pos1 == pos2) else r

    return pixel_orf


pixel_orf = make_pixel_orf(_VECS)


def pixel_prior(f, df, log10_rho, coefs):
    """Free spectrum x a per-pixel weight map -> (nmode, npix)."""
    return jnp.repeat(10.0 ** (2 * log10_rho), 2)[:, None] * jnp.repeat(coefs, 2, axis=0)


def pixel_prior_freqcorr(f, df, log10_rho, coefs, log10_ell):
    """As above but correlating frequency bins -> (nmode, nmode, npix)."""
    S = jnp.repeat(10.0 ** (2 * log10_rho), 2)
    i = jnp.arange(NMODE)
    K = jnp.exp(-0.5 * ((i[:, None] - i[None, :]) / 10 ** log10_ell) ** 2)
    Smat = jnp.sqrt(S)[:, None] * K * jnp.sqrt(S)[None, :]
    C = jnp.repeat(coefs, 2, axis=0)
    return Smat[:, :, None] * jnp.sqrt(C)[:, None, :] * jnp.sqrt(C)[None, :, :]


def hd_traceable(pos1, pos2):
    """hd_orf written without Python branching, so it can be traced/vmapped."""
    z = jnp.clip(jnp.dot(pos1, pos2), -1.0, 1.0)
    omc2 = jnp.clip((1.0 - z) / 2.0, 1e-30, None)
    hd = 1.5 * omc2 * jnp.log(omc2) - 0.25 * omc2 + 0.5
    return jnp.where(jnp.all(pos1 == pos2), 1.0, hd)


def mixed_orf(pos1, pos2, alpha):
    """HD with a sampled monopole admixture -- a parameter-dependent Gamma."""
    mono = jnp.where(jnp.all(pos1 == pos2), 1.0 + 1.0e-6, 1.0)
    return hd_traceable(pos1, pos2) + alpha * mono


PIXPARS = {'gw_log10_rho': jnp.full(NFREQ, -7.0),
           'gw_coefs': jnp.ones((NFREQ, NPIX))}


def _globallikelihood(psrs, gp):
    return ds.GlobalLikelihood(
        [ds.PulsarLikelihood([psr.residuals,
                              ds.makenoise_measurement(psr, psr.noisedict),
                              ds.makegp_timing(psr, svd=True)]) for psr in psrs],
        globalgp=gp)


def _spectrum(psrs, T, log10_A, gamma):
    f, df, _ = ds.fourierbasis(psrs[0], NFREQ, T)
    return np.array(ds.powerlaw(jnp.asarray(f), jnp.asarray(df), log10_A, gamma))


class TestBasisValuedORF:
    """A single basis-valued (pixel) ORF: the stacked, block-per-mode inverse."""

    def _gp(self, psrs, T, prior=pixel_prior):
        return ds.makeglobalgp_fourier(psrs, prior, pixel_orf, components=NFREQ, T=T, name='gw')

    def test_invprior_matches_dense(self):
        """The stacked Phi_inv reproduces the generic dense inverse and its logdet.

        This is the central claim of the fast path: it is the same operation, so any
        divergence here means the block decomposition is wrong, not merely different.
        """
        psrs = _pulsars()
        gp = self._gp(psrs, ds.getspan(psrs))

        fast, ldfast = gp.Phi_inv(PIXPARS)
        ref, ldref = gp.Phi.make_inv()(PIXPARS)

        assert np.allclose(np.array(fast), np.array(ref), rtol=1e-10, atol=0)
        assert np.isclose(float(ldfast), float(ldref), rtol=1e-12)

    def test_invprior_is_actual_inverse(self):
        """Phi_inv @ Phi == I.

        Checked without reference to make_inv, so it also pins the index convention of
        the scatter that puts the per-mode blocks back into the dense layout -- a
        transposed or misplaced block would still look plausible elementwise.
        """
        psrs = _pulsars()
        gp = self._gp(psrs, ds.getspan(psrs))

        Phi = np.array(gp.Phi.getN(PIXPARS))
        Pinv = np.array(gp.Phi_inv(PIXPARS)[0])

        assert np.allclose(Pinv @ Phi, np.eye(Phi.shape[0]), atol=1e-8)

    def test_invprior_params_match_phi(self):
        """Phi_inv must advertise the same parameters as Phi.

        A name missing here does not fail at build time -- it surfaces much later as a
        KeyError when the sampler assembles a parameter dict, or worse as a stale value.
        """
        psrs = _pulsars()
        gp = self._gp(psrs, ds.getspan(psrs))

        assert sorted(gp.Phi_inv.params) == sorted(gp.Phi.params)

    def test_phi_is_block_diagonal_in_frequency(self):
        """Each pulsar-pair block of Phi is diagonal in frequency.

        This is the precondition the stacked inverse relies on. Asserting it directly
        means a change to make2d or priorfunc fails loudly here, rather than silently
        producing a wrong inverse in the tests above.
        """
        psrs = _pulsars()
        gp = self._gp(psrs, ds.getspan(psrs))

        Phi = np.array(gp.Phi.getN(PIXPARS))
        block = Phi[:NMODE, :NMODE]

        assert np.allclose(block, np.diag(np.diag(block)))
        assert np.isclose((np.abs(Phi) > 0).mean(), 1.0 / NMODE, rtol=1e-6)

    def test_pixel_sum_approximates_hd(self):
        """Unit pixel coefficients reproduce HD, fixing the 3/(2 npix) normalization.

        Without this the pixel basis can be self-consistent yet describe a background of
        the wrong amplitude, which no other test in this file would catch. Uses a fine
        near-equal-area grid: the coarse 12-direction set used elsewhere is fine for
        structural tests but far too crude to integrate the antenna pattern.
        """
        psrs = _pulsars(4)
        pos = [p.pos for p in psrs]
        fine = make_pixel_orf(_fibonacci(4000))

        G = np.array([[fine(a, b).sum() for a in pos] for b in pos])
        hd = np.array([[ds.hd_orf(a, b) for a in pos] for b in pos])

        offdiag = ~np.eye(len(pos), dtype=bool)
        assert np.abs(G[offdiag] - hd[offdiag]).max() < 0.02
        assert np.allclose(np.diag(G), 1.0, atol=0.02)

    def test_logL_matches_dense(self):
        """Full logL agrees with Phi_inv suppressed, i.e. the fast path is really wired in.

        A correct inverse that the likelihood never reaches would pass every test above.
        """
        psrs = _pulsars()
        T = ds.getspan(psrs)

        fast = _globallikelihood(psrs, self._gp(psrs, T))
        slow_gp = self._gp(psrs, T)
        slow_gp.Phi_inv = None
        slow = _globallikelihood(psrs, slow_gp)

        p0 = ds.sample_uniform([k for k in fast.logL.params if k not in PIXPARS])
        p0.update(PIXPARS)

        assert np.isclose(float(fast.logL(p0)), float(slow.logL(p0)), rtol=1e-10)

    def test_frequency_correlated_falls_back(self):
        """An (m x m x npix) prior has no block structure, so the dense route is taken.

        The fallback mirrors NoiseMatrix2D_var.make_inv, so the agreement should be exact
        rather than merely close.
        """
        psrs = _pulsars()
        gp = self._gp(psrs, ds.getspan(psrs), prior=pixel_prior_freqcorr)
        pars = {**PIXPARS, 'gw_log10_ell': jnp.array(0.3)}

        fast, ldfast = gp.Phi_inv(pars)
        ref, ldref = gp.Phi.make_inv()(pars)

        assert np.allclose(np.array(fast), np.array(ref), rtol=1e-10, atol=0)
        assert np.isclose(float(ldfast), float(ldref), rtol=1e-12)

        # and the frequency blocks really are dense. Compared relative to the block's own
        # scale: the entries are O(1e-14), so np.allclose's default atol would call any
        # such block diagonal.
        Phi = np.array(gp.Phi.getN(pars))
        block = Phi[:NMODE, :NMODE]
        offdiag = np.abs(block - np.diag(np.diag(block))).max()
        assert offdiag / np.abs(block).max() > 0.1

    def test_cglogL_rejects_basis_orf(self):
        """cglogL needs `factors`, which a basis-valued ORF has no Kronecker structure for.

        It must fail rather than quietly return something wrong.
        """
        psrs = _pulsars()
        T = ds.getspan(psrs)
        gp = self._gp(psrs, T)
        assert gp.factors is None

        al = ds.ArrayLikelihood(
            [ds.PulsarLikelihood([psr.residuals,
                                  ds.makenoise_measurement(psr, psr.noisedict),
                                  ds.makegp_timing(psr, svd=True)]) for psr in psrs],
            commongp=ds.makecommongp_fourier(psrs, ds.powerlaw, NFREQ, T=T, name='rednoise'),
            globalgp=gp)

        with pytest.raises(Exception):
            al.cglogL()

    def test_arraylikelihood_matches_globallikelihood(self):
        """The same pixel model gives the same logL on the matrix and metamath routes.

        These two backends can drift independently, and nothing else here would notice.
        """
        psrs = _pulsars()
        T = ds.getspan(psrs)

        gl = _globallikelihood(psrs, self._gp(psrs, T))
        p0 = ds.sample_uniform([k for k in gl.logL.params if k not in PIXPARS])
        p0.update(PIXPARS)

        al = ds.ArrayLikelihood(
            [ds.PulsarLikelihood([psr.residuals,
                                  ds.makenoise_measurement(psr, psr.noisedict),
                                  ds.makegp_timing(psr, svd=True)]) for psr in psrs],
            commongp=ds.makecommongp_fourier(psrs, ds.powerlaw, NFREQ, T=T, name='rednoise'),
            globalgp=self._gp(psrs, T))

        q0 = dict(p0)
        q0.update(ds.sample_uniform([k for k in al.logL.params if k not in q0]))
        # strip the extra commongp contribution by comparing against a GlobalLikelihood
        # carrying the same per-pulsar red noise
        gl2 = ds.GlobalLikelihood(
            [ds.PulsarLikelihood([psr.residuals,
                                  ds.makenoise_measurement(psr, psr.noisedict),
                                  ds.makegp_timing(psr, svd=True),
                                  ds.makegp_fourier(psr, ds.powerlaw, NFREQ, T=T, name='rednoise')])
             for psr in psrs],
            globalgp=self._gp(psrs, T))

        assert np.isclose(float(gl2.logL(q0)), float(al.logL(q0)), rtol=1e-10)


class TestMultipleORFs:
    """Sums of ORF terms: Phi = sum_t Gamma_t (x) S_t, still block diagonal in frequency."""

    def test_sum_of_kroneckers(self):
        """HD + monopole equals kron(G_hd, diag(S1)) + kron(G_mono, diag(S2)).

        Pins the mathematical content of priorfunc independently of how it is built, so a
        future rewrite of the assembly cannot quietly change the model.
        """
        psrs = _pulsars()
        T = ds.getspan(psrs)
        gp = ds.makeglobalgp_fourier(psrs, [ds.powerlaw, ds.powerlaw],
                                     [ds.hd_orf, ds.monopole_orf], components=NFREQ, T=T, name='gw')
        pars = {'gw_hdorf_log10_A': -14.5, 'gw_hdorf_gamma': 4.33,
                'gw_monopoleorf_log10_A': -15.0, 'gw_monopoleorf_gamma': 4.0}

        Phi = np.array(gp.Phi.getN(pars))
        pos = [p.pos for p in psrs]
        G1 = np.array([[ds.hd_orf(a, b) for a in pos] for b in pos])
        G2 = np.array([[ds.monopole_orf(a, b) for a in pos] for b in pos])
        S1 = _spectrum(psrs, T, pars['gw_hdorf_log10_A'], pars['gw_hdorf_gamma'])
        S2 = _spectrum(psrs, T, pars['gw_monopoleorf_log10_A'], pars['gw_monopoleorf_gamma'])

        assert np.allclose(Phi, np.kron(G1, np.diag(S1)) + np.kron(G2, np.diag(S2)))

    def test_invprior_matches_dense(self):
        """The stacked inverse of a two-term sum matches the generic dense inverse."""
        psrs = _pulsars()
        T = ds.getspan(psrs)
        gp = ds.makeglobalgp_fourier(psrs, [ds.powerlaw, ds.powerlaw],
                                     [ds.hd_orf, ds.monopole_orf], components=NFREQ, T=T, name='gw')
        pars = {'gw_hdorf_log10_A': -14.5, 'gw_hdorf_gamma': 4.33,
                'gw_monopoleorf_log10_A': -15.0, 'gw_monopoleorf_gamma': 4.0}

        fast, ldfast = gp.Phi_inv(pars)
        ref, ldref = gp.Phi.make_inv()(pars)

        assert np.allclose(np.array(fast), np.array(ref), rtol=1e-10, atol=0)
        assert np.isclose(float(ldfast), float(ldref), rtol=1e-12)

    def test_scalar_plus_basis_orf(self):
        """An isotropic HD term summed with a pixel-basis anisotropy term.

        Regression: this combination previously raised
        `TypeError: add got incompatible shapes for broadcasting`, because the multi-term
        assembly scaled by the ORF value instead of contracting the basis axis. It is also
        the model that gives anisotropy searches an isotropic anchor.
        """
        psrs = _pulsars()
        T = ds.getspan(psrs)
        gp = ds.makeglobalgp_fourier(psrs, [ds.powerlaw, pixel_prior],
                                     [ds.hd_orf, pixel_orf], components=NFREQ, T=T, name='gw')
        pars = {'gw_hdorf_log10_A': -14.5, 'gw_hdorf_gamma': 4.33,
                'gw_pixelorf_log10_rho': jnp.full(NFREQ, -7.5),
                'gw_pixelorf_coefs': jnp.ones((NFREQ, NPIX))}

        Phi = np.array(gp.Phi.getN(pars))
        assert Phi.shape == (len(psrs) * NMODE,) * 2
        assert np.isfinite(Phi).all()

        fast, ldfast = gp.Phi_inv(pars)
        ref, ldref = gp.Phi.make_inv()(pars)
        assert np.allclose(np.array(fast), np.array(ref), rtol=1e-10, atol=0)
        assert np.isclose(float(ldfast), float(ldref), rtol=1e-12)

        # the sum decomposes: HD part + pixel part
        pos = [p.pos for p in psrs]
        Ghd = np.array([[ds.hd_orf(a, b) for a in pos] for b in pos])
        Shd = _spectrum(psrs, T, pars['gw_hdorf_log10_A'], pars['gw_hdorf_gamma'])
        Gpix = np.array([[pixel_orf(a, b).sum() for a in pos] for b in pos])
        Spix = np.repeat(10.0 ** (2 * np.asarray(pars['gw_pixelorf_log10_rho'])), 2)
        assert np.allclose(Phi, np.kron(Ghd, np.diag(Shd)) + np.kron(Gpix, np.diag(Spix)))

    def test_frequency_correlated_term_falls_back(self):
        """One matrix-valued spectrum collapses the whole sum to the dense route."""
        psrs = _pulsars()
        T = ds.getspan(psrs)
        gp = ds.makeglobalgp_fourier(psrs, [ds.powerlaw, pixel_prior_freqcorr],
                                     [ds.hd_orf, pixel_orf], components=NFREQ, T=T, name='gw')
        pars = {'gw_hdorf_log10_A': -14.5, 'gw_hdorf_gamma': 4.33,
                'gw_pixelorf_log10_rho': jnp.full(NFREQ, -7.5),
                'gw_pixelorf_coefs': jnp.ones((NFREQ, NPIX)),
                'gw_pixelorf_log10_ell': jnp.array(0.3)}

        fast, ldfast = gp.Phi_inv(pars)
        ref, ldref = gp.Phi.make_inv()(pars)

        assert np.allclose(np.array(fast), np.array(ref), rtol=1e-10, atol=0)
        assert np.isclose(float(ldfast), float(ldref), rtol=1e-12)

    def test_logL_matches_dense(self):
        """Full logL for a two-term model agrees with the dense route."""
        psrs = _pulsars()
        T = ds.getspan(psrs)

        def gp():
            return ds.makeglobalgp_fourier(psrs, [ds.powerlaw, ds.powerlaw],
                                           [ds.hd_orf, ds.monopole_orf],
                                           components=NFREQ, T=T, name='gw')

        fast = _globallikelihood(psrs, gp())
        slow_gp = gp()
        slow_gp.Phi_inv = None
        slow = _globallikelihood(psrs, slow_gp)

        p0 = ds.sample_uniform(fast.logL.params)
        assert np.isclose(float(fast.logL(p0)), float(slow.logL(p0)), rtol=1e-10)


class TestParameterDependentORF:
    """Gamma itself carries sampled parameters."""

    def _gp(self, psrs, T, name='gw'):
        return ds.makeglobalgp_fourier(psrs, ds.powerlaw, mixed_orf, components=NFREQ, T=T, name=name)

    def test_orf_parameter_namespace(self):
        """ORF arguments land in a separate `_orf_` namespace from prior arguments.

        `mixed_orf` takes `alpha` and powerlaw takes `log10_A`/`gamma`; all three must be
        distinct parameters. The namespace exists so that an ORF argument named `gamma`
        cannot silently fuse with the power-law index.
        """
        psrs = _pulsars()
        gp = self._gp(psrs, ds.getspan(psrs))

        assert sorted(gp.Phi.params) == ['gw_gamma', 'gw_log10_A', 'gw_orf_alpha']
        assert sorted(gp.Phi_inv.params) == sorted(gp.Phi.params)

    def test_orf_parameter_namespace_multiterm(self):
        """With several ORFs the names carry the ORF name too: {name}_{orfname}_orf_{arg}."""
        psrs = _pulsars()
        T = ds.getspan(psrs)
        gp = ds.makeglobalgp_fourier(psrs, [ds.powerlaw, ds.powerlaw],
                                     [mixed_orf, ds.monopole_orf], components=NFREQ, T=T, name='gw')

        assert 'gw_mixedorf_orf_alpha' in gp.Phi.params
        assert 'gw_mixedorf_log10_A' in gp.Phi.params

    def test_constant_orf_gains_no_parameters(self):
        """A parameter-free ORF stays a build-time constant and adds nothing to sample.

        Guards the constant-folding path: dragging hd_orf into the traced graph would slow
        every existing model without changing any result, so nothing else would catch it.
        """
        psrs = _pulsars()
        gp = ds.makeglobalgp_fourier(psrs, ds.powerlaw, ds.hd_orf,
                                     components=NFREQ, T=ds.getspan(psrs), name='gw')

        assert sorted(gp.Phi.params) == ['gw_gamma', 'gw_log10_A']
        assert gp.factors is not None      # Kronecker shortcut still available

    def test_matches_manual_phi(self):
        """Phi equals kron(Gamma(alpha), diag(S)) built by hand at the same alpha."""
        psrs = _pulsars()
        T = ds.getspan(psrs)
        gp = self._gp(psrs, T)
        pars = {'gw_log10_A': -14.5, 'gw_gamma': 4.33, 'gw_orf_alpha': 0.3}

        Phi = np.array(gp.Phi.getN(pars))
        pos = [jnp.asarray(p.pos) for p in psrs]
        G = np.array([[float(mixed_orf(a, b, 0.3)) for a in pos] for b in pos])
        S = _spectrum(psrs, T, pars['gw_log10_A'], pars['gw_gamma'])

        assert np.allclose(Phi, np.kron(G, np.diag(S)))

    def test_reduces_to_hd(self):
        """At alpha = 0 the parameterized ORF reproduces the plain HD model.

        A continuity check on the parameterization: the sampled model must contain the
        fixed one as a special case, not merely resemble it.
        """
        psrs = _pulsars()
        T = ds.getspan(psrs)
        pars = {'gw_log10_A': -14.5, 'gw_gamma': 4.33}

        live = np.array(self._gp(psrs, T).Phi.getN({**pars, 'gw_orf_alpha': 0.0}))
        const = np.array(ds.makeglobalgp_fourier(psrs, ds.powerlaw, ds.hd_orf,
                                                 components=NFREQ, T=T, name='gw').Phi.getN(pars))

        assert np.allclose(live, const)

    def test_vmap_matches_python_loop(self):
        """The vectorized ORF evaluation equals a naive double loop over pulsar pairs.

        Pins the index convention as well as the arithmetic: vmap yields orf(pos[i], pos[j])
        while the build-time loop constructs the transpose, which a symmetric ORF hides.
        """
        psrs = _pulsars(4)
        pos = jnp.stack([jnp.asarray(p.pos) for p in psrs])
        alpha = 0.3

        vmapped = jax.vmap(jax.vmap(mixed_orf, in_axes=(None, 0, None)),
                           in_axes=(0, None, None))(pos, pos, alpha)
        looped = np.array([[float(mixed_orf(a, b, alpha)) for b in pos] for a in pos])

        assert np.allclose(np.array(vmapped), looped)

    def test_invprior_matches_dense(self):
        """The parameter-dependent route still inverts correctly."""
        psrs = _pulsars()
        gp = self._gp(psrs, ds.getspan(psrs))
        pars = {'gw_log10_A': -14.5, 'gw_gamma': 4.33, 'gw_orf_alpha': 0.3}

        fast, ldfast = gp.Phi_inv(pars)
        ref, ldref = gp.Phi.make_inv()(pars)

        assert np.allclose(np.array(fast), np.array(ref), rtol=1e-10, atol=0)
        assert np.isclose(float(ldfast), float(ldref), rtol=1e-12)

    def test_logL_gradient_wrt_orf_parameter(self):
        """d logL / d alpha is finite and non-zero.

        The point of a parameter-dependent ORF is to sample it, and an ORF that got
        constant-folded or was silently non-traceable would give exactly zero gradient
        while every value-based test above still passed.
        """
        psrs = _pulsars()
        gp = self._gp(psrs, ds.getspan(psrs))
        gl = _globallikelihood(psrs, gp)

        pars = {'gw_log10_A': -14.5, 'gw_gamma': 4.33, 'gw_orf_alpha': 0.3}
        p0 = ds.sample_uniform([k for k in gl.logL.params if k not in pars])
        p0.update(pars)

        grad = jax.grad(gl.logL)(p0)
        dalpha = float(grad['gw_orf_alpha'])

        assert np.isfinite(dalpha)
        assert abs(dalpha) > 0.0


class TestScalarORFUnchanged:
    """Non-regression on the path that was not modified."""

    def test_kronecker_structure_and_shortcut(self):
        """HD-only Phi is still Gamma (x) S, with the build-time inverse still in place.

        Cheap insurance that adding the basis-valued and parameter-dependent routes left
        the hot path -- the one every existing model uses -- untouched.
        """
        psrs = _pulsars()
        T = ds.getspan(psrs)
        gp = ds.makeglobalgp_fourier(psrs, ds.powerlaw, ds.hd_orf, components=NFREQ, T=T, name='gw')
        pars = {'gw_log10_A': -14.5, 'gw_gamma': 4.33}

        Phi = np.array(gp.Phi.getN(pars))
        pos = [p.pos for p in psrs]
        G = np.array([[ds.hd_orf(a, b) for a in pos] for b in pos])
        S = _spectrum(psrs, T, pars['gw_log10_A'], pars['gw_gamma'])

        assert np.allclose(Phi, np.kron(G, np.diag(S)))
        assert gp.Phi_inv is not None and gp.factors is not None

        fast, ldfast = gp.Phi_inv(pars)
        ref, ldref = gp.Phi.make_inv()(pars)
        assert np.allclose(np.array(fast), np.array(ref), rtol=1e-10, atol=0)
        assert np.isclose(float(ldfast), float(ldref), rtol=1e-12)
