"""Small maps and datasets shared by the tests and make_golden.py."""

import numpy as np
from astropy.io import fits

from trm.doppler import Default, Image, Map, Spectra, Data

TZERO, PERIOD, QUAD, VFINE = 0., 0.1, 0., 10.
HALPHA, HEI = 6562.76, 6678.15


def image(rng, itype, default, nxy=16, nz=1, wave=HALPHA, scale=1.,
          gamma=20., vxy=40., vz=60.):
    shape = (nz, nxy, nxy) if nz > 1 else (nxy, nxy)
    data = rng.uniform(0.5, 1.5, shape)
    wave = np.atleast_1d(np.asarray(wave, dtype=np.float64))
    gamma = np.full(len(wave), gamma, dtype=np.float32)
    # scale must be an ndarray: Image only accepts other iterables through
    # collections.Iterable, which Python 3.10 removed
    scale = np.atleast_1d(np.asarray(scale, dtype=np.float32))
    return Image(data, itype, vxy, wave, gamma, default, scale=scale,
                 vz=vz if nz > 1 else None)


def spectra(rng, w1, w2, nwave, nspec, fwhm, nsub=1):
    wave = np.linspace(w1, w2, nwave) + rng.uniform(-0.05, 0.05, (nspec, 1))
    time = np.sort(rng.uniform(0., 2*PERIOD, nspec))
    expose = np.full(nspec, PERIOD/40, dtype=np.float32)
    nsub = np.resize(np.atleast_1d(nsub), nspec).astype(np.int32)
    flux = np.zeros((nspec, nwave), dtype=np.float32)
    ferr = np.ones((nspec, nwave), dtype=np.float32)
    return Spectra(flux, ferr, wave, time, expose, nsub, fwhm)


def combined(seed=1):
    """A 3D image followed by two 2D ones, on two datasets. The last image
    falls outside the first dataset and the second outside the second."""
    rng = np.random.default_rng(seed)
    images = [
        image(rng, 1, Default.gauss3d(1., 80., 120., 0.3, 100.),
              nxy=20, nz=5, wave=[HALPHA, HEI], scale=[1., 0.4]),
        image(rng, 3, Default.gauss2d(1., 80.)),
        image(rng, 10, Default.uniform(0.9), wave=HEI),
    ]
    dmap = Map(fits.Header(), images, TZERO, PERIOD, QUAD, VFINE)
    data = Data(fits.Header(), [
        spectra(rng, 6540., 6590., 150, 20, 90., nsub=[1, 3]),
        spectra(rng, 6660., 6700., 100, 12, 120.),
    ])
    return dmap, data


def single(itype, seed=2):
    """One 2D image of the given type on one dataset."""
    rng = np.random.default_rng(seed)
    dmap = Map(fits.Header(), [image(rng, itype, Default.uniform(1.))],
               TZERO, PERIOD, QUAD, VFINE)
    data = Data(fits.Header(), [spectra(rng, 6540., 6590., 150, 20, 90.)])
    return dmap, data


def mapvec(dmap):
    return np.concatenate([im.data.ravel() for im in dmap.data])


def setmap(dmap, vec):
    n = 0
    for im in dmap.data:
        im.data[...] = vec[n:n+im.data.size].reshape(im.data.shape)
        n += im.data.size


def datvec(data):
    return np.concatenate([sp.flux.ravel() for sp in data.data])


def setdat(data, vec):
    n = 0
    for sp in data.data:
        sp.flux[...] = vec[n:n+sp.flux.size].reshape(sp.flux.shape)
        n += sp.flux.size
