import numpy as np
import pytest
from astropy.io import fits

import trm.doppler as doppler
from trm.doppler import Default
import fixtures as fx
from make_golden import GOLDEN, outputs

CKMS = 299792.458


def test_adjoint():
    """datcom must be the transpose of comdat: <op(x), y> = <x, tr(y)>."""
    rng = np.random.default_rng(4)
    dmap, data = fx.combined()

    fx.setmap(dmap, rng.normal(size=fx.mapvec(dmap).size))
    x = fx.mapvec(dmap).astype(np.float64)
    doppler.comdat(dmap, data)
    op_x = fx.datvec(data).astype(np.float64)

    fx.setdat(data, rng.normal(size=op_x.size))
    y = fx.datvec(data).astype(np.float64)
    doppler.datcom(data, dmap)
    tr_y = fx.mapvec(dmap).astype(np.float64)

    assert np.dot(op_x, y) == pytest.approx(np.dot(x, tr_y), rel=1e-4)


def test_image_order():
    """The model doesn't depend on the order of the images, including
    images that miss a dataset and a mix of 2D and 3D."""
    dmap, data = fx.combined()
    doppler.comdat(dmap, data)
    model = fx.datvec(data)

    dmap.data.reverse()
    doppler.comdat(dmap, data)
    np.testing.assert_allclose(fx.datvec(data), model, rtol=1e-5,
                               atol=1e-5*np.abs(model).max())


@pytest.fixture(scope='module')
def results():
    return outputs()


@pytest.mark.parametrize('name', sorted(np.load(GOLDEN).files))
def test_golden(results, name):
    gold = np.load(GOLDEN)[name]
    # memsys iterations amplify rounding differences
    rtol = 1e-4 if name == 'memit' else 1e-5
    np.testing.assert_allclose(results[name], gold, rtol=rtol,
                               atol=rtol*np.abs(gold).max())


@pytest.mark.parametrize('nz', [1, 9])
def test_default_conserves_flux(nz):
    """The GAUSS default blurring neither gains nor loses flux."""
    if nz > 1:
        default = Default.gauss3d(1., 120., 60., 0., 1.)
    else:
        default = Default.gauss2d(1., 120.)
    im = fx.image(np.random.default_rng(5), 1, default, nxy=41, nz=nz)
    im.data[...] = 0.
    im.data.flat[im.data.size // 2] = 1.
    dmap = doppler.Map(fits.Header(), [im], fx.TZERO, fx.PERIOD, fx.QUAD,
                       fx.VFINE)

    doppler.comdef(dmap)
    assert im.data.sum() == pytest.approx(1., rel=1e-5)


def test_projection_conserves_flux():
    """Each spectrum integrates to sfac * vxy**2 * the image total."""
    dmap, data = fx.single(1)
    doppler.comdat(dmap, data)
    im, sp = dmap.data[0], data.data[0]

    # doppler.cc divides each pixel by its width in velocity / wavelength
    edge = (sp.wave[:, :-1] + sp.wave[:, 1:]) / 2.
    width = CKMS*np.diff(edge, axis=1)/im.wave[0]/sp.wave[:, 1:-1]
    total = (sp.flux[:, 1:-1]*width).sum(axis=1)

    expect = dmap.sfac*im.vxy**2*im.data.sum(dtype=np.float64)
    np.testing.assert_allclose(total, expect, rtol=1e-4)


def test_single_pixel():
    """One bright pixel at (vx, vy) traces gamma - vx cos 2pi phi
    + vy sin 2pi phi."""
    dmap, data = fx.single(1)
    im, sp = dmap.data[0], data.data[0]
    ny, nx = im.data.shape
    ix, iy = 3, 12
    im.data[...] = 0.
    im.data[iy, ix] = 1.
    doppler.comdat(dmap, data)

    vx = (ix - (nx-1)/2.)*im.vxy
    vy = (iy - (ny-1)/2.)*im.vxy
    phi = 2.*np.pi*(sp.time - fx.TZERO)/fx.PERIOD
    v = im.gamma[0] - vx*np.cos(phi) + vy*np.sin(phi)

    centroid = (sp.flux*sp.wave).sum(axis=1)/sp.flux.sum(axis=1)
    vc = CKMS*(centroid/im.wave[0] - 1.)
    # the fine array is vfine = 10 km/s, and pixels round onto it
    np.testing.assert_allclose(vc, v, atol=6.)
