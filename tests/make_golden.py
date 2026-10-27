"""Records the outputs of the installed extension in golden.npz.

The doppler.cc rewrite must reproduce these. Regenerate only when a change
to the results is intended, from a build of the code being compared against:

    python tests/make_golden.py
"""

import os

import numpy as np

import trm.doppler as doppler
import fixtures as fx

GOLDEN = os.path.join(os.path.dirname(__file__), 'golden.npz')


def outputs():
    out = {}
    rng = np.random.default_rng(3)

    dmap, data = fx.combined()
    doppler.comdat(dmap, data)
    out['comdat'] = fx.datvec(data)

    fx.setdat(data, rng.normal(size=out['comdat'].size))
    doppler.datcom(data, dmap)
    out['datcom'] = fx.mapvec(dmap)

    dmap, data = fx.combined()
    doppler.comdef(dmap)
    out['comdef'] = fx.mapvec(dmap)

    # fit noisy data made from the map, starting from a flat map
    dmap, data = fx.combined()
    doppler.comdat(dmap, data)
    for sp in data.data:
        sp.ferr[...] = 0.02*np.abs(sp.flux).max()
        sp.flux += rng.normal(0., sp.ferr).astype(np.float32)
    fx.setmap(dmap, np.ones(fx.mapvec(dmap).size))
    doppler.memit(dmap, data, 3, 1.)
    out['memit'] = fx.mapvec(dmap)

    for itype in range(1, 11):
        dmap, data = fx.single(itype)
        doppler.comdat(dmap, data)
        out[f'comdat{itype}'] = fx.datvec(data)

    return out


if __name__ == '__main__':
    np.savez(GOLDEN, **outputs())
