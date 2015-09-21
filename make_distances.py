import os
from itertools import izip

import tables
import numpy as np
from astropy.table import Table, vstack
from Chandra.Time import DateTime
import healpy as hp
from agasc import sphere_dist, get_star
import matplotlib.pyplot as plt


NSIDE = 16
MIN_ACA_DIST = 25. / 3600  # 25 arcmin min separation
MAX_ACA_DIST = 2.1  # degrees corner to corner
if 'STAR_CACHE' not in globals():
    STAR_CACHE = {}


def make_distances_h5(max_mag=10.5, outfile='distances.h5', microagasc_file='microagasc.fits'):
    if os.path.exists(outfile):
        raise IOError('{} already exists'.format(outfile))
    agasc = get_microagasc(microagasc_file, max_mag=max_mag)
    t = get_all_dists(agasc)
    make_h5_file(t, outfile)


def make_microagasc(max_mag=10.5, outfile='microagasc.fits'):
    import tables
    with tables.openFile('/proj/sot/ska/data/agasc/miniagasc.h5') as h5:
        print('Reading miniagasc ...')
        mag_aca = h5.root.data.col('MAG_ACA')
        ok = mag_aca < max_mag
        mag_aca = mag_aca[ok]
        ra = h5.root.data.col('RA')[ok]
        ra_pm = h5.root.data.col('PM_RA')[ok]
        dec = h5.root.data.col('DEC')[ok]
        dec_pm = h5.root.data.col('PM_DEC')[ok]
        agasc_id = h5.root.data.col('AGASC_ID')[ok]
        print('   Done.')

    t = Table([agasc_id, ra, dec, ra_pm, dec_pm, mag_aca],
              names=['agasc_id', 'ra', 'dec', 'ra_pm', 'dec_pm', 'mag_aca'])
    t.write(outfile, overwrite=True)


def get_microagasc(filename='microagasc.fits', date=None, max_mag=10.5):
    if not os.path.exists(filename):
        make_microagasc()
    t = Table.read(filename, format='fits')
    t = t[t['mag_aca'] < max_mag]

    # Compute the multiplicative factor to convert from the AGASC proper motion
    # field to degrees.  The AGASC PM is specified in milliarcsecs / year, so this
    # is dyear * (degrees / milliarcsec)
    agasc_equinox = DateTime('2000:001:00:00:00.000')
    dyear = (DateTime(date) - agasc_equinox) / 365.25
    pm_to_degrees = dyear / (3600. * 1000.)

    ra = t['ra']
    dec = t['dec']
    ra_pm = t['ra_pm']
    dec_pm = t['dec_pm']

    ok = ra_pm != -9999  # Select stars with an available PM correction
    ra[ok] = ra[ok] + ra_pm[ok] * pm_to_degrees
    ok = dec_pm != -9999  # Select stars with an available PM correction
    dec[ok] = dec[ok] + dec_pm[ok] * pm_to_degrees

    phi = np.radians(ra)
    theta = np.radians(90 - dec)
    ipix = hp.ang2pix(NSIDE, theta, phi)
    t['ipix'] = ipix.astype(np.uint16 if np.max(ipix) < 65536 else np.uint32)
    t.sort('ipix')
    return t


def get_stars_for_region(agasc, ipix):
    i0, i1 = np.searchsorted(agasc['ipix'], [ipix, ipix + 1])
    return agasc[i0:i1]


def _get_dists(s0, s1):
    idx0, idx1 = np.mgrid[slice(0, len(s0)), slice(0, len(s1))]
    idx0 = idx0.ravel()
    idx1 = idx1.ravel()

    ok = s0['agasc_id'][idx0] < s1['agasc_id'][idx1]
    idx0 = idx0[ok]
    idx1 = idx1[ok]

    dists = sphere_dist(s0['ra'][idx0], s0['dec'][idx0],
                        s1['ra'][idx1], s1['dec'][idx1])
    ok = (dists < MAX_ACA_DIST) & (dists > MIN_ACA_DIST)
    dists = dists[ok] * 3600  # convert to arcsec here
    idx0 = idx0[ok]
    idx1 = idx1[ok]
    dists = dists.astype(np.float32)
    id0 = s0['agasc_id'][idx0].astype(np.int32)
    id1 = s1['agasc_id'][idx1].astype(np.int32)
    # Store mag values in millimag as int16.  Clip for good measure.
    mag0 = np.clip(s0['mag_aca'][idx0] * 1000, -32000, 32000).astype(np.int16)
    mag1 = np.clip(s1['mag_aca'][idx1] * 1000, -32000, 32000).astype(np.int16)

    out = Table([dists, id0, id1, mag0, mag1],
                names=['dists', 'agasc_id0', 'agasc_id1', 'mag0', 'mag1'])
    return out


def get_dists_for_region(agasc, ipix):
    s0 = get_stars_for_region(agasc, ipix)
    ipix_neighbors = hp.get_all_neighbours(NSIDE, ipix)
    s1s = [get_stars_for_region(agasc, ipix_neighbor) for ipix_neighbor in ipix_neighbors]

    t_list = []
    for s1 in [s0] + s1s:
        t_list.append(_get_dists(s0, s1))
    out = vstack(t_list)
    return out


def make_h5_file(t, filename='distances.h5'):
    """Make a new h5 table to hold column from ``dat``."""
    filters = tables.Filters(complevel=5, complib='zlib')
    dat = np.array(t)
    with tables.openFile(filename, mode='a', filters=filters) as h5:
        h5.createTable(h5.root, "data", dat, "Data table", expectedrows=len(dat))
        h5.root.data.flush()

    with tables.openFile(filename, mode='a', filters=filters) as h5:
        print('Creating index')
        h5.root.data.cols.dists.createIndex()
        h5.flush()


def get_all_dists(agasc=None):
    if agasc is None:
        agasc = get_microagasc()
    t_list = []
    for ipix in xrange(hp.nside2npix(NSIDE)):
        print(ipix)
        t_list.append(get_dists_for_region(agasc, ipix))

    out = vstack(t_list)
    out.sort('dists')
    return out


def plot_stars_and_neighbors(agasc, ipix):
    s0 = get_stars_for_region(agasc, ipix)
    ipix_neighbors = hp.get_all_neighbours(NSIDE, ipix)
    s1s = [get_stars_for_region(agasc, ipix_neighbor) for ipix_neighbor in ipix_neighbors]
    plt.plot(s0['ra'], s0['dec'], '.c')
    for s1 in s1s:
        plt.plot(s1['ra'], s1['dec'], '.m')


def plot_dists(agasc, ipix):
    def _get_star(id0):
        if id0 not in STAR_CACHE:
            STAR_CACHE[id0] = get_star(id0)
        return STAR_CACHE[id0]

    dists, id0s, id1s = get_dists_for_region(agasc, ipix)
    plotted = set()
    for dist, id0, id1 in izip(dists, id0s, id1s):
        star0 = _get_star(id0)
        star1 = _get_star(id1)
        ra0 = star0['RA']
        ra1 = star1['RA']
        dec0 = star0['DEC']
        dec1 = star1['DEC']

        phi1 = np.radians(ra1)
        theta1 = np.radians(90 - dec1)
        ipix1 = hp.ang2pix(NSIDE, theta1, phi1)

        plt.plot([ra0, ra1],
                 [dec0, dec1], '-k', alpha=0.2)
        if id0 not in plotted:
            plt.plot([ra0], [dec0], 'ob')
        if id1 not in plotted:
            plt.plot([ra1], [dec1], 'ob' if ipix1 == ipix else 'or')
        dalt = sphere_dist(ra0, dec0,
                           ra1, dec1)
        print('{} {} {} {}'.format(id0, id1, dist, dalt * 3600))
