# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
import warnings

import agasc
import astropy.units as u
import astropy_healpix as hp
import matplotlib.pyplot as plt
import numpy as np
import tables
import tqdm
from astropy.table import Table, vstack

NSIDE = 16
MIN_ACA_DIST = 25.0 / 3600  # 25 arcmin min separation
MAX_ACA_DIST = 2.1  # degrees corner to corner

# Define a global cache for star data which persists in ipython for re-run of module.
if "STAR_CACHE" not in globals():
    STAR_CACHE = {}

HEALPIX_ORDER = "nested"
SKA = os.environ["SKA"]

# Modify this locally before importing if needed.
AGASC_VERSION = "1p8"
DISTANCES_FILE = f"distances_{AGASC_VERSION}.h5"
MICROAGASC_FILE = f"microagasc_{AGASC_VERSION}.fits"


def make_distances_h5(
    max_mag=10.5,
    outfile=DISTANCES_FILE,
    microagasc_file=MICROAGASC_FILE,
):
    if os.path.exists(outfile):
        raise IOError("{} already exists".format(outfile))
    agasc = get_microagasc(microagasc_file, max_mag=max_mag)
    dists = get_all_dists(agasc)
    make_h5_file(dists, outfile)


def make_microagasc(max_mag=10.5, outfile=MICROAGASC_FILE):
    path = agasc.get_agasc_filename("miniagasc_*", version=AGASC_VERSION)
    if path.exists():
        return

    print(f"Reading miniagasc {path}")
    with tables.open_file(path) as h5:
        mag_aca = h5.root.data.col("MAG_ACA")
        ok = mag_aca < max_mag
        mag_aca = mag_aca[ok]
        ra = h5.root.data.col("RA")[ok]
        ra_pm = h5.root.data.col("PM_RA")[ok]
        dec = h5.root.data.col("DEC")[ok]
        dec_pm = h5.root.data.col("PM_DEC")[ok]
        agasc_id = h5.root.data.col("AGASC_ID")[ok]
        epoch = h5.root.data.col("EPOCH")[ok]

    dat = Table(
        [agasc_id, ra, dec, ra_pm, dec_pm, mag_aca, epoch],
        names=["AGASC_ID", "RA", "DEC", "PM_RA", "PM_DEC", "MAG_ACA", "EPOCH"],
    )
    print(f"Writing {outfile}")
    dat.write(outfile, overwrite=True)


def get_microagasc(filename=MICROAGASC_FILE, date="2025:001", max_mag=10.5):
    if not os.path.exists(filename):
        make_microagasc()
    stars = Table.read(filename, format="fits")
    stars = stars[stars["MAG_ACA"] < max_mag]

    agasc.agasc.add_pmcorr_columns(stars, date=date)

    lon = stars["RA_PMCORR"] * u.deg
    lat = stars["DEC_PMCORR"] * u.deg
    ipix = hp.lonlat_to_healpix(lon, lat, nside=NSIDE)
    stars["ipix"] = ipix.astype(np.uint16 if np.max(ipix) < 65536 else np.uint32)
    stars.sort("ipix")
    return stars


def get_stars_for_region(stars, ipix):
    i0, i1 = np.searchsorted(stars["ipix"], [ipix, ipix + 1])
    return stars[i0:i1]


def _get_dists(s0, s1):
    idx0, idx1 = np.mgrid[slice(0, len(s0)), slice(0, len(s1))]
    idx0 = idx0.ravel()
    idx1 = idx1.ravel()

    ok = s0["AGASC_ID"][idx0] < s1["AGASC_ID"][idx1]
    idx0 = idx0[ok]
    idx1 = idx1[ok]

    dists = agasc.sphere_dist(
        s0["RA_PMCORR"][idx0],
        s0["DEC_PMCORR"][idx0],
        s1["RA_PMCORR"][idx1],
        s1["DEC_PMCORR"][idx1],
    )
    ok = (dists < MAX_ACA_DIST) & (dists > MIN_ACA_DIST)
    dists = dists[ok] * 3600  # convert to arcsec here
    idx0 = idx0[ok]
    idx1 = idx1[ok]
    dists = dists.astype(np.float32)
    id0 = s0["AGASC_ID"][idx0].astype(np.int32)
    id1 = s1["AGASC_ID"][idx1].astype(np.int32)
    # Store mag values in millimag as int16.  Clip for good measure.
    mag0 = np.clip(s0["MAG_ACA"][idx0] * 1000, -32000, 32000).astype(np.int16)
    mag1 = np.clip(s1["MAG_ACA"][idx1] * 1000, -32000, 32000).astype(np.int16)

    out = Table(
        [dists, id0, id1, mag0, mag1],
        names=["dists", "agasc_id0", "agasc_id1", "mag0", "mag1"],
    )
    return out


def get_dists_for_region(stars, ipix):
    s0 = get_stars_for_region(stars, ipix)
    ipix_neighbors = hp.neighbours(ipix, nside=NSIDE)
    s1s = [
        get_stars_for_region(stars, ipix_neighbor) for ipix_neighbor in ipix_neighbors
    ]

    t_list = [_get_dists(s0, s1) for s1 in [s0] + s1s]
    out = vstack(t_list)
    return out


def make_h5_file(t, filename=DISTANCES_FILE):
    """Make a new h5 table to hold column from ``dat``."""
    filters = tables.Filters(complevel=5, complib="zlib")
    dat = np.array(t)
    with tables.open_file(filename, mode="a", filters=filters) as h5:
        h5.create_table(h5.root, "data", dat, "Data table", expectedrows=len(dat))
        h5.root.data.flush()

    with tables.open_file(filename, mode="a", filters=filters) as h5:
        print("Creating index")
        h5.root.data.cols.dists.create_index()
        h5.flush()


def get_all_dists(stars=None):
    if stars is None:
        stars = get_microagasc()
    t_list = []
    for ipix in tqdm.tqdm(range(hp.nside_to_npix(NSIDE))):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t_list.append(get_dists_for_region(stars, ipix))

    print("Stacking")
    out = vstack(t_list)
    print("Sorting by dists")
    out.sort("dists")
    return out


def plot_stars_and_neighbors(stars, ipix):
    s0 = get_stars_for_region(stars, ipix)
    ipix_neighbors = hp.neighbours(ipix, nside=NSIDE)
    s1s = [
        get_stars_for_region(stars, ipix_neighbor) for ipix_neighbor in ipix_neighbors
    ]
    plt.plot(s0["RA_PMCORR"], s0["DEC_PMCORR"], ".c")
    for s1 in s1s:
        plt.plot(s1["RA_PMCORR"], s1["DEC_PMCORR"], ".m")


def plot_dists(stars, ipix):
    def _get_star(id0):
        if id0 not in STAR_CACHE:
            STAR_CACHE[id0] = agasc.get_star(id0)
        return STAR_CACHE[id0]

    dists, id0s, id1s = get_dists_for_region(stars, ipix)
    plotted = set()
    for dist, id0, id1 in zip(dists, id0s, id1s):
        star0 = _get_star(id0)
        star1 = _get_star(id1)
        ra0 = star0["RA_PMCORR"]
        ra1 = star1["RA_PMCORR"]
        dec0 = star0["DEC_PMCORR"]
        dec1 = star1["DEC_PMCORR"]

        phi1 = np.deg2rad(ra1) * u.rad
        theta1 = np.deg2rad(90 - dec1) * u.rad
        ipix1 = hp.lonlat_to_healpix(theta1, phi1, nside=NSIDE)

        plt.plot([ra0, ra1], [dec0, dec1], "-k", alpha=0.2)
        if id0 not in plotted:
            plt.plot([ra0], [dec0], "ob")
        if id1 not in plotted:
            plt.plot([ra1], [dec1], "ob" if ipix1 == ipix else "or")
        dalt = stars.sphere_dist(ra0, dec0, ra1, dec1)
        print("{} {} {} {}".format(id0, id1, dist, dalt * 3600))
