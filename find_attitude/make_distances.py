# Licensed under a 3-clause BSD style license - see LICENSE.rst
import argparse
import os

import agasc
import astropy.units as u
import astropy_healpix as hp
import numpy as np
import tables
from astropy.coordinates import SkyCoord
from astropy.table import Table

MIN_ACA_DIST = 25.0 * u.arcsec  # 25 arcsec min separation (FFS won't find closer)
MAX_ACA_DIST = 2.0 * u.deg  # degrees corner to corner (actual max is 7148 arcsec)
MAX_MAG = 10.0  # Max mag to include in the distances file
DATE_DISTANCES = "2025:001"  # Date for proper motion correction in distances
HEALPIX_NSIDE = 64  # Set to have approximately 1 square degree pixels (49152 on sky)
HEALPIX_ORDER = "nested"
SKA = os.environ["SKA"]

# Modify this locally before importing if needed.
AGASC_VERSION = "1p8"
DISTANCES_FILE = f"distances_{AGASC_VERSION}.h5"


def get_options_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Make distances h5 file")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing distances file",
    )
    parser.add_argument(
        "--outfile",
        default=DISTANCES_FILE,
        help=f"Output file name (default={DISTANCES_FILE})",
    )
    return parser


def make_distances_h5(outfile, overwrite=False):
    if os.path.exists(outfile) and not overwrite:
        raise IOError("{} already exists".format(outfile))
    stars = get_microagasc()
    dists = get_dists(stars)
    make_h5_file(dists, outfile)


def get_microagasc():
    """Get subset of miniagasc columns for stars brighter than MAX_MAG.

    Proper motion is corrected to DATE_DISTANCES.
    """
    path = agasc.get_agasc_filename("miniagasc_*", version=AGASC_VERSION)
    print(f"Reading miniagasc {path}")
    with tables.open_file(path) as h5:
        mag_aca = h5.root.data.col("MAG_ACA")
        ok = mag_aca < MAX_MAG
        mag_aca = mag_aca[ok]
        ra = h5.root.data.col("RA")[ok]
        ra_pm = h5.root.data.col("PM_RA")[ok]
        dec = h5.root.data.col("DEC")[ok]
        dec_pm = h5.root.data.col("PM_DEC")[ok]
        agasc_id = h5.root.data.col("AGASC_ID")[ok]
        epoch = h5.root.data.col("EPOCH")[ok]

    stars = Table(
        [agasc_id, ra, dec, ra_pm, dec_pm, mag_aca, epoch],
        names=["AGASC_ID", "RA", "DEC", "PM_RA", "PM_DEC", "MAG_ACA", "EPOCH"],
    )
    agasc.agasc.add_pmcorr_columns(stars, date=DATE_DISTANCES)

    lon = stars["RA_PMCORR"] * u.deg
    lat = stars["DEC_PMCORR"] * u.deg
    ipix = hp.lonlat_to_healpix(lon, lat, nside=HEALPIX_NSIDE, order=HEALPIX_ORDER)
    stars["ipix"] = ipix.astype(
        np.uint16 if hp.nside_to_npix(HEALPIX_NSIDE) < 65536 else np.uint32
    )
    return stars


def get_dists(stars: Table) -> Table:
    # Get all pairs of stars within MAX_ACA_DIST of each other
    print("Finding star pairs")
    sc = SkyCoord(ra=stars["RA_PMCORR"], dec=stars["DEC_PMCORR"], unit="deg")
    idxs0, idxs1, dists, _ = sc.search_around_sky(sc, MAX_ACA_DIST)

    # Remove self-matches and duplicates (i.e. (i, j) and (j, i)), and too-small dists
    ok = (idxs0 < idxs1) & (dists > MIN_ACA_DIST)
    idxs0 = idxs0[ok]
    idxs1 = idxs1[ok]
    dists = dists[ok]

    # Set up for a table
    dists = dists.to_value(u.arcsec).astype(np.float32)
    id0 = stars["AGASC_ID"][idxs0].astype(np.int32)
    id1 = stars["AGASC_ID"][idxs1].astype(np.int32)
    # Store mag values in millimag as int16.  Clip for good measure.
    mag0 = np.clip(stars["MAG_ACA"][idxs0] * 1000, -32000, 32000).astype(np.int16)
    mag1 = np.clip(stars["MAG_ACA"][idxs1] * 1000, -32000, 32000).astype(np.int16)
    pix0 = stars["ipix"][idxs0].astype(np.uint16)
    pix1 = stars["ipix"][idxs1].astype(np.uint16)

    out = Table(
        [dists, id0, id1, mag0, mag1, pix0, pix1],
        names=["dists", "agasc_id0", "agasc_id1", "mag0", "mag1", "pix0", "pix1"],
    )
    print("Sorting by dists")
    out.sort("dists")
    return out


def make_h5_file(dists, filename=DISTANCES_FILE):
    """Make a new h5 table to hold column from ``dat``."""
    filters = tables.Filters(complevel=5, complib="zlib")
    dat = np.array(dists)
    print(f"Writing data to {filename}")
    with tables.open_file(filename, mode="w", filters=filters) as h5:
        h5.create_table(h5.root, "data", dat, "Data table", expectedrows=len(dat))

        # Add some metadata
        h5.root.data.attrs.max_mag = MAX_MAG
        h5.root.data.attrs.min_aca_dist = MIN_ACA_DIST.to_value(u.deg)
        h5.root.data.attrs.max_aca_dist = MAX_ACA_DIST.to_value(u.deg)
        h5.root.data.attrs.date_distances = DATE_DISTANCES
        h5.root.data.attrs.healpix_nside = HEALPIX_NSIDE
        h5.root.data.attrs.healpix_order = HEALPIX_ORDER
        h5.root.data.attrs.agasc_version = AGASC_VERSION

        h5.root.data.flush()

    print("Creating index (takes a minute or two)")
    with tables.open_file(filename, mode="a", filters=filters) as h5:
        h5.root.data.cols.dists.create_index()
        h5.flush()


def main():
    args = get_options_parser().parse_args()
    make_distances_h5(args.outfile, args.overwrite)


if __name__ == "__main__":
    main()
