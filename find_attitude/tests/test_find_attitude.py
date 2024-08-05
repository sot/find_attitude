# Licensed under a 3-clause BSD style license - see LICENSE.rst
from pprint import pformat

import agasc
import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import ascii
from chandra_aca.transform import radec_to_yagzag, yagzag_to_pixels
from Quaternion import Quat

from find_attitude import (
    find_attitude_solutions,
    get_agasc_pairs_attribute,
    get_dists_yag_zag,
    get_stars_from_maude,
    get_stars_from_text,
    logger,
)

MAX_MAG = get_agasc_pairs_attribute("max_mag")

try:
    import maude

    maude.get_msids(
        msids="ccsdsid", start="2016:001:00:00:00", stop="2016:001:00:00:02"
    )
except Exception:
    HAS_MAUDE = False
else:
    HAS_MAUDE = True


def get_stars(
    ra=119.98,
    dec=-78,
    roll=0,
    select=None,
    brightest=True,
    sigma_1axis=0.4,
    sigma_mag=0.2,
    date="2025:001",
):
    # Make test results reproducible
    np.random.seed(int(abs(ra) * 100 + abs(dec) * 10 + abs(roll)))

    if select is None:
        select = slice(None, 8)
    agasc_file = agasc.get_agasc_filename("miniagasc_*")
    stars = agasc.get_agasc_cone(ra, dec, 1.2, date=date, agasc_file=agasc_file)
    stars = stars[stars["MAG_ACA"] < MAX_MAG]
    remove_close_pairs(stars, radius=5 * u.arcsec, both=True)
    remove_close_pairs(stars, radius=25 * u.arcsec, both=False)
    stars = stars[stars["MAG_ACA"] > 5.0]

    if brightest:
        stars.sort("MAG_ACA")
    else:
        index = np.arange(len(stars))
        np.random.shuffle(index)
        stars = stars[index]
    yags, zags = radec_to_yagzag(
        stars["RA_PMCORR"], stars["DEC_PMCORR"], Quat([ra, dec, roll])
    )
    stars["YAG_ERR"] = np.random.normal(scale=sigma_1axis, size=len(stars))
    stars["ZAG_ERR"] = np.random.normal(scale=sigma_1axis, size=len(stars))
    stars["MAG_ERROR"] = np.random.normal(scale=sigma_mag, size=len(stars))
    stars["YAG"] = yags + stars["YAG_ERR"]
    stars["ZAG"] = zags + stars["ZAG_ERR"]
    stars["MAG_ACA"] += stars["MAG_ERROR"]
    stars["RA"] = stars["RA_PMCORR"]
    stars["DEC"] = stars["DEC_PMCORR"]
    stars = stars[
        "AGASC_ID",
        "RA",
        "DEC",
        "YAG",
        "YAG_ERR",
        "ZAG",
        "ZAG_ERR",
        "MAG_ACA",
        "MAG_ERROR",
    ]

    # Make sure stars are on the CCD
    rows, cols = yagzag_to_pixels(stars["YAG"], stars["ZAG"], allow_bad=True)
    ok = (np.abs(rows) < 506.0) & (np.abs(cols) < 506.0)
    stars = stars[ok]

    stars = stars[select].copy()

    return stars


def remove_close_pairs(stars, radius: u.Quantity, both=False):
    """
    Remove one or both stars that are close to each other.

    In AGASC 1.8, very close star pairs can have ASPQ1=0, so we need to remove them
    here. Otherwise we can (and do, in testing) get a mismatch where the solution has
    the wrong star. E.g. ra = 278.33913, dec = -52.29810, roll = 258.51642.
    """
    sc = SkyCoord(ra=stars["RA"], dec=stars["DEC"], unit="deg")
    idxs1, idxs2, d2ds, _ = sc.search_around_sky(sc, radius)
    d2ds = d2ds.to(u.arcsec)
    idxs_drop = set()
    for idx1, idx2, d2d in zip(idxs1, idxs2, d2ds):
        if idx1 < idx2 and idx1 not in idxs_drop:
            if both:
                star = stars[idx1]
                logger.debug(
                    f"Removing star(1) {star['AGASC_ID']} with "
                    f"ra={star['RA']:.5f}, dec={star['DEC']:.5f} mag={star['MAG_ACA']:.2f} dist={d2d:.2f}"
                )
                idxs_drop.add(idx1)
            idxs_drop.add(idx2)
            star = stars[idx2]
            logger.debug(
                f"Removing star(2) {star['AGASC_ID']} with "
                f"ra={star['RA']:.5f}, dec={star['DEC']:.5f} mag={star['MAG_ACA']:.2f} dist={d2d:.2f}"
            )
    stars.remove_rows(list(idxs_drop))


def find_overlapping_distances(min_n_overlap=3, tolerance=3.0):
    while True:
        ra = np.random.uniform(0, 360)
        dec = np.random.uniform(-90, 90)
        roll = np.random.uniform(0, 360)
        logger.debug(ra, dec, roll)
        stars = get_stars(ra, dec, roll, sigma_1axis=0.001, sigma_mag=0.2)
        dist_table = get_dists_yag_zag(stars["YAG"], stars["ZAG"])
        dists = dist_table["dists"]

        n_overlap = 0
        for i, d0 in enumerate(dists):
            for d1 in dists[i + 1 :]:
                if np.abs(d0 - d1) < tolerance:
                    n_overlap += 1
        if n_overlap >= min_n_overlap:
            return ra, dec, roll, stars, dists


def test_overlapping_distances(tolerance=3.0):
    """
    Test a case where distance 5-7 = 864.35 and 5-1 is 865.808
    """
    global ra, dec, roll, stars, agasc_id_star_maps, g_geom_match, g_dist_match
    ra, dec, roll = (
        131.1371559426714,
        65.25369723989581,
        112.4351393383257,
    )  # 3 overlaps, 3.0 tol
    stars = get_stars(ra, dec, roll, sigma_1axis=0.004, sigma_mag=0.2, brightest=True)

    solutions = find_attitude_solutions(stars)
    check_output(solutions, stars, ra, dec, roll)


def _test_random(n_iter=1, sigma_1axis=0.4, sigma_mag=0.2, brightest=True):
    np.random.seed(0)
    for _ in range(n_iter):
        global ra, dec, roll, stars, agasc_id_star_maps, g_geom_match, g_dist_match
        ra = np.random.uniform(0, 360)
        dec = np.random.uniform(-90, 90)
        roll = np.random.uniform(0, 360)
        n_stars = np.random.randint(4, 8)
        stars = get_stars(
            ra,
            dec,
            roll,
            sigma_1axis=sigma_1axis,
            sigma_mag=sigma_mag,
            brightest=brightest,
        )
        stars = stars[:n_stars]
        solutions = find_attitude_solutions(stars)
        check_output(solutions, stars, ra, dec, roll)


def test_multiple_solutions():
    global stars, solutions
    ra, dec, roll = 190.3286989834239, 22.698443628394102, 111.51056234863053
    stars = ascii.read(
        """
 AGASC_ID       RA           DEC           YAG           YAG_ERR          ZAG           ZAG_ERR     MAG_ACA    MAG_ERROR
260863544 189.758890214 22.6594185253  567.401869049  0.615027692698  1811.01764078 -0.293256251994 6.47267   0.280073674992
189804208 191.093682446 21.9926851164 -3294.27782744 -0.181642428933 -1445.81915638  0.812100708212 7.94189   0.199006301588
189800592 190.343952117 21.8305748811 -2925.81601843    -0.381447575  1098.70907129  0.455625742141  9.0004   0.265371250728
260856472 190.787449674 23.2708932848  1363.51998803  0.294404741658 -2168.30508446 -0.261280462907 8.88882  -0.186751057505
260858632 190.173364997 22.9716741441  1103.78516827 -0.370985727476  118.089547978  -0.14060714735 9.50322    0.12366348923
189811352 190.148298323 22.1508704117 -1612.84681732  0.241090402814  1282.01534928 -0.309449604245 9.61865   0.234064435566
260838832 190.053449838 23.5688063028  3249.58724799  0.683018406574 -304.398501017 -0.173295600276 9.46925 -0.0731231730791
260851360 190.235827288  22.625601259 -130.370520041    0.3501954033  383.398889404  0.168175442249 9.73112  0.0393770657269
"""
    )
    solutions = find_attitude_solutions(stars)
    check_output(solutions, stars, ra, dec, roll)


def check_output(solutions, stars, ra, dec, roll):
    logger.debug("*********************************************")
    logger.debug("")

    assert len(solutions) > 0
    for solution in solutions:
        att_fit = solution["att_fit"]

        att_in = Quat([ra, dec, roll])
        d_att = att_in.inv() * att_fit
        d_roll, d_pitch, d_yaw = d_att.roll0, d_att.pitch, d_att.yaw

        logger.debug("============================")
        logger.debug(f"Input: RA Dec Roll = {ra:.5f} {dec:.5f} {roll:.5f}")
        logger.debug(
            f"Solve: RA Dec Roll = {att_fit.equatorial[0]:.5f} {att_fit.equatorial[1]:.5f} {att_fit.equatorial[2]:.5f}"
        )
        logger.debug(solution["summary"])
        if solution["bad_fit"]:
            logger.debug("BAD FIT!")
            continue

        d_roll_lim = {2: 100, 3: 80, 4: 70, 5: 60, 6: 55, 7: 50, 8: 45}[len(stars)]
        assert d_roll < d_roll_lim
        assert d_pitch < 1.0
        assert d_yaw < 1.0

        ok = ~solution["summary"]["m_agasc_id"].mask
        sok = solution["summary"][ok]
        assert np.all(sok["AGASC_ID"] == sok["m_agasc_id"])

    assert sum(1 for s in solutions if not s["bad_fit"]) == 1
    logger.debug("*********************************************\n")


def test_ra_dec_roll(
    ra=115.770455413,
    dec=-75.6580358662,
    roll=86.4089128685,
    brightest=True,
    provide_mags=True,
    sigma_1axis=0.4,
    sigma_mag=0.2,
):
    global stars, agasc_id_star_maps, g_geom_match, g_dist_match, solutions
    stars = get_stars(
        ra, dec, roll, sigma_1axis=sigma_1axis, sigma_mag=sigma_mag, brightest=brightest
    )
    solutions = find_attitude_solutions(stars, tolerance=2.5)
    check_output(solutions, stars, ra, dec, roll)


def test_get_stars_from_greta():
    text = """
         OBSID  17595                    AOACINTT  1697.656   AOACPRGS        0       AOCINTNP      ENAB     AODITHR3  3.7928e-05

                                                                            AOFWAIT  NOWT      Acquisition
       ACA     IMAGE Status  Image   Fid Lt      Centroid Angle     Star    AOACASEQ KALM        Success        GLOBAL STATUS
       MEAS      #   Flags   Functn  Flag        Y         Z        Mag     AOFSTAR  GUID     AORFSTR1    1     AOACPWRF   OK
      IMAGE 0  0      FID    TRAK    FID      922.53   -1004.43      7.2    AONSTARS    5     AORFSTR2    4     AOACRAMF   OK
      IMAGE 1  1      FID    TRAK    FID     2141.40     896.38      7.2    AOKALSTR    5                       AOACROMF   OK
      IMAGE 2  2      FID    TRAK    FID    -1825.12     893.98      7.1                      ENTRY 0    ID     AOACSUMF   OK
      IMAGE 3  3     STAR    TRAK   STAR      223.33      55.83      9.7    SUCCESS FLAGS     ENTRY 1    ID     AOACHIBK   OK
      IMAGE 4  4     STAR    TRAK   STAR     -453.10   -2084.10      9.6    AOACQSUC  SUC     ENTRY 2    ID
      IMAGE 5  5     STAR    TRAK   STAR    -1255.12     196.58      9.2    AOGDESUC  SUC     ENTRY 3    ID     AOACCALF   OK
      IMAGE 6  6     STAR    TRAK   STAR      598.18    2287.97      9.6    AOBRTSUC  SUC     ENTRY 4    ID     AOACRSET   OK
      IMAGE 7  7     STAR    TRAK   STAR     2311.45    1140.60      9.8    AOFIDSUC  SUC     ENTRY 5    ID     AOACSNTY   OK
                                                                            AOACRPT     1     ENTRY 6    ID
                                                                            AORSTART ENAB     ENTRY 7  NOID
    """
    expected = """
    slot type function fid    YAG      ZAG   MAG_ACA
    ---- ---- -------- ---- -------- ------- -------
       3 STAR     TRAK STAR   223.33   55.83     9.7
       4 STAR     TRAK STAR   -453.1 -2084.1     9.6
       5 STAR     TRAK STAR -1255.12  196.58     9.2
       6 STAR     TRAK STAR   598.18 2287.97     9.6
       7 STAR     TRAK STAR  2311.45  1140.6     9.8
    """
    stars = get_stars_from_text(text)
    expected_stars = ascii.read(expected, format="fixed_width_two_line", guess=False)
    assert all(
        np.all(stars[name] == expected_stars[name])
        for name in ("slot", "YAG", "ZAG", "MAG_ACA")
    )
    solutions = find_attitude_solutions(stars, tolerance=2.5)
    assert len(solutions) == 1
    solution = solutions[0]
    logger.debug(pformat(solution))
    logger.debug("RA, Dec, Roll", solutions[0]["att_fit"].equatorial)


def test_get_stars_from_table():
    text = """
      slot yag zag mag
      3     223.33      55.83      9.7
      4    -453.1      -2084.1     9.6
      5   -1255.12     196.58      9.2
      6     598.18    2287.97      9.6
      7    2311.45    1140.60      9.8
    """
    expected = """
    slot    YAG      ZAG   MAG_ACA
    ----  -------- ------- -------
       3    223.33   55.83     9.7
       4   -453.1  -2084.1     9.6
       5  -1255.12  196.58     9.2
       6    598.18 2287.97     9.6
       7   2311.45  1140.6     9.8
    """
    stars = get_stars_from_text(text)
    expected_stars = ascii.read(expected, format="fixed_width_two_line", guess=False)
    assert all(
        np.all(stars[name] == expected_stars[name])
        for name in ("slot", "YAG", "ZAG", "MAG_ACA")
    )
    solutions = find_attitude_solutions(stars, tolerance=2.5)
    assert len(solutions) == 1
    solution = solutions[0]
    logger.debug(pformat(solution))
    logger.debug("RA, Dec, Roll", solutions[0]["att_fit"].equatorial)


@pytest.mark.skipif(not HAS_MAUDE, reason="maude not available")
def test_get_stars_from_maude():
    stars = get_stars_from_maude("2024:001:12:00:00", dt=12.0)
    assert stars.pformat_all() == [
        "slot   YAG      ZAG    MAG_ACA",
        "---- -------- -------- -------",
        "   0    39.69 -1882.49    7.31",
        "   1  2138.01   154.69    7.19",
        "   2 -1826.71   149.87    7.12",
        "   3  2338.57 -1508.50    5.62",
        "   4  1628.89  -269.88    7.69",
        "   5 -1060.80   904.76    7.88",
        "   6  2446.09 -2077.31    8.25",
        "   7 -1562.49  1452.76    8.73",
    ]
    solutions = find_attitude_solutions(stars)
    assert len(solutions) == 1


def check_at_time(time, qatt=None):
    from astropy.table import Table
    from Chandra.Time import DateTime
    from Ska.engarchive import fetch

    msids_all = []
    msids = {}
    typs = ("fid", "yan", "zan", "mag")
    for typ in typs:
        msids[typ] = ["aoac{}{}".format(typ, slot) for slot in range(8)]
        msids_all.extend(msids[typ])

    msids_all.extend(["aoattqt1", "aoattqt2", "aoattqt3", "aoattqt4"])

    tstart = DateTime(time).secs
    tstop = tstart + 60
    dat = fetch.MSIDset(msids_all, tstart, tstop)
    dat.interpolate(2.05)
    sample = {msid: dat[msid].vals[5] for msid in msids_all}

    vals = {}
    slots = [slot for slot in range(8) if sample["aoacfid{}".format(slot)] == "STAR"]
    for typ in typs:
        vals[typ] = [
            sample["aoac{}{}".format(typ, slot)]
            for slot in range(8)
            if sample["aoacfid{}".format(slot)] == "STAR"
        ]

    stars = Table(
        [slots, vals["yan"], vals["zan"], vals["mag"]],
        names=["slots", "YAG", "ZAG", "MAG_ACA"],
    )

    if qatt is None:
        qatt = Quat([dat["aoattqt{}".format(i + 1)].vals[5] for i in range(4)])
    ra, dec, roll = qatt.equatorial

    solutions = find_attitude_solutions(stars)

    assert len(solutions) == 1
    solution = solutions[0]
    dq = qatt.inv() * solution["att_fit"]

    logger.debug(solution["att_fit"].equatorial)
    logger.debug(solution["summary"])

    assert abs(dq.q[0] * 2 * 3600) < 60  # arcsec
    assert abs(dq.q[1] * 2 * 3600) < 1.5  # arcsec
    assert abs(dq.q[2] * 2 * 3600) < 1.5


def test_at_times():
    mcc_results = """
            2015:007:03:00:00   2015:007:03:05:00 - brute
            2015:100:00:00:00   2015:100:00:05:00
            2015:110:00:00:00   2015:110:00:05:00 - brute
            2015:121:00:00:00   2015:121:00:05:00 - brute
            2015:130:00:00:00   2015:130:00:05:00
            2015:152:00:00:00   2015:152:00:05:00
            2015:156:00:00:00   2015:156:00:05:00
            2015:170:00:00:00   2015:170:00:05:00"""
    times = [line.split()[0] for line in mcc_results.strip().splitlines()]
    qatts = [None] * len(times)
    qatts[0] = Quat(
        [300.6576081, 66.73096392, 347.56528804]
    )  # Telem aoattqt* are wrong
    for time, qatt in zip(times, qatts):
        check_at_time(time, qatt)


def test_nsm_constraint():
    """ """
