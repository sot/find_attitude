# Licensed under a 3-clause BSD style license - see LICENSE.rst
from pprint import pformat

import agasc
import astropy.units as u
import numpy as np
import pytest
import ska_sun
from astropy.coordinates import SkyCoord
from astropy.io import ascii
from chandra_aca.transform import radec_to_yagzag, yagzag_to_pixels
from cxotime import CxoTime
from Quaternion import Quat
from ska_helpers.utils import random_radec_in_cone

import find_attitude.find_attitude as fafa
from find_attitude import (
    Constraints,
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


def get_random_attitude(constraints: Constraints) -> Quat:
    """
    Generates a random attitude based on the given constraints.

    Parameters
    ----------
    constraints : Constraints
        The constraints for generating the attitude.

    Returns
    -------
    Quat
        Randomly generated attitude.

    """
    date = constraints.date

    # Back off from constraint off_nom_roll_max by 0.5 degrees to avoid randomly
    # generating an attitude that is too close to the constraint.
    onrm = max(0.0, constraints.off_nom_roll_max - 0.5)
    off_nom_roll = np.random.uniform(-onrm, onrm)

    if constraints.att is not None:
        att0 = Quat(constraints.att)
        # Back off from constraint att_err by 0.5 degrees (same as off_nom_roll).
        att_err = max(0.0, constraints.att_err - 0.5)
        ra, dec = random_radec_in_cone(att0.ra, att0.dec, angle=att_err)
        roll0 = ska_sun.nominal_roll(ra, dec, time=date)
        roll = roll0 + off_nom_roll
        att = Quat([ra, dec, roll])

    elif constraints.pitch is not None:
        pitch_err = max(0.0, constraints.pitch_err - 0.5)
        d_pitch = np.random.uniform(pitch_err, pitch_err)
        pitch = constraints.pitch + d_pitch
        yaw = np.random.uniform(0, 360)
        att = ska_sun.get_att_for_sun_pitch_yaw(
            pitch, yaw, time=date, off_nom_roll=off_nom_roll
        )

    else:
        ra = np.random.uniform(0, 360)
        dec = np.rad2deg(np.arccos(np.random.uniform(-1, 1))) - 90
        roll = ska_sun.nominal_roll(ra, dec, time=date) + off_nom_roll
        att = Quat([ra, dec, roll])

    return att


def check_n_stars(
    n_stars, tolerance=5.0, att_err=5.0, off_nom_roll_max=1.0, min_stars=None
):
    # Get a random sun-pointed attitude anywhere on the sky for a time in 2024 or 2025
    constraints_all_sky = Constraints(
        off_nom_roll_max=0.0,
        date=CxoTime("2024:001") + np.random.uniform(0, 2) * u.yr,
    )
    att_est = get_random_attitude(constraints_all_sky)

    # Now run find attitude test with this constraint where the test attitude will be
    # randomized consistent with the constraints. This selects stars at an attitude
    # which is randomly displaced from the estimated attitude.
    constraints = Constraints(
        off_nom_roll_max=off_nom_roll_max,
        date=constraints_all_sky.date,
        att=att_est,
        att_err=att_err,
        min_stars=min_stars,
    )
    check_random_all_sky(
        constraints=constraints,
        tolerance=tolerance,
        min_stars=n_stars,
        max_stars=n_stars,
    )


def check_random_all_sky(
    sigma_1axis=1.0,
    sigma_mag=0.2,
    brightest=True,
    constraints=None,
    min_stars=4,
    max_stars=8,
    tolerance=3.5,
    log_level="WARNING",
    sherpa_log_level="WARNING",
):
    if constraints is None:
        constraints = Constraints(off_nom_roll_max=20, date="2025:001")

    n_stars = np.random.randint(min_stars, max_stars + 1)

    while True:
        att = get_random_attitude(constraints)
        stars = get_stars(
            att.ra,
            att.dec,
            att.roll,
            sigma_1axis=sigma_1axis,
            sigma_mag=sigma_mag,
            brightest=brightest,
            date=constraints.date,
        )
        if len(stars) >= n_stars:
            break

    solutions = []
    solutions = find_attitude_solutions(
        stars,
        tolerance=tolerance,
        constraints=constraints,
        log_level=log_level,
        sherpa_log_level=sherpa_log_level,
    )

    for solution in solutions:
        solution["att"] = att

    check_output(solutions, stars, att.ra, att.dec, att.roll)


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
        assert np.count_nonzero(sok["AGASC_ID"] == sok["m_agasc_id"]) >= 2

    assert sum(1 for s in solutions if not s["bad_fit"]) == 1
    logger.debug("*********************************************\n")


@pytest.mark.parametrize("seed", range(5))
def test_att_constraint_3_stars(seed):
    np.random.seed(seed)
    check_n_stars(n_stars=3, tolerance=5.0, att_err=5.0, off_nom_roll_max=1.0)


def test_att_constraint_2_stars():
    """Test once for code coverage but do not expect this to always succeed"""
    np.random.seed(10)
    check_n_stars(n_stars=2, tolerance=5.0, att_err=5.0, off_nom_roll_max=1.0)


@pytest.mark.parametrize("seed", range(20, 25))
def test_no_constraints_4_to_8_stars(seed):
    np.random.seed(seed)
    check_random_all_sky(constraints=None, tolerance=3.5)


@pytest.mark.parametrize("seed", range(30, 35))
def test_pitch_constraint_3_stars(seed):
    constraints = Constraints(
        off_nom_roll_max=1.0,
        date=CxoTime("2024:001") + np.random.uniform(0, 2) * u.yr,
        pitch=160,
        pitch_err=1.5,
    )
    check_random_all_sky(
        constraints=constraints, tolerance=4.0, min_stars=3, max_stars=3
    )


def test_nsm_2024036():
    att_nsm = Quat([-0.062141268, -0.054177772, 0.920905180, 0.380968348])
    date_nsm = "2024:036:03:02:38.343"
    slots = [3, 7]
    stars = get_stars_from_maude(date_nsm, slots=slots)
    constraints = Constraints(
        off_nom_roll_max=1.0,
        date=date_nsm,
        att=att_nsm,
        att_err=5.0,
        min_stars=2,
    )

    sols = find_attitude_solutions(
        stars,
        tolerance=2.5,
        constraints=constraints,
        log_level="WARNING",
        sherpa_log_level="WARNING",
    )

    assert len(sols) == 1
    assert not sols[0]["bad_fit"]
    att_exp = Quat([-0.11128925, -0.10379516, 0.90382672, 0.39992314])
    dq = att_exp.dq(sols[0]["att_fit"])
    assert abs(dq.pitch) < 1.0 / 3600
    assert abs(dq.yaw) < 1.0 / 3600
    assert abs(dq.roll0) < 60.0 / 3600

    dq = att_nsm.dq(sols[0]["att_fit"])
    assert abs(dq.pitch) < 4.0
    assert abs(dq.yaw) < 4.0
    assert abs(dq.roll0) < 10.0


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
    summary = solutions[0]["summary"]
    assert summary.pformat_all() == [
        " AGASC_ID     RA      DEC      YAG    YAG_ERR   ZAG    ZAG_ERR MAG_ACA MAG_ERROR  m_yag    m_zag   m_mag   dy    dz   dr  m_agasc_id",
        "---------- -------- -------- -------- ------- -------- ------- ------- --------- -------- -------- ----- ----- ----- ---- ----------",
        "1229590664 113.9040 -75.5443   277.57    0.34  1697.60   -0.19    7.16     -0.27   277.38  1697.49  7.42  0.19  0.11 0.22 1229590664",
        "1229598320 117.5286 -75.5923   311.77    0.36 -1558.71   -0.55    8.33      0.09   311.57 -1558.42  8.24  0.20 -0.30 0.36 1229598320",
        "1229593496 113.9002 -75.1111  1829.68    0.14  1847.63   -0.39    8.73      0.26  1829.68  1847.77  8.47  0.00 -0.14 0.14 1229593496",
        "1229592728 114.4727 -75.4207   765.52   -0.54  1226.08   -0.25    8.57      0.07   766.21  1226.08  8.50 -0.69 -0.00 0.69 1229592728",
        "1229601872 116.9575 -76.1410 -1681.19    0.35 -1131.50   -0.39    8.69      0.02 -1681.39 -1131.36  8.68  0.20 -0.14 0.24 1229601872",
        "1204815872 115.5212 -74.9482  2535.86    0.39   392.52   -0.13    8.69     -0.06  2535.63   392.40  8.75  0.23  0.11 0.26 1204815872",
        "1229593296 113.5808 -75.0731  1936.75   -0.16  2155.14   -0.53    8.99      0.06  1937.06  2155.42  8.93 -0.31 -0.29 0.42 1229593296",
        "1229598408 115.6303 -76.2177 -2018.17    0.33    -5.81    0.38    8.99     -0.18 -2018.36    -6.46  9.17  0.18  0.64 0.67 1229598408",
    ]


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
