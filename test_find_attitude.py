from __future__ import print_function, division

import numpy as np
import agasc
from Ska.quatutil import radec2yagzag
from Quaternion import Quat
from astropy.io import ascii
from find_attitude import get_dists_yag_zag, find_attitude_solutions


def get_stars(ra=119.98, dec=-78, roll=0, select=slice(None, 8), brightest=True,
              sigma_1axis=0.4, sigma_mag=0.2):
    stars = agasc.get_agasc_cone(ra, dec, 1.0)
    ok = (stars['MAG_ACA'] > 5) & (stars['ASPQ1'] == 0) & (stars['MAG_ACA'] < 10.5)
    stars = stars[ok]
    if brightest:
        stars.sort('MAG_ACA')
    else:
        index = np.arange(len(stars))
        np.random.shuffle(index)
        stars = stars[index]
    stars = stars[select]
    yags, zags = radec2yagzag(stars['RA_PMCORR'], stars['DEC_PMCORR'], Quat([ra, dec, roll]))
    stars['YAG_ERR'] = np.random.normal(scale=sigma_1axis, size=len(stars))
    stars['ZAG_ERR'] = np.random.normal(scale=sigma_1axis, size=len(stars))
    stars['MAG_ERROR'] = np.random.normal(scale=sigma_mag, size=len(stars))
    stars['YAG'] = yags * 3600 + stars['YAG_ERR']
    stars['ZAG'] = zags * 3600 + stars['ZAG_ERR']
    stars['MAG_ACA'] += stars['MAG_ERROR']
    stars['RA'] = stars['RA_PMCORR']
    stars['DEC'] = stars['DEC_PMCORR']
    stars = stars['AGASC_ID', 'RA', 'DEC', 'YAG', 'YAG_ERR', 'ZAG', 'ZAG_ERR',
                  'MAG_ACA', 'MAG_ERROR']

    return stars


def find_overlapping_distances(min_n_overlap=3, tolerance=3.0):
    while True:
        ra = np.random.uniform(0, 360)
        dec = np.random.uniform(-90, 90)
        roll = np.random.uniform(0, 360)
        print(ra, dec, roll)
        stars = get_stars(ra, dec, roll, sigma_1axis=0.001, sigma_mag=0.2)
        dist_table = get_dists_yag_zag(stars['YAG'], stars['ZAG'])
        dists = dist_table['dists']

        n_overlap = 0
        for i, d0 in enumerate(dists):
            for d1 in dists[i + 1:]:
                if np.abs(d0 - d1) < tolerance:
                    n_overlap += 1
        if n_overlap >= min_n_overlap:
            return ra, dec, roll, stars, dists


def test_overlapping_distances(tolerance=3.0):
    """
    Test a case where distance 5-7 = 864.35 and 5-1 is 865.808
    """
    global ra, dec, roll, stars, agasc_id_star_maps, g_geom_match, g_dist_match
    ra, dec, roll = (131.1371559426714, 65.25369723989581, 112.4351393383257)  # 3 overlaps, 3.0 tol
    stars = get_stars(ra, dec, roll, sigma_1axis=0.004, sigma_mag=0.2, brightest=True)

    solutions = find_attitude_solutions(stars)
    check_output(solutions, stars, ra, dec, roll)


def test_random(n_iter=1, sigma_1axis=0.4, sigma_mag=0.2, brightest=True):
    for _ in xrange(n_iter):
        global ra, dec, roll, stars, agasc_id_star_maps, g_geom_match, g_dist_match
        ra = np.random.uniform(0, 360)
        dec = np.random.uniform(-90, 90)
        roll = np.random.uniform(0, 360)
        stars = get_stars(ra, dec, roll, sigma_1axis=sigma_1axis,
                          sigma_mag=sigma_mag, brightest=brightest)
        solutions = find_attitude_solutions(stars)
        check_output(solutions, stars, ra, dec, roll)


def test_multiple_solutions():
    global stars, solutions
    ra, dec, roll = 190.3286989834239, 22.698443628394102, 111.51056234863053
    stars = ascii.read("""
 AGASC_ID       RA           DEC           YAG           YAG_ERR          ZAG           ZAG_ERR     MAG_ACA    MAG_ERROR
260863544 189.758890214 22.6594185253  567.401869049  0.615027692698  1811.01764078 -0.293256251994 6.47267   0.280073674992
189804208 191.093682446 21.9926851164 -3294.27782744 -0.181642428933 -1445.81915638  0.812100708212 7.94189   0.199006301588
189800592 190.343952117 21.8305748811 -2925.81601843    -0.381447575  1098.70907129  0.455625742141  9.0004   0.265371250728
260856472 190.787449674 23.2708932848  1363.51998803  0.294404741658 -2168.30508446 -0.261280462907 8.88882  -0.186751057505
260858632 190.173364997 22.9716741441  1103.78516827 -0.370985727476  118.089547978  -0.14060714735 9.50322    0.12366348923
189811352 190.148298323 22.1508704117 -1612.84681732  0.241090402814  1282.01534928 -0.309449604245 9.61865   0.234064435566
260838832 190.053449838 23.5688063028  3249.58724799  0.683018406574 -304.398501017 -0.173295600276 9.46925 -0.0731231730791
260851360 190.235827288  22.625601259 -130.370520041    0.3501954033  383.398889404  0.168175442249 9.73112  0.0393770657269
""")
    solutions = find_attitude_solutions(stars)
    check_output(solutions, stars, ra, dec, roll)


def check_output(solutions, stars, ra, dec, roll):
    print('*********************************************')
    print()
    for solution in solutions:
        att_fit = solution['att_fit']

        att_in = Quat([ra, dec, roll])
        d_att = att_in.inv() * att_fit
        d_roll, d_pitch, d_yaw, _ = 2 * np.degrees(d_att.q) * 3600.

        print('============================')
        print('Input: RA Dec Roll = {} {} {}'.format(ra, dec, roll))
        print('Solve: RA Dec Roll = {} {} {}'.format(*att_fit.equatorial))
        print(solution['summary'])
        if solution['bad_fit']:
            print('BAD FIT!')
            continue

        assert d_roll < 40.
        assert d_pitch < 1.
        assert d_yaw < 1.

        ok = ~solution['summary']['m_agasc_id'].mask
        sok = solution['summary'][ok]
        assert np.all(sok['AGASC_ID'] == sok['m_agasc_id'])

    assert sum(1 for s in solutions if not s['bad_fit']) == 1
    print('*********************************************\n')


def test_ra_dec_roll(ra=115.770455413, dec=-77.6580358662, roll=86.4089128685, brightest=True,
                     provide_mags=True, sigma_1axis=0.4, sigma_mag=0.2):
    global stars, agasc_id_star_maps, g_geom_match, g_dist_match, solutions
    stars = get_stars(ra, dec, roll, sigma_1axis=sigma_1axis, sigma_mag=sigma_mag,
                      brightest=brightest)
    solutions = find_attitude_solutions(stars)
    check_output(solutions, stars, ra, dec, roll)
