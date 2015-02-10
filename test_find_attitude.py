from __future__ import print_function, division

import numpy as np
import agasc
from Ska.quatutil import radec2yagzag
from Quaternion import Quat
# from find_attitude import find_all_matching_agasc_ids, find_attitude_for_agasc_ids


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
    stars['YAG'] = yags * 3600 + np.random.normal(scale=sigma_1axis, size=len(stars))
    stars['ZAG'] = zags * 3600 + np.random.normal(scale=sigma_1axis, size=len(stars))
    stars['MAG'] += np.random.normal(scale=sigma_mag, size=len(stars))
    stars['RA'] = stars['RA_PMCORR']
    stars['DEC'] = stars['DEC_PMCORR']
    stars = stars['AGASC_ID', 'RA', 'DEC', 'YAG', 'ZAG', 'MAG_ACA', 'CLASS', 'ASPQ1']

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
        n_dists = len(dists)

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
    ra, dec, roll = (176.21566235300048, -36.4011650737424, 253.92152697511588) # 2 overlaps, 2.0 tol
    ra, dec, roll = (131.1371559426714, 65.25369723989581, 112.4351393383257)  # 3 overlaps, 3.0 tol
    stars = get_stars(ra, dec, roll, sigma_1axis=0.004, sigma_mag=0.2, brightest=True)
    # stars = stars[[0, 1, 2, 6]]
    agasc_id_star_maps, g_geom_match, g_dist_match = find_all_matching_agasc_ids(
        stars['YAG'], stars['ZAG'], stars['MAG_ACA'], 'distances.h5', tolerance=tolerance)
    check_output(agasc_id_star_maps, stars, ra, dec, roll)


def test_random(n_iter=1, sigma_1axis=0.4, sigma_mag=0.2, brightest=True):
    for _ in xrange(n_iter):
        global ra, dec, roll, stars, agasc_id_star_maps, g_geom_match, g_dist_match
        ra = np.random.uniform(0, 360)
        dec = np.random.uniform(-90, 90)
        roll = np.random.uniform(0, 360)
        stars = get_stars(ra, dec, roll, sigma_1axis=sigma_1axis,
                          sigma_mag=sigma_mag, brightest=brightest)
        agasc_id_star_maps, g_geom_match, g_dist_match = find_all_matching_agasc_ids(
            stars['YAG'], stars['ZAG'], stars['MAG_ACA'], 'distances.h5')
        check_output(agasc_id_star_maps, stars, ra, dec, roll)


def check_output(agasc_id_star_maps, stars, ra, dec, roll):
    for agasc_id_star_map in agasc_id_star_maps:
        yags, zags, m_yags, m_zags, att_fit = find_attitude_for_agasc_ids(
            stars['YAG'], stars['ZAG'], agasc_id_star_map)
        print('Input: RA Dec Roll = {} {} {}'.format(ra, dec, roll))
        print('Solve: RA Dec Roll = {} {} {}'.format(*att_fit.equatorial))
        print('Delta distances {}'.format(np.sqrt((yags - m_yags) ** 2
                                                  + (zags - m_zags) ** 2)))
        att_in = Quat([ra, dec, roll])
        d_att = att_in.inv() * att_fit
        d_roll, d_pitch, d_yaw, _ = 2 * np.degrees(d_att.q) * 3600.
        assert d_roll < 40.
        assert d_pitch < 1.
        assert d_yaw < 1.

    assert len(agasc_id_star_maps) > 0


def test_ra_dec_roll(ra=115.770455413, dec=-77.6580358662, roll=86.4089128685, brightest=True,
                     provide_mags=True, sigma_1axis=0.4, sigma_mag=0.2):
    global stars, agasc_id_star_maps, g_geom_match, g_dist_match
    stars = get_stars(ra, dec, roll, sigma_1axis=sigma_1axis, sigma_mag=sigma_mag, brightest=brightest)
    agasc_id_star_maps, g_geom_match, g_dist_match = find_all_matching_agasc_ids(
        stars['YAG'], stars['ZAG'],
        stars['MAG_ACA'] if provide_mags else None,
        'distances.h5')
    check_output(agasc_id_star_maps, stars, ra, dec, roll)
