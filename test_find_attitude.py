import numpy as np
import agasc
from Ska.quatutil import radec2yagzag
from Quaternion import Quat
# from find_attitude import find_all_matching_agasc_ids, find_attitude_for_agasc_ids


def get_stars(ra=119.98, dec=-78, roll=0, select=slice(None, 8), brightest=True,
              sigma_1axis=0.4, sigma_mag=0.2):
    stars = agasc.get_agasc_cone(ra, dec, 1.0)
    ok = (stars['MAG_ACA'] > 5) & (stars['ASPQ1'] == 0)
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


def test_random():
    ra = np.random.uniform(0, 360)
    dec = np.random.uniform(-90, 90)
    roll = np.random.uniform(0, 360)


def check_output(agasc_id_star_maps, stars, ra, dec, roll):
    for agasc_id_star_map in agasc_id_star_maps:
        yags, zags, m_yags, m_zags, att_fit = find_attitude_for_agasc_ids(
            stars['YAG'], stars['ZAG'], agasc_id_star_map)
        print('Input: RA Dec Roll = {} {} {}'.format(ra, dec, roll))
        print('Solve: RA Dec Roll = {} {} {}'.format(*att_fit.equatorial))
        print('Delta distances {}'.format(np.sqrt((yags - m_yags) ** 2
                                                  + (zags - m_zags) ** 2)))
    assert len(agasc_id_star_maps) == 1


if __name__ == '__main__':
    ra, dec, roll = 115.770455413, -77.6580358662, 86.4089128685
    stars = get_stars(ra, dec, roll, sigma_1axis=0.0005, sigma_mag=0.0001)
    if False:
        stars = get_stars()
        ra=119.98
        dec=-78
        roll=0
    agasc_id_star_maps, g_geom_match, g_dist_match = find_all_matching_agasc_ids(
        stars['YAG'], stars['ZAG'], stars['MAG_ACA'] - 1, 'distances-8.5mag.h5')
        # stars['YAG'], stars['ZAG'], stars['MAG_ACA'] - 1, 'distances.h5')
    check_output(agasc_id_star_maps, stars, ra, dec, roll)
