import numpy as np
import agasc
from Ska.quatutil import radec2yagzag
from Quaternion import Quat
from find_attitude import find_all_matching_agasc_ids, find_attitude_for_agasc_ids


def get_bright_stars(ra=119.98, dec=-78, roll=0, select=slice(None, 8), sigma_1axis=0.4):
    stars = agasc.get_agasc_cone(ra, dec, 1.0)
    ok = (stars['MAG_ACA'] > 5) & (stars['ASPQ1'] == 0)
    stars = stars[ok]
    stars.sort('MAG_ACA')
    stars = stars[select]
    yags, zags = radec2yagzag(stars['RA_PMCORR'], stars['DEC_PMCORR'], Quat([ra, dec, roll]))
    stars['YAG'] = yags * 3600 + sigma_1axis * np.random.normal(scale=sigma_1axis, size=len(stars))
    stars['ZAG'] = zags * 3600 + sigma_1axis * np.random.normal(scale=sigma_1axis, size=len(stars))
    stars = stars['AGASC_ID', 'RA', 'DEC', 'YAG', 'ZAG', 'MAG_ACA', 'CLASS', 'ASPQ1']

    return stars


ra = np.random.uniform(0, 360)
dec = np.random.uniform(-90, 90)
roll = np.random.uniform(0, 360)

stars = get_bright_stars(ra, dec, roll, sigma_1axis=0.5)
stars = get_bright_stars()
agasc_id_star_maps = find_all_matching_agasc_ids(stars['YAG'], stars['ZAG'], stars['MAG_ACA'],
                                                 'distances.h5')
if len(agasc_id_star_maps) == 0:
    print('ARGH, failed to find a match')
else:
    for agasc_id_star_map in agasc_id_star_maps:
        yags, zags, m_yags, m_zags, att_fit = find_attitude_for_agasc_ids(
            stars['YAG'], stars['ZAG'], agasc_id_star_map)
        print('Input: RA Dec Roll = {} {} {}'.format(ra, dec, roll))
        print('Solve: RA Dec Roll = {} {} {}'.format(*att_fit.equatorial))
        print('Delta distances {}'.format(np.sqrt((yags - m_yags) ** 2
                                                  + (zags - m_zags) ** 2)))
