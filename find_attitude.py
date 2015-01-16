import glob
from itertools import izip
import numpy as np
from astropy.table import Table
import agasc
from Ska.quatutil import radec2yagzag
from Quaternion import Quat
# import matplotlib.pyplot as plt
import networkx as nx
import tables


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


def get_dists_yag_zag(yags, zags):
    if np.all(np.abs(yags) < 2.0):
        # Must be in degrees, convert to arcsec
        yags = yags * 3600.
        zags = zags * 3600.

    if len(yags) != len(zags):
        raise ValueError('length of yags and zags must match')

    idx0, idx1 = np.mgrid[slice(0, len(yags)), slice(0, len(yags))]
    idx0 = idx0.ravel()
    idx1 = idx1.ravel()
    ok = idx0 > idx1
    idx0 = idx0[ok]
    idx1 = idx1[ok]

    dists = np.sqrt((yags[idx0] - yags[idx1]) ** 2 + (zags[idx0] - zags[idx1]) ** 2)
    dists = np.array(dists)

    return dists, idx0, idx1


def get_triangles(G):
    result = []
    done = set()
    for n in G:
        done.add(n)
        nbrdone = set()
        nbrs = set(G[n])
        for nbr in nbrs:
            if nbr in done:
                continue
            nbrdone.add(nbr)
            for both in nbrs.intersection(G[nbr]):
                if both in done or both in nbrdone:
                    continue
                result.append((n, nbr, both))
    return result


def get_match_graph(star_idx0, star_idx1, star_dists, agasc_pairs, tolerance=2.0):
    print('Starting get_match_graph')
    gmatch = nx.Graph()
    for i0, i1, dist in izip(star_idx0, star_idx1, star_dists):
        print('Matching {} {} {}'.format(i0, i1, dist))
        aok = agasc_pairs.read_where('(dists > {}) & (dists < {})'
                                     .format(dist - tolerance, dist + tolerance))
        print('  found {} matching pairs'.format(len(aok)))
        for r in aok:
            gmatch.add_edge(r['agasc_id0'], r['agasc_id1'], i0=i0, i1=i1, dist=dist)
    return gmatch


def find_matching_agasc_ids(dists, idx0s, idx1s, agasc_pairs_file):
    with tables.open_file(agasc_pairs_file, 'r') as h5:
        agasc_pairs = h5.root.data
        gmatch = get_match_graph(idx0s, idx1s, dists, agasc_pairs)

    agasc_ids = {}
    match_tris = get_triangles(gmatch)
    for tri in match_tris:
        index_set = set()
        for i0, i1 in ((0, 1), (0, 2), (1, 2)):
            edge_data = gmatch.get_edge_data(tri[i0], tri[i1])
            index_set.add(edge_data['i0'])
            index_set.add(edge_data['i1'])
        if len(index_set) == 3:
            for agasc_id in tri:
                if agasc_id not in agasc_ids:
                    n_triangles = nx.triangles(gmatch, nodes=agasc_id)
                    if n_triangles >= 3:
                        agasc_ids[agasc_id] = n_triangles

    return agasc_ids


def find_all_matching_agasc_ids(yags, zags, agasc_pairs_file):
    dists, idx0s, idx1s = get_dists_yag_zag(stars['YAG'], stars['ZAG'])
    agasc_ids = find_matching_agasc_ids(dists, idx0s, idx1s, agasc_pairs_file)
    return agasc_ids


def find_all_matching_agasc_ids_OLD(yags, zags):
    dists, idx0s, idx1s = get_dists_yag_zag(stars['YAG'], stars['ZAG'])
    for agasc_pairs_file in glob.glob('distances_*.fits'):
        print('Reading and matching against {}'.format(agasc_pairs_file))
        agasc_pairs = Table.read(agasc_pairs_file)
        agasc_ids = find_matching_agasc_ids(dists, idx0s, idx1s, agasc_pairs)
        if agasc_ids:
            print("  {}".format(agasc_ids))


# stars = get_bright_stars(ra=np.random.uniform(0, 360), dec=np.random.uniform(-90, 90), roll=20,
#                          sigma_1axis=0.5)
stars = get_bright_stars(ra=20, dec=30, roll=20, sigma_1axis=0.0001)
# agasc_ids = find_all_matching_agasc_ids(stars['YAG'], stars['ZAG'], 'distances.h5')  # 'agasc_pairs.h5'
