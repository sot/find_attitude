import collections
from itertools import izip, product
import numpy as np
from astropy.table import Table, vstack, Column, MaskedColumn
import networkx as nx
import tables
import pyyaks.logger

DEBUG = False
TEST_OVERLAPPING = False
DELTA_MAG = 1.0  # Accept matches where candidate star is within DELTA_MAG of observed

loglevel = pyyaks.logger.INFO
logger = pyyaks.logger.get_logger(name='find_attidue', level=loglevel,
                                  format="%(asctime)s %(message)s")


def get_dists_yag_zag(yags, zags, mags=None):
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
    if mags is None:
        mags = np.ones_like(dists) * 15

    return Table([dists, idx0, idx1, mags[idx0], mags[idx1]],
                 names=['dists', 'idx0', 'idx1', 'mag0', 'mag1'])


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


def add_edge(graph, id0, id1, i0, i1, dist):
    edge_data = graph.get_edge_data(id0, id1, None)
    if edge_data is None:
        graph.add_edge(id0, id1, i0=[i0], i1=[i1], dist=dist)
    else:
        edge = graph[id0][id1]
        snew = set([i0, i1])
        for i0_old, i1_old in izip(edge['i0'], edge['i1']):
            sold = set([i0_old, i1_old])
            if snew == sold:
                return
        edge['i0'].append(i0)
        edge['i1'].append(i1)


def connected_agasc_ids(ap):
    """
    Return agacs_ids that occur at least 4 times.  Each occurrence indicates
    an edge containing that agasc_id node.
    """
    agasc_ids = np.concatenate((ap['agasc_id0'], ap['agasc_id1']))
    c = Column(agasc_ids)
    cg = c.group_by(c)
    i_big_enough = np.flatnonzero(np.diff(cg.groups.indices) >= 4)
    out = set(cg.groups.keys[i_big_enough].tolist())
    return out


def get_match_graph(aca_stars, agasc_pairs, tolerance):
    idx0s = aca_stars['idx0']
    idx1s = aca_stars['idx1']
    dists = aca_stars['dists']
    mag0s = aca_stars['mag0']
    mag1s = aca_stars['mag1']
    print('Starting get_match_graph')
    gmatch = nx.Graph()
    ap_list = []
    logger.info('Getting matches from file')
    for i0, i1, dist, mag0, mag1 in izip(idx0s, idx1s, dists, mag0s, mag1s):
        logger.verbose('Getting matches from file {} {} {}'.format(i0, i1, dist))
        ap = agasc_pairs.read_where('(dists > {}) & (dists < {})'
                                    .format(dist - tolerance, dist + tolerance))
        # max_mag = max(mag0, mag1) + DELTA_MAG / 10.
        if DELTA_MAG is not None:
            max_mag = max(mag0, mag1) + DELTA_MAG
            mag_ok = (ap['mag0'] < max_mag) & (ap['mag1'] < max_mag)
            ap = ap[mag_ok]

        ones = np.ones(len(ap), dtype=np.uint8)
        ap = Table([ap['dists'], ap['agasc_id0'], ap['agasc_id1'], i0 * ones, i1 * ones],
                   names=['dists', 'agasc_id0', 'agasc_id1', 'i0', 'i1'])
        ap_list.append(ap)
        logger.verbose('  Found {} matching pairs'.format(len(ap)))

    ap = vstack(ap_list)
    connected_ids = connected_agasc_ids(ap)

    if TEST_OVERLAPPING:
        ok = np.zeros(len(ap), dtype=bool)
        for match_id in [541731248, 541466560, 541460960, 541465104,
                         541337704, 541862768, 541736344, 541730152]:
            ok |= (ap['agasc_id0'] == match_id) | (ap['agasc_id1'] == match_id)
        ap = ap[ok]

    agasc_id0 = ap['agasc_id0']
    agasc_id1 = ap['agasc_id1']
    i0 = ap['i0']
    i1 = ap['i1']
    dists = ap['dists']

    logger.info('Adding edges from {} matching distance pairs'.format(len(ap)))

    for i in xrange(len(ap)):
        id0 = agasc_id0[i]
        id1 = agasc_id1[i]
        if id0 in connected_ids and id1 in connected_ids:
            add_edge(gmatch, id0, id1, i0[i], i1[i], dists[i])

    if TEST_OVERLAPPING:
        match_tris = get_triangles(gmatch)
        logger.info('Matching triangles')
        for tri in match_tris:
            e0 = gmatch.get_edge_data(tri[0], tri[1])
            e1 = gmatch.get_edge_data(tri[0], tri[2])
            e2 = gmatch.get_edge_data(tri[1], tri[2])
            logger.info(tri[0], tri[1], tri[2], e0, e1, e2)

        logger.info('Edge data')
        for n0, n1 in nx.edges(gmatch):
            ed = gmatch.get_edge_data(n0, n1)
            print(n0, n1, ed)

    logger.info('Added total of {} nodes'.format(len(gmatch)))

    return gmatch


def get_slot_id_candidates(graph, nodes):
    class BadCandidateError(Exception):
        pass

    n_nodes = len(nodes)
    i0_id0s_list = []
    i1_id1s_list = []
    for node0, node1 in nx.edges(graph, nodes):
        if node0 in nodes and node1 in nodes:
            edge_data = graph.get_edge_data(node0, node1)
            i0_id0s_list.append([(i0, node0) for i0 in edge_data['i0']])
            i1_id1s_list.append([(i1, node1) for i1 in edge_data['i1']])

    id_candidates = []
    for i0_id0s, i1_id1s in izip(product(*i0_id0s_list), product(*i1_id1s_list)):
        logger.info('')
        node_star_index_count = collections.defaultdict(collections.Counter)
        for i0_id0, i1_id1 in izip(i0_id0s, i1_id1s):
            i0, id0 = i0_id0
            i1, id1 = i1_id1
            for i in i0, i1:
                for id_ in id0, id1:
                    node_star_index_count[id_].update([i])

        try:
            agasc_id_star_index_map = {}
            for agasc_id, count_dict in node_star_index_count.items():
                logger.verbose('AGASC ID {} has counts {}'.format(agasc_id, count_dict))
                for index, count in count_dict.items():
                    if count == n_nodes - 1:
                        agasc_id_star_index_map[agasc_id] = index
                        break
                else:
                    logger.verbose('  **** AGASC ID {} is incomplete **** '.format(agasc_id))
                    raise BadCandidateError
        except BadCandidateError:
            pass
        else:
            id_candidates.append(agasc_id_star_index_map)

    return id_candidates


def find_matching_agasc_ids(stars, agasc_pairs_file, g_dist_match=None, tolerance=2.5):
    if g_dist_match is None:
        with tables.open_file(agasc_pairs_file, 'r') as h5:
            agasc_pairs = h5.root.data
            g_dist_match = get_match_graph(stars, agasc_pairs, tolerance)

    logger.info('Getting all triangles from match graph')
    match_tris = get_triangles(g_dist_match)

    # New graph with only plausible triangles
    g_geom_match = nx.Graph()
    logger.info('Finding triangles that match stars pattern')
    for tri in match_tris:
        e0 = g_dist_match.get_edge_data(tri[0], tri[1])
        e1 = g_dist_match.get_edge_data(tri[0], tri[2])
        e2 = g_dist_match.get_edge_data(tri[1], tri[2])
        match_count = 0
        for e0_i, e0_i0 in enumerate(e0['i0']):
            e0_i1 = e0['i1'][e0_i]
            for e1_i, e1_i0 in enumerate(e1['i0']):
                e1_i1 = e1['i1'][e1_i]
                for e2_i, e2_i0 in enumerate(e2['i0']):
                    e2_i1 = e2['i1'][e2_i]
                    if len(set([e0_i0, e0_i1, e1_i0, e1_i1, e2_i0, e2_i1])) == 3:
                        add_edge(g_geom_match, tri[0], tri[1], e0_i0, e0_i1, e0['dist'])
                        add_edge(g_geom_match, tri[0], tri[2], e1_i0, e1_i1, e1['dist'])
                        add_edge(g_geom_match, tri[2], tri[1], e2_i0, e2_i1, e2['dist'])
                        match_count += 1
        # if match_count > 1:
        #     logger.info('** NOTE ** matched multiple plausible triangles for {} {} {}'
        #                 .format(*tri))

    if DEBUG:
        print('g_geom_match: ')
        for n0, n1 in nx.edges(g_geom_match):
            ed = g_geom_match.get_edge_data(n0, n1)
            print(n0, n1, ed)

    out = []

    # Iterate through complete subgraphs (cliques) and require at least
    # four nodes.
    for clique_nodes in nx.find_cliques(g_geom_match):
        n_clique_nodes = len(clique_nodes)
        if n_clique_nodes < 4:
            continue

        logger.info('Checking clique {}'.format(clique_nodes))

        out.extend(get_slot_id_candidates(g_geom_match, clique_nodes))

    logger.info('Done with graph matching')
    return out, g_geom_match, g_dist_match


def find_all_matching_agasc_ids(yags, zags, mags=None, agasc_pairs_file=None, dist_match_graph=None,
                                tolerance=2.0):
    stars = get_dists_yag_zag(yags, zags, mags)
    agasc_id_star_maps, g_geom_match, g_dist_match = find_matching_agasc_ids(
        stars, agasc_pairs_file, dist_match_graph, tolerance=tolerance)
    return agasc_id_star_maps


def find_attitude_for_agasc_ids(yags, zags, agasc_id_star_map):
    """
    Find attitude for a given set of yags and zags and matching agasc_ids
    """
    global yagzag

    from sherpa import ui
    import agasc
    from Quaternion import Quat
    from Ska.quatutil import radec2yagzag

    star_indices = agasc_id_star_map.values()
    yags = yags[star_indices]
    zags = zags[star_indices]

    agasc_ids = agasc_id_star_map.keys()
    agasc_stars = [agasc.get_star(agasc_id) for agasc_id in agasc_ids]

    ras = [s['RA_PMCORR'] for s in agasc_stars]
    decs = [s['DEC_PMCORR'] for s in agasc_stars]

    qatt = Quat([ras[0], decs[0], 0.0])

    def _yag_zag(dra, ddec, droll):
        q = qatt * Quat([dra / 3600., ddec / 3600., droll])
        yags, zags = radec2yagzag(ras, decs, q)
        return yags * 3600, zags * 3600, q

    def yag_zag(pars, x):
        m_yags, m_zags, q = _yag_zag(*pars)
        return np.concatenate([m_yags, m_zags])

    m_yags, m_zags, att_fit = _yag_zag(0, 0, 0)

    stars = get_dists_yag_zag(yags, zags)
    dists = stars['dists']
    idx0 = stars['idx0']
    idx1 = stars['idx1']
    i = np.argmax(dists)
    m_ang = np.arctan2(m_zags[idx0[i]] - m_zags[idx1[i]], m_yags[idx0[i]] - m_yags[idx1[i]])
    s_ang = np.arctan2(zags[idx0[i]] - zags[idx1[i]], yags[idx0[i]] - yags[idx1[i]])
    roll = np.degrees(m_ang - s_ang)

    y = np.concatenate([yags, zags])
    ui.load_arrays(1, np.arange(len(y)), y, np.ones(len(y)))
    ui.load_user_model(yag_zag, "yagzag")
    ui.add_user_pars("yagzag", ["dra", "ddec", "droll"])
    ui.set_model(yagzag)
    ui.set_method('simplex')

    yagzag.dra = 0.0
    yagzag.ddec = 0.0
    yagzag.droll = roll

    ui.freeze(yagzag.droll)
    ui.fit()

    ui.thaw(yagzag.droll)
    ui.fit()
    fit_results = ui.get_fit_results()
    m_yags, m_zags, att_fit = _yag_zag(yagzag.dra.val, yagzag.ddec.val, yagzag.droll.val)

    out = dict(yags=yags,
               zags=zags,
               m_yags=m_yags,
               m_zags=m_zags,
               att_fit=att_fit,
               statval=fit_results.statval,
               agasc_id_star_map=agasc_id_star_map)

    return out


def find_attitude_solutions(stars, tolerance=2.0):
    agasc_id_star_maps = find_all_matching_agasc_ids(stars['YAG'], stars['ZAG'], stars['MAG_ACA'],
                                                     agasc_pairs_file='distances.h5',
                                                     tolerance=tolerance)

    solutions = []
    for agasc_id_star_map in agasc_id_star_maps:
        solution = find_attitude_for_agasc_ids(stars['YAG'], stars['ZAG'], agasc_id_star_map)

        # Check to see if there is another solution that has overlapping
        # stars.  If there are overlaps and this solution has a lower
        # fit statistic then use this solution.
        agasc_ids = set(agasc_id_star_map)
        solution['agasc_ids'] = agasc_ids
        for prev_solution in solutions:
            if agasc_ids.intersection(prev_solution['agasc_ids']):
                if solution['statval'] < prev_solution['statval']:
                    prev_solution.update(solution)
                break
        else:
            solutions.append(solution)

    update_solutions(solutions, stars)
    return solutions


def update_solutions(solutions, stars):
    for sol in solutions:
        summ = Table(stars, masked=True)

        indices = sol['agasc_id_star_map'].values()
        for name in ('m_yag', 'dy', 'm_zag', 'dz', 'dr'):
            summ[name] = MaskedColumn([-99.0] * len(summ), name=name, mask=True)
        summ['m_agasc_id'] = MaskedColumn([-99] * len(summ), name='m_agasc_id', mask=True)

        summ['m_yag'][indices] = sol['m_yags']
        summ['dy'][indices] = sol['yags'] - sol['m_yags']
        summ['m_zag'][indices] = sol['m_zags']
        summ['dz'][indices] = sol['zags'] - sol['m_zags']
        dr = np.sqrt((sol['yags'] - sol['m_yags']) ** 2
                     + (sol['zags'] - sol['m_zags']) ** 2)
        summ['dr'][indices] = dr
        summ['m_agasc_id'][indices] = sol['agasc_id_star_map'].keys()

        for name in summ.colnames:
            if name in ('RA', 'DEC'):
                format = '{:.4f}'
            elif 'agasc' in name.lower():
                format = '{:d}'
            else:
                format = '{:.2f}'
            summ[name].format = format
        sol['summary'] = summ

        # Need at least 4 stars with radial fit residual < 3 arcsec
        sol['bad_fit'] = np.sum(dr < 3.0) < 4
