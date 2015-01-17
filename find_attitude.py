import collections
from itertools import izip
import numpy as np
from astropy.table import Table, vstack
import networkx as nx
import tables
import pyyaks.logger

loglevel = pyyaks.logger.INFO
logger = pyyaks.logger.get_logger(name='find_attidue', level=loglevel,
                                  format="%(asctime)s %(message)s")


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
    aoks = []
    for i0, i1, dist in izip(star_idx0, star_idx1, star_dists):
        logger.info('Getting matches from file {} {} {}'.format(i0, i1, dist))
        aok = agasc_pairs.read_where('(dists > {}) & (dists < {})'
                                     .format(dist - tolerance, dist + tolerance))
        aok = Table([aok['dists'], aok['agasc_id0'], aok['agasc_id1'],
                     i0 * np.ones(len(aok), dtype=np.uint8),
                     i1 * np.ones(len(aok), dtype=np.uint8)],
                    names=['dists', 'agasc_id0', 'agasc_id1', 'i0', 'i1'])
        aoks.append(aok)
        logger.info('  Found {} matching pairs'.format(len(aok)))

    aok = vstack(aoks)

    agasc_id0 = aok['agasc_id0']
    agasc_id1 = aok['agasc_id1']
    i0 = aok['i0']
    i1 = aok['i1']
    dists = aok['dists']

    for i in xrange(len(aok)):
        gmatch.add_edge(agasc_id0[i], agasc_id1[i], i0=i0[i], i1=i1[i], dist=dists[i])
    logger.info('Added edges with {} nodes'.format(len(gmatch)))

    return gmatch


def find_matching_agasc_ids(dists, idx0s, idx1s, agasc_pairs_file, g_dist_match=None):
    if g_dist_match is None:
        with tables.open_file(agasc_pairs_file, 'r') as h5:
            agasc_pairs = h5.root.data
            g_dist_match = get_match_graph(idx0s, idx1s, dists, agasc_pairs)

    logger.info('Getting all triangles from match graph')
    match_tris = get_triangles(g_dist_match)

    # New graph with only plausible triangles
    g_geom_match = nx.Graph()
    logger.info('Finding triangles that match stars pattern')
    for tri in match_tris:
        index_set = set()
        nodes_list = []
        edge_data_list = []
        for i0, i1 in ((0, 1), (0, 2), (1, 2)):
            nodes = tri[i0], tri[i1]
            edge_data = g_dist_match.get_edge_data(*nodes)
            index_set.add(edge_data['i0'])
            index_set.add(edge_data['i1'])
            edge_data_list.append(edge_data)
            nodes_list.append(nodes)
        if len(index_set) == 3:
            for nodes, edge_data in izip(nodes_list, edge_data_list):
                g_geom_match.add_edge(*nodes, **edge_data)

    out = []

    # Iterate through complete subgraphs (cliques) and require at least
    # four nodes.
    for clique_nodes in nx.find_cliques(g_geom_match):
        n_clique_nodes = len(clique_nodes)
        if n_clique_nodes < 4:
            continue

        # Make sure that each node is uniquely associated with the corresponding
        # star catalog entry.
        node_star_index_count = collections.defaultdict(collections.Counter)
        for node0, node1 in nx.edges(g_geom_match, clique_nodes):
            if node0 in clique_nodes and node1 in clique_nodes:
                edge_data = g_dist_match.get_edge_data(node0, node1)
                for node in (node0, node1):
                    for edge_i in [edge_data['i0'], edge_data['i1']]:
                        node_star_index_count[node].update([edge_i])

        agasc_id_star_index_map = {}
        for agasc_id, count_dict in node_star_index_count.items():
            for index, count in count_dict.items():
                if count == n_clique_nodes - 1:
                    agasc_id_star_index_map[agasc_id] = index
                    break
        if len(agasc_id_star_index_map) >= 4:
            out.append(agasc_id_star_index_map)
        else:
            logger.info('Failed clique: {} {}'.format(clique_nodes, node_star_index_count))

    logger.info('Done')
    return out, g_geom_match, g_dist_match


def find_all_matching_agasc_ids(yags, zags, agasc_pairs_file, dist_match_graph=None):
    dists, idx0s, idx1s = get_dists_yag_zag(yags, zags)
    agasc_id_star_maps, g_geom_match, g_dist_match = find_matching_agasc_ids(
        dists, idx0s, idx1s, agasc_pairs_file, dist_match_graph)
    return agasc_id_star_maps


def find_attitude_for_agasc_ids_old(yags, zags, agasc_id_star_map):
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

    # NEED PMCORR versions!!!!
    ras = [s['RA'] for s in agasc_stars]
    decs = [s['DEC'] for s in agasc_stars]

    qatt = Quat([ras[0], decs[0], 0.0])

    def _yag_zag(dra, ddec, droll):
        q = qatt * Quat([dra / 3600., ddec / 3600., droll])
        yags, zags = radec2yagzag(ras, decs, q)
        return yags * 3600, zags * 3600, q

    def yag_zag(pars, x):
        m_yags, m_zags, q = _yag_zag(*pars)
        return np.concatenate([m_yags, m_zags])

    y = np.concatenate([yags, zags])
    ui.load_arrays(1, np.arange(len(y)), y)
    ui.load_user_model(yag_zag, "yagzag")
    ui.add_user_pars("yagzag", ["dra", "ddec", "droll"])
    ui.set_model(yagzag)

    m_yags, m_zags, att_fit = _yag_zag(0, 0, 0)

    yagzag.dra = 0.0
    yagzag.ddec = 0.0
    yagzag.droll = 0.0


    ui.freeze(yagzag.droll)
    ui.fit()

    ui.thaw(yagzag.droll)
    ui.freeze(yagzag.dra)
    ui.freeze(yagzag.ddec)
    ui.fit()

    ui.thaw(yagzag.dra)
    ui.thaw(yagzag.ddec)
    ui.fit()

    m_yags, m_zags, att_fit = _yag_zag(yagzag.dra.val, yagzag.ddec.val, yagzag.droll.val)

    return yags, zags, m_yags, m_zags, att_fit


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

    # NEED PMCORR versions!!!!
    ras = [s['RA'] for s in agasc_stars]
    decs = [s['DEC'] for s in agasc_stars]

    qatt = Quat([ras[0], decs[0], 0.0])

    def _yag_zag(dra, ddec, droll):
        q = qatt * Quat([dra / 3600., ddec / 3600., droll])
        yags, zags = radec2yagzag(ras, decs, q)
        return yags * 3600, zags * 3600, q

    def yag_zag(pars, x):
        m_yags, m_zags, q = _yag_zag(*pars)
        return np.concatenate([m_yags, m_zags])

    m_yags, m_zags, att_fit = _yag_zag(0, 0, 0)

    dists, idx0, idx1 = get_dists_yag_zag(yags, zags)
    i = np.argmax(dists)
    m_ang = np.arctan2(m_zags[idx0[i]] - m_zags[idx1[i]], m_yags[idx0[i]] - m_yags[idx1[i]])
    s_ang = np.arctan2(zags[idx0[i]] - zags[idx1[i]], yags[idx0[i]] - yags[idx1[i]])
    roll = np.degrees(s_ang - m_ang)

    y = np.concatenate([yags, zags])
    ui.load_arrays(1, np.arange(len(y)), y)
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

    m_yags, m_zags, att_fit = _yag_zag(yagzag.dra.val, yagzag.ddec.val, yagzag.droll.val)
    return yags, zags, m_yags, m_zags, att_fit
