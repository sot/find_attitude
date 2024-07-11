# Licensed under a 3-clause BSD style license - see LICENSE.rst
import collections
import logging
import os
from itertools import product
from pathlib import Path

import networkx as nx
import numpy as np
import tables
from astropy.io import ascii
from astropy.table import Column, MaskedColumn, Table, vstack
from ska_helpers.logging import basic_logger

SKA = Path(os.environ["SKA"])

DELTA_MAG = 1.5  # Accept matches where candidate star is within DELTA_MAG of observed

# Get the pre-made list of distances between AGASC stars.  The distances_kadi version
# is a symlink to a file on the kadi machine /export disk (faster).  However, if
# AGASC_PAIRS_FILE env var is defined then use that (for test/development).
try:
    AGASC_PAIRS_FILE = os.environ["AGASC_PAIRS_FILE"]
except KeyError:
    if (path := SKA / "data" / "find_attitude" / "distances-kadi-local.h5").exists():
        AGASC_PAIRS_FILE = path
    else:
        AGASC_PAIRS_FILE = SKA / "data" / "find_attitude" / "distances.h5"

logger = basic_logger(
    name="find_attitude", level="INFO", format="%(asctime)s %(message)s"
)


def get_stars_from_text(text):
    """Get stars table from ``text`` input.

    This can be a hand-entered table or copy/paste from GRETA A_ACA_ALL.  Minimal
    conforming examples are::

       MEAS      #   Flags   Functn  Flag        Y         Z        Mag
      IMAGE 0  0     STAR    TRAK   STAR     -381.28    1479.95      7.2
      IMAGE 1  1     STAR    TRAK   STAR     -582.55    -830.85      8.9
      IMAGE 2  2     STAR    TRAK   STAR     2076.33   -2523.10      8.5
      IMAGE 3  3     STAR    TRAK   STAR     -498.12    -958.33      5.0
      IMAGE 4  4     STAR    TRAK   STAR     -431.68    1600.98      8.2
      IMAGE 5  5     STAR    TRAK   STAR     -282.40     980.50      7.9
      IMAGE 6  6     NULL    NONE   STAR    -3276.80   -3276.80     13.9
      IMAGE 7  7     STAR    TRAK   STAR      573.25   -2411.70      7.1

    or::

      slot yag zag mag
      3     223.33      55.83      9.7
      4    -453.1      -2084.1     9.6
      5   -1255.12     196.58      9.2
      6     598.18    2287.97      9.6
      7    2311.45    1140.60      9.8

    :param text: text representation of input star table information.

    :returns: Table of star data
    """
    greta_fields = "MEAS # Flags Functn Flag Y Z Mag".split()

    # Split input into a list of lists
    text = str(text)
    lines = text.splitlines()
    values_list = [line.split() for line in lines]

    # Get a list of lines that match GRETA ACA status fields start line
    greta_start = [
        i_line
        for i_line, values in enumerate(values_list)
        if greta_fields == values[:8]
    ]
    if greta_start:
        i_start = greta_start[0] + 1
        values_lines = [
            " ".join(values[2:9]) for values in values_list[i_start : i_start + 8]
        ]
        try:
            stars = ascii.read(
                values_lines,
                format="no_header",
                names=["slot", "type", "function", "fid", "YAG", "ZAG", "MAG_ACA"],
            )
        except Exception as err:
            raise ValueError("Could not parse input: {}".format(str(err))) from None
        ok = (
            (stars["type"] == "STAR")
            & (stars["function"] == "TRAK")
            & (stars["fid"] == "STAR")
            & (stars["YAG"] > -3200)
            & (stars["ZAG"] > -3200)
        )

    else:
        # Space-delimited table input with slot, yag, zag, and mag columns
        try:
            stars = ascii.read(text, format="basic", delimiter=" ", guess=False)
        except Exception as err:
            raise ValueError("Could not parse input: {}".format(str(err))) from None

        colnames = ["slot", "yag", "zag", "mag"]
        if not set(stars.colnames).issuperset(colnames):
            raise ValueError(
                "Found column names {} in input but need column names {}".format(
                    stars.colnames, colnames
                )
            ) from None
        stars.rename_column("yag", "YAG")
        stars.rename_column("zag", "ZAG")
        stars.rename_column("mag", "MAG_ACA")
        ok = (stars["YAG"] > -3200) & (stars["ZAG"] > -3200)

    return stars[ok]


def get_dists_yag_zag(yags, zags, mags=None):
    """Get distances between every pair of stars with coordinates ``yags`` and ``zags``.

    Returns a Table with columns 'dists', 'idx0', 'idx1', 'mag0', 'mag1'.

    :param yags: np.array with star Y angles in arcsec
    :param zags: np.array with star Z angles in arcsec
    :param mags: np.array with star magnitudes (optional)

    :returns: Table with pair distances and corollary info
    """
    if np.all(np.abs(yags) < 2.0):
        # Must be in degrees, convert to arcsec
        yags = yags * 3600.0
        zags = zags * 3600.0

    if len(yags) != len(zags):
        raise ValueError("length of yags and zags must match")

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

    return Table(
        [dists, idx0, idx1, mags[idx0], mags[idx1]],
        names=["dists", "idx0", "idx1", "mag0", "mag1"],
    )


def get_triangles(G):
    """
    Get all the triangles in graph ``G``

    This code was taken from a google forum discussion posting by Daniel Schult.
    No license or attribution request was provided.
    https://groups.google.com/forum/#!topic/networkx-discuss/SHjKJFIFNtM

    :param G: input networkx graph

    :returns: list of (node0, node1, node2) tuples
    """
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
    """Add an edge to the graph including metdata values ``i0``, ``i1`` and ``dist``.

    ``id0`` and ``id1`` are the AGASC (and node) ID of the edge that gets inserted.

    The ``i0``, ``i1`` index values represent the index into the ACA stars catalog
    as taken from the ACA distance pair that matches the AGASC.  These catalog index
    values are accumulated in a list because there may be multiple potential
    assignments for each distance.

    Note that at this point there is no order to the identifcation, so it could be
    id0 <=> i0 or id0 <=> i1 (or none of the above).

    :param graph: networkx graph
    :param id0: AGASC / node ID of one node of the edge
    :param id1: AGASC / node ID of the other node of the edge
    :param i0: ACA star catalog index of one node of the edge
    :param i1: ACA star catalog index of ther other node of the edge
    :param dist: distance (arcsec) between nodes

    :returns: None
    """
    # Get previous edge metadata if it exists
    edge_data = graph.get_edge_data(id0, id1, None)

    if edge_data is None:
        graph.add_edge(id0, id1, i0=[i0], i1=[i1], dist=dist)
    else:
        edge = graph[id0][id1]
        snew = {i0, i1}
        for i0_old, i1_old in zip(edge["i0"], edge["i1"]):
            sold = {i0_old, i1_old}
            if snew == sold:
                return
        edge["i0"].append(i0)
        edge["i1"].append(i1)


def connected_agasc_ids(ap):
    """Return agacs_ids that occur at least 4 times.

    Each occurrence indicates an edge containing that agasc_id node.

    :param ap: table of AGASC pairs
    :returns: set of AGASC IDs
    """
    agasc_ids = np.concatenate((ap["agasc_id0"], ap["agasc_id1"]))
    c = Column(agasc_ids)
    cg = c.group_by(c)
    i_big_enough = np.flatnonzero(np.diff(cg.groups.indices) >= 3)
    out = set(cg.groups.keys[i_big_enough].tolist())

    return out


def get_match_graph(aca_pairs, agasc_pairs, tolerance):
    """Return network graph of all AGASC pairs that correspond to an ACA pair.

    Given a table of ``aca_pairs`` representing the distance between every pair in the
    observed ACA centroid data, and the table of AGASC catalog pairs, assemble a network
    graph of all AGASC pairs that correspond to an ACA pair.

    From this initial graph select only nodes that have at least 3 connected edges.

    :param aca_pairs: Table of pairwise distances for observed ACA star data
    :param agasc_pairs: Table of pairwise distances for AGASC catalog stars
    :param tolerance: matching distance (arcsec)

    :returns: networkx graph of distance-match pairs
    """
    idx0s = aca_pairs["idx0"]
    idx1s = aca_pairs["idx1"]
    dists = aca_pairs["dists"]
    mag0s = aca_pairs["mag0"]
    mag1s = aca_pairs["mag1"]

    logger.info("Starting get_match_graph")
    gmatch = nx.Graph()
    ap_list = []

    logger.info("Getting matches from file")
    for i0, i1, dist, mag0, mag1 in zip(idx0s, idx1s, dists, mag0s, mag1s):
        logger.debug("Getting matches from file {} {} {}".format(i0, i1, dist))
        ap = agasc_pairs.read_where(
            "(dists > {}) & (dists < {})".format(dist - tolerance, dist + tolerance)
        )

        # Filter AGASC pairs based on star pairs magnitudes
        if DELTA_MAG is not None:
            max_mag = max(mag0, mag1) + DELTA_MAG
            mag_ok = (ap["mag0"] / 1000.0 < max_mag) & (ap["mag1"] / 1000.0 < max_mag)
            ap = ap[mag_ok]

        ones = np.ones(len(ap), dtype=np.uint8)
        ap = Table(
            [ap["dists"], ap["agasc_id0"], ap["agasc_id1"], i0 * ones, i1 * ones],
            names=["dists", "agasc_id0", "agasc_id1", "i0", "i1"],
        )
        ap_list.append(ap)
        logger.debug("  Found {} matching pairs".format(len(ap)))

    ap = vstack(
        ap_list
    )  # Vertically stack the individual AGASC pairs tables into one big table
    connected_ids = connected_agasc_ids(
        ap
    )  # Find nodes with at least three connections

    agasc_id0 = ap["agasc_id0"]
    agasc_id1 = ap["agasc_id1"]
    i0 = ap["i0"]
    i1 = ap["i1"]
    dists = ap["dists"]

    # Finally make the graph of matching distance pairs
    logger.info("Adding edges from {} matching distance pairs".format(len(ap)))
    for i in range(len(ap)):
        id0 = agasc_id0[i]
        id1 = agasc_id1[i]
        if id0 in connected_ids and id1 in connected_ids:
            add_edge(gmatch, id0, id1, i0[i], i1[i], dists[i])

    logger.info("Added total of {} nodes".format(len(gmatch)))

    return gmatch


def get_slot_id_candidates(graph, nodes):
    """Get list of candidates which map node AGASC ID to ACA star catalog index number.

    For a ``graph`` of nodes / edges that match the stars in distance, and a list of
    ``clique_nodes`` which form a complete subgraph, find a list of identification
    candidates which map node AGASC ID to ACA star catalog index number.  This handles
    possible degenerate solutions and ensures that the outputs are topologically
    sensible.

    This is one of the trickier bits of algorithm, hence the complete lack of code
    comments.  It basically tries all permutations of star catalog index number and sees
    which ones end up as a complete graph (though it does this just by counting instead of
    making graphs).

    :param graph: graph of distance-match pairs
    :param nodes: list of nodes that form a complete subgraph

    :returns: list of dicts for plausible AGASC-ID to star index identifications
    """

    class BadCandidateError(Exception):
        pass

    n_nodes = len(nodes)
    i0_id0s_list = []
    i1_id1s_list = []
    for node0, node1 in nx.edges(graph, nodes):
        if node0 in nodes and node1 in nodes:
            edge_data = graph.get_edge_data(node0, node1)
            i0_id0s_list.append([(i0, node0) for i0 in edge_data["i0"]])
            i1_id1s_list.append([(i1, node1) for i1 in edge_data["i1"]])

    id_candidates = []
    for i0_id0s, i1_id1s in zip(product(*i0_id0s_list), product(*i1_id1s_list)):
        logger.info("")
        node_star_index_count = collections.defaultdict(collections.Counter)
        for i0_id0, i1_id1 in zip(i0_id0s, i1_id1s):
            i0, id0 = i0_id0
            i1, id1 = i1_id1
            for i in i0, i1:
                for id_ in id0, id1:
                    node_star_index_count[id_].update([i])

        try:
            agasc_id_star_index_map = {}
            for agasc_id, count_dict in node_star_index_count.items():
                logger.debug("AGASC ID {} has counts {}".format(agasc_id, count_dict))
                for index, count in count_dict.items():
                    if count == n_nodes - 1:
                        agasc_id_star_index_map[agasc_id] = index
                        break
                else:
                    logger.debug(
                        "  **** AGASC ID {} is incomplete **** ".format(agasc_id)
                    )
                    raise BadCandidateError
        except BadCandidateError:
            pass
        else:
            id_candidates.append(agasc_id_star_index_map)

    return id_candidates


def find_matching_agasc_ids(stars, agasc_pairs_file, g_dist_match=None, tolerance=2.5):
    """Given an input table of ``stars`` find the matching cliques.

    These cliques are completely connected subgraphs in the ``agasc_pairs_file`` of pair
    distances.  Do pair distance matching to within ``tolerance`` arcsec.

    At a minimum the stars table should include the following columns::

        'AGASC_ID', 'RA', 'DEC', 'YAG', 'ZAG', 'MAG_ACA'

    If ``g_dist_match`` is supplied then skip over reading the AGASC pairs and doing
    initial matching.  This is mostly for development.

    :param stars: table of up to 8 stars
    :param agasc_pairs_file: name of AGASC pairs file created with make_distances.py
    :param g_dist_match: graph with matching distances (optional, for development)
    :param tolerance: distance matching tolerance (arcsec)

    :returns: list of possible AGASC-ID to star index maps
    """
    if g_dist_match is None:
        logger.info("Using AGASC pairs file {}".format(agasc_pairs_file))
        h5 = tables.open_file(agasc_pairs_file, "r")
        agasc_pairs = h5.root.data
        g_dist_match = get_match_graph(stars, agasc_pairs, tolerance)
        h5.close()

    logger.info("Getting all triangles from match graph")
    match_tris = get_triangles(g_dist_match)

    # New graph with only plausible triangles
    g_geom_match = nx.Graph()
    logger.info("Finding triangles that match stars pattern")
    for tri in match_tris:
        e0 = g_dist_match.get_edge_data(tri[0], tri[1])
        e1 = g_dist_match.get_edge_data(tri[0], tri[2])
        e2 = g_dist_match.get_edge_data(tri[1], tri[2])

        for e0_i, e0_i0 in enumerate(e0["i0"]):
            e0_i1 = e0["i1"][e0_i]
            for e1_i, e1_i0 in enumerate(e1["i0"]):
                e1_i1 = e1["i1"][e1_i]
                for e2_i, e2_i0 in enumerate(e2["i0"]):
                    e2_i1 = e2["i1"][e2_i]
                    if len({e0_i0, e0_i1, e1_i0, e1_i1, e2_i0, e2_i1}) == 3:
                        add_edge(g_geom_match, tri[0], tri[1], e0_i0, e0_i1, e0["dist"])
                        add_edge(g_geom_match, tri[0], tri[2], e1_i0, e1_i1, e1["dist"])
                        add_edge(g_geom_match, tri[2], tri[1], e2_i0, e2_i1, e2["dist"])

    if logger.level <= logging.DEBUG:
        logger.debug("g_geom_match: ")
        for n0, n1 in nx.edges(g_geom_match):
            ed = g_geom_match.get_edge_data(n0, n1)
            logger.debug(f"{n0=} {n1=} {ed=}")

    out = []

    # Iterate through complete subgraphs (cliques) and require at least
    # four nodes.
    for clique_nodes in nx.find_cliques(g_geom_match):
        n_clique_nodes = len(clique_nodes)
        if n_clique_nodes < 4:
            continue

        logger.info("Checking clique {}".format(clique_nodes))

        out.extend(get_slot_id_candidates(g_geom_match, clique_nodes))

    logger.info("Done with graph matching")
    return out


def find_all_matching_agasc_ids(
    yags, zags, mags=None, agasc_pairs_file=None, dist_match_graph=None, tolerance=2.5
):
    """Given an input table of ``stars`` find the matching cliques.

    These cliques are completely connected subgraphs in the ``agasc_pairs_file`` of pair
    distances.  Do pair distance matching to within ``tolerance`` arcsec.

    If ``g_dist_match`` is supplied then skip over reading the AGASC pairs and doing
    initial matching.  This is mostly for development.

    :param stars: table of up to 8 stars
    :param agasc_pairs_file: name of AGASC pairs file created with make_distances.py
    :param dist_match_graph: graph with matching distances (optional, for development)
    :param tolerance: distance matching tolerance (arcsec)

    :returns: list of possible AGASC-ID to star index maps
    """
    stars = get_dists_yag_zag(yags, zags, mags)
    agasc_id_star_maps = find_matching_agasc_ids(
        stars, agasc_pairs_file, dist_match_graph, tolerance=tolerance
    )
    return agasc_id_star_maps


def find_attitude_for_agasc_ids(yags, zags, agasc_id_star_map):
    """Find the fine attitude for the given inputs.

    Find the fine attitude for a given set of yags and zags and a map of AGASC ID
    to star index (i.e. index into ``yags`` and ``zags`` arrays).

    Returns a dictionary with keys::

      yags : input Y angles
      zags : input Z angles
      m_yags : best fit Y angles
      m_zags : best fit Z angles
      att_fit : best fit attitude quaternion (Quat object)
      statval : final fit statistic
      agasc_id_star_map : input AGASC ID to star index map

    :returns: dict

    """
    # Sherpa model global, leave this alone.
    global yagzag  # noqa: PLW0602

    # Squelch the useless warning below prior to import. Since this application
    # does not dealing with image data we don't worry about silencing these.
    # WARNING: imaging routines will not be available,
    # failed to import sherpa.image.ds9_backend due to
    # 'RuntimeErr: DS9Win unusable: Could not find ds9 on your PATH'
    logging.getLogger("sherpa.image").setLevel(logging.ERROR)

    import agasc
    from Quaternion import Quat
    from sherpa import ui
    from Ska.quatutil import radec2yagzag

    # Set sherpa logger to same level as local logger
    sherpa_logger = logging.getLogger("sherpa")
    # FIXME: this looks like a bug in the original code. Either get rid of the loop or
    # set levels for all handlers along with the sherpa_logger.
    for _hdlr in sherpa_logger.handlers:
        sherpa_logger.setLevel(logger.level)

    star_indices = list(agasc_id_star_map.values())
    yags = yags[star_indices]
    zags = zags[star_indices]

    agasc_ids = list(agasc_id_star_map.keys())
    agasc_stars = [agasc.get_star(agasc_id) for agasc_id in agasc_ids]

    ras = [s["RA_PMCORR"] for s in agasc_stars]
    decs = [s["DEC_PMCORR"] for s in agasc_stars]

    qatt = Quat([ras[0], decs[0], 0.0])

    def _yag_zag(dra, ddec, droll):
        q = qatt * Quat([dra / 3600.0, ddec / 3600.0, droll])
        yags, zags = radec2yagzag(ras, decs, q)
        return yags * 3600, zags * 3600, q

    def yag_zag(pars, x):  # noqa: ARG001
        m_yags, m_zags, q = _yag_zag(*pars)
        return np.concatenate([m_yags, m_zags])

    m_yags, m_zags, att_fit = _yag_zag(0, 0, 0)

    stars = get_dists_yag_zag(yags, zags)
    dists = stars["dists"]
    idx0 = stars["idx0"]
    idx1 = stars["idx1"]
    i = np.argmax(dists)
    m_ang = np.arctan2(
        m_zags[idx0[i]] - m_zags[idx1[i]], m_yags[idx0[i]] - m_yags[idx1[i]]
    )
    s_ang = np.arctan2(zags[idx0[i]] - zags[idx1[i]], yags[idx0[i]] - yags[idx1[i]])
    roll = np.degrees(m_ang - s_ang)

    y = np.concatenate([yags, zags])
    ui.load_arrays(1, np.arange(len(y)), y, np.ones(len(y)))
    ui.load_user_model(yag_zag, "yagzag")
    ui.add_user_pars("yagzag", ["dra", "ddec", "droll"])
    ui.set_model(yagzag)
    ui.set_method("simplex")

    yagzag.dra = 0.0
    yagzag.ddec = 0.0
    yagzag.droll = roll

    ui.freeze(yagzag.droll)
    ui.fit()

    ui.thaw(yagzag.droll)
    ui.fit()
    fit_results = ui.get_fit_results()
    m_yags, m_zags, att_fit = _yag_zag(
        yagzag.dra.val, yagzag.ddec.val, yagzag.droll.val
    )

    out = {
        "yags": yags,
        "zags": zags,
        "m_yags": m_yags,
        "m_zags": m_zags,
        "att_fit": att_fit,
        "statval": fit_results.statval,
        "agasc_id_star_map": agasc_id_star_map,
    }

    return out


def find_attitude_solutions(stars, tolerance=2.5):
    """
    Find attitude solutions given an input table of star data.

    The input star table must have columns 'YAG' (arcsec), 'ZAG' (arcsec), and
    'MAG'.  There must be at least four stars for the matching algorithm to succeed.

    The output is a list of solutions, where each solution is a dict with keys::

      summary : copy of input stars table with new columns of useful info
      yags : input Y angles
      zags : input Z angles
      m_yags : best fit Y angles
      m_zags : best fit Z angles
      agasc_ids : list of AGASC IDs corresponding to inputs
      att_fit : best fit attitude quaternion (Quat object)
      statval : final fit statistic
      agasc_id_star_map : input AGASC ID to star index map

    :param stars: table of star data
    :param tolerance: matching tolerance (arcsec, default=2.5)

    :returns: list of solutions, where each solution is a dict
    """
    if len(stars) < 4:
        raise ValueError(
            "need at least 4 stars for matching, only {} were provided".format(
                len(stars)
            )
        )

    agasc_id_star_maps = find_all_matching_agasc_ids(
        stars["YAG"],
        stars["ZAG"],
        stars["MAG_ACA"],
        agasc_pairs_file=AGASC_PAIRS_FILE,
        tolerance=tolerance,
    )

    solutions = []
    for agasc_id_star_map in agasc_id_star_maps:
        solution = find_attitude_for_agasc_ids(
            stars["YAG"], stars["ZAG"], agasc_id_star_map
        )

        # Check to see if there is another solution that has overlapping
        # stars.  If there are overlaps and this solution has a lower
        # fit statistic then use this solution.
        agasc_ids = set(agasc_id_star_map)
        solution["agasc_ids"] = agasc_ids
        for prev_solution in solutions:
            if agasc_ids.intersection(prev_solution["agasc_ids"]):
                if solution["statval"] < prev_solution["statval"]:
                    prev_solution.update(solution)
                break
        else:
            solutions.append(solution)

    _update_solutions(solutions, stars)
    return solutions


def _update_solutions(solutions, stars):
    for sol in solutions:
        summ = Table(stars, masked=True)

        indices = list(sol["agasc_id_star_map"].values())
        for name in ("m_yag", "dy", "m_zag", "dz", "dr"):
            summ[name] = MaskedColumn([-99.0] * len(summ), name=name, mask=True)
        summ["m_agasc_id"] = MaskedColumn(
            [-99] * len(summ), name="m_agasc_id", mask=True
        )

        summ["m_yag"][indices] = sol["m_yags"]
        summ["dy"][indices] = sol["yags"] - sol["m_yags"]
        summ["m_zag"][indices] = sol["m_zags"]
        summ["dz"][indices] = sol["zags"] - sol["m_zags"]
        dr = np.sqrt(
            (sol["yags"] - sol["m_yags"]) ** 2 + (sol["zags"] - sol["m_zags"]) ** 2
        )
        summ["dr"][indices] = dr
        summ["m_agasc_id"][indices] = list(sol["agasc_id_star_map"].keys())

        for name in summ.colnames:
            if name in ("RA", "DEC"):
                format = "{:.4f}"
            elif "agasc" in name.lower():
                format = "{:d}"
            elif name in ("m_yag", "dy", "m_zag", "dz", "dr"):
                format = "{:.2f}"
            else:
                format = None
            if format:
                summ[name].format = format
        sol["summary"] = summ

        # Need at least 4 stars with radial fit residual < 3 arcsec
        sol["bad_fit"] = np.sum(dr < 3.0) < 4
