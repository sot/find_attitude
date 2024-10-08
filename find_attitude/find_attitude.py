# Licensed under a 3-clause BSD style license - see LICENSE.rst
import collections
import functools
import logging
import os
from itertools import product
from pathlib import Path

# Squelch the useless warning below prior to import. Since this application
# does not dealing with image data we don't worry about silencing these.
# WARNING: imaging routines will not be available,
# failed to import sherpa.image.ds9_backend due to
# 'RuntimeErr: DS9Win unusable: Could not find ds9 on your PATH'
logging.getLogger("sherpa.image").setLevel(logging.ERROR)

import agasc
import astropy.units as u
import astropy_healpix
import networkx as nx
import numpy as np
import sherpa.ui
import tables
from astropy.io import ascii
from astropy.table import Column, MaskedColumn, Table, vstack
from chandra_aca.transform import eci_to_radec, radec_to_yagzag
from cxotime import CxoTimeLike
from Quaternion import Quat
from ska_helpers.logging import basic_logger
from ska_helpers.utils import set_log_level

from find_attitude.constraints import Constraints

# Default faint mag threshold for filtering candidate star pairs. Only applies if no
# constraints are provided.
DELTA_MAG_DEFAULT = 1.5

# Get the pre-made list of distances between AGASC stars.  If AGASC_PAIRS_FILE env var
# is defined then use that (for test/development).
AGASC_PAIRS_FILE = os.environ.get(
    "AGASC_PAIRS_FILE",
    default=Path(os.environ["SKA"]) / "data" / "find_attitude" / "distances.h5",
)

logger = basic_logger(
    name="find_attitude", level="INFO", format="%(asctime)s %(message)s"
)

# Define AGASC pairs h5 file attribute defaults. This is most applicable for testing
# with the legacy distances.h5 file.
ATTRIBUTE_DEFAULTS = {
    "max_mag": 10.5,
    "min_aca_dist": 25.0 / 3600.0,  # deg
    "max_aca_dist": 7200.0 / 3600.0,  # deg
    "date_distances": "2015:001",
    "healpix_nside": 16,
    "healpix_order": "nested",
    "agasc_version": "1p7",
}


@functools.lru_cache()
def get_agasc_pairs_attribute(attr):
    with tables.open_file(AGASC_PAIRS_FILE, "r") as h5:
        try:
            return getattr(h5.root.data.attrs, attr)
        except AttributeError:
            return ATTRIBUTE_DEFAULTS[attr]


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


def get_stars_from_maude(
    date: CxoTimeLike = None, dt: float = 11.0, slots: list | None = None
):
    """Get star data from MAUDE for a given date.

    This gets ``dt`` seconds of star data ending at ``date``. If ``date`` is None then
    get the most recent star data. The star data are AOACYAN, AOACZAN, and AOACMAG for
    each of the 8 slots.

    Parameters
    ----------
    date : CxoTimeLike
        Date for which to get star data
    dt : float
        Time interval (seconds) for which to get star data (default=11.0)
    slots : list of int, optional
        List of slots to get star data for (default is all 8 slots)

    Returns
    -------
    Table
        Table of star data for tracked stars in the 8 slots. Columns are 'slot', 'YAG',
        'ZAG', and 'MAG_ACA'.
    """
    import maude
    from cxotime import CxoTime

    if slots is None:
        slots = list(range(8))

    msids = []
    msids.extend([f"aoacyan{ii}" for ii in slots])
    msids.extend([f"aoaczan{ii}" for ii in slots])
    msids.extend([f"aoacmag{ii}" for ii in slots])
    stop = CxoTime(date)
    start = stop - dt * u.s
    dat = maude.get_msids(msids, start=start, stop=stop)
    results = dat["data"]
    out = {}
    for result in results:
        msid = result["msid"]
        values = result["values"]
        if len(values) == 0:
            # Missing data, this should no happen but just in case
            value = 9999.0
        elif len(values) < 6:
            value = np.median(values)
        else:
            # Throw out the top and bottom 2 values and take the mean of the middle
            values = np.sort(values)
            value = np.mean(values[2:-2])
        out[msid] = value
    tbl = Table()

    tbl["slot"] = slots
    tbl["YAG"] = [out[f"AOACYAN{ii}"] for ii in slots]
    tbl["ZAG"] = [out[f"AOACZAN{ii}"] for ii in slots]
    tbl["MAG_ACA"] = [out[f"AOACMAG{ii}"] for ii in slots]
    tbl.meta["date"] = stop - dt * u.s / 2

    # Filter non-tracking slots which have YAG, ZAG = 3276.8
    ok = (tbl["YAG"] < 3200) & (tbl["ZAG"] < 3200)
    tbl = tbl[ok]
    for name in ["YAG", "ZAG", "MAG_ACA"]:
        tbl[name].format = ".2f"

    return tbl


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
        mags = np.ones_like(yags) * 15

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


def connected_agasc_ids(ap, min_stars):
    """Return agacs_ids that occur at least 4 times.

    Each occurrence indicates an edge containing that agasc_id node.

    :param ap: table of AGASC pairs
    :returns: set of AGASC IDs
    """
    agasc_ids = np.concatenate((ap["agasc_id0"], ap["agasc_id1"]))
    c = Column(agasc_ids)
    cg = c.group_by(c)
    i_big_enough = np.flatnonzero(np.diff(cg.groups.indices) >= min_stars - 1)
    out = set(cg.groups.keys[i_big_enough].tolist())

    return out


def get_match_graph(
    aca_pairs: Table,
    agasc_pairs: tables.table.Table,
    tolerance: float,
    constraints: Constraints | None = None,
) -> nx.Graph:
    """Return network graph of all AGASC pairs that correspond to an ACA pair.

    Given a table of ``aca_pairs`` representing the distance between every pair in the
    observed ACA centroid data, and the table of AGASC catalog pairs, assemble a network
    graph of all AGASC pairs that correspond to an ACA pair.

    From this initial graph select only nodes that have at least 3 connected edges.

    Parameters
    ----------
    aca_pairs : Table
        Table of pairwise distances for observed ACA star data
    agasc_pairs : tables.table.Table
        Table of pairwise distances for AGASC catalog stars from opened h5 object
    tolerance : float
        Matching distance (arcsec)
    constraints : Constraints object or None
        Attitude, normal sun, and date constraints if available

    Returns
    -------
    nx.Graph
        Networkx graph of distance-match pairs
    """

    min_stars = get_min_stars(constraints)

    logger.info("Starting get_match_graph")
    gmatch = nx.Graph()

    ap = get_distance_pairs(aca_pairs, agasc_pairs, tolerance, constraints)

    # Find nodes with at least (min_stars - 1) connections
    connected_ids = connected_agasc_ids(ap, min_stars)

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


def get_distance_pairs(aca_pairs, agasc_pairs, tolerance, constraints) -> Table:
    """
    Get AGASC pairs that match ACA pair distances.

    Output table has columns 'dists', 'agasc_id0', 'agasc_id1', 'i0', 'i1'.

    Parameters
    ----------
    aca_pairs : Table
        Table of pairwise distances for observed ACA star data
    agasc_pairs : tables.table.Table
        Table of pairwise distances for AGASC catalog stars from opened h5 object
    tolerance : float
        Matching distance (arcsec)
    constraints : Constraints object or None
        Attitude, normal sun, and date constraints if available

    Returns
    -------
    Table
        Table of AGASC pairs that match ACA pair distances
    """
    idx0s = aca_pairs["idx0"]
    idx1s = aca_pairs["idx1"]
    dists = aca_pairs["dists"]
    mag0s = aca_pairs["mag0"]
    mag1s = aca_pairs["mag1"]

    logger.info("Getting matches from file")
    ap_list = []
    delta_mag = DELTA_MAG_DEFAULT if constraints is None else constraints.mag_err

    for i0, i1, dist, mag0, mag1 in zip(idx0s, idx1s, dists, mag0s, mag1s):
        logger.debug("Getting matches from file {} {} {}".format(i0, i1, dist))
        ap = agasc_pairs.read_where(
            "(dists > {}) & (dists < {})".format(dist - tolerance, dist + tolerance)
        )
        if constraints is not None and constraints.healpix_indices is not None:
            ok = np.in1d(ap["pix0"], constraints.healpix_indices) | np.in1d(
                ap["pix1"], constraints.healpix_indices
            )
            ap = ap[ok]

        # Filter AGASC pairs based on star pairs magnitudes
        if delta_mag is not None:
            max_mag = max(mag0, mag1) + delta_mag
            mag_ok = (ap["mag0"] / 1000.0 < max_mag) & (ap["mag1"] / 1000.0 < max_mag)
            ap = ap[mag_ok]

        ones = np.ones(len(ap), dtype=np.uint8)
        ap = Table(
            [ap["dists"], ap["agasc_id0"], ap["agasc_id1"], i0 * ones, i1 * ones],
            names=["dists", "agasc_id0", "agasc_id1", "i0", "i1"],
        )
        ap_list.append(ap)
        logger.debug("  Found {} matching pairs".format(len(ap)))

    # Vertically stack the individual AGASC pairs tables into one big table
    ap = vstack(ap_list)
    return ap


def get_min_stars(constraints):
    """Minimum number of stars required for an attitude solution.

    This is somewhat arbitrary, but the number of healpix indices corresponds to sky
    area and is a decent proxy for the number of stars that will be found in the
    matching process.

    Parameters
    ----------
    constraints : Constraints object or None
        Attitude, normal sun, and date constraints if available
    """
    if constraints and constraints.min_stars is not None:
        return constraints.min_stars

    nside = get_agasc_pairs_attribute("healpix_nside")
    npix = astropy_healpix.nside_to_npix(nside)

    if constraints is None or constraints.healpix_indices is None:
        # No constraints or constraints do not restrict the healpix indices
        min_stars = 4
    elif len(constraints.healpix_indices) < npix / 10:
        # Typically for a pitch annulus in normal sun mode or estimated attitude
        min_stars = 3
    else:
        min_stars = 4

    return min_stars


def get_slot_id_candidates(graph, nodes) -> list[dict[int, int]]:
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

    Output is a list of dictionaries where the keys are AGASC IDs and the values are
    indices into the ACA star table. For example::

      [{876482752: 5, 876610304: 3, 876481760: 4, 876486456: 7, 876610432: 6}]

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


def find_matching_agasc_ids(
    aca_pairs, agasc_pairs_file, g_dist_match=None, tolerance=2.5, constraints=None
) -> list[dict[int, int]]:
    """Given an input table of ``stars`` find the matching cliques.

    These cliques are completely connected subgraphs in the ``agasc_pairs_file`` of pair
    distances.  Do pair distance matching to within ``tolerance`` arcsec.

    At a minimum the stars table should include the following columns::

        'AGASC_ID', 'RA', 'DEC', 'YAG', 'ZAG', 'MAG_ACA'

    If ``g_dist_match`` is supplied then skip over reading the AGASC pairs and doing
    initial matching.  This is mostly for development.

    Output is a list of dictionaries where the keys are AGASC IDs and the values are
    indices into the input star table. For example::

      [{876482752: 5, 876610304: 3, 876481760: 4, 876486456: 7, 876610432: 6}]

    :param aca_pairs: table of distances between every star pair
    :param agasc_pairs_file: name of AGASC pairs file created with make_distances.py
    :param g_dist_match: graph with matching distances (optional, for development)
    :param tolerance: distance matching tolerance (arcsec)
    :param constraints: Constraints object or None
        Attitude, normal sun, and date constraints if available

    :returns: list of possible AGASC-ID to star index maps
    """
    # Zero ACA pairs => 0 stars, 1 pair => 2 stars, > 2 pairs => at least 3 stars
    if len(aca_pairs) < 1:
        raise ValueError("need at least 2 stars to match")
    elif len(aca_pairs) == 1:
        func = _find_matching_agasc_ids_two_stars
    else:
        func = _find_matching_agasc_ids_three_or_more_stars

    out = func(aca_pairs, agasc_pairs_file, g_dist_match, tolerance, constraints)
    logger.info(f"Found {len(out)} possible AGASC-ID to star index maps")
    return out


def _find_matching_agasc_ids_two_stars(
    aca_pairs, agasc_pairs_file, g_dist_match=None, tolerance=2.5, constraints=None
):
    """Helper function for find_matching_agasc_ids() for two stars.

    Since for two stars there is no need to find cliques, this function is a bit
    simpler.  It just finds all the pairs of AGASC stars that match the ACA pairs.
    There is a 180 degree ambiguity in the ACA star catalog, so for each pair of ACA
    stars there are two possible AGASC pairs.

    See that function for details.
    """
    if g_dist_match is not None:
        raise ValueError("g_dist_match must be None for two star matching")

    logger.info(f"Opening AGASC pairs file {agasc_pairs_file}")
    with tables.open_file(agasc_pairs_file, "r") as h5:
        agasc_pairs = h5.root.data
        ap = get_distance_pairs(aca_pairs, agasc_pairs, tolerance, constraints)

    out = []
    for row in ap:
        out.append({row["agasc_id0"]: row["i0"], row["agasc_id1"]: row["i1"]})
        out.append({row["agasc_id0"]: row["i1"], row["agasc_id1"]: row["i0"]})

    return out


def _find_matching_agasc_ids_three_or_more_stars(
    aca_pairs, agasc_pairs_file, g_dist_match=None, tolerance=2.5, constraints=None
):
    """Helper function for find_matching_agasc_ids() for three or more stars.

    See that function for details.
    """
    if g_dist_match is None:
        logger.info("Using AGASC pairs file {}".format(agasc_pairs_file))
        with tables.open_file(agasc_pairs_file, "r") as h5:
            agasc_pairs = h5.root.data
            g_dist_match = get_match_graph(
                aca_pairs, agasc_pairs, tolerance, constraints
            )

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
    # min_stars nodes.
    min_nodes = get_min_stars(constraints)
    for clique_nodes in nx.find_cliques(g_geom_match):
        n_clique_nodes = len(clique_nodes)
        if n_clique_nodes < min_nodes:
            continue

        logger.info("Checking clique {}".format(clique_nodes))
        out.extend(get_slot_id_candidates(g_geom_match, clique_nodes))

    logger.info("Done with graph matching")

    if len(out) > 0:
        # Only accept solutions with the maximum number of nodes.
        max_clique_nodes = max(len(c) for c in out)
        out = [c for c in out if len(c) == max_clique_nodes]

    return out


def find_all_matching_agasc_ids(
    yags,
    zags,
    mags=None,
    agasc_pairs_file=None,
    dist_match_graph=None,
    tolerance=2.5,
    constraints=None,
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
    :param constraints: Constraints object or None
        Attitude, normal sun, and date constraints if available

    :returns: list of possible AGASC-ID to star index maps
    """
    aca_pairs = get_dists_yag_zag(yags, zags, mags)
    agasc_id_star_maps = find_matching_agasc_ids(
        aca_pairs,
        agasc_pairs_file,
        dist_match_graph,
        tolerance=tolerance,
        constraints=constraints,
    )
    return agasc_id_star_maps


def find_attitude_for_agasc_ids(yags, zags, agasc_id_star_map, constraints=None):
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

    :param yags: np.array with star Y angles in arcsec
    :param zags: np.array with star Z angles in arcsec
    :param agasc_id_star_map: dict of AGASC ID to star index map
    :param constraints: Constraints object or None
        Attitude, normal sun, and date constraints if available

    :returns: dict

    """
    # Sherpa model global, leave this alone.
    global yagzag  # noqa: PLW0602

    logger.info(f"Finding attitude for {agasc_id_star_map} stars")
    star_indices = list(agasc_id_star_map.values())
    yags = yags[star_indices]
    zags = zags[star_indices]

    date = constraints.date if constraints else None
    agasc_ids = list(agasc_id_star_map.keys())
    agasc_stars = [agasc.get_star(agasc_id, date=date) for agasc_id in agasc_ids]

    ras = [s["RA_PMCORR"] for s in agasc_stars]
    decs = [s["DEC_PMCORR"] for s in agasc_stars]

    qatt = Quat([ras[0], decs[0], 0.0])

    def _yag_zag(dra, ddec, droll):
        q = qatt * Quat([dra / 3600.0, ddec / 3600.0, droll])
        yags, zags = radec_to_yagzag(ras, decs, q)
        return yags, zags, q

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
    ui = sherpa.ui
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

    # Only need to fit roll for more than 2 stars. With 2 stars, original roll is exact.
    if len(agasc_stars) > 2:
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
    logger.info(f"Found attitude {att_fit} with statval {fit_results.statval}")

    return out


def find_attitude_solutions(
    stars,
    tolerance=2.5,
    constraints=None,
    log_level=None,
    sherpa_log_level="WARNING",
):
    """
    Find attitude solutions given an input table of star data.

    The input star table must have columns 'YAG' (arcsec), 'ZAG' (arcsec), and
    'MAG'.  There must be at least four stars for the matching algorithm to succeed.

    This relies on a pre-computed star pair distance file that is created with
    ``make_distances.py``.  This file is an HDF5 file with a table of pair distances
    that is computed for a particular star epoch.

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
    :param constraints: Constraints object or None
        Attitude, normal sun, and date constraints if available
    :param log_level: logging level for find_attidue functions
    :param sherpa_log_level: logging level for sherpa

    :returns: list of solutions, where each solution is a dict
    """
    with (
        set_log_level(logger, log_level),
        set_log_level(logging.getLogger("sherpa"), sherpa_log_level),
    ):
        return _find_attitude_solutions(stars, tolerance, constraints)


def _find_attitude_solutions(
    stars,
    tolerance=2.5,
    constraints=None,
):
    min_stars = get_min_stars(constraints)
    if len(stars) < min_stars:
        raise ValueError(
            f"need at least {min_stars} stars for matching,"
            f" only {len(stars)} were provided"
        )

    agasc_id_star_maps = find_all_matching_agasc_ids(
        stars["YAG"],
        stars["ZAG"],
        stars["MAG_ACA"],
        agasc_pairs_file=AGASC_PAIRS_FILE,
        tolerance=tolerance,
        constraints=constraints,
    )

    solutions = []
    for agasc_id_star_map in agasc_id_star_maps:
        solution = find_attitude_for_agasc_ids(
            stars["YAG"], stars["ZAG"], agasc_id_star_map, constraints
        )

        if constraints and not constraints.check_off_nom_roll(solution["att_fit"]):
            continue

        # Check to see if there is another solution that has overlapping
        # stars.  If there are overlaps and this solution has a lower
        # fit statistic then use this solution.
        agasc_ids = set(agasc_id_star_map)
        solution["agasc_ids"] = agasc_ids
        for prev_solution in solutions:
            if agasc_ids.intersection(prev_solution["agasc_ids"]):
                if solution["statval"] < prev_solution["statval"]:
                    logger.info(
                        f"Updating solution for {prev_solution['agasc_ids']} with "
                        f"{agasc_ids} for better statval {solution['statval']}"
                    )
                    prev_solution.update(solution)
                else:
                    logger.info(f"Skipping solution for {agasc_ids} (worse statval)")
                break
        else:
            logger.info(
                f"Adding solution for {agasc_ids} with statval {solution['statval']}"
            )
            solutions.append(solution)

    _update_solutions(solutions, stars, min_stars)
    return solutions


def _update_solutions(solutions, stars, min_stars):
    for sol in solutions:
        summ = Table(stars, masked=True)

        indices = list(sol["agasc_id_star_map"].values())
        for name in ("m_yag", "m_zag", "m_mag", "dy", "dz", "dr"):
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
        m_agasc_ids = list(sol["agasc_id_star_map"].keys())
        summ["m_agasc_id"][indices] = m_agasc_ids

        stars = agasc.get_stars(m_agasc_ids)
        summ["m_mag"][indices] = [star["MAG_ACA"] for star in stars]

        for name in summ.colnames:
            if name in ("RA", "DEC"):
                format = "{:.4f}"
            elif "agasc" in name.lower():
                format = "{:d}"
            elif name in ("m_yag", "m_zag", "m_mag", "dy", "dz", "dr"):
                format = "{:.2f}"
            elif name in ("YAG", "YAG_ERR", "ZAG", "ZAG_ERR", "MAG_ACA", "MAG_ERROR"):
                format = "{:.2f}"
            else:
                format = None
            if format:
                summ[name].format = format
        sol["summary"] = summ

        # Need at least min_stars stars with radial fit residual < 5.0 arcsec
        sol["bad_fit"] = np.sum(dr < 5.0) < min_stars


def get_healpix_indices_within_annulus(
    ra: float,
    dec: float,
    *,
    radius0: float,
    radius1: float,
    nside: int = 64,
    order: str = "nested",
):
    """Get healpix indices within a range of radii from a given RA, Dec.

    This is a much faster, somewhat approximate version of the astropy_healpix method
    cone_search_lonlat(). It works by computing the positions of the four corners of
    each pixels and checking if that is within the annulus. This requires that
    ``radius0`` and ``radius1`` are both much bigger than the healpix pixel size.

    Parameters
    ----------
    ra : float
        Right ascension in degrees
    dec : float
        Declination in degrees
    radius0 : float
        Inner radius in degrees
    radius1 : float
        Outer radius in degrees
    nside : int
        Healpix nside parameter (default=64)
    order : str
        Healpix order parameter (default='nested')

    Returns
    -------
    healpix_indices : ndarray
        List of healpix indices
    """
    npix = astropy_healpix.nside_to_npix(nside)

    ok = np.zeros(npix, dtype=bool)
    for dx, dy in zip([0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]):
        x, y, z = astropy_healpix.healpix_to_xyz(
            np.arange(npix), dx=dx, dy=dy, nside=64, order=order
        )
        ras, decs = eci_to_radec(np.array([x, y, z]).T)
        distances = agasc.sphere_dist(ra, dec, ras, decs)
        ok |= (distances >= radius0) & (distances <= radius1)

    return np.where(ok)[0]
