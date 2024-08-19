# Licensed under a 3-clause BSD style license - see LICENSE.rst
import functools
import logging
from dataclasses import dataclass

import astropy.units as u
import astropy_healpix
import numpy as np
import ska_sun
from cxotime import CxoTime, CxoTimeLike
from Quaternion import Quat, QuatLike

logger = logging.getLogger(__name__)


@dataclass
class Constraints:
    """Constraints on attitude for attitude solution.

    Parameters
    ----------
    att : QuatLike or None
        Estimated attitude, assumed to be within ``att_err`` of true attitude.
    att_err : float
        Radial uncertainty of the attitude in degrees (default=4.0).
    pitch : float | None
        If not None, constrain sun pitch angle to ``pitch +/- pitch_err`` degrees.
        Default is None => no pitch constraint.
    pitch_err : float
        Uncertainty in pitch in degrees (default=1.5).
    off_nom_roll_max : float or None
        Maximum off-nominal roll angle in degrees (default=2.0).
    min_stars : int or None
        Minimum number of stars required for a valid solution. Default is None, meaning
        the minimum number of stars is determined from the constraints in the
        ``get_min_stars()`` function. Key use case for this parameter is to set it to 2
        for a two-star solution.
    mag_err : float or None
        Faint mag threshold for filtering candidate star pairs (default=1.5).
    date : CxoTimeLike
        Date for sun position calculation (default=NOW).
    """

    att: QuatLike | None = None
    att_err: float = 4.0
    pitch: float | None = None
    pitch_err: float = 1.5
    off_nom_roll_max: float | None = 2.0
    min_stars: int | None = None
    mag_err: float | None = 1.5
    date: CxoTimeLike = None

    @functools.cached_property
    def healpix_indices(self) -> np.ndarray | None:
        """Return a list of healpix indices based on the attitude and date.

        If ``att`` is not None then restrict to the cone within ``att_err`` of the
        attitude quaternion.  If ``pitch`` is not None then restrict to an annulus
        ``pitch +/- pitch_err`` corresponding to the sun pitch angle. Both can be
        supplied at the same time.

        If both ``att`` and ``pitch`` are None then return ``None``.

        Returns
        -------
        healpix_indices : ndarray | None
            List of healpix indices or None
        """
        from find_attitude import (
            get_agasc_pairs_attribute,
            get_healpix_indices_within_annulus,
        )

        if self.att is None and self.pitch is None:
            return None

        nside = get_agasc_pairs_attribute("healpix_nside")
        order = get_agasc_pairs_attribute("healpix_order")
        max_aca_radius = get_agasc_pairs_attribute("max_aca_dist") / 2

        hp = astropy_healpix.HEALPix(nside=nside, order=order)

        if self.att is None:
            idxs = np.arange(hp.npix)
        else:
            att = Quat(self.att)
            idxs = hp.cone_search_lonlat(
                att.ra * u.deg,
                att.dec * u.deg,
                (self.att_err + max_aca_radius) * u.deg,
            )

        if self.pitch is not None:
            # Get healpix indices of a cone out to pitch +/- pitch_err.
            sun_ra, sun_dec = ska_sun.position(self.date)
            radius0 = self.pitch - self.pitch_err - max_aca_radius
            radius1 = self.pitch + self.pitch_err + max_aca_radius
            idxs_nsun = get_healpix_indices_within_annulus(
                sun_ra,
                sun_dec,
                radius0=radius0,
                radius1=radius1,
                nside=nside,
                order=order,
            )
            idxs = np.intersect1d(idxs, idxs_nsun, assume_unique=True)

        return idxs

    def check_off_nom_roll(self, q_att: Quat):
        """Check if the off-nominal roll angle is within the maximum allowed.

        Parameters
        ----------
        q_att : Quat
            Attitude quaternion

        Returns
        -------
        ok : bool
            True if the off-nominal roll is within the maximum allowed
        """
        if self.off_nom_roll_max is None:
            return True

        off_nom_roll = ska_sun.off_nominal_roll(q_att, CxoTime(self.date))
        ok = abs(off_nom_roll) <= self.off_nom_roll_max
        if not ok:
            logger.info(
                f"Att {q_att} off-nominal roll {off_nom_roll:.2f} exceeds maximum"
                f" {self.off_nom_roll_max:.2f}"
            )
        return ok
