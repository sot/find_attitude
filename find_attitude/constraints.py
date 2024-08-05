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
        If supplied, restrict to cone within ``attitude_radius`` of attitude quaternion
    att_radius : float
        Radius of the attitude cone in degrees (default=4.0)
    normal_sun : bool
        If True, restrict to normal sun region (default=False)
    normal_sun_radius : float
        Radius of the normal sun annulus in degrees (default=1.5)
    normal_sun_pitch : float
        Pitch angle of normal sun in degrees (default=90.0). E.g. 160.0 for offset NSM.
    off_nom_roll_max : float or None
        Maximum off-nominal roll angle in degrees (default=2.0)
    date : CxoTimeLike
        Date for normal sun calculation (default=now)
    """

    att: QuatLike | None = None
    att_radius: float = 4.0
    normal_sun: bool = False
    normal_sun_radius: float = 1.5
    normal_sun_pitch: float = 90.0
    off_nom_roll_max: float | None = 2.0
    date: CxoTimeLike = None

    @functools.cached_property
    def healpix_indices(self) -> np.ndarray | None:
        """Return a list of healpix indices based on the attitude and date.

        If ``att`` is supplied then restrict to the cone within ``att_radius`` of the
        attitude quaternion.  If ``normal_sun`` is True then restrict to an annulus
        corresponding to the normal sun pitch angle. Both can be supplied at the same time.

        If neither ``att`` nor ``normal_sun`` are supplied then return ``None``.

        Returns
        -------
        healpix_indices : ndarray | None
            List of healpix indices or None
        """
        from find_attitude import (
            get_agasc_pairs_attribute,
            get_healpix_indices_within_annulus,
        )

        if self.att is None and not self.normal_sun:
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
                (self.att_radius + max_aca_radius) * u.deg,
            )

        if self.normal_sun:
            # Get healpix indices of a cone out to normal_sun_pitch + margin, and then
            # do the same for the anti-sun.  The normal sun indices are the intersection
            # of these two lists.
            sun_ra, sun_dec = ska_sun.position(self.date)
            radius0 = self.normal_sun_pitch - self.normal_sun_radius - max_aca_radius
            radius1 = self.normal_sun_pitch + self.normal_sun_radius + max_aca_radius
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
