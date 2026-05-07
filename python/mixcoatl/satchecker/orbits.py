import numpy as np
import lsst.geom as geom
from lsst.afw.cameraGeom import FOCAL_PLANE, PIXELS


def sky_to_focal_plane(ra_array, dec_array, pvs, camera):
    """
    Convert RA/Dec arrays to focal plane (mm) coordinates.
    """
    x = []
    y = []
    for ra, dec in zip(ra_array, dec_array):
        sky = geom.SpherePoint(ra, dec, geom.degrees)

        for row in pvs:
            wcs = row.getWcs()
            pix = wcs.skyToPixel(sky)
            detector = camera[row.getId()]

            if detector.getBBox().contains(geom.Point2I(pix)):
                fp = detector.transform(pix, PIXELS, FOCAL_PLANE)
                x.append(fp.getX())
                y.append(fp.getY())
                break

    return np.array(x), np.array(y)
