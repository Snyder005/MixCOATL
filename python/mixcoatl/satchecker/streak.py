from typing import Self

import lsst.geom as geom
import numpy as np
from lsst.afw.cameraGeom import FOCAL_PLANE, PIXELS

from .utils import regularize_line


class Streak:
    """A linear feature in a detector image.

    Useful
    ------
    camera
        detector
    detector:
        - bbox
            To get intersect points with the detector bounding box
        - transform
            To transform from PIXELS to FOCAL_PLANE
    """
    def __init__(self, rho: float, theta: float, visit: int, detector):
        self.rho, self.theta = regularize_line(rho, theta)
        self.visit = visit
        self.detector = detector

    @classmethod
    def from_table(cls, table, camera) -> list[Self]:
        visit = table.meta["visit"]
        streaks = []

        for row in table:
            detector_id = int(row["detector"])
            detector = camera[detector_id]

            xc, yc = detector.getBBox().getCenter()
            theta = np.deg2rad(row["theta"])
            rho = row["rho"] + xc * np.cos(theta) + yc * np.sin(theta)
            streaks.append(cls(rho, theta, visit, detector))
            
        return streaks

    def get_bbox_intercepts(self):
        bbox = self.detector.getBBox()
        xmin, xmax = bbox.getMinX(), bbox.getMaxX()
        ymin, ymax = bbox.getMinY(), bbox.getMaxY()

        sin_t = np.sin(self.theta)
        cos_t = np.cos(self.theta)

        pts = []
        if abs(sin_t) > 1e-12:
            for x in (xmin, xmax):
                y = (self.rho - x * cos_t) / sin_t
                if ymin <= y <= ymax:
                    pts.append(geom.Point2D(x, y))

        if abs(cos_t) > 1e-12:
            for y in (ymin, ymax):
                x = (self.rho - y * sin_t) / cos_t
                if xmin <= x <= xmax:
                    pts.append(geom.Point2D(x, y))

        unique = []
        for p in pts:
            if all(p.distanceSquared(q) >= 2e-18 for q in unique):
                unique.append(p)

        if len(unique) < 2:
            raise RuntimeError("Streak line does not intersect bbox properly.")

        return unique[:2]

    def get_focal_plane_line(self):

        n = geom.Extent2D([np.cos(self.theta), np.sin(self.theta)])
        d = geom.Extent2D([-np.sin(self.theta), np.cos(self.theta)])
        p0 = geom.Point2D(self.rho * n)
        p1 = p0 + d 

        p0_fp = self.detector.transform(p0, PIXELS, FOCAL_PLANE)
        p1_fp = self.detector.transform(p1, PIXELS, FOCAL_PLANE)

        d_fp = p1_fp - p0_fp
        d_fp /= d_fp.computeNorm()
        n_fp = geom.Extent2D(-d_fp.getY(), d_fp.getX())

        theta_fp = np.arctan2(n_fp.getY(), n_fp.getX())
        rho_fp = np.dot(n_fp, p0_fp)

        return regularize_line(rho_fp, theta_fp)
