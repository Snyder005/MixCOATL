from __future__ import annotations

import math
import numpy as np
from numpy.typing import ArrayLike, NDArray

import lsst.geom as geom
from lsst.afw.geom import TransformPoint2ToPoint2


class Line2D:

    def __init__(self, rho: float, theta: geom.Angle):
        rho, theta = self._canonicalize(rho, theta)
        self._rho = rho
        self._theta = theta

    @classmethod
    def from_points(cls, p1: geom.Point2D, p2: geom.Point2D) -> Line2D:
        """Create a `Line2D` instance from two points.

        Parameters
        ----------
        p1 : `lsst.geom.Point2D`
            A point on the line.
        p2 : `lsst.geom.Point2D`
            A second point on the line.

        Returns
        -------
        line : `Line2D`
            An instance of `Line2D` defined by the two points.
        """
        return cls.from_point_and_direction(p1, p1 - p2)

    @classmethod
    def from_point_and_direction(cls, point: geom.Point2D, direction: geom.Extent2D) -> Line2D:
        """Create a `Line2D` instance from a point and a direction.

        Parameters
        ----------
        point : `lsst.geom.Point2D`
            A point on the line.
        direction : `lsst.geom.Extent2D`
            The direction of the line.

        Returns
        -------
        line : `Line2D`
            An instance of `Line2D` defined by the point and direction.

        Raises
        ------
        ValueError
            Raised if the direction vector is zero.
        """
        if (norm := direction.computeNorm()) == 0:
            raise ValueError("direction vector must be non-zero")

        normal = geom.Extent2D(-direction.y, direction.x) / norm
        rho = point.x * normal.x + point.y * normal.y
        theta = math.atan2(normal.y, normal.x) * geom.radians

        return cls(rho, theta)

    @property
    def rho(self) -> float:
        """The signed perpendicular distance from the origin to the line."""
        return self._rho

    @property
    def theta(self) -> geom.Angle:
        """The angle of the line normal vector."""
        return self._theta

    @property
    def normal(self) -> geom.Extent2D:
        """The line normal vector."""
        t = self.theta.asRadians()
        return geom.Extent2D(math.cos(t), math.sin(t))

    @property
    def direction(self) -> geom.Extent2D:
        """The line direction vector."""
        t = self.theta.asRadians()
        return geom.Extent2D(-math.sin(t), math.cos(t))

    @property
    def point(self) -> geom.Point2D:
        """The point on the line closest to the origin."""
        return geom.Point2D(self.normal * self.rho)

    def rescale(self, factor: float) -> None:
        """Rescale the perpendicular distance to the origin.
        
        Parameters
        ----------
        factor : `float`
            The factor to rescale by.
        """
        self._rho *= factor

    def shift(self) -> None:
        """Translate origin and update rho."""
        ...

    def transformed(self, transform: TransformPoint2ToPoint2) -> Line2D:
        """Transform this line into a new coordinate system.
        
        The transformation is applied by transforming two points on the line
        and constructing a new line from the transformed points. This allows
        the line parameters in Hesse normal form to be mapped through 
        arbitrary 2D coordinate transforms

        Parameters
        ----------
        transform : `lsst.afw.geom.TransformPoint2ToPoint2`
            Transform that maps points from the current coordinate system into
            the target coordinate system.

        Returns
        -------
        line : `Line2D`
            A new line in the target coordinate system.    
        """
        p0 = self.point
        p1 = p0 + self.direction

        p0_t = transform.applyForward(p0)
        p1_t = transform.applyForward(p1)

        return Line2D.from_points(p0_t, p1_t)

    def get_box_intercepts(self, box: geom.Box2I) -> tuple[geom.Point2D, geom.Point2D]:
        """Calculate the intersection points between the line and a box.

        Parameters
        ----------
        box : `lsst.geom.Box2I`
            The bounding box to intersect with the line.

        Returns
        -------
        p1 : `lsst.geom.Point2D`
            First intersection point between the line and the box bounds.
        p2 : `lsst.geom.Point2D`
            Second intersection point between the line and the box bounds.

        Returns
        -------
        ValueError
            Raised if the line does not intersect the box in exactly two
            points.
        """
        xmin = bbox.minX
        xmax = bbox.maxX
        ymin = bbox.minY
        ymax = bbox.maxY

        p0 = self.point
        d = self.direction

        points = []
        if abs(d.x) > 1e-14:
            for x in (xmin, xmax):
                y = p0.y + (x - p0.x) * d.y / d.x
                if ymin - 1e-12 <= y <= ymax + 1e-12:
                    points.append(geom.Point2D(x, y))

        if abs(d.y) > 1e-14:
            for y in (ymin, ymax):
                x = p0.x + (y - p0.y) * d.x / d.y
                if xmin - 1e-12 <= x <= xmax + 1e-12:
                    points.append(geom.Point2D(x, y))

        unique = []
        for p in points:
            if all(p.distanceSquared(q) >= 2e-18 for q in unique):
                unique.append(p)

        if len(unique) < 2:
            raise ValueError("line does not intersect box in exactly two points")

        return unique[0], unique[1]

    @staticmethod
    def _canonicalize(rho: float, theta: geom.Angle) -> tuple[float, geom.Angle]:
        """Convert line parameters to a canonical Hesse normal form.

        Parameters
        ----------
        rho : `float`
            Signed perpendicular distance from the origin to the line.
        theta : `float`
            Angle of the line normal vector.

        Returns
        -------
        rho : `float`
            Canonical signed perpendicular distance.
        theta : `lsst.geom.Angle`
            Canonical normal angle in the interval [0, pi).
        """
        theta_rad = theta.asRadians() % (2 * math.pi)

        if theta_rad >= math.pi:
            theta_rad -= math.pi
            rho = -rho
        
        return rho, theta_rad * geom.radians


def embed_rho_theta(
        rho: ArrayLike,
        theta: ArrayLike,
        rho_tol: float,
        theta_tol: float,
    ) -> NDArray[np.float64]:
    """Embed in euclidean space."""
    if rho_tol <= 0:
        raise ValueError(f"rho_tol must be > 0: {rho_tol}")

    if theta_tol <= 0:
        raise ValueError(f"theta_tol must be > 0: {theta_tol}")

    if theta_tol < 1e-12:
        raise ValueError(f"theta_tol too small for stable embedding: {theta_tol}")
    
    rho = np.asarray(rho, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)

    dist_scale = 1.0 / rho_tol
    angle_scale = 1.0 / (math.sqrt(2.0) * math.sin(theta_tol / 2.0))

    embedded_points = np.empty((rho.shape[0], 3), dtype=np.float64)
    embedded_points[:, 0] = angle_scale * np.cos(theta)
    embedded_points[:, 1] = angle_scale * np.sin(theta)
    embedded_points[:, 2] = dist_scale * rho

    return embedded_points

def fit_line_from_xy(x: ArrayLike, y: ArrayLike) -> Line2D:
    """Fit a line to x/y arrays.

    Used by SatChecker, should exist outside DM code.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    points = np.column_stack((x, y))
    centroid = points.mean(axis=0)

    _, _, vh = np.linalg.svd(points - centroid)
    normal = vh[-1]

    rho = float(np.dot(centroid, normal))
    theta = math.atan2(normal[1], normal[0])

    return Line2D(rho, theta * geom.radians)

