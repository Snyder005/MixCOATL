from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

import lsst.geom as geom

@dataclass
class LineFitResult:
    line_segment: LineSegment2D
    rms: float
    width: float
    aspect_ratio: float

class LineGeometry2D(ABC):
    """Abstract base class for transformable line geometries."""

    _TRANSFORM_BASELINE: float = 10.0

    @classmethod
    @abstractmethod
    def from_points(cls, p0: geom.Point2D, p1: geom.Point2D) -> Self:
        """Construct line geometry from two defining points."""
        ...

    @abstractmethod
    def as_line(self) -> Line2D:
        ...

    @abstractmethod
    def at(self, s: float) -> geom.Point2D:
        """Evaluate geometry at parameter s."""
        ...
    
    @abstractmethod
    def contains(self, point: geom.Point2D, atol: float = 1e-12) -> bool:
        """Return ``True`` if point lies on line geometry."""
    ...

    @abstractmethod
    def intersections_with_box_edges(
        self,
        box: geom.Box2I | geom.Box2D,
        atol: float = 1e-12,
    ) -> list[geom.Point2D]:
        """Return intersection points with a box boundary."""
        ...

    @abstractmethod
    def _defining_points(self) -> tuple[geom.Point2D, geom.Point2D]:
        """Return the two points defining the line geometry."""
        ...

    @abstractmethod
    def _interval_in_box(self, box: geom.Box2I | geom.Box2D) -> geom.IntervalD | None:
        """Return valid parameter interval in box."""
        ...

    def clipped_to(self, box: geom.Box2D | geom.Box2I) -> LineSegment2D | None:
        """Clip line geometry to a box."""
        interval = self._interval_in_box(box)

        if interval is None:
            return None

        return LineSegment2D(self.as_line(), interval=interval)

    def intersection(self, box: geom.Box2D | geom.Box2I) -> LineSegment2D | None:
        """Return intersection with a box.

        Parameters
        ----------
        box : `lsst.geom.Box2I` or `lsst.geom.Box2D`
            The box to intersect the line geometry with.
        """
        return self.clipped_to(box)

    def transformed(self, transform: Any, baseline: float = 10.0) -> Self:
        """Apply a geometric transformation.
        
        The supplied transform maps points expressed in the current coordinate
        system into points in the target coordinate system.

        Parameters
        ----------
        transform : `lsst.afw.geom.TransformPoint2ToPoint2`
            Transform that maps points from the current coordinate system into
            the target coordinate system.
        baseline : `float`, optional
            The distance between the two points to be transformed (10.0, by
            default).

        Returns
        -------
        transformed : `Line2D`
            A new line in the target coordinate system.    
        """
        p0, p1 = self.defining_points()

        p0_t = _apply_transform(transform, p0)
        p1_t = _apply_transform(transform, p1)

        return type(self).from_points(p0_t, p1_t)

    def rotated(self, angle: geom.Angle) -> Self:
        """Apply a rotational transformation.

        Parameters
        ----------
        angle : `lsst.geom.Angle`
            The angle to rotate by.

        Returns
        -------
        transformed :
            The transformed object.
        """
        return self.transformed(geom.AffineTransform.makeRotation(angle))

    def scaled(self, factor: float) -> Self:
        """Apply a scaling transformation.
        
        Parameters
        ----------
        factor : `float`
            The factor to scale by.

        Returns
        -------
        transformed :
            The transformed object.
        """
        return self.transformed(geom.AffineTransform.makeScaling(factor))

    def translated(self, offset: geom.Extent2D) -> Self:
        """Apply a translation transformation.
        
        Parameters
        ----------
        offset : `lsst.geom.Extent2D`
            The offset to translate by.

        Returns
        -------
        transformed :
            The transformed object.
        """
        return self.transformed(geom.AffineTransform.makeTranslation(translation))



class Line2D(LineGeometry2D):

    def __init__(self, rho: float, theta: geom.Angle):
        rho, theta = _canonicalize(rho, theta)
        self._rho = rho
        self._theta = theta

    @classmethod
    def from_points(cls, p0: geom.Point2D, p1: geom.Point2D) -> Self:
        """Create a `Line2D` instance from two points.

        Parameters
        ----------
        p0 : `lsst.geom.Point2D`
            A point on the line.
        p1 : `lsst.geom.Point2D`
            A second point on the line.

        Returns
        -------
        line : `Line2D`
            An instance of `Line2D` defined by the two points.
        """
        return cls.from_point_and_direction(p0, p1 - p0)

    @classmethod
    def from_point_and_direction(cls, point: geom.Point2D, direction: geom.Extent2D) -> Self:
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

        direction = direction / norm
        normal = geom.Extent2D(-direction.y, direction.x)

        rho = _dot(geom.Extent2D(point), normal)
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
    def origin(self) -> geom.Point2D:
        """The point on the line closest to the origin."""
        return geom.Point2D(self.normal * self.rho)

    def as_line(self) -> Line2D:
        return self

    def at(self, s: float) -> geom.Point2D:
        return self.origin + self.direction * s

    def contains(self, point: geom.Point2D, atol: float = 1e-12) -> bool:
        """Return ``True`` if point lies on the line."""
        return abs(self.distance(point)) <= atol

    def intersections_with_box_edges(
        self,
        box: geom.Box2I | geom.Box2D,
        atol: float = 1e-12,
    ) ->list[geom.Point2D]:
        """Return intersection points with box boundary."""
        o = self.origin
        d = self.direction

        xmin = box.minX
        xmax = box.maxX
        ymin = box.minY
        ymax = box.maxY

        points: list[geom.Point2D] = []
        if abs(d.x) > atol:
            for x in (xmin, xmax):
                p = self.at((x - o.x) / d.x)
                if ymin - atol <= p.y <= ymax + atol:
                    points.append(p)

        if abs(d.y) > atol:
            for y in (ymin, ymax):
                p = self.at((y - o.y) / d.y)
                if xmin - atol <= p.x <= xmax + atol:
                    points.append(p)

        unique: list[geom.Point2D] = []
        for p in points:
            for q in unique:
                if p.distanceSquared(q) <= atol * atol:
                    break

        else:
            unique.append(p)

        return unique

    def project(self, point: geom.Point2D) -> float:
        """Project a point onto the line coordinate system.

        Parameters
        ----------
        point : `lsst.geom.Point2D`
            Point to project.

        Returns
        -------
        s : `float`
            Signed longitudinal coordinate along the line direction relative
            to the point closest to the origin.
        """
        delta = point - self.origin
        return _dot(delta, self.direction)

    def signed_distance(self, point: geom.Point2D) -> float:
        """Compute the signed perpendicular distance to the line."""
        delta = point - self.origin
        return _dot(delta, self.normal)

    def _defining_points(self) -> tuple[geom.Point2D, geom.Point2D]:
        """Return two points defining the line."""
        return self.origin, self.at(self._TRANSFORM_BASELINE)

    def _interval_in_box(self, box: geom.Box2I | geom.Box2D) -> geom.IntervalD | None:
        return _line_box_interval(self, box)


class LineSegment2D(LineGeometry2D):
    """A line segment geometric primitive."""

    def __init__(self, line: Line2D, interval: geom.IntervalD):
        self._line = line
        self._interval = interval

    @classmethod    
    def from_center_length(cls, line: Line2D, u_center: float, length: float) -> Self:
        h = 0.5 * length
        interval = geom.IntervalD.fromSpannedPoints((u_center - h, u_center + h))
        return cls(line, interval)

    @classmethod
    def from_points(cls, p0: geom.Point2D, p1: geom.Point2D) -> Self:
        line = Line2D.from_points(p0, p1)
        interval = geom.IntervalD.fromSpannedPoints([line.project(p0), line.project(p1)])

        return cls(line, interval)

    @property
    def line(self) -> Line2D:
        return self._line

    @property
    def interval(self) -> geom.IntervalD:
        return self._interval

    @property
    def length(self) -> float:
        return self.interval.size

    @property
    def u_center(self) -> float:
        return self.interval.center

    @property
    def u_max(self) -> float:
        return self.interval.max

    @property
    def u_min(self) -> float:
        return self.interval.min

    @property
    def center(self) -> float:
        return self.at(self.u_center)

    @property
    def p0(self) -> geom.Point2D:
        return self.at(self.u_min)

    @property
    def p1(self) -> geom.Point2D:
        return self.at(self.u_max)

    def as_line(self) -> Line2D:
        return self.line

    def at(self, s: float) -> geom.Point2D:
        return self.line.at(s)

    def contains(self, point: geom.Point2D, atol: float = 1e-12) -> bool:
        """Return ``True`` if point lies on segment."""
        if not self.line.contains(point, atol=atol):
            return False

        s = self.line.project(point)
        return (self.interval.min - atol) <= s <= (self.interval.max + atol)

    def intersections_with_box_edges(
        self,
        box: geom.Box2I | geom.Box2D,
        atol: float = 1e-12,
    ) -> list[geom.Point2D]:

        return [
            p for p in self.line.intersections_with_box_edges(box, atol=atol) if self.contains(p, atol=atol)
        ]

    def _defining_points(self) -> tuple[geom.Point2D, geom.point2D]:
        """Return segment endpoints."""
        return self.p0, self.p1

    def _interval_in_box(self, box: geom.Box2I | geom.Box2D) -> geom.IntervalD | None:
        box_interval = _line_box_interval(self.line, box)

        if box_interval is None:
            return None

        smin = max(self.interval.min, box_interval.min)
        smax = min(self.interval.max, box_interval.max)

        if smin > smax:
            return None

        return geom.IntervalD.fromSpannedPoints([smin, smax])


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


def fit_line_segment_from_xy(x: ArrayLike, y: ArrayLike, weights: ArrayLike | None = None) -> LineSegment2D:
    """Fit a weighted line segment to 2D points.

    Parameters
    ----------
    x, y : array-like
        Point coordinates.
    weights : array-like, optional
        Non-negative point weights.

    Returns
    -------
    segment : `LineSegment2D`
        Best-fit finite line segment.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if weights is None:
        weights = np.ones_like(x, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    if weights.shape != x.shape:
        raise ValueError("weights must have same shape as x/y")

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (weights > 0)
    if np.count_nonzero(valid) < 2:
        raise ValueError("need at least two valid weighted points")

    x = x[valid]
    y = y[valid]
    w = weights[valid]

    wsum = np.sum(w)
    centroid = np.array([np.sum(w * x) / wsum, np.sum(w * y) / wsum], dtype=np.float64)

    points = np.column_stack((x, y))
    centered = points - centroid
    weighted = centered * np.sqrt(w[:, np.newaxis])

    _, _, vh = np.linalg.svd(weighted)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)

    rho = float(np.dot(centroid, normal))
    theta = math.atan2(normal[1], normal[0])
    line = Line2D(rho, theta * geom.radians)

    distances = np.array([line.signed_distance(geom.Point2D(px, py)) for px, py in zip(x, y, strict=True)])
    projections = [line.project(geom.Point2D(px, py)) for px, py in zip(x, y, strict=True)]
    interval = geom.IntervalD.fromSpannedPoints(projections)

    line_segment = LineSegment2D(line=line, interval=interval)
    rms = np.sqrt(np.average(np.array(distances)**2, weights=w))
    width = 2.355 * rms
    aspect_ratio = line_segment.length / width

    return LineFitResult(
        line_segment=line_segment,
        rms=rms,
        width=width,
        aspect_ratio=aspect_ratio,
    )


def _apply_transform(transform: Any, point: geom.Point2D) -> geom.Point2D:
    """Apply a transform to a point.

    Apply a transformation to a point either by calling the applyForward 
    method of the object or calling the object directly.

    Parameters
    ----------
    transform:
        The object the encapsulates the transformation.
    point: `lsst.geom.Point2D`
        The point to transform.

    Returns
    -------
    transformed : `lsst.geom.Point2D`
        The transformed point.

    Raises
    ------
    TypeError
        Raised if ``transform`` does not have a valid callable interface.
    """
    if hasattr(transform, "applyForward"):
        return transform.applyForward(point)

    if callable(transform):
        return transform(point)

    raise TypeError("transform is invalid callable or object")


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
    canonical_rho : `float`
        Canonical signed perpendicular distance.
    canonical_theta : `lsst.geom.Angle`
        Canonical normal angle in the interval [0, pi).
    """
    theta_rad = theta.asRadians() % (2 * math.pi)

    if theta_rad >= math.pi:
        theta_rad -= math.pi
        rho = -rho
    
    return rho, theta_rad * geom.radians


def _dot(v0: geom.Extent2D, v1: geom.Extent2D) -> float:
    """Compute the Euclidean dot product of two vectors.
    
    Parameters
    ----------
    v0 : `lsst.geom.Extent2D`
        The first vector.
    v1 : `lsst.geom.Extent2D`
        The second vector.
        
    Returns
    -------
    dot_product : `float`
        The dot product of the two vectors.
    """
    return v0.x * v1.x + v0.y * v1.y


def _line_box_interval(line: Line2D, box: geom.Box2D | geom.Box2I) -> geom.IntervalD | None:
    """Compute the line parameter interval inside a box.

    Uses slab intersection in parametric form.
    """
    origin = line.origin
    direction = line.direction

    xmin = box.minX
    xmax = box.maxX
    ymin = box.minY
    ymax = box.maxY

    smin = -math.inf
    smax = math.inf

    def clip_axis(p0: float, dp: float, lo: float, hi: float) -> tuple[float, float] | None:
        if abs(dp) < 1e-15:
            if p0 < lo or p0 > hi:
                return None

            return -math.inf, math.inf

        s0 = (lo - p0) / dp
        s1 = (hi - p0) / dp

        return min(s0, s1), max(s0, s1)

    if (x_interval := clip_axis(origin.x, direction.x, xmin, xmax)) is None:
        return None

    if (y_interval := clip_axis(origin.y, direction.y, ymin, ymax)) is None:
        return None

    smin = max(smin, x_interval[0], y_interval[0])
    smax = min(smax, x_interval[1], y_interval[1])

    if smin > smax:
        return None

    return geom.IntervalD.fromSpannedPoints([smin, smax])
