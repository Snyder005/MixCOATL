from __future__ import annotations

import math
import numpy as np
from numpy.typing import ArrayLike, NDArray

import lsst.geom as geom


class Line2D:

    def __init__(self, rho: float, theta: lsst.geom.Angle):
        rho, theta = self._canonicalize(rho, theta)
        self._rho = rho
        self._theta = theta

    @classmethod
    def from_points(cls, p0: lsst.geom.Point2D, p1: lsst.geom.Point2D) -> Line2D:
        return cls.from_point_and_direction(p0, p1 - p0)

    @classmethod
    def from_point_and_direction(cls, point: lsst.geom.Point2D, direction: lsst.geom.Extent2D) -> Line2D:
        norm = direction.computeNorm()
        if norm == 0:
            raise ValueError("direction vector must be non-zero")

        normal = geom.Extent2D(-direction.getY(), direction.getX()) / norm
        rho = point.getX() * normal.getX() + point.getY() * normal.getY()
        theta = math.atan2(normal.getY(), normal.getX()) * geom.radians

        return cls(rho, theta)

    @property
    def rho(self) -> float:
        return self._rho

    @property
    def theta(self) -> geom.Angle:
        return self._theta

    @property
    def normal(self) -> geom.Extent2D:
        t = self.theta.asRadians()
        return geom.Extent2D(math.cos(t), math.sin(t))

    @property
    def direction(self) -> geom.Extent2D:
        t = self.theta.asRadians()
        return geom.Extent2D(-math.sin(t), math.cos(t))

    def scale(self) -> None:
        """Scale rho parameter (change of units or binning)."""
        ...

    def shift(self) -> None:
        """Translate origin and update rho."""
        ...

    @staticmethod
    def _canonicalize(rho: float, theta: lsst.geom.Angle) -> tuple[float, lsst.geom.Angle]:
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

def fit_line_from_xy(x: ArrayLike, y: ArrayLike) -> tuple[float, float]:
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

    return rho, theta

def standardize_rho_theta(
        rho: ArrayLike, 
        theta: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Standardizes arrays of rho/theta.

    May not be needed if tables are constructed from Line2D objects.
    """
    theta_arr = np.asarray(theta, dtype=np.float64)
    rho_arr = np.asarray(rho, dtype=np.float64)
    if rho_arr.shape != theta_arr.shape:
        raise ValueError("rho and theta must have the same shape")

    theta_arr = theta_arr % (2 * np.pi)

    mask = theta_arr >= np.pi
    theta_arr = np.where(mask, theta_arr - np.pi, theta_arr)
    rho_arr = np.where(mask, -rho_arr, rho_arr)

    return rho_arr, theta_arr
