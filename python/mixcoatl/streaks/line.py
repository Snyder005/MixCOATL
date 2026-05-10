from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

def fit_line_from_xy(x: ArrayLike, y: ArrayLike) -> Line:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have the same shape")

    pts_arr = np.column_stack((x_arr, y_arr))
    centroid = pts_arr.mean(axis=0)

    _, _, vh = np.linalg.svd(pts_arr - centroid)
    normal = vh[-1]

    rho = float(np.dot(centroid, normal))
    theta = math.atan2(normal[1], normal[0])

    return Line(rho=rho, theta=theta)

def get_standardized_line(rho: float, theta: float) -> Line:
    theta = theta % (2 * math.pi)

    if theta >= math.pi:
        theta -= math.pi
        rho = -rho

    return Line(rho=rho, theta=theta)

def standardize_rho_theta(
        rho: ArrayLike, 
        theta: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    theta_arr = np.asarray(theta, dtype=np.float64)
    rho_arr = np.asarray(rho, dtype=np.float64)
    if rho_arr.shape != theta_arr.shape:
        raise ValueError("rho and theta must have the same shape")

    theta_arr = theta_arr % (2 * np.pi)

    mask = theta_arr >= np.pi
    theta_arr = np.where(mask, theta_arr - np.pi, theta_arr)
    rho_arr = np.where(mask, -rho_arr, rho_arr)

    return rho_arr, theta_arr

@dataclass(slots=True, frozen=True)
class Line:
    rho: float
    theta: float
