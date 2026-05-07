import math
import numpy as np


def fit_line(x_array, y_array):
    """Fit a line in Hesse normal form."""
    pts = np.column_stack([x_array, y_array])
    centroid = pt_array.mean(axis=0)
    X = pts - centroid

    _, _, Vh = np.linalg.svd(X)
    n = Vh[-1]
    
    rho = np.dot(centroid, n)
    theta = np.arctan2(n[1], n[0])
    return rho, theta

def regularize_line(rho, theta):
    """Regularize Hesse normal form parameters to a canonical domain."""
    theta = theta % (2 * math.pi)

    if theta >= math.pi:
        theta -= math.pi
        rho = -rho

    return rho, theta

def embed_line(rho, theta, rho_tol, theta_tol):
    """Embed a Hesse normal form parameterization in a Euclidean space.

    Parameters
    ----------

    """
    w_rho = 1.0 / (rho_tol ** 2)
    w_angle = 1.0 / (1.0 - math.cos(theta_tol))

    embedding = np.array(
        [np.sqrt(w_angle) * np.cos(theta), np.sqrt(w_angle) * np.sin(theta), np.sqrt(w_rho) * rho]
    )
    return embedding
