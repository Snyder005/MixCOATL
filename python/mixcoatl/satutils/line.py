import numpy as np

def fit_hesse_line(x_array, y_array):
    """Fit a line in Hesse normal form."""
    xc = np.mean(x_array)
    yc = np.mean(y_array)

    X = np.vstack([x_array - xc, y_array - yc]).T
    u, s, vh = np.linalg.svd(X)
    n = geom.Extent2D(vh[-1])
    n /= n.computeNorm()

    rho = xc * n.getX() + yc * n.getY()
    theta = np.arctan2(n.getY(), n.getX())

    return rho, theta

def embed_hesse_line(rho, theta, rho_tol, theta_tol):
    """Embed a line in Hesse normal form in a scaled Euclidean space."""
    w_rho = 1.0 / (rho_tol ** 2)
    w_angle = 1.0 / (1.0 - np.cos(theta_tol))

    embedding = np.array(
        [np.sqrt(w_angle) * np.cos(theta), np.sqrt(w_angle) * np.sin(theta), np.sqrt(w_rho) * rho]
    )
    return embedding

def _canonical_hesse_form(rho, theta):
    """Convert Hesse normal form parameters to a canonical domain."""
    theta = theta % (2 * np.pi)

    if theta >= np.pi:
        theta -= np.pi
        rho = -rho

    return rho, theta
