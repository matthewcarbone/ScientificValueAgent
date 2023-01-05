from itertools import product
import warnings

import numpy as np


def sigmoid(x, x0, a):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = 1.0 / (1.0 + np.exp(-a * (x - x0)))
    return v


def limited_time_budget(N, dims):
    """Returns the grid of the limited time budget (LTB) "policy".

    Parameters
    ----------
    N : int
        The number of points to sample _per dimension_.
    dims : int
        The number of dimensions.

    Returns
    -------
    np.ndarray
    """

    deltax = 1.0 / (N - 1)
    arr = [deltax * ii for ii in range(N)]
    return np.array(list(product(*[arr for _ in range(dims)])))


def mu_Gaussians(p, E=np.linspace(-1, 1, 100), x0=0.5, sd=0.05):
    """Returns a dummy "spectrum" which is just two Gaussian functions. The
    proportion of the two functions is goverened by ``p``.

    Parameters
    ----------
    p : float
        The proportion of the first phase.
    E : numpy.ndarray
        Energy grid.

    Returns
    -------
    numpy.ndarray
        The spectrum on the provided grid.
    """

    p2 = 1.0 - p
    return p * np.exp(-((x0 - E) ** 2) / sd) + p2 * np.exp(
        -((x0 + E) ** 2) / sd
    )
