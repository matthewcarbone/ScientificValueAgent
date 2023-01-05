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
