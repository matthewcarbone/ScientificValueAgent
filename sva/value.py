"""Module containing the value functions.

Value functions should always have the signature: value(X, Y, **kwargs)
"""

from abc import ABC, abstractmethod

import numpy as np
from monty.json import MSONable
from scipy.spatial import distance_matrix


class BaseValue(ABC):
    @abstractmethod
    def __call__(self, X, Y):
        ...


def svf(X, Y, sd=None, multiplier=1.0):
    """The value of two datasets, X and Y. Both X and Y must have the same
    number of rows. The returned result is a value of value for each of the
    data points.

    Parameters
    ----------
    X : numpy.ndarray
        The input data of shape N x d.
    Y : numpy.ndarray
        The output data of shape N x d'. Note that d and d' can be different
        and they also do not have to be 1.
    sd : float, optional
        Controls the length scale decay.
    multiplier : float, optional
        Multiplies the automatically derived length scale if ``sd`` is
        ``None``.

    Returns
    -------
    array_like
        The value for each data point.
    """

    X_dist = distance_matrix(X, X)

    # If sd is None, we automatically determine the length scale from the data
    if sd is None:
        distance = X_dist.copy()
        distance[distance == 0.0] = np.inf
        sd = distance.min(axis=1).reshape(1, -1) * multiplier

    Y_dist = distance_matrix(Y, Y)

    v = Y_dist * np.exp(-(X_dist**2) / sd**2 / 2.0)

    return v.mean(axis=1)


class SVF(BaseValue, MSONable):
    def __init__(self, sd=None, multiplier=1.0):
        self._sd = sd
        self._multiplier = multiplier

    def __call__(self, X, Y):
        return svf(X, Y, self._sd, self._multiplier)
