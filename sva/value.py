"""Module containing the value functions.

Value functions should always have the signature: value(X, Y, **kwargs)
"""

import numpy as np
from attrs import define, field
from scipy.spatial import distance_matrix

from sva.monty.json import MSONable


def sigmoid(d, x0, a):
    return 1.0 / (1.0 + np.exp(-(d - x0) / a))


def svf(X, Y, sd=None, multiplier=1.0, proximity_penalty=None):
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
    proximity_penalty : float, optional

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

    v = Y_dist * np.exp(-X_dist / sd)  # Removed factor of 2

    if proximity_penalty is not None:
        l_max = distance.min(axis=1).max()
        v = v * sigmoid(X_dist, l_max / proximity_penalty, 0.05)

    return v.mean(axis=1)


@define
class SVF(MSONable):
    sd = field(default=None)
    multiplier = field(default=1.0)
    proximity_penalty = field(default=None)

    def __call__(self, X, Y):
        return svf(X, Y, self.sd, self.multiplier, self.proximity_penalty)
