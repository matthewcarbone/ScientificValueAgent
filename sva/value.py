"""Module containing the value functions.

Value functions should always have the signature: value(X, Y, **kwargs)
"""

import numpy as np
from attrs import define, field
from scipy.spatial import distance_matrix

from sva.monty.json import MSONable


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

    v = Y_dist * np.exp(-(X_dist**2) / sd**2)  # Removed factor of 2

    return v.mean(axis=1)


def global_penalty(X):
    X_dist = distance_matrix(X, X)
    X_dist[X_dist == 0] = np.inf
    h_min = X_dist.min(axis=1).reshape(-1, 1)
    h_max = h_min.max()  # Maximum nearest neighbor distance

    # Trying exponential decay
    # return (1.0 - np.exp(-h_min / h_max)).squeeze()

    # Trying sidmoid
    return (1.0 / (1.0 + np.exp(-(h_min - h_max) / 1.0))).squeeze()


def svf_global_penalty(X, v_i):
    """Takes the result of the standard scientific value function and applies
    a global penalty on the points: each point x_i, constraining it such that
    the value will be low if that point it too close to neighboring points.
    The length scale of this penalty is set by looking at the largest
    distance in the dataset."""

    return global_penalty(X) * v_i


@define
class SVF(MSONable):
    sd = field(default=None)
    multiplier = field(default=1.0)

    def __call__(self, X, Y):
        return svf(X, Y, self.sd, self.multiplier)


@define
class SVFGlobalPenalty(SVF):
    def __call__(self, X, Y):
        v_i = svf(X, Y, self.sd, self.multiplier)
        return svf_global_penalty(X, v_i)
