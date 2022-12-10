"""Module containing the value functions.

Value functions should always have the signature: value(X, Y, **kwargs)
"""

import numpy as np
from scipy.spatial import distance_matrix


def value_function(X, Y, sd=None, multiplier=1.0):
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

    if sd is None:
        # Automatic determination
        distance = X_dist.copy()
        distance[distance == 0.0] = np.inf
        sd = distance.min(axis=1).reshape(1, -1) * multiplier
    
    Y_dist = distance_matrix(Y, Y)

    v = Y_dist * np.exp(-X_dist**2 / sd**2 / 2.0)

    return v.mean(axis=1)


def symmetric_value_function(X, Y):
    """A similar value function to the asymmetric ``value_function``.
    
    Parameters
    ----------
    X : numpy.ndarray
    Y : numpy.ndarray

    Returns
    -------
    array_like
        The value for each data point. 
    """

    X_dist = distance_matrix(X, X)  # (N, N)

    distance = X_dist.copy()
    distance[distance == 0.0] = np.inf
    sd = distance.min(axis=1, keepdims=True)  # (N, 1)
    sd = sd * sd.T  # Symmetric piece (N, 1) * (1, N) = (N, N)

    Y_dist = distance_matrix(Y, Y)

    v = Y_dist * np.exp(-X_dist**2 / sd**2 / 2.0)

    return v.mean(axis=1)
