"""Module containing the value functions.

Value functions should always have the signature: value(X, Y, **kwargs)
"""

import numpy as np
from scipy.spatial import distance_matrix


def default_asymmetric_value_function(
    X, Y, sd=None, multiplier=1.0, characteristic_length="min", density=False
):
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
    characteristic_length : {"min", "max", "mean", "median"}
        The operation to perform on the input data in order to get the
        characteristic length. Default is min.
    density : bool
        If True, normalizes each point by the local density of nearby points.

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
        if characteristic_length == "min":
            sd = distance.min(axis=1).reshape(1, -1) * multiplier
        elif characteristic_length == "max":
            # Why would one do this? I don't know. Maybe a control experiment?
            sd = distance.max(axis=1).reshape(1, -1) * multiplier
        elif characteristic_length == "mean":
            sd = distance.mean(axis=1).reshape(1, -1) * multiplier
        elif characteristic_length == "median":
            sd = np.median(distance, axis=1).reshape(1, -1) * multiplier
        else:
            raise ValueError(
                f"Unknown characteristic length {characteristic_length}"
            )

    Y_dist = distance_matrix(Y, Y)

    w = np.exp(-X_dist / sd)
    v = Y_dist * w

    if not density:
        return v.mean(axis=1)

    # Otherwise, normalize
    return v.mean(axis=1) / w.mean(axis=1)


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

    # Length scale
    ls = sd * sd.T  # Symmetric piece (N, 1) * (1, N) = (N, N)

    Y_dist = distance_matrix(Y, Y)

    v = Y_dist * np.exp(-(X_dist**2) / ls / 2.0)

    return v.mean(axis=1)
