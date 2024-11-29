"""Module containing the value functions.

Value functions should always have the signature: value(X, Y, **kwargs)
"""

import numpy as np
from attrs import define, field
from monty.json import MSONable
from scipy.spatial import distance_matrix


def svf(
    X,
    Y,
    sd=None,
    multiplier=1.0,
    characteristic_length="min",
    density=False,
    symmetric=False,
    scale=True,
    square_exponent=False,
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
    symmetric : bool
        If True, uses the symmetric value function. Otherwise, uses an
        asymmetric representation. Default is False.
    scale : bool
        If True, scales the data to the range [-1, 1].
    square_exponent: bool
        If True, squares the argument of the exponential weight of the
        distances in the input space.

    Returns
    -------
    array_like
        The value for each data point.
    """

    X_dist = distance_matrix(X, X)

    if sd is None:
        # Automatic determination
        distance = X_dist.copy()
        if characteristic_length == "min":
            sd = (
                np.nanmin(
                    np.where(distance == 0, np.nan, distance), axis=1
                ).reshape(1, -1)
                * multiplier
            )
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

    if symmetric:
        ls = np.sqrt(sd * sd.T)  # Symmetric piece (N, 1) * (1, N) = (N, N)
    else:
        ls = sd

    if square_exponent:
        arg = -(X_dist**2) / ls**2
    else:
        arg = -X_dist / ls
    w = np.exp(arg)
    v = Y_dist * w

    if not density:
        v = v.mean(axis=1)
    else:
        v = v.mean(axis=1) / w.mean(axis=1)

    if scale:
        b = 1.0
        a = -1.0
        vmin = v.min()
        vmax = v.max()
        v_scaled = (b - a) * (v - vmin) / (vmax - vmin) + a
        return v_scaled
    return v


@define
class SVF(MSONable):
    # TODO: really silly do to it this way
    params = field(
        default={
            "sd": None,
            "multiplier": 1.0,
            "characteristic_length": "min",
            "density": False,
            "symmetric": False,
            "scale": True,
            "square_exponent": False,
        }
    )

    def __call__(self, X, Y):
        return svf(X, Y, **self.params)
