"""Module containing the value functions.

Value functions should always have the signature: value(X, Y, **kwargs)
"""

import numpy as np
from attrs import define, field
from monty.json import MSONable
from scipy.spatial import distance_matrix


def distance_matrix_pbc_vectorized(coords, box_side_lengths):
    """
    Vectorized calculation of pairwise distances under periodic boundary conditions.

    Parameters:
        coords (numpy.ndarray): NxD array of N points in D-dimensional space.
        box_length (float): Length of the sides of the simulation box.

    Returns:
        numpy.ndarray: NxN matrix of pairwise distances.
    """
    coords = np.array(coords)
    delta = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    box_side_lengths = box_side_lengths.reshape(1, 1, -1)
    assert box_side_lengths.shape[-1] == delta.shape[-1]
    delta = delta - np.rint(delta / box_side_lengths) * box_side_lengths
    dist_matrix = np.sqrt((delta**2).sum(axis=-1))
    return dist_matrix


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
    denominator_pbc=False,
    box_side_lengths=None,
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
    denominator_pbc: bool
        If True, calculates the denominator of the density approximation
        using periodic boundary conditions for the distances, where the
        distance between two points i and j along each dimension is taken
        to be the minimum of the distance d_ij and its closest image,
        d_ij'.
    box_side_lengths: array_like
        If denominator_pbc is True, this argument is required to calculate
        the periodic boundary conditions properly. Should be the length
        of each side of the box by coordinate.

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
        if denominator_pbc:
            X_dist = distance_matrix_pbc_vectorized(X, box_side_lengths)
            arg = -(X_dist**2) / ls**2 if square_exponent else -X_dist / ls
            denominator = np.exp(arg)
            denominator = denominator.mean(axis=1)
        else:
            denominator = w.mean(axis=1)
        v = v.mean(axis=1) / denominator

    if scale:
        b = 1.0
        a = 0.0
        vmin = v.min()
        vmax = v.max()
        v_scaled = (b - a) * (v - vmin) / (vmax - vmin) + a
        return v_scaled
    return v


@define
class SVF(MSONable):
    sd = field(default=None)
    multiplier = field(default=1.0)
    characteristic_length = field(default="min")
    density = field(default=False)
    symmetric = field(default=False)
    scale = field(default=True)
    square_exponent = field(default=False)
    denominator_pbc = field(default=False)
    box_side_lengths = field(default=None)

    def __call__(self, X, Y):
        return svf(
            X,
            Y,
            sd=self.sd,
            multiplier=self.multiplier,
            characteristic_length=self.characteristic_length,
            density=self.density,
            symmetric=self.symmetric,
            scale=self.scale,
            square_exponent=self.square_exponent,
            denominator_pbc=self.denominator_pbc,
            box_side_lengths=self.box_side_lengths,
        )
