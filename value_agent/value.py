import numpy as np
from scipy.spatial import distance_matrix


def next_closest_raster_scan_point(
    proposed_points,
    observed_points,
    possible_coordinates,
    eps=1e-8
):
    """A helper function which determines the closest grid point for every
    proposed points, under the constraint that the proposed point is not
    present in the currently observed points, given possible coordinates.
    
    Parameters
    ----------
    proposed_points : array_like
        The proposed points. Should be of shape N x d, where d is the dimension
        of the space (e.g. 2-dimensional for a 2d raster). N is the number of
        proposed points (i.e. the batch size).
    observed_points : array_like
        Points that have been previously observed. N1 x d, where N1 is the
        number of previously observed points.
    possible_coordinates : array_like
        A grid of possible coordinates, options to choose from. N2 x d, where
        N2 is the number of coordinates on the grid.
    eps : float, optional
        The cutoff for determining that two points are the same, as computed
        by the L2 norm via scipy's ``distance_matrix``.

    Returns
    -------
    numpy.ndarray
        The new proposed points.
    """

    assert proposed_points.shape[1] == observed_points.shape[1]
    assert proposed_points.shape[1] == possible_coordinates.shape[1]

    D2 = distance_matrix(observed_points, possible_coordinates) > eps
    D2 = np.all(D2, axis=0)

    actual_points = []
    for possible_point in proposed_points:
        p = possible_point.reshape(1, -1)
        D = distance_matrix(p, possible_coordinates).squeeze()
        argsorted = np.argsort(D)
        for index in argsorted:
            if D2[index]:
                actual_points.append(possible_coordinates[index])
                break

    return np.array(actual_points)


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
