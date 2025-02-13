import datetime
import hashlib
import json
import random
from itertools import product
from time import perf_counter

import numpy as np
import torch
from scipy.spatial import distance_matrix
from scipy.stats import qmc

GLOBAL_STATE = {"seed": None}


class Timer:
    def __enter__(self):
        self._time = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self._time = perf_counter() - self._time

    @property
    def dt(self):
        return self._time


def seed_everything(seed):
    """Manually seeds everything needed for reproducibility using torch,
    numpy and the Python standard library."""

    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # The scipy seed somehow needs to be seeded separately
        GLOBAL_STATE["seed"] = seed


def random_indexes(array_size, samples=10):
    """Executes a random downsampling of the provided array along the first
    axis, without replacement.

    Parameters
    ----------
    array_size : int
        The total number of possible points to sample from
    samples : int, optional

    Returns
    -------
    np.ndarray
    """

    return np.random.choice(range(array_size), size=samples, replace=False)


def scale_by_domain(x, domain):
    """Scales provided data x by the bounds provided in the domain. Note that
    all dimensions must perfectly match.

    Parameters
    ----------
    x : np.ndarray
        The input data of shape (N, d). This data should be scaled between 0
        and 1.
    domain : np.ndarray
        The domain to scale to. Should be of shape (2, d), where domain[0, :]
        is the minimum along each axis and domain[1, :] is the maximum.

    Returns
    -------
    np.ndarray
        The scaled data of shape (N, d).
    """

    if x.ndim != 2:
        raise ValueError("Dimension of x must be == 2")

    if domain.shape[1] != x.shape[1]:
        raise ValueError("Domain and x shapes mismatched")

    if domain.shape[0] != 2:
        raise ValueError("Domain shape not equal to 2")

    if x.min() < 0.0:
        raise ValueError("x.min() < 0 (should be >= 0)")

    if x.max() > 1.0:
        raise ValueError("x.max() > 0 (should be <= 0)")

    return (domain[1, :] - domain[0, :]) * x + domain[0, :]


def get_coordinates(points_per_dimension, domain):
    """Gets a grid of equally spaced points on each dimension.
    Returns these results in coordinate representation.

    Parameters
    ----------
    points_per_dimension : int or list
        The number of points per dimension. If int, assumed to be 1d.
    domain : np.ndarray
        A 2 x d array indicating the domain along each axis.

    Returns
    -------
    np.ndarray
        The available points for sampling.
    """

    if isinstance(points_per_dimension, int):
        points_per_dimension = [points_per_dimension] * domain.shape[1]
    gen = product(*[np.linspace(0.0, 1.0, nn) for nn in points_per_dimension])
    return scale_by_domain(np.array([xx for xx in gen]), domain)


def get_random_points(domain, n=1):
    """Gets a random selection of points on a provided domain. The dimension
    of the data is inferred from the shape of the domain.

    Parameters
    ----------
    domain : np.ndarray
        The domain to scale to. Should be of shape (2, d).
    n : int
        Total number of points.

    Returns
    -------
    np.ndarray
    """

    X = np.random.random(size=(n, domain.shape[1]))
    return scale_by_domain(X, domain)


def get_latin_hypercube_points(domain, n=5):
    """Gets a random selection of points in the provided domain using the
    Latin Hypercube sampling algorithm.

    Parameters
    ----------
    domain : np.ndarray
        The domain to scale to. Should be of shape (2, d).
    n : int
        Total number of points.

    Returns
    -------
    np.ndarray
    """

    sampler = qmc.LatinHypercube(d=domain.shape[1], optimization="random-cd")
    sample = sampler.random(n=n)
    return qmc.scale(sample, *domain)


def next_closest_raster_scan_point(
    proposed_points, observed_points, possible_coordinates, eps=1e-8
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
        The new proposed points. REturns None if no new points were found.
    """

    # TODO: replace this by raise ValueError statements
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

    if len(actual_points) == 0:
        return None

    return np.array(actual_points)


def save_json(d, path):
    with open(path, "w") as outfile:
        json.dump(d, outfile, indent=4, sort_keys=True)


def read_json(path):
    """
    Test

    test

    Parameters
    ----------
    path : os.PathLike
        Path to the json file to read.
    """
    with open(path, "r") as infile:
        dat = json.load(infile)
    return dat


def get_hash(s):
    h = hashlib.new("sha256")
    s = s.encode("utf8")
    h.update(s)
    return h.hexdigest()


def get_dt_now():
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")
