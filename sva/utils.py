import json
import random
from importlib import import_module
from itertools import product
from os import environ
from time import perf_counter
from warnings import warn

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits import axes_grid1

    MPL_INSTALLED = True
except ImportError:
    mpl = None
    plt = None
    axes_grid1 = None
    MPL_INSTALLED = False
import numpy as np
import torch
from scipy.spatial import distance_matrix


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


def get_function_from_signature(signature):
    """Parases a function of the form module.submodule:function to import
    and get the actual function as defined.

    Parameters
    ----------
    signature : str

    Returns
    -------
    callable
    """

    module, function = signature.split(":")
    module = import_module(module)
    return getattr(module, function)


def set_mpl_defaults(labelsize=12, dpi=250):
    if not MPL_INSTALLED:
        warn("Matplotlib is not installed")
        return
    mpl.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams["font.family"] = "STIXGeneral"
    if environ.get("DISABLE_LATEX"):
        mpl.rcParams["text.usetex"] = False
    else:
        mpl.rcParams["text.usetex"] = True
    plt.rc("xtick", labelsize=labelsize)
    plt.rc("ytick", labelsize=labelsize)
    plt.rc("axes", labelsize=labelsize)
    mpl.rcParams["figure.dpi"] = dpi
    plt.rcParams["figure.figsize"] = (3, 2)


def set_mpl_grids(
    ax,
    minorticks=True,
    grid=False,
    bottom=True,
    left=True,
    right=True,
    top=True,
):
    if not MPL_INSTALLED:
        warn("Matplotlib is not installed")
        return
    if minorticks:
        ax.minorticks_on()

    ax.tick_params(
        which="both",
        direction="in",
        bottom=bottom,
        left=left,
        top=top,
        right=right,
    )

    if grid:
        ax.grid(which="minor", alpha=0.2, linestyle=":")
        ax.grid(which="major", alpha=0.5)


def legend_without_duplicate_labels(ax, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    unique = [
        (h, _label)
        for i, (h, _label) in enumerate(zip(handles, labels))
        if _label not in labels[:i]
    ]
    ax.legend(*zip(*unique), **kwargs)


def add_colorbar(
    im, aspect=10, pad_fraction=0.5, integral_ticks=None, **kwargs
):
    """Add a vertical color bar to an image plot."""

    if not MPL_INSTALLED:
        warn("Matplotlib is not installed")
        return

    # https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1.0 / aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    cbar = im.axes.figure.colorbar(im, cax=cax, **kwargs)
    if integral_ticks is not None:
        L = len(integral_ticks)
        cbar.set_ticks(
            [
                cbar.vmin
                + (cbar.vmax - cbar.vmin) / L * ii
                - (cbar.vmax - cbar.vmin) / L / 2.0
                for ii in range(1, L + 1)
            ]
        )
        cbar.set_ticklabels(integral_ticks)
    return cbar


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
