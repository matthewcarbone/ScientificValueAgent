import numpy as np

from sva.truth.common import sigmoid


def get_2d_grids(
    x1_grid=np.linspace(0, 1, 51),
    x2_grid=np.linspace(0, 1, 51),
    n_downsample=100,
    seed=123,
):
    """Gets some two-dimensional grids for testing and other work.

    Parameters
    ----------
    x1_grid : numpy.ndarray, optional
    x2_grid : numpy.ndarray, optional
    n_downsample : int, optional
        The number of points to downsample to from the full grid.
    seed : int, optional
        The random seed for the downsampling.

    Returns
    -------
    dict
    """

    np.random.seed(seed)
    N = len(x1_grid) * len(x2_grid)
    random_sample = np.sort(np.random.choice(range(N), 100, replace=False))
    full_input_coords = np.array(
        [[_x1, _x2] for _x1 in x1_grid for _x2 in x2_grid]
    )
    input_coords = full_input_coords[random_sample, :]
    return {
        "x1": x1_grid,
        "x2": x2_grid,
        "input_coords": input_coords,
        "full_input_coords": full_input_coords,
        "seed": seed,
    }


def get_phase_plot_info(truth, **kwargs):
    grids = get_2d_grids()
    x1_grid = grids["x1"]
    x2_grid = grids["x2"]
    X, Y = np.meshgrid(x1_grid, x2_grid)
    Z = truth(X, Y, **kwargs)
    return x1_grid, x2_grid, Z


def _sine(x):
    return 0.25 * np.sin(2.0 * np.pi * x)


def mu_Gaussians(p, E=np.linspace(-1, 1, 100), x0=0.5, sd=0.05):
    """Returns a dummy "spectrum" which is just two Gaussian functions. The
    proportion of the two functions is goverened by ``p``.

    Parameters
    ----------
    p : float
        The proportion of the first phase.
    E : numpy.ndarray
        Energy grid.

    Returns
    -------
    numpy.ndarray
        The spectrum on the provided grid.
    """

    p2 = 1.0 - p
    return p * np.exp(-((x0 - E) ** 2) / sd) + p2 * np.exp(
        -((x0 + E) ** 2) / sd
    )


def phase_1_sine_on_2d_raster(x, y, x0=0.5, a=30.0):
    """Takes the y-distance between a sigmoid function and the provided
    point."""

    distance = y - _sine(x)
    return sigmoid(distance, x0, a)


def truth_sine2phase(X):
    phase_1 = [phase_1_sine_on_2d_raster(*c) for c in X]
    return np.array([mu_Gaussians(p) for p in phase_1])


def points_in_10_percent_range(X):
    """The proportion of points contained in the 10% area specified by two
    sine functions at 0.5 + 0.25 sin(2 pi x) +/- 0.05.

    Parameters
    ----------
    X : numpy.ndarray

    Returns
    -------
    float
    """

    total_points = X.shape[0]
    x = X[:, 0]
    y = X[:, 1]
    y_upper = 0.55 + 0.25 * np.sin(2.0 * np.pi * x)
    y_lower = 0.45 + 0.25 * np.sin(2.0 * np.pi * x)
    where = np.where((y < y_upper) & (y > y_lower))[0]
    L = len(where)
    return L / total_points
