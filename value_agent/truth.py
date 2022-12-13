from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor


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


def sigmoid(x, x0, a):
    return 1.0 / (1.0 + np.exp(-a * (x - x0)))


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
    """Takes the y-distance between a sigmoid function and the provided point."""

    distance = y - _sine(x)
    return sigmoid(distance, x0, a)


def truth_sine2phase(X):
    phase_1 = [phase_1_sine_on_2d_raster(*c) for c in X]
    return np.array([mu_Gaussians(p) for p in phase_1])


def theta_phase(x, y, x0=1, a=10):
    # Radially symmetric about theta=0
    # x, y = scale_coords(x, y)
    angle = np.arctan2(y - 0.5, x - 0.5)
    return 1.0 - sigmoid(np.abs(angle), x0=x0, a=a)


def corner_phase(x, y, x0=0.25, a=20, loc_x=0.0, loc_y=1.0):
    # Distance from the top left corner
    # x, y = scale_coords(x, y)
    r = np.sqrt((loc_x - x) ** 2 + (loc_y - y) ** 2)
    return 1.0 - sigmoid(r, x0=x0, a=a)


def circle_phase(x, y, x0=0.5, a=20, loc_x=0.125, loc_y=0.125):
    # Distance from a point near the bottom right quadrant
    # x, y = scale_coords(x, y)
    r = np.sqrt((loc_x - x) ** 2 + (loc_y - y) ** 2)
    return 1.0 - sigmoid(r, x0=x0, a=a)


@cache
def _get_xrd_map():

    # Utterly horrendous practice here but for this it's fine...
    p = Path(__file__).absolute().parent / "xrd_map.xlsx"
    df = pd.read_excel(p, header=0, index_col=0)
    Y = df.to_numpy().T
    indexes = [0, 100, 150, 230]
    return Y[indexes, ::20]


def truth_4phase(X):
    pure_phases = _get_xrd_map()
    # Gets the actual "value" of the observation
    prop2 = theta_phase(X[:, 0], X[:, 1], x0=0.5, a=5.0)
    prop1 = corner_phase(X[:, 0], X[:, 1], x0=0.5, a=30.0)
    prop3 = circle_phase(X[:, 0], X[:, 1], x0=0.05, a=50.0)
    total_prop = prop1 + prop2 + prop3
    total_prop[total_prop > 1.0] = 1.0
    prop4 = 1.0 - total_prop
    return np.array([prop1, prop2, prop3, prop4]).T @ pure_phases


@cache
def _get_uv_model():

    p = Path(__file__).absolute().parent / "uv_data.csv"
    df = pd.read_csv(p)
    X = df[["NCit", "pH", "HA"]].to_numpy()
    X[:, 1] += 16.0
    Y = df.iloc[:, 4:].to_numpy()
    knn = KNeighborsRegressor(n_neighbors=2, weights="distance")
    knn.fit(X, Y)
    return knn


def truth_uv(x):
    model = _get_uv_model()
    return model.predict(x)


@cache
def _get_1d_phase_data():
    """Construct 1-D phase diagram akin to
    |---A---|---A  B (linear)---|---B---|---B+C (quadratic)---|---C---|---D---|
    Where the dataset

    Returns
    -------
    np.ndarray:
        (4, N) array of data describing the phases
    """
    return np.concatenate(
        (
            np.zeros((1, 100)),
            np.ones((1, 100)),
            np.ones((1, 100)) * 0.5,
            np.ones((1, 100)) * 3,
        ),
        axis=0,
    )  # TODO: load some real data


def _get_1d_phase_fractions(
    X: np.ndarray, b_start=10, a_stop=50, c_start=60, b_stop=80, c_stop=90
):
    """Construct the weights for a 1d phase diagram
    |---A---|---A+B (linear)---|---B---|---B+C (quadratic)---|---C---|---D---|

    Parameters
    ----------
    X : np.ndarray
        1-d array of points to gather fractions of
    b_start : int, optional
        Where the A+B phase begins, by default 10
    a_stop : int, optional
        Where pure B phase begins, by default 50
    c_start : int, optional
        Where the B+C phase begins, by default 60
    b_stop : int, optional
        Where the pure C phase begins, by default 80
    c_stop : int, optional
        Where C will abruptly transition to D., by default 90

    Returns
    -------
    _type_
        _description_
    """

    weights = np.zeros((4, X.shape[0]))
    weights[0, :] += X < b_start  # Pure A
    weights[1, :] += (a_stop < X) & (X < c_start)  # Pure B
    weights[2, :] += (b_stop < X) & (X < c_stop - 1)  # Pure C
    weights[3, :] += X > c_stop + 1  # Pure D

    # A+B Linear
    weights[0, :] += ((X < a_stop) & (X > b_start)) * (
        1 - (X - b_start) / (a_stop - b_start)
    )
    weights[1, :] += (
        ((X < a_stop) & (X > b_start)) * (X - b_start) / (a_stop - b_start)
    )

    # B+C Quadratic
    weights[1, :] += ((X > c_start) & (X < b_stop)) * (
        1 - ((X - c_start) / (b_stop - c_start)) ** 2
    )
    weights[2, :] += ((X > c_start) & (X < b_stop)) * (
        (X - c_start) / (b_stop - c_start)
    ) ** 2

    # C+D Sigmoidal
    weights[2, :] += ((X > c_stop - 1) & (X < c_stop + 1)) * (
        1 - sigmoid(X, c_stop, 100)
    )
    weights[3, :] += ((X > c_stop - 1) & (X < c_stop + 1)) * sigmoid(
        X, c_stop, 100
    )
    return weights


def truth_1d_phase(X):
    phases = _get_1d_phase_data()
    weights = _get_1d_phase_fractions(X)
    return (phases.T @ weights).T
