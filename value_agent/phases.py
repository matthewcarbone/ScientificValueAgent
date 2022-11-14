from pathlib import Path

import numpy as np
import pandas as pd


def get_2d_grids(
    x1_grid=np.linspace(0, 1, 51),
    x2_grid=np.linspace(0, 1, 51),
    n_downsample=100,
    seed=123
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
    full_input_coords = np.array([
        [_x1, _x2] for _x1 in x1_grid for _x2 in x2_grid
    ])
    input_coords = full_input_coords[random_sample, :]
    return {
        "x1": x1_grid,
        "x2": x2_grid,
        "input_coords": input_coords,
        "full_input_coords": full_input_coords,
        "seed": seed
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
    return p * np.exp(-(x0 - E)**2 / sd) + p2 * np.exp(-(x0 + E)**2 / sd)


def phase_1_sine_on_2d_raster(x, y, x0=0.5, a=30.0):
    """Takes the y-distance between a sigmoid function and the provided point.
    """
    
    distance = y - _sine(x)
    return sigmoid(distance, x0, a)


def sine_on_2d_raster_observations(X):
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
    r = np.sqrt((loc_x - x)**2 + (loc_y - y)**2)
    return 1.0 - sigmoid(r, x0=x0, a=a)


def circle_phase(x, y, x0=0.5, a=20, loc_x=0.125, loc_y=0.125):
    # Distance from a point near the bottom right quadrant
    # x, y = scale_coords(x, y)
    r = np.sqrt((loc_x - x)**2 + (loc_y - y)**2)
    return 1.0 - sigmoid(r, x0=x0, a=a)


# Utterly horrendous practice here but for this it's fine...
p = Path(__file__).absolute().parent / "xrd_map.xlsx"
df = pd.read_excel(p, header=0, index_col=0)
Y = df.to_numpy().T
indexes = [0, 100, 150, 230]
pure_phases = Y[indexes, ::20]


def truth_4phase(X):
    # Gets the actual "value" of the observation
    prop2 = theta_phase(X[:, 0], X[:, 1], x0=0.5, a=5.0)
    prop1 = corner_phase(X[:, 0], X[:, 1], x0=0.5, a=30.0)
    prop3 = circle_phase(X[:, 0], X[:, 1], x0=0.05, a=50.0)
    total_prop = prop1 + prop2 + prop3
    total_prop[total_prop > 1.0] = 1.0
    prop4 = 1.0 - total_prop
    return np.array([prop1, prop2, prop3, prop4]).T @ pure_phases