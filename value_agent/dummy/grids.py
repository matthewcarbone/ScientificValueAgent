import numpy as np


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
