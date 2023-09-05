from itertools import product
from tqdm import tqdm
import warnings

import numpy as np
from scipy.interpolate import griddata


def get_2d_grids(
    x1_grid=np.linspace(0, 1, 101),
    x2_grid=np.linspace(0, 1, 101),
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = 1.0 / (1.0 + np.exp(-a * (x - x0)))
    return v


def limited_time_budget(N, dims):
    """Returns the grid of the limited time budget (LTB) "policy". Points are
    returned in the range [0, 1).

    Parameters
    ----------
    N : int
        The number of points to sample _per dimension_.
    dims : int
        The number of dimensions.

    Returns
    -------
    np.ndarray
    """

    deltax = 1.0 / (N - 1)
    arr = [deltax * ii for ii in range(N)]
    return np.array(list(product(*[arr for _ in range(dims)])))


def random_sampling(N, dims):
    """Returns a random sampling of the space. Points are returned in the range
    [0, 1).

    Parameters
    ----------
    N : int
        The number of points to sample _per dimension_.
    dims : int
        The number of dimensions.

    Returns
    -------
    np.ndarray
    """

    total_points = N**dims
    return np.random.random(size=(total_points, dims))


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


def _interpolant_2d(
    X,
    grid_points,
    phase_truth,
    interpolation_method="linear",
):
    """Returns a 2-dimensional, linear interpolant.

    Parameters
    ----------
    X : np.ndarray
        The points on the grid, of shape (n x d).
    grid_points : int
        The number of grid points to use for the linear interpolant.
    phase_truth : callable
        A function that takes as input meshgrids x and y and returns an array
        containing the phase proportions of phase 1.
    interpolation_method : str, optional
        The interpolation method to pass to ``griddata``. Recommendation is to
        use "linear", "nearest" and "cubic".

    No Longer Returned
    ------------------
    np.ndarray, np.ndarray
        The "true" (dense grid) and interpolated (sampled points) results.
    """

    g = np.linspace(0, 1, grid_points)
    dense_x, dense_y = np.meshgrid(g, g)
    space = np.vstack([dense_x.ravel(), dense_y.ravel()]).T
    true = phase_truth(space[:, 0], space[:, 1])
    known = phase_truth(X[:, 0], X[:, 1])
    interpolated = griddata(
        X, known, (dense_x, dense_y), method=interpolation_method
    )
    return true.reshape(grid_points, grid_points), interpolated


def _residual_2d_phase_mse(
    X,
    grid_points,
    phase_truth,
    interpolation_method="linear",
):
    true, interpolated = _interpolant_2d(
        X, grid_points, phase_truth, interpolation_method
    )
    assert np.sum(np.isnan(true)) == 0
    return np.nanmean((true - interpolated) ** 2)


def _residual_2d_phase_relative_mae(
    X,
    grid_points,
    phase_truth,
    interpolation_method="linear",
):
    true, interpolated = _interpolant_2d(
        X, grid_points, phase_truth, interpolation_method
    )
    assert np.sum(np.isnan(true)) == 0
    not_nans = ~np.isnan(interpolated)
    d = np.sum(true * not_nans)
    return np.nanmean(np.abs(true - interpolated) / d)


def _compute_metrics_all_acquisition_functions_and_LTB(
    *,
    results_by_acqf,
    metric_function,
    metrics_grid=list(range(3, 251, 3)),
    metrics_grid_linear=[ii for ii in range(2, 16)],
    metric_function_kwargs={
        "grid_points": 100,
        "interpolation_method": "linear",
    },
    disable_pbar=False,
):
    """Computes the metric provided across all data.

    Parameters
    ----------
    results_by_acqf : dict
        Result of ``sva.postprocessing.parse_results_by_acquisition_function``.
    metrics_grid : array_like, optional
        The n-grid for the metric.
    metrics_grid_linear : array_like, optional
        The n-grid for the LTB metric.
    metric_function : function, optional
    metric_function_kwargs : dict, optional
    disable_pbar : bool, optional

    Returns
    -------
    dict
    """

    all_metrics = dict()
    for acquisition_function_name, values in results_by_acqf.items():
        tmp_metrics = []
        for N in tqdm(metrics_grid, disable=disable_pbar):
            tmp_metrics.append(
                [
                    metric_function(exp.data.X[:N], **metric_function_kwargs)
                    for exp in values
                ]
            )
        all_metrics[acquisition_function_name] = np.array(tmp_metrics)

    all_metrics["Linear"] = []
    for N_per_dim in tqdm(metrics_grid_linear, disable=disable_pbar):
        arr = limited_time_budget(N_per_dim, 2)
        res = metric_function(arr, **metric_function_kwargs)
        all_metrics["Linear"].append(res)
    all_metrics["Linear"] = np.array(all_metrics["Linear"]).reshape(-1, 1)
    metrics_grid_linear = np.array(metrics_grid_linear) ** 2

    return {
        "metrics": all_metrics,
        "metrics_grid": metrics_grid,
        "metrics_grid_linear": metrics_grid_linear,
    }
