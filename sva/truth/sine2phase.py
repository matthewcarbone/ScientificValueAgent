import numpy as np
from tqdm import tqdm

from scipy.interpolate import griddata

from sva.truth.common import sigmoid, limited_time_budget


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


def sine2phase_interpolant_2d(
    X,
    grid_points,
    phase_truth=phase_1_sine_on_2d_raster,
    interpolation_method="linear",
):
    """Returns a 2-dimensional, linear interpolant for the sine2phase example.

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
        X,
        known,
        (dense_x, dense_y),
        method=interpolation_method,
        fill_value=0.0,
    )
    return true.reshape(grid_points, grid_points), interpolated


def sine2phase_residual_2d_phase_mse(
    X,
    grid_points,
    phase_truth=phase_1_sine_on_2d_raster,
    interpolation_method="linear",
):
    true, interpolated = sine2phase_interpolant_2d(
        X, grid_points, phase_truth, interpolation_method
    )
    return np.mean((true - interpolated) ** 2)


def sine2phase_residual_2d_phase_relative_mae(
    X,
    grid_points,
    phase_truth=phase_1_sine_on_2d_raster,
    interpolation_method="linear",
):
    true, interpolated = sine2phase_interpolant_2d(
        X, grid_points, phase_truth, interpolation_method
    )
    d = np.sum(true, axis=-1, keepdims=True)
    return np.mean(np.abs(true - interpolated) / d)


def sine2phase_compute_metrics_all_acquisition_functions_and_LTB(
    results_by_acqf,
    metrics_grid=list(range(3, 251, 3)),
    metrics_grid_linear=[ii for ii in range(2, 16)],
    metric_function=sine2phase_residual_2d_phase_mse,
    metric_kwargs={
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
    metric_kwargs : dict, optional
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
                    metric_function(exp.data.X[:N], **metric_kwargs)
                    for exp in values
                ]
            )
        all_metrics[acquisition_function_name] = np.array(tmp_metrics)

    all_metrics["Linear"] = []
    for N_per_dim in tqdm(metrics_grid_linear, disable=disable_pbar):
        arr = limited_time_budget(N_per_dim, 2)
        res = sine2phase_residual_2d_phase_mse(arr, **metric_kwargs)
        all_metrics["Linear"].append(res)
    all_metrics["Linear"] = np.array(all_metrics["Linear"]).reshape(-1, 1)
    metrics_grid_linear = np.array(metrics_grid_linear) ** 2

    return {
        "metrics": all_metrics,
        "metrics_grid": metrics_grid,
        "metrics_grid_linear": metrics_grid_linear,
    }
