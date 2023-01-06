import numpy as np

from sva.truth.common import (
    sigmoid,
    mu_Gaussians,
    _interpolant_2d,
    _residual_2d_phase_mse,
    _residual_2d_phase_relative_mae,
    _compute_metrics_all_acquisition_functions_and_LTB,
)


def _sine(x):
    return 0.25 * np.sin(2.0 * np.pi * x)


def phase_1_sine_on_2d_raster(x, y, x0=0.5, a=100.0):
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
    interpolation_method="linear",
):
    return _interpolant_2d(
        X,
        grid_points,
        phase_truth=phase_1_sine_on_2d_raster,
        interpolation_method=interpolation_method,
    )


def sine2phase_residual_2d_phase_mse(
    X,
    grid_points,
    interpolation_method="linear",
):
    return _residual_2d_phase_mse(
        X,
        grid_points,
        phase_truth=phase_1_sine_on_2d_raster,
        interpolation_method=interpolation_method,
    )


def sine2phase_residual_2d_phase_relative_mae(
    X,
    grid_points,
    interpolation_method="linear",
):
    return _residual_2d_phase_relative_mae(
        X,
        grid_points,
        phase_truth=phase_1_sine_on_2d_raster,
        interpolation_method=interpolation_method,
    )


def sine2phase_compute_metrics_all_acquisition_functions_and_LTB(
    results_by_acqf,
    metrics_grid=list(range(3, 251, 3)),
    metrics_grid_linear=[ii for ii in range(2, 16)],
    metric="mse",
    grid_points=100,
    interpolation_method="linear",
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
    metric : {"mse", "rmae"}, optional
    grid_points : int, optional
    interpolation_method : str, optional
    disable_pbar : bool, optional

    Returns
    -------
    dict
    """

    if metric == "mse":
        metric_function = sine2phase_residual_2d_phase_mse
    elif metric == "rmae":
        metric_function = sine2phase_residual_2d_phase_relative_mae
    else:
        raise ValueError(f"Unknown metric={metric}")

    return _compute_metrics_all_acquisition_functions_and_LTB(
        results_by_acqf=results_by_acqf,
        metric_function=metric_function,
        metrics_grid=metrics_grid,
        metrics_grid_linear=metrics_grid_linear,
        metric_function_kwargs={
            "grid_points": grid_points,
            "interpolation_method": interpolation_method,
        },
        disable_pbar=False,
    )
