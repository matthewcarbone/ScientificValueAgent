import numpy as np

from sva.truth.common import (
    sigmoid,
    mu_Gaussians,
    _interpolant_2d,
    _residual_2d_phase_mse,
    _residual_2d_phase_relative_mae,
    _compute_metrics_all_acquisition_functions_and_LTB,
)


def phase_1_linear_on_2d_raster(x, y, x0=0.5, a=30.0):
    """Takes the y-distance between a sigmoid function and the provided
    point."""

    distance = y + 1.5 * (x - 0.5)
    return sigmoid(distance, x0, a)


def truth_linear2phase(X):
    phase_1 = [phase_1_linear_on_2d_raster(*c) for c in X]
    return np.array([mu_Gaussians(p) for p in phase_1])


def linear2phase_interpolant_2d(
    X,
    grid_points,
    interpolation_method="linear",
):
    return _interpolant_2d(
        X,
        grid_points,
        phase_truth=phase_1_linear_on_2d_raster,
        interpolation_method=interpolation_method,
    )


def linear2phase_residual_2d_phase_mse(
    X,
    grid_points,
    interpolation_method="linear",
):
    return _residual_2d_phase_mse(
        X,
        grid_points,
        phase_truth=phase_1_linear_on_2d_raster,
        interpolation_method=interpolation_method,
    )


def linear2phase_residual_2d_phase_relative_mae(
    X,
    grid_points,
    interpolation_method="linear",
):
    return _residual_2d_phase_relative_mae(
        X,
        grid_points,
        phase_truth=phase_1_linear_on_2d_raster,
        interpolation_method=interpolation_method,
    )


def linear2phase_compute_metrics_all_acquisition_functions_and_LTB(
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
        metric_function = linear2phase_residual_2d_phase_mse
    elif metric == "rmae":
        metric_function = linear2phase_residual_2d_phase_relative_mae
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
