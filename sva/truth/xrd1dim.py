from functools import cache

import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm

from sva.truth.common import sigmoid


def gaussian(x, mu, sig, a):
    return a * np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))


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
    phases = np.zeros((4, 1000))
    x = np.linspace(0, 9, 1000)
    phases[0, :] += gaussian(x, 2, 0.03, 3) + gaussian(x, 6, 0.04, 1)
    phases[1, :] += (
        gaussian(x, 3, 0.05, 2)
        + gaussian(x, 5, 0.05, 1)
        + gaussian(x, 6.5, 0.05, 0.7)
    )
    phases[2, :] += (
        gaussian(x, 1, 0.02, 5)
        + gaussian(x, 4, 0.05, 0.8)
        + gaussian(x, 5.8, 0.03, 1)
    )
    phases[3, :] += (
        gaussian(x, 1.5, 0.03, 4)
        + gaussian(x, 6, 0.02, 1)
        + gaussian(x, 7.5, 0.04, 0.8)
        + gaussian(x, 8.5, 0.04, 0.4)
    )

    return phases + np.random.normal(0, 0.1, phases.shape) ** 2


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
    np.array
        Array of phase weights for all X points. Shape is (4, X.shape[0])
    """

    if len(X.shape) == 2 and X.shape[1] > 1:
        raise ValueError(f"X shape {X.shape} should be 1d, or (n, 1).")

    X = np.array(X)  # Type enforcement
    X = X.squeeze()

    weights = np.zeros((4, X.shape[0]))
    weights[0, :] += X <= b_start  # Pure A
    weights[1, :] += (a_stop <= X) & (X <= c_start)  # Pure B
    weights[2, :] += (b_stop <= X) & (X <= c_stop - 1)  # Pure C
    weights[3, :] += X >= c_stop + 1  # Pure D

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


def truth_xrd1dim(X):
    phases = _get_1d_phase_data()
    weights = _get_1d_phase_fractions(X)
    return (phases.T @ weights).T


def _residual_1d_phase_get_weights(X, linspace_points=100_000):
    X = np.unique(X.squeeze())
    known_weights = _get_1d_phase_fractions(X)
    linspace = np.linspace(0.0, 100.0, linspace_points)
    true_weights = _get_1d_phase_fractions(linspace)
    f = interp1d(
        X,
        known_weights,
        bounds_error=False,
        fill_value=(known_weights[:, 0], known_weights[:, -1]),
    )
    interpolated_weights = f(linspace)
    return true_weights, interpolated_weights


def residual_1d_phase_mse(X, linspace_points=10_000, use_only=None):
    """Get residuals of what is known from the sampled locations in
    comparison to the whole phase space. This makes an assumption that a good
    scientist could work out the phase fractions from the patterns provided
    and linearly interpolate those fractions to fill out all of phase space.
    In the high sample limit this mean squared error will tend toward zero.

    The highest drivers of this error will be poorly sampled regions of
    transition.

    Parameters
    ----------
    X : np.array
        1-d array of all data points queried by the agent.
    linspace_points : int
        The number of points to take on a linear grid to construct the
        "ground truth".

    Returns
    -------
    float
        Mean squared residual error from interpolating knoledge of space.
    """

    true_weights, interpolated_weights = _residual_1d_phase_get_weights(
        X, linspace_points
    )

    assert np.sum(np.isnan(true_weights)) == 0

    # tmp is of shape n_phases x linspace_points
    tmp = (true_weights - interpolated_weights) ** 2
    if use_only is not None:
        tmp = tmp[:, use_only]

    return np.nanmean(tmp)


def residual_1d_phase_relative_mae(X, linspace_points=10_000, use_only=None):
    """Similar to ``residual_1d_phase_mse`` but returns the relative mean
    absolute deviation relative to the ground truth (the ``true_weights``).
    This is a common metric in the crystallography community and known as
    Profile Residual (Rp).
     https://en.wikipedia.org/wiki/Rietveld_refinement#Figures_of_merit

    Parameters
    ----------
    X : np.array
        1-d array of all data points queried by the agent.
    linspace_points : int
        The number of points to take on a linear grid to construct the
        "ground truth".

    Returns
    -------
    float
        Relative mean absolute deviation from interpolating knoledge of space.
    """

    true_weights, interpolated_weights = _residual_1d_phase_get_weights(
        X, linspace_points
    )
    assert np.sum(np.isnan(true_weights)) == 0
    not_nans = ~np.isnan(interpolated_weights)
    d = np.sum(true_weights * not_nans, axis=-1, keepdims=True)

    # tmp is of shape n_phases x linspace_points
    tmp = np.abs(true_weights - interpolated_weights) / d
    if use_only is not None:
        tmp = tmp[:, use_only]

    return np.nanmean(tmp)


def xrd1dim_compute_metrics_all_acquisition_functions_and_LTB(
    results_by_acqf,
    metrics_grid=list(range(3, 251, 10)),
    metrics_grid_linear=list(range(3, 251, 10)),
    metric="rmae",
    grid_points=10000,
    disable_pbar=False,
    xmin=0.0,
    xmax=100.0,
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
        metric_function = residual_1d_phase_mse
    elif metric == "rmae":
        metric_function = residual_1d_phase_relative_mae
    else:
        raise ValueError(f"Unknown metric={metric}")

    G = np.linspace(0.0, 100.0, grid_points)
    if xmin == 0.0 and xmax == 100.0:
        use_only = None
    else:
        use_only = np.where((G <= xmax) & (G >= xmin))[0]

    all_metrics = dict()
    for acquisition_function_name, values in results_by_acqf.items():
        tmp_metrics = []
        for n in tqdm(metrics_grid, disable=disable_pbar):
            tmp_list_1 = []
            for exp in values:
                X_tmp = exp.data.X[:n].squeeze()
                try:
                    r = metric_function(
                        X_tmp,
                        linspace_points=grid_points,
                        use_only=use_only,
                    )
                except IndexError:
                    r = np.nan
                tmp_list_1.append(r)
            tmp_metrics.append(tmp_list_1)
        all_metrics[acquisition_function_name] = np.array(tmp_metrics)

    all_metrics["Linear"] = []
    for N in tqdm(metrics_grid_linear, disable=disable_pbar):
        res = metric_function(
            np.linspace(0, 100, N),
            linspace_points=grid_points,
            use_only=use_only,
        )
        all_metrics["Linear"].append(res)
    all_metrics["Linear"] = np.array(all_metrics["Linear"]).reshape(-1, 1)

    return {
        "metrics": all_metrics,
        "metrics_grid": metrics_grid,
        "metrics_grid_linear": metrics_grid_linear,
    }
