from functools import cache
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d
from tqdm import tqdm


@cache
def _load_bto_data():
    """Loads BTO data into an xarray.
    These are diffraction patterns over ~0 to 25 AA.
    The termperature range is 150 to 445.
    """

    data_path = Path(__file__).parent / "bto_data.nc"
    weights_path = Path(__file__).parent / "bto_xca_weights.nc"
    return xr.open_dataarray(data_path), xr.open_dataarray(weights_path)


def truth_bto(temperature) -> np.ndarray:
    """
    Returns interpolated X-ray diffraction pattern for BTO at
    temperature between 150 and 445.

    Parameters
    ----------
    temperature : Union[float, list, np.ndarray]
        temperature points to estimate patterns for
    """
    da, _ = _load_bto_data()
    return da.interp(temperature=temperature.squeeze()).data.T


def _get_cmf_predicted_phase_fractions(X: np.ndarray):
    """Construct the predicted weights as a ground truth from the cmf data

    Parameters
    ----------
    X : np.ndarray
        1-d array of points to gather fractions of

    Returns
    np.ndarray
        Array of phase weights for all X points. Shape is (X.shape[0], 4)
    """
    _, weights = _load_bto_data()
    return weights.interp(temperature=X)


def _get_cmf_predicted_phase_fractions_linspace(X: np.ndarray):
    """Construct the predicted weights as a ground truth from the cmf data
    Exact copy of the above, but only caching for the linspace which is
    guarenteed to repeat.

    Parameters
    ----------
    X : np.ndarray
        1-d array of points to gather fractions of

    Returns
    np.ndarray
        Array of phase weights for all X points. Shape is (X.shape[0], 4)
    """

    _, weights = _load_bto_data()
    return weights.interp(temperature=X)


def cmf_predicted_mse(X, linspace_points=5_000, use_only=None):
    """Get the mean squared error between the set of queries and a linspace,
    treating the CMF data for weights are ground truth for phase fractions.

    Parameters
    ----------
    X : np.ndarray
        1-d array of all data points queried by the agent.
    linspace_points : int, optional
        The number of points to take on a linear grid to construct the
        "ground truth", by default 5_000
    """

    X = np.unique(X.squeeze())
    known_weights = _get_cmf_predicted_phase_fractions(X).T
    linspace = np.linspace(150.0, 445.0, linspace_points)
    true_weights = _get_cmf_predicted_phase_fractions_linspace(linspace).T
    f = interp1d(
        X,
        known_weights,
        bounds_error=False,
        fill_value=(known_weights[:, 0], known_weights[:, -1]),
    )
    interpolated_weights = f(linspace)

    tmp = (true_weights - interpolated_weights) ** 2
    if use_only is not None:
        tmp = tmp[:, use_only]

    return np.nanmean(tmp)


def bto_compute_metrics_all_acquisition_functions_and_LTB(
    results_by_acqf,
    metrics_grid=list(range(3, 251, 10)),
    metrics_grid_linear=list(range(3, 251, 10)),
    metric="mse",
    grid_points=10000,
    disable_pbar=False,
    xmin=150.0,
    xmax=445.0,
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
        metric_function = cmf_predicted_mse
    else:
        raise ValueError(f"Unknown metric={metric}")

    G = np.linspace(150.0, 445.0, grid_points)
    if xmin == 150.0 and xmax == 445.0:
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
            np.linspace(150.0, 445.0, N),
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
