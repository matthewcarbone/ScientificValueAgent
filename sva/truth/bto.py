from functools import cache
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d


@cache
def _load_bto_data():
    """Loads BTO data into an xarray.
    These are diffraction patterns over ~0 to 25 AA.
    The termperature range is 150 to 445.
    """

    data_path = Path(__file__).parent / "bto_data.nc"
    weights_path = Path(__file__).parent / "bto_weights.nc"
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


@cache
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


def cmf_predicted_mse(X, linspace_points=5_000):
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
    known_weights = _get_cmf_predicted_phase_fractions(X)
    linspace = np.linspace(0.0, 100.0, linspace_points)
    true_weights = _get_cmf_predicted_phase_fractions_linspace(linspace)
    f = interp1d(
        X,
        known_weights,
        bounds_error=False,
        fill_value=(known_weights[:, 0], known_weights[:, -1]),
    )
    interpolated_weights = f(linspace)
    return np.nanmean((true_weights - interpolated_weights) ** 2)
