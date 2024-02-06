import warnings

import numpy as np


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
    e = -((x0 + E) ** 2) / sd
    return p * np.exp(-((x0 - E) ** 2) / sd) + p2 * np.exp(e)


def _sine(x):
    return 0.25 * np.sin(2.0 * np.pi * x)


def sigmoid(x, x0, a):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = 1.0 / (1.0 + np.exp(-a * (x - x0)))
    return v


def phase_1_sine_on_2d_raster(x, y, x0=0.5, a=100.0):
    """Takes the y-distance between a sigmoid function and the provided
    point."""

    distance = y - _sine(x)
    return sigmoid(distance, x0, a)


def truth_sine2phase(X):
    phase_1 = [phase_1_sine_on_2d_raster(*c) for c in X]
    return np.array([mu_Gaussians(p) for p in phase_1])
