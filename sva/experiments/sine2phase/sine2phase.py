import warnings

import numpy as np
from attrs import define, field

from ..base import Experiment, ExperimentProperties

E_GRID = np.linspace(-1, 1, 100)


def _mu_Gaussians(p, E=E_GRID, x0=0.5, sd=0.05):
    """Returns a dummy "spectrum" which is just two Gaussian functions. The
    proportion of the two functions is goverened by ``p``.

    Parameters
    ----------
    p : float
        The proportion of the first phase.
    E : numpy.ndarray
        Energy grid.
    x0 : float
        The absolute value of the location of each Gaussian.
    sd : float
        The standard deviation of the Gaussians.

    Returns
    -------
    numpy.ndarray
        The spectrum on the provided grid.
    """

    p2 = 1.0 - p
    sd = sd**2
    e = -((x0 + E) ** 2) / sd
    e2 = -((x0 - E) ** 2) / sd
    return p * np.exp(e) + p2 * np.exp(e2)


def _sine(x):
    return 0.25 * np.sin(2.0 * np.pi * x)


def _sigmoid(x, x0, a):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = 1.0 / (1.0 + np.exp(-a * (x - x0)))
    return v


def _phase_1_sine_on_2d_raster(x, y, x0=0.5, a=100.0):
    """Takes the y-distance between a _sigmoid function and the provided
    point."""

    distance = y - _sine(x)
    return _sigmoid(distance, x0, a)


def _get_phase_from_proportion(x, x0, a, gaussian_x0, gaussian_sd):
    phase_1 = [_phase_1_sine_on_2d_raster(*c, x0, a) for c in x]
    return np.array(
        [_mu_Gaussians(p, x0=gaussian_x0, sd=gaussian_sd) for p in phase_1]
    )


@define
class Sine2Phase(Experiment):
    properties = field(
        factory=lambda: ExperimentProperties(
            n_input_dim=2,
            n_output_dim=len(E_GRID),
            domain=np.array([[0.0, 1.0], [0.0, 1.0]]).T,
        )
    )
    x0 = field(default=0.5)
    a = field(default=100.0)
    gaussian_x0 = field(default=0.5)
    gaussian_sd = field(default=0.22)

    def get_phase(self, x):
        return np.array(
            [_phase_1_sine_on_2d_raster(*c, self.x0, self.a) for c in x]
        ).reshape(-1, 1)

    def _truth(self, x):
        return _get_phase_from_proportion(
            x, self.x0, self.a, self.gaussian_x0, self.gaussian_sd
        )
