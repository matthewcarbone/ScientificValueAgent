import warnings

import numpy as np
from attrs import field, validators, define
from monty.json import MSONable

from sva.experiments.base import (
    NOISE_TYPES,
    ExperimentData,
    ExperimentMixin,
    ExperimentProperties,
)


E_GRID = np.linspace(-1, 1, 100)
def mu_Gaussians(p, E=E_GRID, x0=0.5, sd=0.05):
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


@define
class Sine2Phase(ExperimentMixin, MSONable):
    properties = field(
        factory=lambda: ExperimentProperties(
            n_input_dim=2,
            n_output_dim=len(E_GRID),
            valid_domain=None,
            experimental_domain=np.array([[0.0, 1.0], [0.0, 1.0]]).T,
        )
    )
    noise = field(default=None, validator=validators.instance_of(NOISE_TYPES))
    data = field(factory=lambda: ExperimentData())
    x0 = field(default=0.5)
    a = field(default=100.0)

    def _truth(self, x):
        phase_1 = [phase_1_sine_on_2d_raster(*c, self.x0, self.a) for c in x]
        return np.array([mu_Gaussians(p) for p in phase_1])
