import warnings

import numpy as np
from attrs import define, field, validators
from PyAstronomy.pyasl import broadGaussFast

from sva.monty.json import MSONable

from ..base import (
    NOISE_TYPES,
    ExperimentData,
    ExperimentHistory,
    ExperimentMixin,
    ExperimentProperties,
    MultimodalExperimentMixin,
)
from ..campaign import CampaignBaseMixin, MultimodalCampaignMixin

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
class Sine2Phase(ExperimentMixin, CampaignBaseMixin, MSONable):
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
    history = field(factory=lambda: ExperimentHistory())
    x0 = field(default=0.5)
    a = field(default=100.0)
    gaussian_x0 = field(default=0.5)
    gaussian_sd = field(default=0.22)

    def _truth(self, x):
        return _get_phase_from_proportion(
            x, self.x0, self.a, self.gaussian_x0, self.gaussian_sd
        )


@define
class Sine2Phase2Resolutions(
    MultimodalExperimentMixin, MultimodalCampaignMixin, MSONable
):
    """A modification of the Sine2Phase experiment which produces a multimodal
    output. The low-resolution output is a broadened Gaussian with artificial
    Gaussian noise injected. The noise paramter controls how strong this noise
    is. The high-resolution output is the original signal, with no noise."""

    properties = field(
        factory=lambda: ExperimentProperties(
            n_input_dim=2,
            n_output_dim=len(E_GRID),
            valid_domain=None,
            experimental_domain=np.array([[0.0, 1.0], [0.0, 1.0]]).T,
        )
    )
    noise = None
    history = field(factory=lambda: ExperimentHistory())
    low_resolution_noise = field(default=0.05)
    data = field(factory=lambda: ExperimentData())
    x0 = field(default=0.5)
    a = field(default=100.0)
    gaussian_x0 = field(default=0.5)
    gaussian_sd = field(default=0.05)
    low_resolution_broadening = field(default=0.1)
    n_modalities = 2

    def _truth(self, x):
        if self.noise is not None:
            raise ValueError(
                "noise parameter should be unset here, "
                "set low_resolution_noise instead"
            )
        low_y = None
        low_ii = np.where(x[:, -1] == 0)[0]
        if len(low_ii) > 0:
            low_y = _get_phase_from_proportion(
                x[low_ii, :-1],
                self.x0,
                self.a,
                self.gaussian_x0,
                self.gaussian_sd,
            )
            for ii in range(len(low_y)):
                low_y[ii, :] = broadGaussFast(
                    E_GRID, low_y[ii, :], self.low_resolution_broadening
                )
            low_y += np.random.normal(
                scale=self.low_resolution_noise, size=low_y.shape
            )

        high_y = None
        high_ii = np.where(x[:, -1] == 1)[0]
        if len(high_ii) > 0:
            high_y = _get_phase_from_proportion(
                x[high_ii, :-1],
                self.x0,
                self.a,
                self.gaussian_x0,
                self.gaussian_sd,
            )

        # Have to be careful to order the output in the same way as the input
        to_return = np.empty((x.shape[0], self.properties.n_output_dim))
        if low_y is not None:
            to_return[low_ii, :] = low_y
        if high_y is not None:
            to_return[high_ii, :] = high_y
        return to_return
