import numpy as np
from monty.json import MSONable

from sva.experiments.base import ExperimentMixin, ExperimentData
from ._gpax import (
    low_fidelity_sinusoidal,
    high_fidelity_sinusoidal,
    get_gpax_sinusoidal_dataset,
)


# Excellent test functions
# http://www.sfu.ca/~ssurjano/optimization.html


class SimpleSigmoidExperiment(ExperimentMixin, MSONable):
    """A simple 1d experimental response to a 1d input. This is a sigmoid
    function centered at 0, with a range (-0.5, 0.5). The sharpness of the
    sigmoid function is adjustable by setting the parameter a."""

    n_input_dim = 1
    n_output_dim = 1
    valid_domain = None
    experimental_domain = np.array([-2.0, 2.0]).reshape(2, 1)

    @property
    def noise(self):
        return self._noise

    def __init__(self, a=10.0, noise=None, data=ExperimentData()):
        self._a = a
        self._noise = noise
        self._data = data

    def _truth(self, x):
        return 2.0 / (1.0 + np.exp(-self._a * x)) - 1.0

    def _dtruth(self, x):
        e = np.exp(-self._a * x)
        d = 1.0 + e
        return 2.0 * self._a * e / d**2


class WavySinusoidalGPax(ExperimentMixin, MSONable):

    n_input_dim = 1
    n_output_dim = 2
    valid_domain = None
    experimental_domain = np.array([0.0, 100.0]).reshape(2, 1)

    @staticmethod
    def get_default_dataset():
        """Returns the dataset used in Maxim Ziatdinov's notebook which can
        be found in his code here: https://github.com/ziatdinovmax/gpax.

        Returns
        -------
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
            x_train, y_train, x_test_full_range, y_test_full_range
        """
        return get_gpax_sinusoidal_dataset()

    def __init__(self, data=ExperimentData()):
        self._data = data

    def _truth(self, x):
        low = low_fidelity_sinusoidal(x, noise=self._noise)
        high = high_fidelity_sinusoidal(x, noise=self._noise)
        return np.array([low, high]).T
