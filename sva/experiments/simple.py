import numpy as np
from monty.json import MSONable

from sva.experiments.base import Experiment, ExperimentData


class SimpleSigmoidExperiment(Experiment, MSONable):
    """A simple 1d experimental response to a 1d input. This is a sigmoid
    function centered at 0, with a range (-0.5, 0.5). The sharpness of the
    sigmoid function is adjustable by setting the parameter a."""

    n_input_dim = 1
    n_output_dim = 1
    valid_domain = None
    experimental_domain = np.array([-2.0, 2.0]).reshape(2, 1)

    def __init__(self, a=10.0, data=ExperimentData()):
        self._a = a
        self._data = data

    def _truth(self, x):
        return 2.0 / (1.0 + np.exp(-self._a * x)) - 1.0

    def _dtruth(self, x):
        e = np.exp(-self._a * x)
        d = 1.0 + e
        return 2.0 * self._a * e / d**2
