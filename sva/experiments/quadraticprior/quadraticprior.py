import gpytorch
import numpy as np
from attrs import define, field

from ..base import Experiment, ExperimentProperties


def mean(x1, x2):
    return -(x1**2) - x2**2


def func(x1, x2):
    return mean(x1, x2) + 0.5 * np.sin(x1 * 8) * np.cos(x1 + x2 * 3)


class QuadraticPriorMean(gpytorch.means.Mean):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = -(x[..., 0] ** 2) - x[..., 1] ** 2
        return y


@define
class QuadraticPrior(Experiment):
    properties = field(
        factory=lambda: ExperimentProperties(
            n_input_dim=2,
            n_output_dim=1,
            domain=np.array([[-1.0, 1.0], [-1.0, 1.0]]).T,
        )
    )

    true_optima = np.array([0.183931, -0.0423739])

    def _truth(self, X):
        x = X[:, 0]
        y = X[:, 1]
        return func(x, y).reshape(-1, 1)
