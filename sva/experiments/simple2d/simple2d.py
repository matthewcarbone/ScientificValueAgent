import numpy as np
from attrs import define, field

from ..base import Experiment, ExperimentProperties


@define
class Simple2d(Experiment):
    properties = field(
        factory=lambda: ExperimentProperties(
            n_input_dim=2,
            n_output_dim=1,
            domain=np.array([[-4.0, 5.0], [-5.0, 4.0]]).T,
        )
    )

    true_optima = np.array([2.0, -4.0])

    def _truth(self, X):
        x = X[:, 0]
        y = X[:, 1]
        res = (1 - x / 3.0 + x**5 + y**5) * np.exp(
            -(x**2) - y**2
        ) + 2.0 * np.exp(-((x - 2) ** 2) - (y + 4) ** 2)
        # Constant offset so that the minimum is roughly 0
        const = 0.737922
        return (res.reshape(-1, 1) + const) / (2.0 + const)
