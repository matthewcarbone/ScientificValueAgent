import numpy as np
from attrs import define, field

from ..base import Experiment, ExperimentProperties


@define
class NegatedRosenbrock2d(Experiment):
    properties = field(
        factory=lambda: ExperimentProperties(
            n_input_dim=2,
            n_output_dim=1,
            valid_domain=None,
            experimental_domain=np.array([[-2.048, 2.048], [-2.048, 2.048]]).T,
        )
    )

    def _truth(self, X):
        x = X[:, 0]
        y = X[:, 1]
        res = -100 * ((y - (x**2)) ** 2) - (1 - x) ** 2
        return res.reshape(-1, 1)
