import numpy as np
from attrs import define, field

from ..base import Experiment, ExperimentProperties


@define
class Rastrigin2d(Experiment):
    properties = field(
        factory=lambda: ExperimentProperties(
            n_input_dim=2,
            n_output_dim=1,
            valid_domain=None,
            experimental_domain=np.array([[-1.0, 1.0], [-1.0, 1.0]]).T,
        )
    )

    def _truth(self, X):
        x = X[:, 0]
        y = X[:, 1]
        res = (
            (10 * 2)
            + (x**2 - (10 * np.cos(2 * np.pi * x)))
            + (y**2 - (10 * np.cos(2 * np.pi * y)))
        )
        return res.reshape(-1, 1)
