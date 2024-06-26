import numpy as np
from attrs import define, field

from ..base import Experiment, ExperimentProperties


@define
class NegatedAckley2d(Experiment):
    properties = field(
        factory=lambda: ExperimentProperties(
            n_input_dim=2,
            n_output_dim=1,
            domain=np.array([[-5.0, 5.0], [-5.0, 5.0]]).T,
        )
    )

    def _truth(self, X):
        x = X[:, 0]
        y = X[:, 1]
        res = (
            -(-20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))))
            + np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
            - np.e
            - 20
        )
        return res.reshape(-1, 1)
