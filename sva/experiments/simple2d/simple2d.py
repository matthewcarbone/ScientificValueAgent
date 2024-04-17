import numpy as np
from attrs import define, field, validators
from monty.json import MSONable

from ..base import (
    NOISE_TYPES,
    ExperimentData,
    ExperimentHistory,
    ExperimentMixin,
    ExperimentProperties,
)
from ..campaign import CampaignBaseMixin


@define
class Simple2d(ExperimentMixin, CampaignBaseMixin, MSONable):
    history = field(factory=lambda: ExperimentHistory())
    properties = field(
        factory=lambda: ExperimentProperties(
            n_input_dim=2,
            n_output_dim=1,
            valid_domain=None,
            experimental_domain=np.array([[-4.0, 5.0], [-5.0, 4.0]]).T,
        )
    )
    noise = field(default=None, validator=validators.instance_of(NOISE_TYPES))
    data = field(factory=lambda: ExperimentData())

    def _truth(self, X):
        x = X[:, 0]
        y = X[:, 1]
        res = (1 - x / 3.0 + x**5 + y**5) * np.exp(
            -(x**2) - y**2
        ) + 2.0 * np.exp(-((x - 2) ** 2) - (y + 4) ** 2)
        return res.reshape(-1, 1)
