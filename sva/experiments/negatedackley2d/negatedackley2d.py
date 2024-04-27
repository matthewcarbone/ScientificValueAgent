import numpy as np
from attrs import define, field

from sva.monty.json import MSONable

from ..base import ExperimentMixin, ExperimentProperties
from ..campaign import CampaignBaseMixin


@define
class negatedAckley2d(ExperimentMixin, CampaignBaseMixin, MSONable):
    properties = field(
        factory=lambda: ExperimentProperties(
            n_input_dim=2,
            n_output_dim=1,
            valid_domain=None,
            experimental_domain=np.array([[-5.0, 5.0], [-5.0, 5.0]]).T,
        )
    )

    def _truth(self, X):
        x = X[:, 0]
        y = X[:, 1]
        res = -(-20 * np.exp(-0.2 * np.sqrt(0.5*(x**2 + y**2)))) \
               + np.exp(0.5*(np.cos(2*np.pi*x) + np.cos(2*np.pi*y))) \
               - np.e - 20
        return res.reshape(-1, 1)
