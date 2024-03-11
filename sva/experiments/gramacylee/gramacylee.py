import numpy as np
from attrs import define, field, validators
from monty.json import MSONable

from sva.experiments.base import (
    NOISE_TYPES,
    ExperimentData,
    ExperimentMixin,
    ExperimentProperties,
)


@define
class NegatedGramacyLee2012(ExperimentMixin, MSONable):
    """The Gramacy & Lee function details and citations can be found here:
    https://www.sfu.ca/~ssurjano/grlee12.html"""

    properties = field(
        factory=lambda: ExperimentProperties(
            n_input_dim=1,
            n_output_dim=1,
            valid_domain=None,
            experimental_domain=np.array([[0.5, 2.5]]).T,
        )
    )
    noise = field(default=None, validator=validators.instance_of(NOISE_TYPES))
    data = field(factory=lambda: ExperimentData())

    def _truth(self, x):
        t1 = -np.sin(10.0 * np.pi * x) / 2.0 / x
        t2 = -((x - 1.0) ** 4)
        return t1 + t2
