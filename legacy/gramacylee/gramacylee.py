import numpy as np
from attrs import define, field

from sva.monty.json import MSONable

from ..base import ExperimentMixin, ExperimentProperties


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

    def _truth(self, x):
        t1 = -np.sin(10.0 * np.pi * x) / 2.0 / x
        t2 = -((x - 1.0) ** 4)
        return t1 + t2
