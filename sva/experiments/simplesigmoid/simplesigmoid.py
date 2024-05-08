import numpy as np
from attrs import define, field, validators

from ..base import Experiment, ExperimentProperties


@define
class SimpleSigmoid(Experiment):
    """A simple 1d experimental response to a 1d input. This is a sigmoid
    function centered at 0, with a range (-0.5, 0.5). The sharpness of the
    sigmoid function is adjustable by setting the parameter a."""

    properties = field(
        factory=lambda: ExperimentProperties(
            n_input_dim=1,
            n_output_dim=1,
            domain=np.array([-2.0, 2.0]).reshape(2, 1),
        )
    )
    a = field(default=10.0, validator=validators.instance_of(float))

    def _truth(self, x):
        return 2.0 / (1.0 + np.exp(-self.a * x)) - 1.0

    def _dtruth(self, x):
        e = np.exp(-self.a * x)
        d = 1.0 + e
        return 2.0 * self.a * e / d**2
