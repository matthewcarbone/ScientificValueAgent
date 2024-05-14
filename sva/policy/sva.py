from attrs import define, field

from sva.value import SVF

from .base import FixedPolicy


@define
class FixedSVAPolicy(FixedPolicy):
    """Executes a Scientific Value Function-driven experiment."""

    svf = field(factory=SVF)

    def _get_data(self, experiment, data):
        return data.X, self.svf(data.X, data.Y).reshape(-1, 1)
