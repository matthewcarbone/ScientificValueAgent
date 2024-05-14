import numpy as np
from attrs import define, field

from ..base import Experiment, ExperimentProperties

try:
    from ggce import DenseSolver, Model, System
    from ggce.logger import disable_logger
except ImportError as e:
    raise ImportError(
        "To use the ggce experiment, you must install ggce. We recommend "
        "installing via 'pip install ggce==0.1.3'. "
        f"Exception caught: {e}"
    )


@define
class Peierls(Experiment):
    phonon_frequency = field(default=1.0)
    phonon_extent = field(default=2)
    phonon_number = field(default=3)
    dimensionless_coupling_strength = field(default=1.0)
    hopping = field(default=1.0)
    eta = field(default=0.05)
    properties = field(
        factory=lambda: ExperimentProperties(
            n_input_dim=2,
            n_output_dim=1,
            domain=np.array([[0.0, np.pi], [-3.0, 0.0]]).T,
        )
    )

    def __attrs_post_init__(self):
        self._model = Model.from_parameters(hopping=self.hopping)
        self._model.add_(
            "Peierls",
            phonon_frequency=self.phonon_frequency,
            phonon_extent=self.phonon_extent,
            phonon_number=self.phonon_number,
            dimensionless_coupling_strength=self.dimensionless_coupling_strength,
        )
        with disable_logger():
            self._system = System(self._model)
            self._solver = DenseSolver(self._system)

    def _truth(self, X):
        result = []
        for k, w in X:
            g = self._solver.greens_function([k], [w], eta=self.eta, pbar=False)
            result.append((-np.imag(g) / np.pi).item())
        return np.array(result).reshape(-1, 1)
