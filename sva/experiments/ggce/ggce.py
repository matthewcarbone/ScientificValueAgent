import numpy as np
from attrs import define, field
from attrs.validators import instance_of

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
    disable_ggce_logger = field(default=True)
    properties = field(
        factory=lambda: ExperimentProperties(
            n_input_dim=2,
            n_output_dim=1,
            domain=np.array([[0.0, np.pi], [-3.0, 0.0]]).T,
        )
    )
    y_log = field(default=True, validator=instance_of(bool))

    def _init_system_solver(self):
        self._system = System(self._model)
        self._solver = DenseSolver(self._system)

    def __attrs_post_init__(self):
        self._model = Model.from_parameters(hopping=self.hopping)
        dcc = self.dimensionless_coupling_strength
        self._model.add_(
            "Peierls",
            phonon_frequency=self.phonon_frequency,
            phonon_extent=self.phonon_extent,
            phonon_number=self.phonon_number,
            dimensionless_coupling_strength=dcc,
        )

        if self.disable_ggce_logger:
            with disable_logger():
                self._init_system_solver()
        else:
            self._init_system_solver()

    @property
    def model(self):
        return self._model

    @property
    def system(self):
        return self._system

    @property
    def solver(self):
        return self._solver

    def _truth(self, X):
        result = []
        for k, w in X:
            g = self._solver.greens_function([k], [w], eta=self.eta, pbar=False)
            result.append((-np.imag(g) / np.pi).item())
        A = np.array(result).reshape(-1, 1)
        if self.y_log:
            A = np.log10(A)
        return A
