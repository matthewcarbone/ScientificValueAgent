from functools import cached_property

import gpytorch
import numpy as np
from attrs import define, field

from sva.models.gp import EasySingleTaskGP
from sva.models.gp.gp import get_covar_module
from sva.utils import get_coordinates, seed_everything

from ..base import Experiment, ExperimentProperties


@define
class GPDream(Experiment):
    model_params = field(default=None)
    ppd_factor = field(default=5.0)
    seed = field(default=None)

    @cached_property
    def gp(self):
        seed_everything(self.seed)
        lengthscale = self.model_params["lengthscale"]
        kernel = get_covar_module(self.model_params)
        kernel = gpytorch.kernels.ScaleKernel(kernel)
        d = self.properties.n_input_dim

        gp = EasySingleTaskGP.from_default(
            X=None, Y=None, covar_module=kernel, input_dims=d
        )

        ppd = int(self.ppd_factor / lengthscale)

        domain = self.properties.domain

        X = get_coordinates(ppd, domain)
        Y = gp.sample(X, samples=1)
        Y = Y.reshape(-1, 1)

        del gp

        true_gp = EasySingleTaskGP.from_default(X, Y, covar_module=kernel)

        self.metadata["true_optimum"] = true_gp.find_optima(domain)

        return true_gp

    @classmethod
    def from_default(cls, d=1, seed=1, ppd_factor=5.0, **kwargs):
        # Available options for now
        assert set(list(kwargs.keys())).issubset(
            set(["kernel", "lengthscale", "period_length"])
        )
        properties = ExperimentProperties(
            n_input_dim=d,
            n_output_dim=1,
            domain=np.array([[-1.0, 1.0] for _ in range(d)]).T,
        )
        klass = cls(
            properties=properties,
            model_params=kwargs,
            ppd_factor=ppd_factor,
            seed=seed,
        )
        return klass

    def find_optima(self, **kwargs):
        return self.gp.find_optima(self.properties.domain, **kwargs)

    def _truth(self, X):
        return self.gp.predict(X)[0].reshape(-1, 1)
