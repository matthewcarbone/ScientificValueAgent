import gpytorch
import numpy as np
from attrs import define, field

from sva.models.gp import EasySingleTaskGP
from sva.models.gp.gp import get_covar_module
from sva.utils import get_coordinates, seed_everything

from ..base import Experiment, ExperimentProperties


@define
class GPDream(Experiment):
    gp = field(default=None)
    seed = field(default=None)

    @classmethod
    def from_default(cls, d=1, seed=1, **kwargs):
        # Available options for now
        assert set(list(kwargs.keys())).issubset(
            set(["kernel", "lengthscale", "period_length"])
        )
        properties = ExperimentProperties(
            n_input_dim=d,
            n_output_dim=1,
            domain=np.array([[-1.0, 1.0] for _ in range(d)]).T,
        )
        seed_everything(seed)
        lengthscale = kwargs["lengthscale"]
        kernel = get_covar_module(kwargs)
        kernel = gpytorch.kernels.ScaleKernel(kernel)

        gp = EasySingleTaskGP.from_default(
            X=None, Y=None, covar_module=kernel, input_dims=d
        )

        ppd = int(5.0 / lengthscale)

        X = get_coordinates(ppd, properties.domain)
        Y = gp.sample(X, samples=1)
        Y = Y.reshape(-1, 1)
        klass = cls(
            properties=properties,
            gp=EasySingleTaskGP.from_default(X, Y, covar_module=kernel),
            seed=seed,
        )
        klass.metadata["true_optimum"] = klass.gp.find_optima(properties.domain)
        return klass

    def find_optima(self, **kwargs):
        return self.gp.find_optima(self.properties.domain, **kwargs)

    def _truth(self, X):
        return self.gp.predict(X)[0].reshape(-1, 1)
