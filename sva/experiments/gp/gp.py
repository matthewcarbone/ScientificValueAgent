import gpytorch
import numpy as np
import torch
from attrs import define

from sva.models.gp import EasySingleTaskGP
from sva.utils import get_coordinates, seed_everything

from ..base import Experiment, ExperimentProperties


@define
class GPDream(Experiment):
    gp = None

    @classmethod
    def dream_from_RBF_prior(cls, d=1, length_scale=1.0, seed=1):
        properties = ExperimentProperties(
            n_input_dim=d,
            n_output_dim=1,
            domain=np.array([[-1, 1] for _ in range(d)]).T,
        )
        seed_everything(seed)
        rbf = gpytorch.kernels.RBFKernel()
        rbf.lengthscale = torch.tensor(length_scale)
        kernel = gpytorch.kernels.ScaleKernel(rbf)

        gp = EasySingleTaskGP.from_default(
            X=None, Y=None, covar_module=kernel, input_dims=d
        )

        ppd = int(5.0 / length_scale)

        X = get_coordinates(ppd, properties.domain)
        Y = gp.sample(X, samples=1)
        Y = Y.reshape(-1, 1)
        klass = cls(properties=properties)
        klass.gp = EasySingleTaskGP.from_default(X, Y, covar_module=kernel)
        return klass

    def find_optima(self, **kwargs):
        return self.gp.find_optima(self.properties.domain, **kwargs)

    def _truth(self, X):
        return self.gp.predict(X)[0].reshape(-1, 1)
