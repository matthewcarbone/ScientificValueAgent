from functools import cached_property

import gpytorch
import numpy as np
from attrs import define, field
from sklearn.neighbors import KNeighborsRegressor

from sva.models.gp.gp import EasySingleTaskGP, get_covar_module
from sva.utils import get_coordinates, seed_everything

from ..base import Experiment, ExperimentProperties


@define
class GPDream(Experiment):
    gp_model_params = field(default=None)
    ppd_factor = field(default=5.0)
    seed = field(default=None)

    @property
    def optimum(self):
        return self.metadata["optimum"]

    def _get_training_data(self):
        """Helper method which return X, Y (training data) as well as the
        kernel used to generate it."""

        assert set(list(self.gp_model_params.keys())).issubset(
            set(["kernel", "lengthscale", "period_length"])
        )
        seed_everything(self.seed)
        lengthscale = self.gp_model_params["lengthscale"]
        kernel = get_covar_module(self.gp_model_params)
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

        return X, Y, kernel

    @cached_property
    def model(self):
        X, Y, kernel = self._get_training_data()
        domain = self.properties.domain
        true_gp = EasySingleTaskGP.from_default(X, Y, covar_module=kernel)
        self.metadata["optimum"] = true_gp.find_optima(domain)
        return true_gp

    @classmethod
    def from_default(cls, gp_model_params, d=1, seed=1, ppd_factor=5.0):
        # Available options for now
        properties = ExperimentProperties(
            n_input_dim=d,
            n_output_dim=1,
            domain=np.array([[-1.0, 1.0] for _ in range(d)]).T,
        )
        klass = cls(
            properties=properties,
            gp_model_params=gp_model_params,
            ppd_factor=ppd_factor,
            seed=seed,
        )
        return klass

    def _truth(self, X):
        return self.model.predict(X)[0].reshape(-1, 1)


@define
class GPDreamKNN(GPDream):
    knn_params = field(factory=dict)

    @cached_property
    def model(self):
        X, Y, _ = self._get_training_data()
        # domain = self.properties.domain
        model = KNeighborsRegressor(**self.knn_params)
        model.fit(X, Y.flatten())
        self.metadata["optimum"] = None  # TODO:
        return model

    @classmethod
    def from_default(
        cls, gp_model_params, knn_params, d=1, seed=1, ppd_factor=5.0
    ):
        # Available options for now
        properties = ExperimentProperties(
            n_input_dim=d,
            n_output_dim=1,
            domain=np.array([[-1.0, 1.0] for _ in range(d)]).T,
        )
        klass = cls(
            properties=properties,
            gp_model_params=gp_model_params,
            ppd_factor=ppd_factor,
            seed=seed,
            knn_params=knn_params,
        )
        return klass

    def _truth(self, X):
        return self.model.predict(X).reshape(-1, 1)
