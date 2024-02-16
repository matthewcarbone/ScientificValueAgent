"""Module for running campaigns."""

from abc import ABC, abstractproperty, abstractmethod
from attrs import define, field, validators
# from botorch.acquisition.penalized import PenalizedAcquisitionFunction
# from botorch.exceptions.errors import ModelFittingError
# from botorch.fit import fit_gpytorch_mll
# from botorch.models.transforms.input import Normalize
# from botorch.models.transforms.outcome import Standardize
# from botorch.models import SingleTaskGP
# from botorch.optim import optimize_acqf
# from functools import cached_property
# import gpytorch
import numpy as np
from monty.json import MSONable
# from scipy.spatial import distance_matrix
# from sklearn.preprocessing import StandardScaler, MinMaxScaler


import torch ###
from tqdm import tqdm


from sva.experiments.base import ExperimentMixin

@define(kw_only=True)
class BaseCampaign(ABC, MSONable):

    experiment = field()
    @experiment.validator  # noqa
    def valid_experiment(self, _, value):
        if not issubclass(value.__class__, ExperimentMixin):
            raise ValueError(
                f"Invalid experiment: {value}. Must inherit from "
                "ExperimentMixin"
            )

    predict_every = field(
        default=10,
        validator=validators.instance_of((type(None), int))
    )
    predict_points_per_dimension = field(
        default=100,
        validator=validators.instance_of(int)
    )

    record = field(factory=list)

    seed = field(default=None)
    @seed.validator  # noqa
    def valid_seed(self, _, value):
        if value is None:
            return
        assert value >= 0
        assert isinstance(value, int)

    @abstractproperty
    def sampled_points(self):
        ...

    @abstractproperty
    def dense_grid(self):
        ...

    @abstractmethod
    def _acquire_data_(self, state: dict):
        ...

    @abstractmethod
    def _transform_data_(self, state: dict):
        ...

    @abstractmethod
    def _initialize_model_(self, state: dict):
        ...

    @abstractmethod
    def _fit_model_(self, state: dict):
        ...

    @abstractmethod
    def _ask_model_(self, state: dict):
        ...

    @abstractmethod
    def _predict_(self, state: dict):
        ...

    @abstractmethod
    def _update_state_(self, state: dict):
        ...

    def initialize_data(self, n, protocol="random"):
        """Initializes the data via some provided protocol. This is an
        optional component which is only needed if the experiment does not
        contain any points already."""

        if self.experiment.data.is_initialized:
            raise RuntimeError("Data is already initialized")

        seed = self.seed
        self.experiment._initialize_data_(n=n, seed=seed, protocol=protocol)

    def run(self, n_experiments, pbar=False):
        """The run method has a set of sequential steps that occur for
        every experiment. Each of these sequential steps should return a
        dictionary, that is ultimately passed to the next step and added to
        as the procedure continues. Each time we iterate, this current state
        dictionary can be appended to a log via an appending method.

        Suppose we are at iteration n:

        1. ``_acquire_data(dict)`` Get the data X_n, Y_n, both of which
        are of shape (N, d) and (N, d').
        2. ``_transform_data_(dict)`` Executes any transforms that might
        be required (except scaling, which is always performed by the GP). This
        could be a scientific value agent transformation, for example, or some
        reduction on the target space from a vector to a scalar (such as
        picking out a particular peak location).
        3. ``_initialize_model_(dict)`` Creates the model itself.
        4. ``_fit_model_(dict)`` Sets the model to training mode and fits it.
        5. ``_ask_model_(dict)`` Sets the model to evaluation mode and "asks"
        the model what the next experiment should be.
        6. ``_predict_(dict)`` Runs any predictions.
        7. ``_update_state_(dict)`` Updates the current data with the new
        point(s).

        The simulation then continues to iterate until completion.
        """

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        for ii in tqdm(range(n_experiments), disable=not pbar):

            state = dict(iteration=ii, n_experiments=n_experiments)
            self._acquire_data_(state)
            self._transform_data_(state)
            self._initialize_model_(state)
            self._fit_model_(state)
            self._ask_model_(state)
            self._predict_(state)
            self._update_state_(state)
            self.record.append(state)


# @define(kw_only=True)
# class Campaign(BaseCampaign, MSONable):

#     def sampled_points(self):
#         return self.experiment.data.X

#     @cached_property
#     def dense_grid(self):
#         domain = self.experiment.properties.experimental_domain
#         return get_grid(self.predict_points_per_dimension, domain)

#     def _acquire_data_(self, state):
#         state["X"] = torch.tensor(self.experiment.data.X.copy())
#         state["Y"] = torch.tensor(self.experiment.data.Y.copy())

#     def _transform_data_(self, state):
#         return

#     # def _get_in_out_transforms(self, X, Y):

#     #     input_transform = Normalize(
#     #         X.shape[1], transform_on_eval=True
#     #     )
#     #     output_transform = Standardize(Y.shape[1])

#     #     # Initialize the transforms
#     #     # input_transform = None
#     #     # if self.input_transform is not None:
#     #     #     input_transform = self.input_transform(
#     #     #         X.shape[1], transform_on_eval=True
#     #     #     )
#     #     # output_transform = None
#     #     # if self.output_transform is not None:
#     #     #     output_transform = self.output_transform(Y.shape[1])
#     #     return input_transform, output_transform

#     def _initialize_model_(self, state):

#         X = state["X"]
#         Y = state["Y"]

#         # print(X.shape, Y.shape)

#         # input_transform, output_transform = self._get_in_out_transforms(X, Y)

#         # default mean_prior: gpytorch.means.ConstantMean()
#         # default kernel: gpytorch.kernels.ScaleKernel(...)
#         # default likelihood: gpytorch.likelihoods.GaussianLikelihood()
#         model = get_model(
#             X=X,
#             Y=Y,
#             transform_input=True,
#             transform_output=True,
#         )
#         state["model"] = model

#     def _fit_model_(self, state):
#         # optimizer = self.optimizer.lower()
#         # losses = None
#         # match optimizer:
#         #     case "mll":
#         #         try:
#         #             _fit_gp_gpytorch_mll_(gp, fit_kwargs)
#         #         except ModelFittingError:
#         #             self._fit_(gp, X, fit_kwargs, "adam")
#         #     case "adam":
#         #         losses = _fit_gp_Adam_(gp, X, fit_kwargs)
#         #     case _:
#         #         raise ValueError(f"Invalid choice of optimizer {optimizer}")
#         # return losses
#         losses = fit_gp_gpytorch_mll_(state["model"])
#         # losses = fit_gp_Adam_(state["model"], state["X"])
#         state["losses"] = losses

#     def _ask_model_(self, state):

#         gp = state["model"]

#         bounds = torch.tensor(self.experiment.properties.experimental_domain)
#         # print(bounds)

#         # Figure out a way to detect EI
#         # kwargs = {}
#         # if None:
#         #     kwargs = {"best_f": None}

#         # Otherwise
#         # TODO: undo this
#         # HARDCODE FOR TESTING
#         acqf = UpperConfidenceBound(gp, beta=100000.0)
#         optimize_acqf_kwargs = {"q": 1, "num_restarts": 5, "raw_samples": 20}
#         next_points, value = optimize_acqf(
#             acqf,
#             bounds=bounds,
#             **optimize_acqf_kwargs,
#         )
#         state["next_points"] = next_points
#         state["value"] = value
#         state["acquisition_function"] = acqf

#     def _predict_(self, state):
#         """Runs forward prediction on a grid to get the objective function.
#         Returns the mean and standard deviation of the gp."""

#         if self.predict_every is None:
#             pass

#         elif self.predict_every == 0:
#             return None

#         elif state["n_experiments"] - 1 == state["iteration"]:
#             pass

#         elif state["iteration"] == 0:
#             pass

#         elif state["X"].shape[0] % self.run_predict_every != 0:
#             return None

#         gp = state["model"]

#         grid = torch.tensor(self.dense_grid)
#         with torch.no_grad(), gpytorch.settings.fast_pred_var():
#             posterior = gp.posterior(grid, observation_noise=True)
#         mu = posterior.mean.detach().numpy().squeeze()
#         sd = np.sqrt(posterior.variance.detach().numpy().squeeze())
#         state["mu"] = mu
#         state["sd"] = sd

#     def _update_state_(self, state):

#         # Update X and Y
#         self.experiment.update_data_(state["next_points"].detach().numpy())

#         # Get rid of information we don't need every iteration
#         state.pop("X")
#         state.pop("Y")
#         state.pop("n_experiments")

#     # def __init__(
#     #     self,
#     #     experiment,
#     #     predict_every=None,
#     #     predict_points_per_dimension=100,
#     #     # value,
#     #     # acquisition_function,
#     #     # predict_points_per_dimension,
#     #     # run_predict_every,
#     #     # optimizer=None,
#     #     # optimize_acqf_kwargs=None,
#     #     # fit_kwargs=None,
#     #     # record=None,
#     # ):
#     #     """Summary

#     #     Parameters
#     #     ----------
#     #     data : sva.data.CampaignData
#     #     experiment : sva.truth.Truth
#     #     value : sva.value.BaseValue
#     #     acquisition_function : None, optional
#     #         Description
#     #     acquisition_function_kwargs : None, optional
#     #         Description
#     #     points_per_dimension : int, optional
#     #         Description
#     #     """

#     #     self.experiment = experiment
#     #     self.predict_every = predict_every
#     #     # self.value = value

#     #     # self.acquisition_function = acquisition_function

#     #     self.predict_points_per_dimension = predict_points_per_dimension
#     #     # self.run_predict_every = run_predict_every
#     #     # assert self.run_predict_every >= 0

#     #     # self.optimizer = optimizer
#     #     # self.optimize_acqf_kwargs = optimize_acqf_kwargs if \
#     #     #     optimize_acqf_kwargs is not None else {}
#     #     # self.fit_kwargs = fit_kwargs if fit_kwargs is not None else {}
#     #     # self.record = record if record is not None else []


# @define(kw_only=True)
# class SVACampaign(Campaign, MSONable):
#     """Runs an active learning campaign using the SVA procedure."""

#     def _transform_data_(self, state):
#         value = svf(state["X"], state["Y"])
#         state["Y"] = value.reshape(-1, 1)
