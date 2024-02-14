"""Module for running campaigns."""

from abc import ABC, abstractmethod, abstractproperty
from functools import cached_property

import gpytorch
import numpy as np
import torch
from botorch.acquisition.penalized import PenalizedAcquisitionFunction
from botorch.exceptions.errors import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from monty.json import MSONable
from scipy.spatial import distance_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm

from sva.data import CampaignData, _get_grid
from sva.utils import Timer, get_function_from_signature
from sva.value import svf as value_function


class BaseCampaign(ABC, MSONable):

    # Abstract properties

    @abstractproperty
    def sampled_points(self): ...

    @abstractproperty
    def dense_grid(self): ...

    # Abstract methods

    @abstractmethod
    def _acquire_data(self, state: dict) -> dict: ...

    @abstractmethod
    def _transform_data(self, state: dict): ...

    @abstractmethod
    def _initialize_model_(self, state: dict): ...

    @abstractmethod
    def _fit_model_(self, state: dict): ...

    @abstractmethod
    def _ask_model_(self, state: dict): ...

    @abstractmethod
    def _predict_(self, state: dict): ...

    @abstractmethod
    def _update_state_(self, state: dict): ...

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


class Campaign(BaseCampaign):
    def sampled_points(self):
        return self.data.X

    @cached_property
    def dense_grid(self):
        domain = self.experiment.experimental_domain
        return _get_grid(self.predict_points_per_dimension, domain)

    def _acquire_data_(self, state):
        state["X"] = torch.tensor(self.data.X.copy())
        state["Y"] = torch.tensor(self.data.Y.copy())

    def _transform_data_(self, state):
        return state

    def _get_in_out_transforms(self, X, Y):

        # Initialize the transforms
        input_transform = None
        if self.input_transform is not None:
            input_transform = self.input_transform(
                X.shape[1], transform_on_eval=True
            )
        output_transform = None
        if self.output_transform is not None:
            output_transform = self.output_transform(Y.shape[1])
        return input_transform, output_transform

    def _initialize_model_(self, state):

        X = torch.tensor(state["X"])
        Y = torch.tensor(state["Y"])

        input_transform, output_transform = self._get_in_out_transforms(X, Y)

        # default mean_prior: gpytorch.means.ConstantMean()
        # default kernel: gpytorch.kernels.ScaleKernel(...)
        # default likelihood: gpytorch.likelihoods.GaussianLikelihood()
        state["model"] = self.model(
            train_X=X,
            train_Y=Y,
            input_transform=input_transform,
            outcome_transform=output_transform,
        )

    def _fit_model_(self, state):
        # optimizer = self.optimizer.lower()
        # losses = None
        # match optimizer:
        #     case "mll":
        #         try:
        #             _fit_gp_gpytorch_mll_(gp, fit_kwargs)
        #         except ModelFittingError:
        #             self._fit_(gp, X, fit_kwargs, "adam")
        #     case "adam":
        #         losses = _fit_gp_Adam_(gp, X, fit_kwargs)
        #     case _:
        #         raise ValueError(f"Invalid choice of optimizer {optimizer}")
        # return losses
        losses = _fit_gp_Adam_(state["model"], state["X"], self.fit_kwargs)
        state["losses"] = losses

    def _ask_model_(self, state):

        gp = state["model"]

        bounds = torch.tensor(self.experiment.experimental_domain)

        # Figure out a way to detect EI
        kwargs = {}
        if None:
            kwargs = {"best_f": None}

        # Otherwise
        acqf = self.acquisition_function(gp, **kwargs)
        next_points, value = optimize_acqf(
            acqf,
            bounds=bounds,
            **self.optimize_acqf_kwargs,
        )
        state["next_points"] = next_points
        state["value"] = value
        state["acquisition_function"] = acqf

    def _predict_(self, state):
        """Runs forward prediction on a grid to get the objective function.
        Returns the mean and standard deviation of the gp."""

        if self.run_predict_every == 0:
            return None

        elif state["n_experiments"] - 1 == state["iterations"]:
            pass

        elif state["iterations"] == 0:
            pass

        elif state["X"].shape[0] % self.run_predict_every != 0:
            return None

        gp = state["model"]

        grid = torch.tensor(self.dense_grid)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = gp.posterior(grid, observation_noise=True)
        mu = posterior.mean.detach().numpy().squeeze()
        sd = np.sqrt(posterior.variance.detach().numpy().squeeze())
        state["mu"] = mu
        state["sd"] = sd

    def _update_state_(self, state):

        # Update X and Y
        self.data.update_X_(state["next_points"].detach().numpy())
        self.data.update_Y_(self.experiment)

        # Get rid of information we don't need every iteration
        state.pop("X")
        state.pop("Y")
        state.pop("n_experiments")

    def __init__(
        self,
        *,
        data,
        experiment,
        value,
        model,
        # acquisition_function,
        # predict_points_per_dimension,
        # run_predict_every,
        # optimizer=None,
        # optimize_acqf_kwargs=None,
        # fit_kwargs=None,
        # record=None,
    ):
        """Summary

        Parameters
        ----------
        data : sva.data.CampaignData
        experiment : sva.truth.Truth
        value : sva.value.BaseValue
        acquisition_function : None, optional
            Description
        acquisition_function_kwargs : None, optional
            Description
        points_per_dimension : int, optional
            Description
        """

        self.data = data
        self.experiment = experiment
        self.value = value
        self.model = model

        # self.acquisition_function = acquisition_function

        # self.predict_points_per_dimension = predict_points_per_dimension
        # self.run_predict_every = run_predict_every
        # assert self.run_predict_every >= 0

        # self.optimizer = optimizer
        # self.optimize_acqf_kwargs = optimize_acqf_kwargs if \
        #     optimize_acqf_kwargs is not None else {}
        # self.fit_kwargs = fit_kwargs if fit_kwargs is not None else {}
        # self.record = record if record is not None else []


class SVACampaign(Campaign):
    """Runs an active learning campaign using the SVA procedure."""

    ...
