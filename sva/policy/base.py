from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial
from math import ceil

import numpy as np
import torch
from attrs import define, field
from attrs.validators import ge, instance_of
from botorch.acquisition.penalized import PenalizedAcquisitionFunction
from botorch.optim import optimize_acqf

from sva.bayesian_optimization import parse_acquisition_function
from sva.logger import logger
from sva.monty.json import MSONable


@define(kw_only=True)
class PolicyState(MSONable):
    """A helper class for state information to be passed back to the
    Campaign after every step.

    Parameters
    ----------
    terminate : bool
        If True, signals to the Campaign to terminate the simulation early.
    """

    terminate = field(default=False)


@define(kw_only=True)
class Policy(ABC, MSONable):
    """A core class for defining how a campaign is run. The Policy defines
    exactly how the next experiment should be determined, whether via some
    meta-policy or Bayesian Optimization (or some other method).

    Parameters
    ----------
    n_max : int
        The size of the data past which the campaign will terminate.
    prime_kwargs : dict
        Defines the call to CampaignData.prime. Primes the experiment with
        some seed points.
    """

    # General parameters
    n_max = field(validator=[instance_of(int), ge(1)])

    # Initialization parameters
    prime_kwargs = field(validator=instance_of(dict))

    @abstractmethod
    def step(self, experiment, data):
        """Step determines what to do given the current dataset."""

        ...

    def __attrs_post_init__(self):
        logger.debug(f"{self.name} initialized as: {self}")

    @property
    def name(self):
        return self.__class__.__name__


@define(kw_only=True)
class RandomPolicy(Policy):
    """Self-explanatory random policy. Selects random points in the domain
    at every step. Expect this policy to be pretty bad!"""

    def step(self, experiment, data):
        X = experiment.get_random_coordinates(n=self.n_max)
        Y = experiment(X)
        metadata = {"experiment": experiment.name, "policy": self.name}
        metadata = [metadata] * X.shape[0]
        data.update(X, Y, metadata)
        return PolicyState()


@define(kw_only=True)
class GridPolicy(Policy):
    """Self-explanatory grid-search policy. Selects points based on an
    even grid."""

    def step(self, experiment, data):
        ppd = ceil((self.n_max) ** (1.0 / experiment.n_input_dim))
        X = experiment.get_dense_coordinates(ppd=ppd)
        Y = experiment(X)
        metadata = {"experiment": experiment.name, "policy": self.name}
        metadata = [metadata] * X.shape[0]
        data.update(X, Y, metadata)
        return PolicyState()


@define(kw_only=True)
class RequiresBayesOpt(Policy):
    """Defines a few methods that are required for non-trivial policies.
    In particular, this class defines the step method, the _get_acqf_at_state
    method and a variety of required keyword-only parameters required to
    make the step work."""

    model_factory = field()

    @model_factory.validator
    def validate_model_factory(self, _, value):
        if not isinstance(value, partial):
            raise ValueError(
                f"Provided model_factory: {value} must be of type partial"
            )

    optimize_kwargs = field(validator=instance_of(dict))

    model_fitting_function = field()

    @model_fitting_function.validator
    def validate_model_fitting_function(self, _, value):
        if not isinstance(value, partial):
            raise ValueError(
                f"Provided model_fitting_function {value} must be of type "
                "partial"
            )

    save_acquisition_function = field(
        default=False, validator=instance_of(bool)
    )
    save_model = field(default=False, validator=instance_of(bool))

    def _get_data(self, experiment, data):
        """Retrieves the current data at every step. This might include some
        unorthodox transformations like SVA."""

        if data.N > 0:
            return data.X, data.Y

        # Otherwise, it's cold start
        # We provide the model with
        X = torch.empty(0, experiment.n_input_dim)
        Y = torch.empty(0, 1)
        return X, Y

    @abstractmethod
    def _get_acqf_at_state(self, experiment, data): ...

    def _penalize(self, acqf, experiment, data):
        # Default is no penalty
        return acqf

    @abstractmethod
    def _get_metadata(self, experiment, data): ...

    def _get_acquisition_function_kwargs(self, experiment, data): ...

    def step(self, experiment, data):
        # Get the model and the data
        X, Y = self._get_data(experiment, data)
        N = X.shape[0]
        model = self.model_factory(X, Y)

        # Fit the model if we have at least 3 data points
        # Otherwise, we simply use the default length scale (which is 1).
        # The reason for this is because when attempting to fit the
        # model with less that a few data points, we can possibly overfit
        # and mess up the rest of the process
        if N > 2:
            fit_results = self.model_fitting_function(model)
            logger.debug(f"Model fit, output is: {fit_results}")
        else:
            logger.debug("Not enough data provided, model fitting skipped")

        # Important step to get the acquisition function as a function of
        # the current step, experiment and data
        acqf_rep = self._get_acqf_at_state(experiment, data)
        logger.debug(f"Acquisition function is: {acqf_rep}")

        # Parse the acquisition function, penalty and produce the final
        # acquisition function at this step
        r = parse_acquisition_function(acqf_rep)
        acqf_factory = r["acqf_factory"]
        requires_bf = r["requires_best_f"]

        kwargs = {"best_f": Y.max() if N > 0 else 0.0} if requires_bf else {}
        acqf = acqf_factory(model.model, **kwargs)
        acqf = self._penalize(acqf, experiment, data)

        # Get the bounds from the experiment and find the next point
        bounds = torch.tensor(experiment.domain)
        X, v = optimize_acqf(acqf, bounds=bounds, **self.optimize_kwargs)
        X = X.numpy()
        array_str = np.array_str(X, precision=5)
        logger.debug(f"Next points {array_str} with value {v}")

        # "Run" the next experiment
        Y = experiment(X)

        # Get the metadata
        metadata = self._get_metadata(experiment, data)

        if self.save_acquisition_function:
            metadata["acquisition_function"] = deepcopy(acqf)

        if self.save_model:
            metadata["model"] = deepcopy(model)

        # Note need to actually copy metadata for each of the samples in the
        # case that q > 1
        data.update(X, Y, [metadata for _ in range(X.shape[0])])

        return PolicyState()


@define(kw_only=True)
class FixedPolicy(RequiresBayesOpt):
    """Executes a fixed-policy Bayesian Optimization experiment. The
    acquisition function is either a partial object from hydra, or a simple
    string alias for that partial. A dictionary for converting between
    these representations is provided in sva.bayesian_optimization."""

    acquisition_function = field()
    penalty_function = field(default=None)
    penalty_strength = field(default=1000.0)

    @property
    def name(self):
        return parse_acquisition_function(self.acquisition_function)["name"]

    def _get_acqf_at_state(self, experiment, data):
        """Gets the acquisition function on the current state, given the
        experiment and data"""

        return self.acquisition_function

    def _penalize(self, acqf, experiment, data):
        if self.penalty_function is None:
            return acqf
        return PenalizedAcquisitionFunction(
            acqf,
            self.penalty_function(experiment, data),
            regularization_parameter=self.penalty_strength,
        )

    def _get_metadata(self, experiment, data):
        return {"experiment": experiment.name, "policy": self.name}
