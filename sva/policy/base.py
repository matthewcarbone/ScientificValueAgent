from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial

import numpy as np
import torch
from attrs import define, field
from attrs.validators import ge, instance_of
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
        X = experiment.get_random_coordinates(n=1)
        Y = experiment(X)
        metadata = {"experiment": experiment.name, "policy": self.name}
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

    @abstractmethod
    def _get_acqf_at_state(self, experiment, data): ...

    @abstractmethod
    def _get_metadata(self, experiment, data): ...

    def step(self, experiment, data):
        # Get the model
        model = self.model_factory(data.X, data.Y)

        # Fit the model
        fit_results = self.model_fitting_function(model)
        logger.debug(f"Model fit, output is: {fit_results}")

        # Important step to get the acquisition function as a function of
        # the current step, experiment and data
        acquisition_function = self._get_acqf_at_state(experiment, data)
        logger.debug(f"Acquisition function is: {acquisition_function}")

        # Ask for the next point
        factory, is_EI = parse_acquisition_function(acquisition_function)
        kwargs = {"best_f": data.Y.max()} if is_EI else {}
        acqf = factory(model.model, **kwargs)
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
    """Executes a fixed-policy Bayesian Optimization experiment."""

    acquisition_function = field()

    @property
    def name(self):
        acqf = self.acquisition_function

        if isinstance(acqf, dict):
            acqf_kwargs = acqf["kwargs"]
            acqf = acqf["acquisition_function"]
            if "EI" in acqf:
                beta = None
            else:
                beta = acqf_kwargs["beta"]

        elif isinstance(acqf, str):
            if "EI" in acqf:
                beta = None
            else:
                acqf, beta = acqf.split("-")

        elif isinstance(acqf, partial):
            if "qExpectedImprovement" == acqf.func.__name__:
                beta = None
                acqf = "qEI"
            elif "ExpectedImprovement" == acqf.func.__name__:
                beta = None
                acqf = "EI"
            elif "qUpperConfidenceBound" == acqf.func.__name__:
                beta = acqf.keywords["beta"]
                acqf = "qUCB"
            elif "UpperConfidenceBound" == acqf.func.__name__:
                beta = acqf.keywords["beta"]
                acqf = "UCB"
            else:
                raise ValueError(f"Invalid acqf {acqf}")

        if beta is None:
            return acqf
        beta = float(beta)
        return f"{acqf}-{beta:.02f}"

    def _get_acqf_at_state(self, experiment, data):
        """Gets the acquisition function on the current state, given the
        experiment and data"""

        return self.acquisition_function

    def _get_metadata(self, experiment, data):
        return {"experiment": experiment.name, "policy": self.name}
