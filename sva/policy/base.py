from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
import torch
from attrs import define, field
from attrs.validators import ge, instance_of
from botorch.optim import optimize_acqf

from sva.bayesian_optimization import (
    get_acquisition_function_name,
    parse_acquisition_function,
)
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
class FixedPolicy(Policy):
    """Executes a fixed-policy Bayesian Optimization experiment. Requires a
    model factory which is re-fit at every step of the optimization. The model
    factory should take arguments for X and Y, and should have the fit() method
    defined on it.

    Parameters
    ----------
    model_factory : callable
        Callable that produces the model as a function of X and Y.
    model_fitting_function : callable
        Function used for fitting the model. Takes an EasyGP object as input
        (or at least one with the model attribute).
    """

    model_factory = field()
    acquisition_function = field()
    optimize_kwargs = field()
    model_fitting_function = field()
    save_acquisition_function = field(
        default=False, validator=instance_of(bool)
    )
    save_model = field(default=False, validator=instance_of(bool))

    def __attrs_post_init__(self):
        logger.debug(f"{self.name} initialized as: {self}")

    @property
    def name(self):
        acqf = get_acquisition_function_name(self.acquisition_function)
        return f"{self.__class__.__name__}-{acqf}"

    def step(self, experiment, data):
        model = self.model_factory(data.X, data.Y)
        # TODO: eventually we want to pass Yvar too

        # Fit the model
        fit_results = self.model_fitting_function(model)
        logger.debug(f"Model fit, output is: {fit_results}")

        # Ask for the next point
        factory, is_EI = parse_acquisition_function(self.acquisition_function)
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
        metadata = {"experiment": experiment.name, "policy": self.name}

        if self.save_acquisition_function:
            metadata["acquisition_function"] = deepcopy(acqf)

        if self.save_model:
            metadata["model"] = deepcopy(model)

        data.update(X, Y, [metadata])

        return PolicyState()
