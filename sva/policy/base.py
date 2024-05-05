from abc import ABC, abstractmethod
from copy import deepcopy

from attrs import define, field, frozen
from attrs.validators import ge, instance_of, optional

from sva.bayesian_optimization import ask
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
    model_fitting_method : str
        The method accessed from the model via getattr used to fit the model
        on the data.
    model_fitting_kwargs : dict
        Keyword arguments to pass to the fitting procedure.
    """

    model_factory = field()
    model_fitting_method = field(default="fit_mll")
    model_fitting_kwargs = field(factory=dict)
    acquisition_function = field()
    acquisition_function_kwargs = field()
    optimize_kwargs = field()
    save_acquisition_function = field(
        default=False, validator=instance_of(bool)
    )
    save_model = field(default=False, validator=instance_of(bool))

    def step(self, experiment, data):
        model = self.model_factory(data.X, data.Y)
        # TODO: eventually we want to pass Yvar too
        #
        # Fit the model
        getattr(model, self.model_fitting_method)(**self.model_fitting_kwargs)

        # ask
        ask_state = ask(
            model,
            self.acquisition_function,
            bounds=experiment.domain,
            acquisition_function_kwargs=self.acquisition_function_kwargs,
            optimize_kwargs=self.optimize_kwargs,
        )
        X = ask_state["next_points"]
        Y = experiment(X)
        if not self.save_acquisition_function:
            ask_state.pop("acquisition_function")

        metadata = {
            "experiment": experiment.name,
            "policy": self.name,
            "acquisition_function": self.acquisition_function,
            "ask_state": ask_state,
        }

        if self.save_model:
            metadata["model"] = deepcopy(model)

        data.update(X, Y, metadata)
        return PolicyState()
