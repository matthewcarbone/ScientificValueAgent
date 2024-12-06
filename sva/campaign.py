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
from monty.json import MSONable
from scipy.stats import qmc

from sva.bayesian_optimization import parse_acquisition_function
from sva.logger import logger
from sva.models import DEVICE, EasySingleTaskGP, fit_EasyGP_mll
from sva.utils import Timer, seed_everything
from sva.value import SVF


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
    """

    # General parameters
    n_max = field(validator=[instance_of(int), ge(1)], default=100)

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
        Y, Y_std = experiment(X)
        metadata = {"experiment": experiment.name, "policy": self.name}
        metadata = [metadata] * X.shape[0]
        data.update(X, Y, Y_std, metadata)
        return PolicyState()


@define(kw_only=True)
class GridPolicy(Policy):
    """Self-explanatory grid-search policy. Selects points based on an
    even grid."""

    def step(self, experiment, data):
        ppd = ceil((self.n_max) ** (1.0 / experiment.n_input_dim))
        X = experiment.get_dense_coordinates(ppd=ppd)
        Y, Y_std = experiment(X)
        metadata = {"experiment": experiment.name, "policy": self.name}
        metadata = [metadata] * X.shape[0]
        data.update(X, Y, Y_std, metadata)
        return PolicyState()


DEFAULT_OPTIMIZE_KWARGS = {"q": 1, "num_restarts": 200, "raw_samples": 1000}


def importance_sampling(acqf, bounds, **kwargs):
    """Perform importance sampling over some acquisition function by discritizing space according to
    `sampling_grid_n` and constructing a discrete probability distribution over the grid points.
    """

    d = bounds.shape[1]
    halton = qmc.Halton(d=d)
    n = kwargs.get("n_samples", d * 10)  # Default 10d points
    q = kwargs.get("q", 1)
    assert q < n
    samples = halton.random(n=n)
    qual = qmc.discrepancy(samples)
    logger.debug(f"qmc discrepancy (sample quality index) = {qual:.02e}")
    samples = qmc.scale(samples, bounds[0, :].squeeze(), bounds[1, :].squeeze())

    with torch.no_grad():
        values = acqf(samples)
        probabilities = torch.softmax(values, dim=0).detach().numpy()
    sampled_indices = np.random.choice(
        samples.shape[0], size=q, p=probabilities, replace=False
    )
    next_points = samples[sampled_indices]
    return next_points, values[sampled_indices]


@define(kw_only=True)
class RequiresBayesOpt(Policy):
    """Defines a few methods that are required for non-trivial policies.
    In particular, this class defines the step method, the _get_acqf_at_state
    method and a variety of required keyword-only parameters required to
    make the step work."""

    model_factory = field(default=partial(EasySingleTaskGP.from_default))
    optimize_kwargs = field(
        default=DEFAULT_OPTIMIZE_KWARGS, validator=instance_of(dict)
    )
    model_fitting_function = field(default=partial(fit_EasyGP_mll))
    save_acquisition_function = field(
        default=False, validator=instance_of(bool)
    )
    save_model = field(default=False, validator=instance_of(bool))
    calculate_model_optimum = field(default=True, validator=instance_of(bool))
    use_importance_sampling = field(default=False, validator=instance_of(bool))

    def _get_data(self, experiment, data):
        """Retrieves the current data at every step. This might include some
        unorthodox transformations like SVA."""

        if data.N > 0:
            return data.X, data.Y, data.Y_std

        # Otherwise, it's cold start
        # We provide the model with
        X = torch.empty(0, experiment.n_input_dim)
        Y = torch.empty(0, 1)
        Y_std = torch.empty(0, 1)
        return X, Y, Y_std

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
        X, Y, Y_std = self._get_data(experiment, data)
        N = X.shape[0]
        model = self.model_factory(X, Y, Y_std)

        # Fit the model if we have at least 3 data points
        # Otherwise, we simply use the default length scale (which is 1).
        # The reason for this is because when attempting to fit the
        # model with less that a few data points, we can possibly overfit
        # and mess up the rest of the process
        if self.model_fitting_function is None:
            logger.debug("Model fitting function is None, skipping")
        elif N > 2:
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
        bounds = torch.tensor(experiment.domain).to(DEVICE)
        if self.use_importance_sampling:
            X, v = importance_sampling(
                acqf, bounds=bounds, **self.optimize_kwargs
            )
        else:
            X, v = optimize_acqf(acqf, bounds=bounds, **self.optimize_kwargs)
        X = X.cpu().numpy()
        array_str = np.array_str(X, precision=5)
        logger.debug(f"Next points {array_str} with value {v}")

        # "Run" the next experiment
        Y, Y_std = experiment(X)

        # Get the metadata
        metadata = self._get_metadata(experiment, data)

        if self.calculate_model_optimum:
            o, v = model.find_optima(experiment.domain)
            o = o.cpu().numpy()
            v = v.cpu().numpy()
            metadata["model_optimum"] = (o, v)

        if self.save_acquisition_function:
            metadata["acquisition_function"] = deepcopy(acqf)

        if self.save_model:
            metadata["model"] = deepcopy(model)

        # Note need to actually copy metadata for each of the samples in the
        # case that q > 1
        data.update(X, Y, Y_std, [metadata for _ in range(X.shape[0])])

        return PolicyState()


@define(kw_only=True)
class FixedPolicy(RequiresBayesOpt):
    """Executes a fixed-policy Bayesian Optimization experiment. The
    acquisition function is either a partial object from hydra, or a simple
    string alias for that partial. A dictionary for converting between
    these representations is provided in sva.bayesian_optimization."""

    acquisition_function = field(default="EI")
    penalty_function_factory = field(default=None)
    penalty_strength = field(default=1000.0)

    @property
    def name(self):
        return parse_acquisition_function(self.acquisition_function)["name"]

    def _get_acqf_at_state(self, experiment, data):
        """Gets the acquisition function on the current state, given the
        experiment and data"""

        return self.acquisition_function

    def _penalize(self, acqf, experiment, data):
        if self.penalty_function_factory is None:
            return acqf
        return PenalizedAcquisitionFunction(
            acqf,
            self.penalty_function_factory(experiment, data),
            regularization_parameter=self.penalty_strength,
        )

    def _get_metadata(self, experiment, data):
        return {"experiment": experiment.name, "policy": self.name}


@define
class FixedSVAPolicy(FixedPolicy):
    """Executes a Scientific Value Function-driven experiment."""

    svf = field(factory=SVF)

    def _get_data(self, experiment, data):
        # TODO: propagation of errors for SVF!
        # Right now, we're just returning None all the time!
        return data.X, self.svf(data.X, data.Y).reshape(-1, 1), None


@define
class CampaignData(MSONable):
    """Container for sampled data during experiments. This is a serializable
    abstraction over the data used during the experiments. This includes
    updating it and saving it to disk. Note that data contained here must
    always be two-dimensional (x.ndim == 2)."""

    X: np.ndarray = field(default=None)
    Y: np.ndarray = field(default=None)
    Y_std: np.ndarray = field(default=None)
    metadata = field(factory=list)

    @property
    def N(self):
        if self.X is None:
            return 0
        return self.X.shape[0]

    @property
    def is_initialized(self):
        if self.X is None and self.Y is None:
            return False
        return True

    def update_X(self, X):
        """Updates the current input data with new inputs.

        Parameters
        ----------
        X : np.ndarray
            Two-dimensional data to update the X values with.
        """

        assert X.ndim == 2
        if self.X is not None:
            self.X = np.concatenate([self.X, X], axis=0)
        else:
            self.X = X

    def update_Y(self, Y):
        """Updates the current output data with new outputs.

        Parameters
        ----------
        Y : np.array
            Two-dimensional data to update the Y values with.
        """

        if self.Y is not None:
            self.Y = np.concatenate([self.Y, Y], axis=0)
        else:
            self.Y = Y

    def update_Y_std(self, Y_std):
        if self.Y_std is not None:
            self.Y_std = np.concatenate([self.Y_std, Y_std], axis=0)
        else:
            self.Y_std = Y_std

    def update_metadata(self, new_metadata):
        self.metadata.extend(new_metadata)

    def update(self, X, Y, Y_std=None, metadata=None):
        """Helper method for updating the data attribute with new data.

        Parameters
        ----------
        X, Y : numpy.ndarray
            The input and output data to update with.
        metadata : list, optional
            Should be a list of Any or None.
        """

        assert X.shape[0] == Y.shape[0]
        if metadata is not None:
            assert X.shape[0] == len(metadata)
        else:
            metadata = [None] * X.shape[0]

        self.update_X(X)
        self.update_Y(Y)
        if Y_std is None:
            self.update_Y_std(Y_std)
        self.update_metadata(metadata)

    def prime(self, experiment, protocol, seed=None, **kwargs):
        """Initializes the data via some provided protocol.

        Current options are "random", "LatinHypercube" and "dense". In
        addition, there is the "cold_start" option, which does nothing. This
        will force the campaign model to use the unconditioned prior.

        Parameters
        ----------
        experiment : sva.experiments.Experiment
            Callable experiment.
        protocol : str, optional
            The method for using to initialize the data.
        kwargs
            To pass to the particular method.
        """

        if protocol == "cold_start":
            return  # do nothing!

        if seed is not None:
            seed_everything(seed)

        if protocol == "random":
            X = experiment.get_random_coordinates(**kwargs)
        elif protocol == "LatinHypercube":
            X = experiment.get_latin_hypercube_coordinates(**kwargs)
        elif protocol == "dense":
            X = experiment.get_dense_coordinates(**kwargs)
        else:
            raise NotImplementedError(f"Unknown provided protocol {protocol}")

        Y, Y_std = experiment(X)

        d = {"experiment": experiment.name, "policy": protocol}
        self.update(X, Y, Y_std, metadata=[d] * X.shape[0])

    def __eq__(self, exp):
        # XOR for when things aren't initialized
        if (exp.X is None) ^ (self.X is None):
            return False
        if (exp.Y is None) ^ (self.Y is None):
            return False

        if exp.X is not None and self.X is not None:
            if exp.X.shape != self.X.shape:
                return False
            if not np.all(exp.X == self.X):
                return False
        if exp.Y is not None and self.Y is not None:
            if exp.Y.shape != self.Y.shape:
                return False
            if not np.all(exp.Y == self.Y):
                return False

        return True


@define(kw_only=True)
class Campaign(MSONable):
    """Core executor for running an experiment.

    Parameters
    ----------
    seed : int
        Random number generator seed to ensure reproducibility. The Campaign
        ensures the seed is passed through to all generators that require it
        such that the entire campaign is reproducible.
    """

    experiment = field()
    policy = field()
    data = field(factory=lambda: CampaignData())
    seed = field(validator=[instance_of(int), ge(0)])

    def get_model_at_iteration(self, iter=-1):
        """Retrievew the model at the provided iteration. This only works if
        you're saving the model in the policy."""

        if not self.policy.save_model:
            logger.warning("policy.save_model is False, returning None")
            return None

        return self.data.metadata[iter]["model"]

    def _run(self):
        seed_everything(self.seed)

        while self.data.N < self.policy.n_max:
            self.policy.step(self.experiment, self.data)

    @property
    def name(self):
        return f"{self.experiment.name}_{self.policy.name}_{self.seed}"

    def run(self):
        with Timer() as timer:
            self._run()
        logger.success(f"[{timer.dt:.02f} s] {self.name}")
