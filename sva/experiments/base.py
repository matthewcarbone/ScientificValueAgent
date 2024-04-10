import pickle
from abc import ABC, abstractmethod, abstractproperty
from copy import deepcopy
from pathlib import Path
from typing import Callable
from warnings import warn

import numpy as np
import torch
from attrs import define, field, frozen, validators
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from monty.json import MSONable
from tqdm import tqdm

from sva import __version__
from sva.models.gp import EasyMultiTaskGP, EasySingleTaskGP
from sva.models.gp.bo import ask, is_EI
from sva.utils import (
    get_coordinates,
    get_function_from_signature,
    get_random_points,
    read_json,
)


@define
class ExperimentData(MSONable):
    """Container for sampled data during experiments. This is a serializable
    abstraction over the data used during the experiments. This includes
    updating it and saving it to disk. Note that data contained here must
    always be two-dimensional (x.ndim == 2)."""

    X: np.ndarray = field(default=None)
    Y: np.ndarray = field(default=None)
    Yvar: np.ndarray = field(default=None)

    @property
    def N(self):
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

    def update_Y(self, experiment):
        """Updates the current output data with new outputs.

        Parameters
        ----------
        callable
            Must be callable and take a 2d input array as input.
        """

        assert self.X is not None

        if self.Y is not None:
            diff = self.X.shape[0] - self.Y.shape[0]
            new_Y, new_Yvar = experiment(self.X[-diff:])
            self.Y = np.concatenate([self.Y, new_Y], axis=0)
        else:
            self.Y, new_Yvar = experiment(self.X)

        # If we compute variances
        if new_Yvar is not None:
            # And we have existing variances
            if self.Yvar is not None:
                self.Yvar = np.concatenate([self.Yvar, new_Yvar], axis=0)

            # Otherwise, we don't have existing variances
            else:
                self.Yvar = new_Yvar


@define
class ExperimentHistory(MSONable):
    """Container for the history of the experiment. Note that this is not
    MSONable as it will contain a variety of objects that can only be
    pickled, so that is the protocol we'll use for this."""

    history: list = field(factory=list)

    def append(self, x):
        assert isinstance(x, dict)
        self.history.append(x)

    def extend(self, x):
        assert all([isinstance(xx, dict) for xx in x])
        self.history.extend(x)

    # def as_dict(self):
    #     """Override MSONable here. We have to save the history separately."""
    #
    #     return {"history": "_UNSET"}

    def __len__(self):
        return len(self.history)

    def __getitem__(self, ii):
        return self.history[ii]


@frozen
class ExperimentProperties(MSONable):
    """Defines the core set of experiment properties, which are frozen after
    setting. These are also serializable and cannot be changed after they
    are set."""

    n_input_dim = field(validator=validators.instance_of(int))
    n_output_dim = field(validator=validators.instance_of(int))
    valid_domain = field(
        validator=validators.instance_of((type(None), np.ndarray))
    )
    experimental_domain = field(validator=validators.instance_of(np.ndarray))


NOISE_TYPES = (Callable, float, np.ndarray, list, type(None))


class ExperimentMixin(ABC):
    """Abstract base class for a source of truth. These sources of truth are
    for a single modality."""

    @abstractproperty
    def noise(self): ...

    @abstractproperty
    def history(self): ...

    @abstractproperty
    def properties(self): ...

    @abstractproperty
    def data(self): ...

    @abstractmethod
    def _truth(self, x: np.ndarray) -> np.ndarray:
        """Vectorized truth function. Should return the value of the truth
        function for all rows of the provided x input."""

        raise NotImplementedError

    def _variance(self, _):
        """Vectorized truth of the variance of the experiment. Distinct from
        the 'noise' property, this method returns the errorbar (one standard
        deviation) on the observed results. By default, this returns None,
        which is distinct from 0 (there's a difference between having a
        completely precise observation, returning 0, and having no knowledge
        of the noise, returning None)."""

        return None

    def _dtruth(self):
        """The derivative of the truth value with respect to x. May not be
        implemented, depending on the function."""

        raise NotImplementedError(
            "The derivative of the truth function is not implemented for "
            "this experiment."
        )

    def _validate_input(self, x):
        # Ensure x has the right shape
        if not x.ndim == 2:
            raise ValueError("x must have shape (N, d)")

        # Ensure that the second dimension of x is the right size for the
        # chosen experiment
        if not x.shape[1] == self.properties.n_input_dim:
            raise ValueError(
                f"x second dimension must be {self.properties.n_input_dim}"
            )

        # Ensure that the input is in the bounds
        # Domain being None implies the domain is the entirety of the reals
        # This is a fast way to avoid the check
        if self.properties.valid_domain is not None:
            check1 = self.properties.valid_domain[0, :] <= x
            check2 = x <= self.properties.valid_domain[1, :]
            if not np.all(check1 & check2):
                raise ValueError("Some inputs x were not in the domain")

    def _validate_output(self, y):
        # Ensure y has the right shape
        if not y.ndim == 2:
            raise ValueError("y must have shape (N, d')")

        # Assert that the second dimension of the output has the right size for
        # the chosen experiment
        if not y.shape[1] == self.properties.n_output_dim:
            raise ValueError(
                f"y second dimension must be {self.properties.n_output_dim}"
            )

    def variance(self, x: np.ndarray, validate=False):
        if validate:
            self._validate_input(x)
        return self._variance(x)

    def truth(self, x: np.ndarray) -> np.ndarray:
        """Access the noiseless results of an "experiment"."""

        self._validate_input(x)
        y = self._truth(x)
        self._validate_output(y)
        # No need to validate x again since it's validated above
        yvar = self._variance(x)
        return y, yvar

    def dtruth(self, x: np.ndarray) -> np.ndarray:
        """Access the derivative of the noiseless results of the experiment."""

        self._validate_input(x)
        y = self._dtruth(x)
        self._validate_output(y)
        return y

    def get_random_coordinates(self, n=1):
        """Runs n random input points."""

        domain = self.properties.experimental_domain
        return get_random_points(domain, n=n)

    def get_dense_coordinates(self, ppd):
        """Gets a set of dense coordinates.

        Parameters
        ----------
        ppd : int or list
            Points per dimension.

        Returns
        -------
        np.ndarray
        """

        domain = self.properties.experimental_domain
        return get_coordinates(ppd, domain)

    def get_experimental_domain_mpl_extent(self):
        """This is a helper for getting the "extent" for matplotlib's
        imshow.
        """

        if self.properties.n_input_dim != 2:
            raise NotImplementedError(
                "get_mpl_extent only implemented for 2d inputs."
            )

        x0 = self.properties.experimental_domain[0, 0]
        x1 = self.properties.experimental_domain[1, 0]

        y0 = self.properties.experimental_domain[0, 1]
        y1 = self.properties.experimental_domain[1, 1]

        return [x0, x1, y0, y1]

    def update_data(self, x):
        """Helper method for updating the data attribute with new data.

        Parameters
        ----------
        x : numpy.ndarray
        """

        self.data.update_X(x)
        self.data.update_Y(self)

    def initialize_data(self, n, protocol="random"):
        """Initializes the X data via some provided protocol.

        Parameters
        ----------
        n : int
            The number of points to use initially.
        protocol : str, optional
            The method for using to initialize the data.
        """

        if protocol == "random":
            X = self.get_random_coordinates(n=n)
        else:
            raise NotImplementedError(f"Unknown provided protocol {protocol}")

        self.update_data(X)

    def __call__(self, x):
        """The (possibly noisy) result of the experiment. Also returns the
        variance of the observations."""

        y, yvar = self.truth(x)

        if self.noise is None:
            return y, yvar

        if isinstance(self.noise, Callable):
            noise = self.noise(x)
            return y + np.random.normal(scale=noise, size=y.shape), yvar

        if isinstance(self.noise, float):
            pass
        elif isinstance(self.noise, np.ndarray):
            assert self.noise.ndim == 1
            assert len(self.noise) == self.properties.n_output_dim
        elif isinstance(self.noise, list):
            assert len(self.noise) == self.properties.n_output_dim
        else:
            raise ValueError("Incompatible noise type")

        return y + np.random.normal(scale=self.noise, size=y.shape), yvar


#     def save(self, directory):
#         """Saves itself to a provided directory."""
#
#         directory = Path(directory)
#         directory.mkdir(exist_ok=True, parents=True)
#
#         j = self.to_json()
#
#         # this is a list containing un-MSONable objects
#         history = self.history.history
#
#         with open(directory / "experiment.json", "w") as f:
#             f.write(j)
#         pickle.dump(
#             history,
#             open(directory / "experiment_history.pkl", "wb"),
#             protocol=pickle.HIGHEST_PROTOCOL,
#         )
#
#
# def load_experiment(directory):
#     """Loads an experiment from disk. Each experiment should get its own
#     directory."""
#
#     directory = Path(directory)
#
#     d = read_json(directory / "experiment.json")
#     history = pickle.load(open(directory / "experiment_history.pkl", "rb"))
#
#     module = d["@module"]
#     klass = d["@class"]
#     version = d["@version"]
#
#     if version != __version__:
#         warn(
#             f"Loaded experiment has version {version}, which is different "
#             f"than current sva version {__version__}"
#         )
#
#     klass = get_function_from_signature(f"{module}:{klass}").from_dict(d)
#
#     # Manually set the history
#     klass.history = history
#
#     return klass


class CampaignBaseMixin:
    """This class acts as a mixin for experiments in which there is a single
    output."""

    def _calculate_remaining_loops(self, n, q):
        # Calculate the number of remaining experiments, including the
        # number of experiment loops to perform
        if self.data.X is not None:
            initial_size = self.data.X.shape[0]
        else:
            initial_size = 0
        remaining = n - initial_size
        if remaining <= 0:
            warn("No experiments performed, set n higher for more experiments")
            return
        loops = np.ceil(remaining / q)
        return int(loops)

    @staticmethod
    def _fit(gp, fit_with, fit_kwargs):
        fit_kwargs = fit_kwargs if fit_kwargs is not None else {}
        if fit_with == "mll":
            gp.fit_mll(**fit_kwargs)
        elif fit_with == "Adam":
            gp.fit_Adam(**fit_kwargs)
        else:
            raise ValueError(
                f"train_with is {fit_with} but must be one of mll or Adam"
            )

    def _ask(
        self,
        gp,
        Y,
        acquisition_function,
        acquisition_function_kwargs,
        optimize_acqf_kwargs,
    ):
        if is_EI(acquisition_function):
            acquisition_function_kwargs["best_f"] = Y.max()
        return ask(
            gp.model,
            acquisition_function,
            bounds=self.properties.experimental_domain,
            acquisition_function_kwargs=acquisition_function_kwargs,
            optimize_acqf_kwargs=optimize_acqf_kwargs,
        )

    def run(
        self,
        n,
        acquisition_function,
        acquisition_function_kwargs=None,
        svf=None,
        model_factory=EasySingleTaskGP,
        fit_with="mll",
        fit_kwargs=None,
        optimize_acqf_kwargs=None,
        optimize_gp=False,
        num_restarts=150,
        raw_samples=150,
        pbar=True,
    ):
        """Executes the campaign by running many sequential experiments.

        Parameters
        ----------
        n : int
            The total number of experiments to run.
        acquisition_function
            Either a string representation of a model signature, an alias for
            an acquisition function defined in sva.models.gp.bo or a factory
            for an acquisition function.
        acquisition_function_kwargs
            Keyword arguments to pass to the acquisition function factory.
        svf
            If None, does standard optimization. Otherwise, uses the
            Scientific Value Agent transformation.
        fit_with : str
            Either "mll" or "Adam". Defines how the GP is fit
        fit_kwargs : dict, optional
            The keyword arguments to fit to the fitting procedure. Default
            is None.
        optimize_acqf_kwargs : dict, optional
            Keyword arguments to pass to the optimizer of the acquisition
            function. Sensible defaults are set if this is not provided,
            including sequential (q=1) optimization.
        optimize_gp : bool
            If True, will perform an optimization step over the fitted GP at
            every step of the experiment, finding its maxima using the fully
            exploitative UCB(beta=0) acquisition function.
        num_restarts, raw_samples : int
            Parameters to pass to the BoTorch optimization scheme.
        pbar : bool
            Whether or not to display the tqdm progress bar.
        """

        if optimize_acqf_kwargs is None:
            optimize_acqf_kwargs = {
                "q": 1,
                "num_restarts": 20,
                "raw_samples": 100,
            }

        if acquisition_function_kwargs is None:
            acquisition_function_kwargs = {}

        loops = self._calculate_remaining_loops(n, optimize_acqf_kwargs["q"])

        # The for loop runs over the maximum possible number of experiments
        for ii in tqdm(range(loops), disable=not pbar):
            # Get the data
            X = self.data.X
            Y = self.data.Y
            Yvar = self.data.Yvar

            # Simple fitting of a Gaussian process
            # using some pretty simple default values for things, which we
            # can always change later
            # TODO: enable the SVA to use Yvar when present
            if svf:
                Y = svf(X, Y).reshape(-1, 1)

            # The factory instantiates a model. It must have the from_default
            # method defined on it. It must also be compatible with the
            # from_default method (so for example, if your model has noisy
            # observations, you must use a noise-compatible GP, such as
            # a fixed-noise GP).
            args = [X, Y]
            if Yvar is None:
                args.append(Yvar)
            gp = model_factory.from_default(*args)

            # Fit the model
            self._fit(gp, fit_with, fit_kwargs)

            # Ask the model what to do next, we're also careful to check for
            # the best_f required keyword argument in the case of an EI
            # acquisition function
            # Update the internal data store with the next points
            state = self._ask(
                gp,
                Y,
                acquisition_function,
                acquisition_function_kwargs,
                optimize_acqf_kwargs,
            )

            # Append the history with everything we want to keep
            # Note that the complete state of the GP is saved in the
            # acquisition function model
            d = {
                "iteration": ii,
                "N": X.shape[0],
                "state": state,
                "easy_gp": deepcopy(gp),
            }

            # Optionally, we can run an additional optimization step on the GP
            # to get the maximum value of that GP in the experimental space.
            # This is useful for simulated campaigning and is disabled by
            # default.
            if optimize_gp:
                r = gp.optimize(
                    experiment=self,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                )
                r.pop("acquisition_function")
                d["optimize_gp"] = r

            self.history.append(d)

            self.update_data(state["next_points"])


@define
class DynamicExperiment(ExperimentMixin, CampaignBaseMixin):
    gp = field()
    properties = field()
    data = field()
    noise = field(default=None, validator=validators.instance_of(NOISE_TYPES))
    history = field(factory=lambda: ExperimentHistory())
    metadata = field(factory=dict)

    def _truth(self, x):
        mu, _ = self.gp.predict(x)
        return mu.reshape(-1, 1)


def get_dreamed_experiment(
    X,
    Y,
    domain,
    train_with="mll",
    adam_kwargs=None,
    ppd=20,
    num_restarts=150,
    raw_samples=150,
):
    """Creates an Experiment object from data alone. This is done via the
    following steps.

    1. A standard single task GP is fit to the data.
    2. A sample is drawn from that GP.
    3. That sample itself is fit by another GP.
    4. The mean of this GP is now the experiment in question. Running
    experiment(x) will produce the mean of this function as the prediction.

    Parameters
    ----------
    X, Y : np.ndarray
        The input and output data of the experiment. Since this will be
        approximated with a single task GP, Y must be one-dimensional.
    domain : np.ndarray
        The experimental domain of the problem. Must be of shape (2, d), where
        d is the dimensionality of the input.
    train_with : str, optional
        The training protocol for the GP approximator. Must be in
        {"Adam", "mll"}. Default is "mll".
    adam_kwargs : dict, optional
        Keyword arguments to pass to the Adam optimizer, if selected.
    ppd : int
        The number of points-per-dimension used in the dreamed GP.
    num_restarts, raw_samples : int
        Keyword arguments to pass to BoTorch's acquisition function optimizers.

    Returns
    -------
    DynamicExperiment
    """

    if train_with not in ["Adam", "mll"]:
        raise ValueError("train_with must be one of Adam or mll")

    gp = EasySingleTaskGP.from_default(X, Y)

    if train_with == "mll":
        gp.fit_mll()
    else:
        adam_kwargs = adam_kwargs if adam_kwargs is not None else dict()
        gp.fit_Adam(**adam_kwargs)

    dreamed_gp = gp.dream(ppd=ppd, domain=domain)
    n_input_dim = dreamed_gp.model.train_inputs[0].shape[1]

    properties = ExperimentProperties(
        n_input_dim=n_input_dim,
        n_output_dim=1,
        valid_domain=None,
        experimental_domain=domain,
    )
    data = ExperimentData(X=X, Y=Y)

    exp = DynamicExperiment(gp=dreamed_gp, properties=properties, data=data)
    exp.metadata["optima"] = exp.gp.optimize(
        domain=domain, num_restarts=num_restarts, raw_samples=raw_samples
    )
    return exp


# WARNING: This is not really functional yet
class MultimodalExperimentMixin(ExperimentMixin):
    @abstractproperty
    def n_modalities(self):
        raise NotImplementedError

    def _validate_input(self, x):
        # Ensure x has the right shape
        if not x.ndim == 2:
            raise ValueError(
                "x must have shape (N, d + 1), where d is the dimension of "
                "the feature, and the added extra dimension is the task index"
            )

        # Ensure that the second dimension of x is the right size for the
        # chosen experiment
        if not x.shape[1] == self.properties.n_input_dim + 1:
            raise ValueError(
                f"x second dimension must be {self.properties.n_input_dim+1}"
            )

        # Ensure that the input is in the bounds
        # Domain being None implies the domain is the entirety of the reals
        # This is a fast way to avoid the check
        if self.properties.valid_domain is not None:
            check1 = self.properties.valid_domain[0, :] <= x[:, :-1]
            check2 = x[:, :-1] <= self.properties.valid_domain[1, :]
            if not np.all(check1 & check2):
                raise ValueError("Some inputs x were not in the domain")

    def get_random_coordinates(self, n, domain=None, modality=0):
        x = super().get_random_coordinates(n, domain)
        n = x.shape[0]
        modality_array = np.zeros((n, 1)) + modality
        return np.concatenate([x, modality_array], axis=1)

    def get_dense_coordinates(self, ppd, modality=0, domain=None):
        """Gets a set of dense coordinates, augmented with the modality
        index, which defaults to 0.

        Parameters
        ----------
        ppd : int or list
            Points per dimension.
        modality : int
            Indexes the modality to use in multi-modal experiments.
        domain : np.ndarray, torch.tensor, optional
            The experimental domain of interest. If not provided defaults to
            that of the self experiment.

        Returns
        -------
        np.ndarray
        """

        x = super().get_dense_coordinates(ppd, domain=domain)
        n = x.shape[0]
        modality_array = np.zeros((n, 1)) + modality
        return np.concatenate([x, modality_array], axis=1)

    def initialize_data(self, n, modality=0, protocol="random"):
        """Initializes the X data via some provided protocol. Takes care to
        initialize the multi-modal.

        Parameters
        ----------
        n : int
            The number of points to use initially.
        modality : int
            The modality to use during initialization. Defaults to 0, which
            can be assumed to be the low-fidelity experiment inforomation.
        protocol : str, optional
        """

        if protocol == "random":
            X = self.get_random_coordinates(n=n)
        else:
            raise NotImplementedError(f"Unknown provided protocol {protocol}")
        self.update_data(X)

    def run_gp_experiment(
        self,
        max_experiments,
        modality_callback=lambda x: 0,
        task_feature=-1,
        svf=None,
        acquisition_function="UCB",
        acquisition_function_kwargs={"beta": 10.0},
        optimize_acqf_kwargs={"q": 1, "num_restarts": 20, "raw_samples": 100},
        optimize_results=False,
        pbar=True,
    ):
        """A special experiment runner which works on multimodal data. The
        user must specify a special callback function which provides the
        modality index as a function of iteration. By default, this is just
        0 for any iteration index (returns the low-fidelity experiment index).
        """

        if len(self.data.history) > 0:
            start = self.data.history[-1]["iteration"] + 1
        else:
            start = 0

        # First, check to see if the data is initialized
        if self.data.X is None:
            raise ValueError("You must initialize starting data first")

        # Run the experiment
        for ii in tqdm(range(start, start + max_experiments), disable=not pbar):
            # Get the data
            X = self.data.X
            Y = self.data.Y

            if X.shape[0] > max_experiments:
                break

            # Simple fitting of a Gaussian process
            # using some pretty simple default values for things, which we
            # can always change later
            if svf:
                new_target = np.empty(shape=(Y.shape[0], 1))
                # Assign each of the values based on the individual modal
                # experiments. The GP should take care of the rest
                for modality_index in range(self.n_modalities):
                    where = np.where(X[:, task_feature] == modality_index)[0]
                    if len(where) > 0:
                        target = svf(X[where, :], Y[where, :])
                        new_target[where, :] = target.reshape(-1, 1)
                target = new_target
            else:
                target = Y

            if target.ndim > 1 and target.shape[1] > 1:
                raise ValueError("Can only predict on a scalar target")
            if target.ndim == 1:
                target = target.reshape(-1, 1)
            gp = EasyMultiTaskGP.from_default(
                X, target, task_feature=task_feature
            )

            # Should be able to change how the gp is fit here
            gp.fit_mll()

            # Get the current modality of the experiment we're currently
            # running
            modality_index = modality_callback(ii)

            # Need to use a posterior transform here to tell the acquisition
            # function how to weight the multiple outputs
            weights = np.zeros(shape=(self.n_modalities,))
            weights[modality_index] = 1.0
            weights = torch.tensor(weights)
            transform = ScalarizedPosteriorTransform(weights=weights)
            acquisition_function_kwargs["posterior_transform"] = transform

            # Ask the model what to do next
            if acquisition_function in ["EI", "qEI"]:
                acquisition_function_kwargs["best_f"] = Y[
                    :, modality_index
                ].max()

            state = ask(
                gp.model,
                acquisition_function,
                bounds=self.properties.experimental_domain,
                acquisition_function_kwargs=acquisition_function_kwargs,
                optimize_acqf_kwargs=optimize_acqf_kwargs,
            )

            # Update the internal data store with the next points
            X2 = state["next_points"]
            self.update_data(X2)

            # Append the history with everything we want to keep
            # Note that the complete state of the GP is saved in the
            # acquisition function model

            d = {
                "iteration": ii,
                "next_points": state["next_points"],
                "value": state["value"],
                "acquisition_function": deepcopy(state["acquisition_function"]),
                "easy_gp": deepcopy(gp),
            }

            self.data.history.append(d)
