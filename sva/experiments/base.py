from abc import ABC, abstractmethod, abstractproperty
from copy import deepcopy
from typing import Callable, Optional, Union

import numpy as np
from attrs import define, field, frozen, validators
from monty.json import MSONable
from tqdm import tqdm

from sva.models.gp.bo import ask
from sva.models.gp.single_task_gp import EasySingleTaskGP
from sva.utils import get_coordinates, get_random_points


@define
class ExperimentData(MSONable):
    """Container for sampled data during experiments. Does essentially nothing
    except hold the data and provide methods for dealing with it, including
    updating it and saving it to disk. Note that data contained here must
    always be two-dimensional (x.ndim == 2)."""

    X: np.ndarray = field(default=None)
    Y: np.ndarray = field(default=None)
    history: list = field(factory=list)

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
            new_Y = experiment(self.X[-diff:])
            self.Y = np.concatenate([self.Y, new_Y], axis=0)
        else:
            self.Y = experiment(self.X)


NOISE_TYPES = (Callable, float, np.ndarray, list, type(None))


class ExperimentMixin(ABC):
    """Abstract base class for a source of truth. These sources of truth are
    for a single modality."""

    @abstractproperty
    def noise(self):
        ...

    @abstractproperty
    def properties(self):
        ...

    @abstractproperty
    def data(self):
        ...

    @abstractmethod
    def _truth(self, x: np.ndarray) -> np.ndarray:
        """Vectorized truth function. Should return the value of the truth
        function for all rows of the provided x input."""

        raise NotImplementedError

    def _dtruth(self, x: np.ndarray) -> np.ndarray:
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

    def truth(self, x: np.ndarray) -> np.ndarray:
        """Access the noiseless results of an "experiment"."""

        self._validate_input(x)
        y = self._truth(x)
        self._validate_output(y)
        return y

    def dtruth(self, x: np.ndarray) -> np.ndarray:
        """Access the derivative of the noiseless results of the experiment."""

        self._validate_input(x)
        y = self._dtruth(x)
        self._validate_output(y)
        return y

    def random_inputs(self, n=1, seed=None):
        """Runs n random input points."""

        return get_random_points(
            self.properties.experimental_domain, n=n, seed=seed
        )

    def get_dense_coordinates(self, ppd, domain=None):
        """Gets a set of dense coordinates.

        Parameters
        ----------
        ppd : int or list
            Points per dimension.

        Returns
        -------
        np.ndarray
        """

        if domain is None:
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

    def initialize_data(self, n, seed=None, protocol="random"):
        """Initializes the X data via some provided protocol.

        Parameters
        ----------
        n : int
            The number of points to use initially.
        seed : int
            The random seed to ensure reproducibility.
        protocol : str, optional
        """

        if protocol == "random":
            X = self.random_inputs(n=n, seed=seed)
        else:
            raise NotImplementedError(f"Unknown provided protocol {protocol}")

        self.update_data(X)

    def __call__(self, x):
        """The (possibly noisy) result of the experiment."""

        if self.noise is None:
            return self.truth(x)

        if isinstance(self.noise, Callable):
            noise = self.noise(x)
            y = self.truth(x)
            return y + np.random.normal(scale=noise, size=y.shape)

        if isinstance(self.noise, float):
            pass
        elif isinstance(self.noise, np.ndarray):
            assert self.noise.ndim == 1
            assert len(self.noise) == self.properties.n_output_dim
        elif isinstance(self.noise, list):
            assert len(self.noise) == self.properties.n_output_dim
        else:
            raise ValueError("Incompatible noise type")

        y = self.truth(x)
        return y + np.random.normal(scale=self.noise, size=y.shape)

    def run_gp_experiment(
        self,
        max_experiments,
        svf=None,
        acquisition_function="UCB",
        acquisition_function_kwargs={"beta": 10.0},
        optimize_acqf_kwargs={"q": 1, "num_restarts": 20, "raw_samples": 100},
        pbar=True,
    ):
        if len(self.data.history) > 0:
            start = self.data.history[-1]["iteration"] + 1
        else:
            start = 0

        # First, check to see if the data is initialized
        if self.data.X is None:
            raise ValueError("You must initialize starting data first")

        # Run the experiment
        for ii in tqdm(
            range(start, start + max_experiments), disable=not pbar
        ):
            # Get the data
            X = self.data.X
            Y = self.data.Y

            if X.shape[0] > max_experiments:
                break

            # Simple fitting of a Gaussian process
            # using some pretty simple default values for things, which we
            # can always change later
            target = Y if not svf else svf(X, Y)
            if target.ndim > 1 and target.shape[1] > 1:
                raise ValueError("Can only predict on a scalar target")
            if target.ndim == 1:
                target = target.reshape(-1, 1)
            gp = EasySingleTaskGP.from_default(X, target)

            # Should be able to change how the gp is fit here
            gp.fit_mll()

            # Ask the model what to do next
            if acquisition_function in ["EI", "qEI"]:
                acquisition_function_kwargs["best_f"] = Y.max()
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
            self.data.history.append(
                {
                    "iteration": ii,
                    "next_points": state["next_points"],
                    "value": state["value"],
                    "acquisition_function": deepcopy(
                        state["acquisition_function"]
                    ),
                    "easy_gp": deepcopy(gp),
                }
            )


class MultimodalExperimentMixin(ExperimentMixin):

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

    def get_dense_coordinates(self, ppd, modality=0, domain=None):
        """Gets a set of dense coordinates, augmented with the modality
        index, which defaults to 0.

        Parameters
        ----------
        ppd : int or list
            Points per dimension.
        modality : int
            Indexes the modality to use in multi-modal experiments. 

        Returns
        -------
        np.ndarray
        """

        x = super().get_dense_coordinates(ppd, domain=domain)
        n = x.shape[0]
        modality_array = np.zeros((n, 1)) + modality
        return np.concatenate([x, modality_array], axis=1)

@frozen
class ExperimentProperties(MSONable):
    """Defines the core set of experiment properties, which are frozen after
    setting."""

    n_input_dim = field(validator=validators.instance_of(int))
    n_output_dim = field(validator=validators.instance_of(int))
    valid_domain = field(
        validator=validators.instance_of((type(None), np.ndarray))
    )
    experimental_domain = field(validator=validators.instance_of(np.ndarray))
