from abc import ABC, abstractmethod, abstractproperty
from attrs import define, field, frozen, validators
from typing import Optional, Union, Callable

from monty.json import MSONable
import numpy as np

from sva.utils import get_random_points, get_coordinates


@define
class ExperimentData(MSONable):
    """Container for sampled data during experiments. Does essentially nothing
    except hold the data and provide methods for dealing with it, including
    updating it and saving it to disk. Note that data contained here must
    always be two-dimensional (x.ndim == 2)."""

    X: np.ndarray = field(default=None)
    Y: np.ndarray = field(default=None)

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
            raise NotImplementedError(
                f"Unknown provided protocol {protocol}"
            )

        self.update_data(X)

    def __call__(self, x):
        """The (possibly noisy) result of the experiment."""

        if self.noise is None:
            return self.truth(x)

        if isinstance(self.noise, Callable):
            noise = self.noise(x)
            return self.truth(x) + np.random.normal(scale=noise, size=x.shape)

        if isinstance(self.noise, float):
            pass
        elif isinstance(self.noise, np.ndarray):
            assert self.noise.ndim == 1
            assert len(self.noise) == self.properties.n_output_dim
        elif isinstance(self.noise, list):
            assert len(self.noise) == self.properties.n_output_dim
        else:
            raise ValueError("Incompatible noise type")

        return self.truth(x) + np.random.normal(scale=self.noise, size=x.shape)


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
